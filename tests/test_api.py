"""Tests for the FastAPI backend."""

import json
import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestChatRequest:
    def test_valid_request(self):
        from api.schemas import ChatRequest
        req = ChatRequest(thread_id="abc-123", message="What is 2+2?")
        assert req.thread_id == "abc-123"
        assert req.message == "What is 2+2?"

    def test_missing_thread_id_raises(self):
        from pydantic import ValidationError
        from api.schemas import ChatRequest
        with pytest.raises(ValidationError):
            ChatRequest(message="hello")

    def test_missing_message_raises(self):
        from pydantic import ValidationError
        from api.schemas import ChatRequest
        with pytest.raises(ValidationError):
            ChatRequest(thread_id="abc")

    def test_empty_message_is_valid(self):
        from api.schemas import ChatRequest
        req = ChatRequest(thread_id="abc", message="")
        assert req.message == ""


# ---------------------------------------------------------------------------
# Streaming helper tests
# ---------------------------------------------------------------------------

class TestBuildInput:
    def test_new_thread_seeds_state(self):
        from tutor.graph import build_graph
        from api.streaming import _build_input
        graph = build_graph()
        result = _build_input(graph, "brand-new-thread-xyz-987", "Hello")
        assert result["student_input"] == "Hello"
        assert result["strategies_tried"] == []
        assert result["conversation_history"] == []
        assert result["session_paused"] is False
        assert result["concepts_needing_review"] == []

    def test_student_input_always_present(self):
        from tutor.graph import build_graph
        from api.streaming import _build_input
        graph = build_graph()
        result = _build_input(graph, "another-new-thread-abc", "What is 3+3?")
        assert result["student_input"] == "What is 3+3?"


class TestExtractMetadata:
    def _make_snapshot(self, values):
        class FakeSnapshot:
            pass
        snap = FakeSnapshot()
        snap.values = values
        return snap

    def test_extracts_all_fields(self):
        from api.streaming import _extract_metadata
        snap = self._make_snapshot({
            "session_paused": True,
            "concept": "fractions",
            "concepts_needing_review": ["fractions"],
            "conversation_history": [{"role": "user", "content": "Help!"}],
        })
        result = _extract_metadata(snap)
        assert result == {
            "session_paused": True,
            "concept": "fractions",
            "concepts_needing_review": ["fractions"],
            "conversation_history": [{"role": "user", "content": "Help!"}],
        }

    def test_missing_fields_return_defaults(self):
        from api.streaming import _extract_metadata
        snap = self._make_snapshot({})
        result = _extract_metadata(snap)
        assert result["session_paused"] is False
        assert result["concept"] == ""
        assert result["concepts_needing_review"] == []
        assert result["conversation_history"] == []

    def test_partial_fields(self):
        from api.streaming import _extract_metadata
        snap = self._make_snapshot({"concept": "multiplication"})
        result = _extract_metadata(snap)
        assert result["concept"] == "multiplication"
        assert result["session_paused"] is False


# ---------------------------------------------------------------------------
# Endpoint integration tests
# ---------------------------------------------------------------------------

async def _fake_stream(graph, thread_id: str, message: str):
    """Minimal fake SSE stream for endpoint tests — no LLM calls."""
    yield 'event: token\ndata: {"chunk": "Hello"}\n\n'
    yield 'event: token\ndata: {"chunk": " there"}\n\n'
    yield (
        'event: done\ndata: '
        + json.dumps({
            "session_paused": False,
            "concept": "addition",
            "concepts_needing_review": [],
            "conversation_history": [],
        })
        + "\n\n"
    )


@pytest.fixture
def client():
    from api.main import app
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_returns_ok_json(self, client):
        r = client.get("/health")
        assert r.json() == {"status": "ok"}


class TestChatEndpoint:
    def test_returns_200(self, client):
        with patch("api.router.stream_chat", side_effect=_fake_stream):
            r = client.post("/chat", json={"thread_id": "t1", "message": "hi"})
        assert r.status_code == 200

    def test_content_type_is_event_stream(self, client):
        with patch("api.router.stream_chat", side_effect=_fake_stream):
            r = client.post("/chat", json={"thread_id": "t1", "message": "hi"})
        assert "text/event-stream" in r.headers["content-type"]

    def test_stream_contains_token_events(self, client):
        with patch("api.router.stream_chat", side_effect=_fake_stream):
            r = client.post("/chat", json={"thread_id": "t1", "message": "hi"})
        events = [e for e in r.text.split("\n\n") if e.strip()]
        token_events = [e for e in events if e.startswith("event: token")]
        assert len(token_events) == 2

    def test_stream_contains_exactly_one_done_event(self, client):
        with patch("api.router.stream_chat", side_effect=_fake_stream):
            r = client.post("/chat", json={"thread_id": "t1", "message": "hi"})
        events = [e for e in r.text.split("\n\n") if e.strip()]
        done_events = [e for e in events if e.startswith("event: done")]
        assert len(done_events) == 1

    def test_done_event_payload_has_required_keys(self, client):
        with patch("api.router.stream_chat", side_effect=_fake_stream):
            r = client.post("/chat", json={"thread_id": "t1", "message": "hi"})
        events = [e for e in r.text.split("\n\n") if e.strip()]
        done_event = next(e for e in events if e.startswith("event: done"))
        data_line = done_event.split("\n")[1]
        payload = json.loads(data_line.replace("data: ", ""))
        assert "session_paused" in payload
        assert "concept" in payload
        assert "concepts_needing_review" in payload
        assert "conversation_history" in payload

    def test_invalid_request_body_returns_422(self, client):
        r = client.post("/chat", json={"message": "missing thread_id"})
        assert r.status_code == 422
