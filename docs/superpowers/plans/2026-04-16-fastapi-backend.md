# FastAPI Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wrap the existing LangGraph AI tutor in a FastAPI HTTP server with SSE streaming, Redis session persistence, and Docker containerisation ready for Railway deployment.

**Architecture:** A new `api/` package acts as a thin adapter over the unchanged `tutor/` package. `tutor/graph.py` gains a `build_graph(checkpointer=None)` factory so the API can inject a `RedisSaver` while tests keep using `MemorySaver`. SSE streaming uses LangGraph's `astream_events()` to emit tokens in real time, followed by a `done` event carrying session metadata.

**Tech Stack:** FastAPI 0.115+, uvicorn, LangGraph `astream_events`, `langgraph-checkpoint-redis`, Redis 7, httpx (tests), Docker, docker-compose

**Spec:** `docs/superpowers/specs/2026-04-16-fastapi-backend-design.md`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `tutor/graph.py` | Modify | Add `build_graph(checkpointer=None)` factory; keep `tutor = build_graph()` for CLI |
| `requirements.txt` | Modify | Add fastapi, uvicorn, langgraph-checkpoint-redis, redis, httpx |
| `api/__init__.py` | Create | Empty package marker |
| `api/schemas.py` | Create | Pydantic `ChatRequest` model |
| `api/streaming.py` | Create | `_build_input`, `_extract_metadata`, `stream_chat` async SSE generator |
| `api/main.py` | Create | FastAPI app, CORS middleware, lifespan (Redis or MemorySaver) |
| `api/router.py` | Create | `GET /health`, `POST /chat` endpoints |
| `tests/test_api.py` | Create | TestClient integration tests (health, chat content-type, SSE events) |
| `Dockerfile` | Create | Single-stage Python 3.12-slim image |
| `docker-compose.yml` | Create | Local dev: api + redis services |
| `.dockerignore` | Create | Exclude .env, __pycache__, .git, tests |

---

## Task 1: Refactor tutor/graph.py — add build_graph() factory

**Files:**
- Modify: `tutor/graph.py`

The current file compiles the graph at module level with a hardcoded `MemorySaver`. We need a factory function so the API can inject a `RedisSaver`. The module-level `tutor = build_graph()` must stay so that `cli.py` (`from tutor.graph import tutor`) and all existing tests continue to work without changes.

- [ ] **Step 1: Run existing tests to establish baseline**

```bash
pytest tests/ -v
```

Expected: all tests pass (green). If any fail, do not proceed — fix them first.

- [ ] **Step 2: Rewrite tutor/graph.py with build_graph() factory**

Replace the entire file with:

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from tutor.state import TutorState
from tutor.nodes import (
    classify_question,
    factual_react_loop,
    reasoning_react_loop,
    escalate,
    resume_session,
)


# ---------------------------------------------------------------------------
# Routing functions (pure — no LLM calls, safe to unit test)
# ---------------------------------------------------------------------------

def entry_router(state: TutorState) -> str:
    if state.get("session_paused"):
        return "resume_session"
    return "classify_question"


def route_after_classify(state: TutorState) -> str:
    if state.get("question_type") == "factual":
        return "factual_react_loop"
    return "reasoning_react_loop"


def route_after_reasoning(state: TutorState) -> str:
    if state.get("session_paused"):
        return "escalate"
    return END


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

def build_graph(checkpointer=None):
    """Compile the tutor graph with the given checkpointer.

    Pass a RedisSaver in production; omit (or pass None) to use MemorySaver.
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    builder = StateGraph(TutorState)

    builder.add_node("classify_question", classify_question)
    builder.add_node("factual_react_loop", factual_react_loop)
    builder.add_node("reasoning_react_loop", reasoning_react_loop)
    builder.add_node("escalate", escalate)
    builder.add_node("resume_session", resume_session)

    builder.add_conditional_edges(START, entry_router)
    builder.add_conditional_edges("classify_question", route_after_classify)
    builder.add_conditional_edges("reasoning_react_loop", route_after_reasoning)
    builder.add_edge("resume_session", END)
    builder.add_edge("factual_react_loop", END)
    builder.add_edge("escalate", END)

    return builder.compile(checkpointer=checkpointer)


# Module-level instance — used by cli.py and existing tests
tutor = build_graph()
```

- [ ] **Step 3: Run existing tests to verify nothing broke**

```bash
pytest tests/ -v
```

Expected: same results as Step 1 — all tests still pass.

- [ ] **Step 4: Commit**

```bash
git add tutor/graph.py
git commit -m "refactor: extract build_graph() factory for injectable checkpointer"
```

---

## Task 2: Update requirements.txt

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add new dependencies**

Append to `requirements.txt`:

```
fastapi>=0.115
uvicorn[standard]>=0.30
langgraph-checkpoint-redis>=0.1
redis>=5.0
httpx>=0.27
```

- [ ] **Step 2: Install and verify**

```bash
pip install -r requirements.txt
```

Expected: installs without errors. If `langgraph-checkpoint-redis` is not found on PyPI, try `langgraph-redis` instead and update the line accordingly.

After install, verify the Redis import resolves:

```bash
python -c "from langgraph.checkpoint.redis import RedisSaver; print('OK')"
```

Expected: `OK` (if it fails, check the correct import path in the installed package: `python -c "import langgraph.checkpoint.redis; print(dir(langgraph.checkpoint.redis))"`)

- [ ] **Step 3: Run existing tests**

```bash
pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore: add fastapi, uvicorn, redis, httpx dependencies"
```

---

## Task 3: Create api/schemas.py — Pydantic models

**Files:**
- Create: `api/__init__.py`
- Create: `api/schemas.py`
- Create (tests): `tests/test_api.py` (schemas section)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_api.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_api.py::TestChatRequest -v
```

Expected: `ModuleNotFoundError: No module named 'api'` or `ImportError`. That's the correct failure.

- [ ] **Step 3: Create api/__init__.py**

Create `api/__init__.py` as an empty file.

- [ ] **Step 4: Create api/schemas.py**

```python
from pydantic import BaseModel


class ChatRequest(BaseModel):
    thread_id: str
    message: str
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_api.py::TestChatRequest -v
```

Expected: 4 tests pass.

- [ ] **Step 6: Commit**

```bash
git add api/__init__.py api/schemas.py tests/test_api.py
git commit -m "feat: add api package with ChatRequest schema"
```

---

## Task 4: Create api/streaming.py — SSE helpers and generator

**Files:**
- Create: `api/streaming.py`
- Modify: `tests/test_api.py` (add streaming helper tests)

- [ ] **Step 1: Write the failing tests**

Add these test classes to `tests/test_api.py` (append after the existing content):

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_api.py::TestBuildInput tests/test_api.py::TestExtractMetadata -v
```

Expected: `ImportError: cannot import name '_build_input' from 'api.streaming'`

- [ ] **Step 3: Create api/streaming.py**

```python
"""SSE streaming utilities for the tutor API."""

import json
from typing import AsyncIterator

from tutor.graph import build_graph


def _build_input(graph, thread_id: str, message: str) -> dict:
    """Build the LangGraph input dict, seeding list fields on the first turn."""
    config = {"configurable": {"thread_id": thread_id}}
    state_update = {"student_input": message}
    if not graph.get_state(config).values:
        state_update.update({
            "strategies_tried": [],
            "concepts_needing_review": [],
            "conversation_history": [],
            "session_paused": False,
        })
    return state_update


def _extract_metadata(state_snapshot) -> dict:
    """Pull session metadata from a LangGraph state snapshot."""
    v = state_snapshot.values
    return {
        "session_paused": v.get("session_paused", False),
        "concept": v.get("concept", ""),
        "concepts_needing_review": v.get("concepts_needing_review", []),
        "conversation_history": v.get("conversation_history", []),
    }


async def stream_chat(graph, thread_id: str, message: str) -> AsyncIterator[str]:
    """Async generator that yields SSE-formatted strings.

    Emits:
      event: token  — one per LLM output token (skips classify_question tokens)
      event: done   — once, with session metadata after the graph finishes
      event: error  — if an exception occurs
    """
    config = {"configurable": {"thread_id": thread_id}}
    input_state = _build_input(graph, thread_id, message)

    try:
        async for event in graph.astream_events(input_state, config, version="v2"):
            if event["event"] == "on_chat_model_stream":
                node = event.get("metadata", {}).get("langgraph_node", "")
                if node == "classify_question":
                    continue
                chunk = event["data"]["chunk"].content
                if chunk:
                    yield f"event: token\ndata: {json.dumps({'chunk': chunk})}\n\n"

        state = await graph.aget_state(config)
        yield f"event: done\ndata: {json.dumps(_extract_metadata(state))}\n\n"
    except Exception:
        yield f"event: error\ndata: {json.dumps({'message': 'Something went wrong. Please try again.'})}\n\n"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_api.py::TestBuildInput tests/test_api.py::TestExtractMetadata -v
```

Expected: 7 tests pass.

- [ ] **Step 5: Run full test suite to check nothing regressed**

```bash
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add api/streaming.py tests/test_api.py
git commit -m "feat: add streaming helpers and SSE generator"
```

---

## Task 5: Create api/main.py and api/router.py — FastAPI app and endpoints

**Files:**
- Create: `api/router.py`
- Create: `api/main.py`
- Modify: `tests/test_api.py` (add endpoint integration tests)

- [ ] **Step 1: Write the failing integration tests**

Append to `tests/test_api.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_api.py::TestHealthEndpoint tests/test_api.py::TestChatEndpoint -v
```

Expected: `ImportError: cannot import name 'app' from 'api.main'`

- [ ] **Step 3: Create api/router.py**

```python
"""API route handlers."""

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from api.schemas import ChatRequest
from api.streaming import stream_chat

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/chat")
async def chat(req: ChatRequest, request: Request):
    tutor = request.app.state.tutor
    return StreamingResponse(
        stream_chat(tutor, req.thread_id, req.message),
        media_type="text/event-stream",
    )
```

- [ ] **Step 4: Create api/main.py**

```python
"""FastAPI application factory."""

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.router import router
from tutor.graph import build_graph

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        from langgraph.checkpoint.redis import RedisSaver
        checkpointer = RedisSaver.from_conn_string(redis_url)
        app.state.tutor = build_graph(checkpointer)
    else:
        app.state.tutor = build_graph()  # MemorySaver — local dev / tests
    yield


app = FastAPI(title="AI Tutor API", lifespan=lifespan)

cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_api.py::TestHealthEndpoint tests/test_api.py::TestChatEndpoint -v
```

Expected: 8 tests pass.

- [ ] **Step 6: Run full test suite**

```bash
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 7: Smoke-test the server manually**

In one terminal:

```bash
uvicorn api.main:app --reload
```

In another terminal:

```bash
curl -s http://localhost:8000/health
```

Expected: `{"status":"ok"}`

```bash
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"thread_id":"test-123","message":"What is 2 plus 2?"}'
```

Expected: a stream of `event: token` lines followed by `event: done`.

- [ ] **Step 8: Commit**

```bash
git add api/router.py api/main.py tests/test_api.py
git commit -m "feat: add FastAPI app with /health and /chat SSE endpoints"
```

---

## Task 6: Docker — Dockerfile, docker-compose.yml, .dockerignore

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`
- Create: `.dockerignore`

No automated tests for Docker config. Manual verification steps are provided.

- [ ] **Step 1: Create .dockerignore**

```
.env
.git
.gitignore
__pycache__
*.pyc
*.pyo
.pytest_cache
tests/
docs/
*.md
.worktrees
```

- [ ] **Step 2: Create Dockerfile**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 3: Create docker-compose.yml**

```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    env_file: .env
    environment:
      REDIS_URL: redis://redis:6379
    depends_on:
      redis:
        condition: service_healthy

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5
```

- [ ] **Step 4: Verify Docker build (requires Docker Desktop running)**

```bash
docker build -t ai-tutor-api .
```

Expected: builds successfully, no errors.

- [ ] **Step 5: Verify docker-compose starts cleanly**

```bash
docker compose up --build
```

Expected: redis starts, api starts, `Application startup complete.` logged.

Test health check in another terminal:

```bash
curl http://localhost:8000/health
```

Expected: `{"status":"ok"}`

Stop with `Ctrl+C`.

- [ ] **Step 6: Commit**

```bash
git add Dockerfile docker-compose.yml .dockerignore
git commit -m "feat: add Dockerfile and docker-compose for local dev and Railway deployment"
```

---

## Verification

After all tasks complete, run the full suite one final time:

```bash
pytest tests/ -v
```

Expected: all tests pass, no failures.

End-to-end SSE smoke test with docker-compose (Redis active):

```bash
docker compose up -d
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"thread_id":"smoke-test-001","message":"What is 3 plus 3?"}'
```

Expected: token events stream in, followed by `event: done` with `"concept"`, `"session_paused"`, etc.

---

## Railway Deployment Checklist

After the branch is merged:

1. Connect repo to Railway, point to `Dockerfile`
2. Add Railway Redis plugin — Railway auto-sets `REDIS_URL`
3. Set environment variables: `OPENAI_API_KEY`, `TAVILY_API_KEY`, `CORS_ORIGINS` (your frontend URL)
4. Deploy — Railway will build the Docker image and run `uvicorn api.main:app --host 0.0.0.0 --port 8000`
5. Hit `https://<your-service>.railway.app/health` to confirm it's live
