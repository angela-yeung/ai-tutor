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
