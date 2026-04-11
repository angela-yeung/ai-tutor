"""Tests for routing functions in tutor/graph.py.

All routing functions are pure (no LLM calls), so no mocking is needed.
"""

import pytest
from langgraph.graph import END
from tutor.graph import entry_router, route_after_classify, route_after_reasoning


# ---------------------------------------------------------------------------
# entry_router
# ---------------------------------------------------------------------------

def test_entry_router_paused_goes_to_resume():
    state = {
        "session_paused": True,
        "strategies_tried": [],
        "conversation_history": [],
        "concepts_needing_review": [],
    }
    assert entry_router(state) == "resume_session"


def test_entry_router_normal_goes_to_classify():
    state = {
        "session_paused": False,
        "strategies_tried": [],
        "conversation_history": [],
        "concepts_needing_review": [],
    }
    assert entry_router(state) == "classify_question"


def test_entry_router_no_paused_key_goes_to_classify():
    assert entry_router({}) == "classify_question"


def test_entry_router_paused_false_explicit():
    """Explicit False on session_paused still routes to classify_question."""
    assert entry_router({"session_paused": False}) == "classify_question"


# ---------------------------------------------------------------------------
# route_after_classify
# ---------------------------------------------------------------------------

def test_route_after_classify_factual():
    assert route_after_classify({"question_type": "factual"}) == "factual_react_loop"


def test_route_after_classify_reasoning():
    assert route_after_classify({"question_type": "reasoning"}) == "reasoning_react_loop"


def test_route_after_classify_unknown_defaults_to_reasoning():
    assert route_after_classify({"question_type": "unknown"}) == "reasoning_react_loop"


def test_route_after_classify_missing_key_defaults_to_reasoning():
    assert route_after_classify({}) == "reasoning_react_loop"


def test_route_after_classify_empty_string_defaults_to_reasoning():
    assert route_after_classify({"question_type": ""}) == "reasoning_react_loop"


# ---------------------------------------------------------------------------
# route_after_reasoning
# ---------------------------------------------------------------------------

def test_route_after_reasoning_paused_goes_to_escalate():
    assert route_after_reasoning({"session_paused": True}) == "escalate"


def test_route_after_reasoning_normal_goes_to_end():
    assert route_after_reasoning({"session_paused": False}) == END


def test_route_after_reasoning_no_paused_key_goes_to_end():
    assert route_after_reasoning({}) == END


def test_route_after_reasoning_paused_false_goes_to_end():
    """Explicit False routes to END."""
    assert route_after_reasoning({"session_paused": False}) == END
