"""Tests for routing functions in tutor/graph.py.

All routing functions are pure (no LLM calls), so no mocking is needed.
"""

import pytest
from tutor.graph import (
    entry_router,
    route_after_classify,
    route_after_reasoning,
    route_after_input_guard,
)


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


def test_route_after_reasoning_normal_goes_to_output_guard():
    assert route_after_reasoning({"session_paused": False}) == "output_guard"


def test_route_after_reasoning_no_paused_key_goes_to_output_guard():
    assert route_after_reasoning({}) == "output_guard"


def test_route_after_reasoning_paused_false_goes_to_output_guard():
    """Explicit False routes to output_guard."""
    assert route_after_reasoning({"session_paused": False}) == "output_guard"


# ---------------------------------------------------------------------------
# route_after_input_guard
# ---------------------------------------------------------------------------

def test_route_after_input_guard_blocked():
    assert route_after_input_guard({"input_blocked": True}) == "output_guard"


def test_route_after_input_guard_clean():
    assert route_after_input_guard({"input_blocked": False}) == "entry_router"


def test_route_after_input_guard_missing_key_goes_to_entry_router():
    assert route_after_input_guard({}) == "entry_router"
