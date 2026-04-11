"""CI-safe eval tests for the v2 architecture.

All tests use mocks — no live LLM calls. The file is structured in five sections:
  1. Routing tests (parametrized)
  2. Golden example routing validation
  3. State shape tests
  4. Age-appropriateness prompt rule verification
  5. Node output shape tests (mocked)
"""

import json
import inspect
import operator
from typing import get_type_hints
from unittest.mock import MagicMock, patch

import pytest
from langgraph.graph import END

from tutor.graph import entry_router, route_after_classify, route_after_reasoning
from tutor.state import TutorState
from tutor import nodes

# ---------------------------------------------------------------------------
# Load golden examples
# ---------------------------------------------------------------------------

with open("tests/evals/fixtures/golden_examples.json", encoding="utf-8") as f:
    GOLDEN = json.load(f)


# ---------------------------------------------------------------------------
# Section 1: Routing tests (parametrized)
# ---------------------------------------------------------------------------

ROUTING_CASES = [
    ("resume_session",  {"session_paused": True},   "entry_router",           "resume_session"),
    ("classify_normal", {"session_paused": False},  "entry_router",           "classify_question"),
    ("factual_route",   {"question_type": "factual"},   "route_after_classify", "factual_react_loop"),
    ("reasoning_route", {"question_type": "reasoning"}, "route_after_classify", "reasoning_react_loop"),
    ("escalate_route",  {"session_paused": True},   "route_after_reasoning",  "escalate"),
    ("reasoning_done",  {"session_paused": False},  "route_after_reasoning",  END),
]


@pytest.mark.parametrize("name,state,router_name,expected", ROUTING_CASES)
def test_routing(name, state, router_name, expected):
    routers = {
        "entry_router": entry_router,
        "route_after_classify": route_after_classify,
        "route_after_reasoning": route_after_reasoning,
    }
    result = routers[router_name](state)
    assert result == expected, (
        f"{router_name}({state}) → {result!r}, expected {expected!r}"
    )


# ---------------------------------------------------------------------------
# Section 2: Golden example routing validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "scenario",
    [s for s in GOLDEN if s["id"] in ("resume_session", "distress_detection")],
)
def test_golden_entry_router(scenario):
    state = scenario["input_state"]
    if state.get("session_paused"):
        assert entry_router(state) == "resume_session"


@pytest.mark.parametrize(
    "scenario",
    [s for s in GOLDEN if s["id"] in ("factual_web_search", "reasoning_first_hint")],
)
def test_golden_classify_routing(scenario):
    state = scenario["input_state"]
    result = route_after_classify(state)
    expected = scenario["expected_node"]
    assert result == expected


# ---------------------------------------------------------------------------
# Section 3: State shape tests
# ---------------------------------------------------------------------------

def test_state_has_strategies_tried():
    hints = get_type_hints(TutorState, include_extras=True)
    assert "strategies_tried" in hints


def test_state_has_concepts_needing_review():
    hints = get_type_hints(TutorState, include_extras=True)
    assert "concepts_needing_review" in hints


def test_state_has_conversation_history():
    hints = get_type_hints(TutorState, include_extras=True)
    assert "conversation_history" in hints


def test_state_no_hints_given():
    hints = get_type_hints(TutorState, include_extras=True)
    assert "hints_given" not in hints


def test_state_no_session_complete():
    hints = get_type_hints(TutorState, include_extras=True)
    assert "session_complete" not in hints


# ---------------------------------------------------------------------------
# Section 4: Age-appropriateness prompt rule verification
# ---------------------------------------------------------------------------

def test_age_rule_in_factual_loop():
    source = inspect.getsource(nodes.factual_react_loop)
    assert "_AGE_RULE" in source


def test_age_rule_in_reasoning_loop():
    source = inspect.getsource(nodes.reasoning_react_loop)
    assert "_AGE_RULE" in source


def test_age_rule_constant_defined():
    assert hasattr(nodes, "_AGE_RULE")
    assert "10 words" in nodes._AGE_RULE


def test_classify_prompt_arithmetic_is_reasoning():
    """Classifier prompt must explicitly state arithmetic is always reasoning."""
    source = inspect.getsource(nodes.classify_question)
    assert "arithmetic" in source.lower() or "calculation" in source.lower()


# ---------------------------------------------------------------------------
# Section 5: Node output shape tests (mocked)
# ---------------------------------------------------------------------------

def _full_state(**overrides) -> dict:
    """Return a minimal valid state dict, with optional field overrides."""
    base = {
        "student_input": "",
        "concept": "",
        "question_type": "",
        "strategies_tried": [],
        "conversation_history": [],
        "current_response": "",
        "session_paused": False,
        "concepts_needing_review": [],
    }
    base.update(overrides)
    return base


def test_classify_question_returns_required_keys():
    with patch("tutor.nodes._classifier_llm") as mock_llm:
        mock_llm.invoke.return_value = MagicMock(
            content="factual|capital of France", tool_calls=[]
        )
        result = nodes.classify_question(
            _full_state(student_input="What is the capital of France?")
        )
    assert "question_type" in result
    assert "concept" in result


def test_classify_question_factual_type():
    with patch("tutor.nodes._classifier_llm") as mock_llm:
        mock_llm.invoke.return_value = MagicMock(
            content="factual|spider legs", tool_calls=[]
        )
        result = nodes.classify_question(
            _full_state(student_input="How many legs does a spider have?")
        )
    assert result["question_type"] == "factual"


def test_classify_question_reasoning_type():
    with patch("tutor.nodes._classifier_llm") as mock_llm:
        mock_llm.invoke.return_value = MagicMock(
            content="reasoning|addition to 20", tool_calls=[]
        )
        result = nodes.classify_question(
            _full_state(student_input="What is 7 + 8?")
        )
    assert result["question_type"] == "reasoning"


def test_escalate_sets_session_paused():
    result = nodes.escalate(
        _full_state(concept="addition", session_paused=False)
    )
    assert result["session_paused"] is True
    assert result["current_response"] != ""


def test_escalate_response_is_non_empty():
    result = nodes.escalate(_full_state())
    assert isinstance(result["current_response"], str)
    assert len(result["current_response"]) > 10


def test_resume_session_clears_session_paused():
    with patch("tutor.nodes._format_llm") as mock_llm:
        mock_llm.invoke.return_value = MagicMock(
            content="Welcome back! We were working on addition.", tool_calls=[]
        )
        result = nodes.resume_session(
            _full_state(concept="addition", session_paused=True)
        )
    assert result["session_paused"] is False


def test_resume_session_returns_response():
    with patch("tutor.nodes._format_llm") as mock_llm:
        mock_llm.invoke.return_value = MagicMock(
            content="Welcome back! We were working on addition.", tool_calls=[]
        )
        result = nodes.resume_session(
            _full_state(concept="addition", session_paused=True)
        )
    assert "current_response" in result
    assert result["current_response"] != ""


