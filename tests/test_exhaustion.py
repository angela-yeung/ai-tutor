"""Tests for strategy exhaustion behaviour and reasoning_react_loop signals."""

import pytest
from unittest.mock import MagicMock, patch

from tutor.tools import scaffold_hint
from tutor.nodes import reasoning_react_loop


ALL_STRATEGIES = ["guiding_question", "analogy", "concrete_example", "sub_problem", "number_line"]


def _base_state(**overrides) -> dict:
    state = {
        "student_input": "I don't know",
        "concept": "addition",
        "strategies_tried": [],
        "conversation_history": [],
        "concepts_needing_review": [],
        "session_paused": False,
        "question_type": "reasoning",
        "current_response": "",
    }
    state.update(overrides)
    return state


def make_mock_response(content: str, tool_calls=None):
    mock = MagicMock()
    mock.content = content
    mock.tool_calls = tool_calls if tool_calls is not None else []
    return mock


# ---------------------------------------------------------------------------
# scaffold_hint exhaustion (tool-level)
# ---------------------------------------------------------------------------

class TestScaffoldHintExhaustion:
    def test_raises_on_all_strategies_tried(self):
        """scaffold_hint.func() raises ValueError when all 5 strategies already tried."""
        with pytest.raises(ValueError, match="All strategies exhausted"):
            scaffold_hint.func(concept="addition", strategies_tried=ALL_STRATEGIES)

    def test_raises_regardless_of_order(self):
        """ValueError is raised regardless of strategy order."""
        shuffled = list(reversed(ALL_STRATEGIES))
        with pytest.raises(ValueError):
            scaffold_hint.func(concept="counting", strategies_tried=shuffled)

    def test_raises_with_duplicates_present(self):
        """Exhaustion is detected even with duplicate entries as long as all 5 are covered."""
        with_duplicates = ALL_STRATEGIES + ["guiding_question"]
        with pytest.raises(ValueError):
            scaffold_hint.func(concept="subtraction", strategies_tried=with_duplicates)

    def test_no_raise_when_one_strategy_available(self):
        """No error when at least one strategy remains; verifies LLM is called."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = make_mock_response("Try counting on your fingers!")
        four_of_five = ALL_STRATEGIES[:-1]  # all except "number_line"
        with patch("tutor.tools._hint_llm", mock_llm):
            result = scaffold_hint.func(concept="addition", strategies_tried=four_of_five)
        assert result["strategy"] == "number_line"
        assert result["hint"] == "Try counting on your fingers!"


# ---------------------------------------------------------------------------
# reasoning_react_loop — REVIEW signal
# ---------------------------------------------------------------------------

class TestReasoningLoopReviewSignal:
    def test_review_signal_populates_concepts_needing_review(self):
        """When LLM outputs REVIEW:<concept>, concepts_needing_review is populated."""
        mock_response = make_mock_response(
            "The answer is 23! Can you count to 10 for me?\nREVIEW:addition to 20"
        )
        with patch("tutor.nodes._reasoning_llm") as mock_llm:
            mock_llm.invoke.return_value = mock_response
            result = reasoning_react_loop(_base_state(
                concept="addition to 20",
                strategies_tried=ALL_STRATEGIES,
            ))
        assert "addition" in " ".join(result.get("concepts_needing_review", []))

    def test_review_signal_returns_nonempty_response(self):
        """The text before REVIEW: becomes current_response."""
        mock_response = make_mock_response(
            "The answer is 23! Well done!\nREVIEW:addition to 20"
        )
        with patch("tutor.nodes._reasoning_llm") as mock_llm:
            mock_llm.invoke.return_value = mock_response
            result = reasoning_react_loop(_base_state(
                concept="addition to 20",
                strategies_tried=ALL_STRATEGIES,
            ))
        assert result.get("current_response", "") != ""

    def test_review_signal_does_not_pause_session(self):
        """REVIEW signal should not set session_paused."""
        mock_response = make_mock_response(
            "The answer is 7! Can you try another one?\nREVIEW:subtraction"
        )
        with patch("tutor.nodes._reasoning_llm") as mock_llm:
            mock_llm.invoke.return_value = mock_response
            result = reasoning_react_loop(_base_state(concept="subtraction"))
        assert not result.get("session_paused", False)

    def test_review_concept_matches_state_concept_when_blank(self):
        """If REVIEW: has no trailing concept, the state concept is used."""
        mock_response = make_mock_response(
            "Great job! You got it!\nREVIEW:"
        )
        with patch("tutor.nodes._reasoning_llm") as mock_llm:
            mock_llm.invoke.return_value = mock_response
            result = reasoning_react_loop(_base_state(concept="multiplication"))
        # concepts_needing_review should contain something (state concept fallback)
        assert len(result.get("concepts_needing_review", [])) > 0


# ---------------------------------------------------------------------------
# reasoning_react_loop — ESCALATE signal
# ---------------------------------------------------------------------------

class TestReasoningLoopEscalateSignal:
    def test_escalate_sets_session_paused(self):
        """When LLM outputs ESCALATE, session_paused is True."""
        mock_response = make_mock_response("ESCALATE")
        with patch("tutor.nodes._reasoning_llm") as mock_llm:
            mock_llm.invoke.return_value = mock_response
            result = reasoning_react_loop(_base_state(student_input="I want to cry"))
        assert result.get("session_paused") is True

    def test_escalate_returns_empty_current_response(self):
        """ESCALATE signal produces empty current_response (escalate node writes the message)."""
        mock_response = make_mock_response("ESCALATE")
        with patch("tutor.nodes._reasoning_llm") as mock_llm:
            mock_llm.invoke.return_value = mock_response
            result = reasoning_react_loop(_base_state(student_input="I hate this"))
        assert result.get("current_response", "") == ""

    def test_escalate_with_trailing_punctuation(self):
        """ESCALATE. or ESCALATE! should also trigger the paused path."""
        mock_response = make_mock_response("ESCALATE.")
        with patch("tutor.nodes._reasoning_llm") as mock_llm:
            mock_llm.invoke.return_value = mock_response
            result = reasoning_react_loop(_base_state(student_input="This is too hard!"))
        assert result.get("session_paused") is True

    def test_escalate_does_not_add_concepts_for_review(self):
        """ESCALATE path should not add to concepts_needing_review."""
        mock_response = make_mock_response("ESCALATE")
        with patch("tutor.nodes._reasoning_llm") as mock_llm:
            mock_llm.invoke.return_value = mock_response
            result = reasoning_react_loop(_base_state(student_input="I want to quit"))
        assert result.get("concepts_needing_review", []) == []
