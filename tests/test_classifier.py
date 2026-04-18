"""Tests for classify_question node.

All LLM calls are mocked — no real API calls are made.
"""

import pytest
from unittest.mock import MagicMock, patch

from tutor.nodes import classify_question


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mock_response(content: str):
    mock = MagicMock()
    mock.content = content
    mock.tool_calls = []
    return mock


def _base_state(student_input: str) -> dict:
    return {
        "student_input": student_input,
        "strategies_tried": [],
        "conversation_history": [],
        "concepts_needing_review": [],
        "session_paused": False,
        "concept": "",
        "question_type": "",
        "current_response": "",
    }


# ---------------------------------------------------------------------------
# Factual questions
# ---------------------------------------------------------------------------

class TestFactualClassification:
    def test_capital_of_france(self):
        with patch("tutor.nodes._classifier_llm") as mock_llm:
            mock_llm.invoke.return_value = make_mock_response("factual|capital of France")
            result = classify_question(_base_state("What is the capital of France?"))
        assert result["question_type"] == "factual"
        assert "france" in result["concept"].lower() or "capital" in result["concept"].lower()

    def test_spider_legs(self):
        with patch("tutor.nodes._classifier_llm") as mock_llm:
            mock_llm.invoke.return_value = make_mock_response("factual|spider legs")
            result = classify_question(_base_state("How many legs does a spider have?"))
        assert result["question_type"] == "factual"
        assert "spider" in result["concept"].lower() or "legs" in result["concept"].lower()

    def test_harry_potter_author(self):
        with patch("tutor.nodes._classifier_llm") as mock_llm:
            mock_llm.invoke.return_value = make_mock_response("factual|Harry Potter author")
            result = classify_question(_base_state("Who wrote Harry Potter?"))
        assert result["question_type"] == "factual"
        assert "harry" in result["concept"].lower() or "author" in result["concept"].lower()

    def test_sky_colour(self):
        with patch("tutor.nodes._classifier_llm") as mock_llm:
            mock_llm.invoke.return_value = make_mock_response("factual|sky colour")
            result = classify_question(_base_state("What colour is the sky?"))
        assert result["question_type"] == "factual"
        assert "sky" in result["concept"].lower() or "colour" in result["concept"].lower()

    def test_days_in_week(self):
        with patch("tutor.nodes._classifier_llm") as mock_llm:
            mock_llm.invoke.return_value = make_mock_response("factual|days in week")
            result = classify_question(_base_state("How many days are in a week?"))
        assert result["question_type"] == "factual"
        assert "days" in result["concept"].lower() or "week" in result["concept"].lower()


# ---------------------------------------------------------------------------
# Reasoning questions
# ---------------------------------------------------------------------------

class TestReasoningClassification:
    def test_addition_14_plus_9(self):
        with patch("tutor.nodes._classifier_llm") as mock_llm:
            mock_llm.invoke.return_value = make_mock_response("reasoning|addition")
            result = classify_question(_base_state("What is 14 + 9?"))
        assert result["question_type"] == "reasoning"
        assert "addition" in result["concept"].lower()

    def test_subtraction_apples(self):
        with patch("tutor.nodes._classifier_llm") as mock_llm:
            mock_llm.invoke.return_value = make_mock_response("reasoning|subtraction")
            result = classify_question(_base_state("If I have 3 apples and eat 1, how many are left?"))
        assert result["question_type"] == "reasoning"
        assert "subtraction" in result["concept"].lower()

    def test_number_sequence(self):
        with patch("tutor.nodes._classifier_llm") as mock_llm:
            mock_llm.invoke.return_value = make_mock_response("reasoning|number sequence")
            result = classify_question(_base_state("Which number comes after 19?"))
        assert result["question_type"] == "reasoning"
        assert "number" in result["concept"].lower() or "sequence" in result["concept"].lower()

    def test_addition_sweets(self):
        with patch("tutor.nodes._classifier_llm") as mock_llm:
            mock_llm.invoke.return_value = make_mock_response("reasoning|addition")
            result = classify_question(_base_state("I have 5 sweets and my friend has 3. How many altogether?"))
        assert result["question_type"] == "reasoning"
        assert "addition" in result["concept"].lower()

    def test_subtraction_10_minus_4(self):
        with patch("tutor.nodes._classifier_llm") as mock_llm:
            mock_llm.invoke.return_value = make_mock_response("reasoning|subtraction")
            result = classify_question(_base_state("What is 10 minus 4?"))
        assert result["question_type"] == "reasoning"
        assert "subtraction" in result["concept"].lower()


# ---------------------------------------------------------------------------
# Fallback behaviour
# ---------------------------------------------------------------------------

class TestClassifyFallback:
    def test_garbage_output_defaults_to_reasoning(self):
        with patch("tutor.nodes._classifier_llm") as mock_llm:
            mock_llm.invoke.return_value = make_mock_response("invalid_output_no_pipe")
            result = classify_question(_base_state("How much is 7 plus 8?"))
        assert result["question_type"] == "reasoning"

    def test_garbage_output_concept_is_truncated_input(self):
        long_input = "How much is 7 plus 8 take away 3 and add 5 and multiply by 2 altogether?"
        with patch("tutor.nodes._classifier_llm") as mock_llm:
            mock_llm.invoke.return_value = make_mock_response("invalid_output_no_pipe")
            result = classify_question(_base_state(long_input))
        assert result.get("concept", "") != ""
        # Concept is the first 50 chars of the input when fallback fires
        assert result["concept"] == long_input[:50]

    def test_unknown_type_normalised_to_reasoning(self):
        """If the LLM outputs an unrecognised type label, fall back to reasoning."""
        with patch("tutor.nodes._classifier_llm") as mock_llm:
            mock_llm.invoke.return_value = make_mock_response("mystery|some concept")
            result = classify_question(_base_state("Some question"))
        assert result["question_type"] == "reasoning"
        assert result["concept"] == "some concept"


# ---------------------------------------------------------------------------
# Topic-change detection
# ---------------------------------------------------------------------------

class TestTopicChangeDetection:
    def test_no_reset_on_initial_turn(self):
        """First turn (no previous concept): strategies_tried must not appear in result."""
        state = _base_state("What is 5 + 3?")
        # _base_state already sets concept="" and strategies_tried=[]
        with patch("tutor.nodes._classifier_llm") as mock_llm:
            mock_llm.invoke.return_value = make_mock_response("reasoning|simple addition")
            result = classify_question(state)
        assert "strategies_tried" not in result

    def test_no_reset_on_same_topic(self):
        """Same topic as previous turn: strategies_tried must not appear in result."""
        state = _base_state("What is 7 + 4?")
        state["concept"] = "simple addition"
        state["strategies_tried"] = ["guiding_question"]
        with patch("tutor.nodes._classifier_llm") as mock_llm:
            mock_llm.invoke.return_value = make_mock_response("reasoning|simple addition|true")
            result = classify_question(state)
        assert "strategies_tried" not in result

    def test_reset_on_topic_change(self):
        """Different topic from previous turn: strategies_tried must be reset to []."""
        state = _base_state("How far is the moon?")
        state["concept"] = "simple addition"
        state["strategies_tried"] = ["guiding_question", "analogy"]
        with patch("tutor.nodes._classifier_llm") as mock_llm:
            mock_llm.invoke.return_value = make_mock_response("factual|moon distance|false")
            result = classify_question(state)
        assert result.get("strategies_tried") == []

    def test_no_reset_when_llm_returns_two_parts_on_subsequent_turn(self):
        """If LLM drops same_topic field on subsequent turn, no reset should fire."""
        state = _base_state("What is 7 + 4?")
        state["concept"] = "simple addition"
        state["strategies_tried"] = ["guiding_question"]
        with patch("tutor.nodes._classifier_llm") as mock_llm:
            mock_llm.invoke.return_value = make_mock_response("reasoning|simple addition")
            result = classify_question(state)
        assert "strategies_tried" not in result

    def test_no_incorrect_reset_when_concept_contains_pipe(self):
        """If concept contains a pipe, same_topic must still be read from the last field."""
        state = _base_state("What is 7 + 4?")
        state["concept"] = "simple addition"
        state["strategies_tried"] = ["guiding_question"]
        with patch("tutor.nodes._classifier_llm") as mock_llm:
            # Concept contains a pipe; same_topic is the last field = "true" (no reset)
            mock_llm.invoke.return_value = make_mock_response("reasoning|addition|subtraction|true")
            result = classify_question(state)
        assert "strategies_tried" not in result
