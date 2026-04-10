"""Tests for tutor/tools.py."""
from unittest.mock import MagicMock, patch

import pytest

from tutor.tools import calculator, web_search, scaffold_hint


# ---------------------------------------------------------------------------
# calculator
# ---------------------------------------------------------------------------

class TestCalculator:
    def test_addition(self):
        assert calculator.invoke({"expression": "2 + 3"}) == "5"

    def test_division(self):
        result = calculator.invoke({"expression": "10 / 4"})
        assert result == "2.5"

    def test_blocks_object_traversal(self):
        # AST walker rejects attribute access — object graph traversal blocked
        result = calculator.invoke({"expression": "(1).__class__"})
        assert result == ""

    def test_blocks_string_expression(self):
        # AST walker only accepts numeric literals — strings are rejected
        result = calculator.invoke({"expression": "'hello'"})
        assert result == ""

    def test_syntax_error_returns_empty(self):
        result = calculator.invoke({"expression": "2 +"})
        assert result == ""


# ---------------------------------------------------------------------------
# web_search
# ---------------------------------------------------------------------------

class TestWebSearch:
    def test_formatted_output(self):
        fake_response = {
            "results": [
                {"title": "Result One", "content": "First snippet."},
                {"title": "Result Two", "content": "Second snippet."},
            ]
        }
        mock_client = MagicMock()
        mock_client.search.return_value = fake_response

        with patch("tutor.tools.TavilyClient", return_value=mock_client):
            result = web_search.invoke({"query": "test query"})

        assert "Result One" in result
        assert "First snippet." in result
        assert "Result Two" in result
        assert "Second snippet." in result
        # Two results separated by double newline
        assert "\n\n" in result

    def test_error_returns_empty(self):
        with patch("tutor.tools.TavilyClient", side_effect=Exception("network error")):
            result = web_search.invoke({"query": "anything"})
        assert result == ""

    def test_search_exception_returns_empty(self):
        mock_client = MagicMock()
        mock_client.search.side_effect = Exception("timeout")

        with patch("tutor.tools.TavilyClient", return_value=mock_client):
            result = web_search.invoke({"query": "anything"})
        assert result == ""


# ---------------------------------------------------------------------------
# scaffold_hint
# ---------------------------------------------------------------------------

class TestScaffoldHint:
    def _make_llm_response(self, text: str):
        msg = MagicMock()
        msg.content = text
        return msg

    def test_returns_dict_with_strategy_and_hint(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = self._make_llm_response("Think of sharing cookies equally.")

        with patch("tutor.tools._hint_llm", mock_llm):
            result = scaffold_hint.func(
                concept="division",
                strategies_tried=[],
            )

        assert isinstance(result, dict)
        assert result["strategy"] == "guiding_question"
        assert result["hint"] == "Think of sharing cookies equally."

    def test_skips_tried_strategies(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = self._make_llm_response("Like slicing a pizza.")

        with patch("tutor.tools._hint_llm", mock_llm):
            result = scaffold_hint.func(
                concept="fractions",
                strategies_tried=["guiding_question"],
            )

        assert result["strategy"] == "analogy"
        assert result["hint"] == "Like slicing a pizza."

    def test_raises_value_error_when_all_strategies_exhausted(self):
        all_strategies = [
            "guiding_question",
            "analogy",
            "concrete_example",
            "sub_problem",
            "number_line",
        ]
        # Use .func() to call the raw function directly — .invoke() may swallow
        # exceptions depending on the LangChain version.
        with pytest.raises(ValueError, match="All strategies exhausted"):
            scaffold_hint.func(
                concept="addition",
                strategies_tried=all_strategies,
            )
