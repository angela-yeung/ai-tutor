"""Unit tests for tools and gather_context node.

No network calls or LLM calls are made — everything is patched.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tutor.tools import execute_code, web_search


# ---------------------------------------------------------------------------
# execute_code tests
# ---------------------------------------------------------------------------

def test_execute_code_addition():
    result = execute_code.invoke({"code": "2 + 3"})
    assert result == "5"


def test_execute_code_division():
    result = execute_code.invoke({"code": "10 / 4"})
    assert result == "2.5"


def test_execute_code_integer_result():
    result = execute_code.invoke({"code": "3.0 * 4.0"})
    assert result == "12"


def test_execute_code_blocks_open():
    # open() is not in safe builtins — should return empty string
    result = execute_code.invoke({"code": "open('secret.txt')"})
    assert result == ""


def test_execute_code_blocks_import():
    result = execute_code.invoke({"code": "import os; os.getcwd()"})
    assert result == ""


def test_execute_code_handles_syntax_error():
    result = execute_code.invoke({"code": "def ("})
    assert result == ""


def test_execute_code_none_result():
    # An assignment produces None from eval of the last line
    result = execute_code.invoke({"code": "x = 5\nx"})
    assert result == "5"


# ---------------------------------------------------------------------------
# web_search tests
# ---------------------------------------------------------------------------

@patch("tutor.tools.MultiServerMCPClient")
def test_web_search_returns_text(mock_cls):
    mock_tool = MagicMock()
    mock_tool.name = "tavily-search"
    mock_tool.invoke.return_value = [{"content": "Spiders have 8 legs."}]

    mock_client = MagicMock()
    mock_client.get_tools.return_value = [mock_tool]

    mock_instance = AsyncMock()
    mock_instance.__aenter__.return_value = mock_client
    mock_cls.return_value = mock_instance

    result = web_search.invoke({"query": "how many legs does a spider have"})
    assert len(result) > 0


@patch("tutor.tools.MultiServerMCPClient")
def test_web_search_on_failure(mock_cls):
    mock_cls.side_effect = Exception("network error")
    result = web_search.invoke({"query": "anything"})
    assert result == ""


# ---------------------------------------------------------------------------
# gather_context node tests
# ---------------------------------------------------------------------------

BASE_STATE = {
    "student_input": "What is 48 divided by 6?",
    "concept": "",
    "hints_given": 0,
    "understanding_level": "",
    "session_history": [],
    "current_response": "",
    "incorrect_attempts": 0,
    "session_paused": False,
    "session_complete": False,
    "tool_context": "",
}


@patch("tutor.nodes._get_llm_with_tools")
@patch("tutor.nodes._TOOL_MAP")
def test_gather_context_calls_tool(mock_tool_map, mock_get_llm):
    mock_llm = MagicMock()
    mock_get_llm.return_value = mock_llm
    mock_response = MagicMock()
    mock_response.tool_calls = [{"name": "execute_code", "args": {"code": "48 / 6"}}]
    mock_llm.invoke.return_value = mock_response

    mock_tool = MagicMock()
    mock_tool.invoke.return_value = "8"
    mock_tool_map.get.return_value = mock_tool

    from tutor.nodes import gather_context
    result = gather_context(BASE_STATE)

    assert result["tool_context"] == "8"
    mock_tool_map.get.assert_called_once_with("execute_code")
    mock_tool.invoke.assert_called_once_with({"code": "48 / 6"})


@patch("tutor.nodes._get_llm_with_tools")
def test_gather_context_no_tool_calls(mock_get_llm):
    mock_llm = MagicMock()
    mock_get_llm.return_value = mock_llm
    mock_response = MagicMock()
    mock_response.tool_calls = []
    mock_llm.invoke.return_value = mock_response

    from tutor.nodes import gather_context
    state = {**BASE_STATE, "student_input": "What rhymes with cat?"}
    result = gather_context(state)

    assert result["tool_context"] == ""


@patch("tutor.nodes._get_llm_with_tools")
@patch("tutor.nodes._TOOL_MAP")
def test_gather_context_tool_exception_returns_empty(mock_tool_map, mock_get_llm):
    mock_llm = MagicMock()
    mock_get_llm.return_value = mock_llm
    mock_response = MagicMock()
    mock_response.tool_calls = [{"name": "web_search", "args": {"query": "spider legs"}}]
    mock_llm.invoke.return_value = mock_response

    mock_tool = MagicMock()
    mock_tool.invoke.side_effect = Exception("tool error")
    mock_tool_map.get.return_value = mock_tool

    from tutor.nodes import gather_context
    result = gather_context(BASE_STATE)

    assert result["tool_context"] == ""
