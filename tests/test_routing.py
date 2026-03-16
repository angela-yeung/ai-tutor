"""Tests for check_understanding routing logic.

These tests are pure unit tests — no LLM calls are made.
`route_after_check` is extracted as a standalone function in graph.py
so it can be tested independently of the compiled graph.
"""

import pytest
from tutor.graph import route_after_check


BASE_STATE = {
    "student_input": "",
    "concept": "addition",
    "hints_given": 1,
    "understanding_level": "",
    "session_history": [],
    "current_response": "",
    "incorrect_attempts": 0,
    "session_paused": False,
    "session_complete": False,
}


@pytest.mark.parametrize(
    "understanding_level, expected_node",
    [
        ("got_it",      "reinforce_concept"),
        ("progressing", "scaffold_hint"),
        ("stuck",       "scaffold_hint"),
        ("incorrect",   "scaffold_hint"),
        ("frustrated",  "encourage"),
        ("distressed",  "escalate"),
    ],
)
def test_route_after_check(understanding_level: str, expected_node: str) -> None:
    state = {**BASE_STATE, "understanding_level": understanding_level}
    result = route_after_check(state)
    assert result == expected_node, (
        f"Expected '{expected_node}' for level '{understanding_level}', got '{result}'"
    )


def test_unknown_level_defaults_to_scaffold_hint() -> None:
    """Unrecognised classification should fall back to scaffold_hint."""
    state = {**BASE_STATE, "understanding_level": "some_unknown_value"}
    assert route_after_check(state) == "scaffold_hint"


def test_empty_level_defaults_to_scaffold_hint() -> None:
    state = {**BASE_STATE, "understanding_level": ""}
    assert route_after_check(state) == "scaffold_hint"
