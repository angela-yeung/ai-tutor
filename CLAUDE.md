# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project is a LangGraph-based AI tutor application using Python. The main frameworks are LangGraph for agent orchestration and OpenAI for LLM calls. Always consider LangGraph state management patterns when making architectural suggestions.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest tests/

# Run a single test
pytest tests/test_routing.py::test_entry_router_paused_goes_to_resume -v

# Start a new tutoring session
python -m tutor.cli

# Resume a paused session
python -m tutor.cli --resume <thread_id>

# Run Phoenix experiment evals (requires OPENAI_API_KEY and `phoenix serve` running)
python tests/evals/run_phoenix_evals.py
```

Requires `OPENAI_API_KEY` env var. All LLM calls use `gpt-4o`. `TAVILY_API_KEY` required for web search in factual loop.

## Architecture

A LangGraph `StateGraph` compiled with `MemorySaver`. Each CLI turn is one `invoke()` call.

**Entry routing** (`graph.py: entry_router`):
- `session_paused=True` → `resume_session`
- otherwise → `classify_question`

**`classify_question`** runs on every turn — binary classification (`factual` / `reasoning`) with concept extraction. Constrained output format: `<type>|<concept>`.

**Post-classify routing** (`graph.py: route_after_classify`):
- `factual` → `factual_react_loop` → `format_response` → END
- `reasoning` → `reasoning_react_loop`

**`reasoning_react_loop` routing** (`graph.py: route_after_reasoning`):
- `session_paused=True` → `escalate` → END
- otherwise → `format_response` → END

**`resume_session`** → END (shows welcome back, clears `session_paused`; next turn goes to `classify_question`)

**ReAct loops**: both loops use a while-loop pattern (LLM invoked with bound tools → if tool_calls → execute tool via `.invoke()` → append ToolMessage → continue → if no tool_calls → done). Tools are called via `.invoke()` for Phoenix auto-instrumentation.

**`reasoning_react_loop` Reason step** instructs the LLM to: (1) read `strategies_tried` — never repeat, (2) assess progress from `conversation_history`, (3) detect distress → output ESCALATE, (4) confirm understanding if demonstrated, (5) handle exhaustion (all 5 strategies tried → output direct answer + REVIEW:<concept>), (6) otherwise call `scaffold_hint` tool.

**`format_response`**: dedicated post-processing node. Rewrites `current_response` for Grade 1 audience. Applied to output from both loops.

**`route_after_classify`, `route_after_reasoning`, `entry_router`** are pure functions (no LLM calls) — independently testable.

## State fields

Defined in `tutor/state.py` as a `TypedDict`. Key fields:
- `student_input` — latest message from the student
- `concept` — extracted by `classify_question` each turn
- `question_type` — `"factual"` or `"reasoning"`
- `strategies_tried` — `Annotated[list, operator.add]`; grows each reasoning turn; never reset mid-session
- `conversation_history` — `Annotated[list, operator.add]`; `{"role": ..., "content": ...}` message log
- `current_response` — latest assistant response
- `session_paused` — set `True` by `reasoning_react_loop` on distress; cleared by `resume_session`
- `concepts_needing_review` — `Annotated[list, operator.add]`; populated on strategy exhaustion; printed at session end

## Testing

Always run the full test suite (`pytest`) after making code changes. Ensure all tests pass before considering a task complete.

## Code Conventions

Use lazy initialization for LLM clients (e.g., ChatOpenAI) — never instantiate at module level. This prevents tests from requiring API keys at import time.

## Workflow Preferences

When the user presents multiple implementation options (e.g., Option A vs Option B), ask which they prefer before starting implementation. Do not assume based on existing test expectations.

Before making changes to any feature area, explain how the current LangGraph state flows through the graph for that area — show the relevant nodes and edges. Do NOT make any code changes yet; wait for the user to confirm they understand.

Before making any changes, run the existing test suite and show which tests pass/fail. Then propose your changes and predict which tests will be affected.

## Debugging Guidelines

When debugging, check the actual root cause (API credits, network connectivity, Python version) before assuming config file format issues.

## Prompt constraints

All student-facing LLM calls include `_AGE_RULE` (sentences ≤10 words, Grade 1 vocabulary, analogies from toys/food/animals/home/playground only, warm tone). Do not remove or weaken this rule — it is a core product requirement for the Grade 1 target audience.
