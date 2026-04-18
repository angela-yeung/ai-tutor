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

`REDIS_URL` optional — if set, uses `RedisSaver` for persistence (production); otherwise `MemorySaver` (local dev/tests).

## Frontend Commands

```bash
cd frontend
npm install          # first time only
npm run dev          # Next.js dev server on http://localhost:3000
npm run test         # Vitest unit tests (frontend/tests/)
npm run build        # Production build
```

## API Server

```bash
uvicorn api.main:app --reload   # FastAPI dev server on http://localhost:8000
# Or use Docker Compose for full stack:
docker-compose up
```

**Endpoints:** `GET /health`, `POST /chat` (SSE streaming response).

**API architecture** (`/api/`):
- `main.py` — FastAPI app factory; initialises tutor graph in lifespan; selects `RedisSaver` vs `MemorySaver` based on `REDIS_URL`
- `router.py` — route handlers; `/chat` returns `StreamingResponse` (text/event-stream)
- `streaming.py` — drives LangGraph `invoke()`, emits SSE `data:` lines, sends `done` event with metadata (including `concepts_needing_review`)
- `schemas.py` — Pydantic request/response models

**Frontend architecture** (`/frontend/`):
- Next.js 14 app router; components in `components/`; SSE client in `lib/chat.ts`
- `ReviewPanel` component — collapsible; only renders when `concepts_needing_review` is non-empty
- Tests use Vitest + Testing Library (`frontend/tests/`)

## Architecture

A LangGraph `StateGraph` compiled with `MemorySaver`. Each CLI turn is one `invoke()` call.

**Entry routing** (`graph.py: entry_router`):
- `session_paused=True` → `resume_session`
- otherwise → `classify_question`

**`classify_question`** runs on every turn — binary classification (`factual` / `reasoning`) with concept extraction. Constrained output format: `<type>|<concept>`.

**Post-classify routing** (`graph.py: route_after_classify`):
- `factual` → `factual_react_loop` → `output_guard` → END
- `reasoning` → `reasoning_react_loop`

**`reasoning_react_loop` routing** (`graph.py: route_after_reasoning`):
- `session_paused=True` → `escalate` → `output_guard` → END
- otherwise → `output_guard` → END

**`resume_session`** → `output_guard` → END (shows welcome back, clears `session_paused`; next turn goes to `classify_question`)

**ReAct loops**: both loops use a while-loop pattern (LLM invoked with bound tools → if tool_calls → execute tool via `.invoke()` → append ToolMessage → continue → if no tool_calls → done). Tools are called via `.invoke()` for Phoenix auto-instrumentation.

**`reasoning_react_loop` Reason step** instructs the LLM to: (1) read `strategies_tried` — never repeat, (2) assess progress from `conversation_history`, (3) detect distress → output ESCALATE, (4) confirm understanding if demonstrated, (5) handle exhaustion (all 5 strategies tried → output direct answer + REVIEW:<concept>), (6) otherwise call `scaffold_hint` tool.

**`input_guard`**: First node after START. Runs all input guardrails (injection detection, length cap, NLI topic scope, OpenAI moderation). On block: sets `input_blocked=True` and `current_response` to safe deflection, then routes to `output_guard`. On pass: routes to `entry_router`.

**`output_guard`**: Terminal node before END. Runs OpenAI moderation on `current_response`. On flag: replaces with safe fallback. On pass: state unchanged.

**`route_after_classify`, `route_after_reasoning`, `entry_router`, `route_after_input_guard`** are pure functions (no LLM calls) — independently testable.

**`guardrails.py`**: Houses all guardrail logic (injection detection, NLI topic scope, length cap, OpenAI moderation) called by `input_guard` and `output_guard` nodes.

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
- `input_blocked` — `bool`; set `True` by `input_guard` on block; cleared each new turn

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
