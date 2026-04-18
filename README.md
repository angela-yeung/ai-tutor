# Personal AI Tutoring Assistant

A LangGraph-based AI tutor for Grade 1 students (age 6) that adapts its behavior based on what kind of question is being asked — and what the child actually needs.

## Problem Statement

When children use AI tools for homework help, they typically receive direct answers — bypassing the cognitive effort that builds genuine understanding. Current tutoring tools compound this: they either hand over answers too readily, follow rigid scripts that can't
adapt to a child's reasoning, or lack the emotional awareness to recognise when a student is frustrated rather than just stuck.

The deeper problem is that not all questions are the same. *"What is the capital of France?"* needs a fact. *"I don't understand why 3 + 4 = 7"* needs something else entirely — an adaptive guide that meets the child where they are, tries a different approach if the first doesn't land, and knows when to slow down.

This Personal AI Tutoring Assistant is a conversational learning tool for Grade 1 students that treats these two situations differently by design. Factual questions get direct, age-appropriate answers. Conceptual questions get Socratic scaffolding — probing questions, analogies, and progressive hints — that guides a child toward understanding rather than just giving it to them.

## Features

- **Dual ReAct loops** — factual questions use `web_search` (Tavily); reasoning questions use `calculator` + `scaffold_hint`
- **5-strategy Socratic scaffolding** — guiding question, analogy, concrete example, sub-problem, number line; strategies are never repeated within a session
- **Strategy exhaustion handling** — after all 5 strategies, the tutor gives a direct answer and flags the concept for adult review
- **Distress detection and human-in-the-loop** — reasoning loop detects student frustration; session pauses gracefully and preserves full state
- **Defense-in-depth input guardrails** — regex/fuzzy injection detection, zero-shot NLI topic classifier, OpenAI Moderation API
- **Output moderation** — every response passes through OpenAI Moderation before being shown
- **Phoenix observability** — every CLI turn is traced; LLM calls and tool invocations appear as nested spans, supporting an evaluation-driven development approach
- **LLM-as-judge eval suite** — 5 custom evaluators, 29 labeled examples, run as Phoenix experiments
- **Session review** — `concepts_needing_review` printed at session end for parent/teacher follow-up
- **FastAPI backend** — REST + SSE streaming wrapper with Redis checkpointer support for production

## Architecture

The assistant is a LangGraph `StateGraph` with 8 nodes and 4 pure routing functions, compiled with a `MemorySaver` or `RedisSaver` checkpointer. Each CLI turn is one `invoke()` call.

### Graph Flow

```
START
  │
  └─ input_guard
       ├─ (blocked)  → output_guard → END
       └─ (pass)     → entry_router
                          ├─ (session_paused)  → resume_session → output_guard → END
                          └─ (normal)          → classify_question
                                                    ├─ factual    → factual_react_loop → output_guard → END
                                                    └─ reasoning  → reasoning_react_loop
                                                                        ├─ (distressed) → escalate → output_guard → END
                                                                        └─ (ok)         → output_guard → END
```

### Nodes

| Node | Purpose |
|------|---------|
| `input_guard` | Runs all input guardrails; blocks or passes to entry routing |
| `entry_router` | Routes based on `session_paused`: resume or classify |
| `classify_question` | Binary factual/reasoning classification + concept extraction |
| `factual_react_loop` | ReAct loop with `web_search` tool; web results are moderation-filtered |
| `reasoning_react_loop` | ReAct loop with `calculator` + `scaffold_hint`; detects `ESCALATE` and `REVIEW:<concept>` signals |
| `escalate` | Sets `session_paused=True`; emits warm pause message |
| `resume_session` | Welcome-back message; clears `session_paused` |
| `output_guard` | Runs OpenAI Moderation on `current_response` before every END |

### Routing

All 4 routing functions (`route_after_input_guard`, `entry_router`, `route_after_classify`, `route_after_reasoning`) are pure functions — they read from state only, make no LLM calls, and are independently unit-testable.

### State

Key fields in `TutorState` (TypedDict in `tutor/state.py`):

| Field | Type | Reducer |
|-------|------|---------|
| `student_input` | str | — |
| `question_type` | `"factual"` \| `"reasoning"` | — |
| `concept` | str | — |
| `strategies_tried` | list[str] | `operator.add` — grows across turns, never resets mid-session |
| `conversation_history` | list[dict] | `operator.add` |
| `current_response` | str | — |
| `session_paused` | bool | — |
| `concepts_needing_review` | list[str] | `operator.add` |
| `input_blocked` | bool | — |

## Observability

Install and start Phoenix:

```bash
pip install arize-phoenix
phoenix serve
```

Open `http://localhost:6006` to view traces.

Each CLI turn creates a trace. Tool calls (`web_search`, `calculator`, `scaffold_hint`) appear as child spans within the loop spans, enabling step-by-step inspection of the ReAct reasoning.

### Eval Suite

A custom LLM-as-judge eval suite runs against the live graph via Phoenix experiments (`tests/evals/run_phoenix_evals.py`). Each run uploads a fresh dataset snapshot and appends a new experiment run — compare runs across iterations for regression testing.

**5 evaluators across 29 labeled examples:**

| Evaluator | Node | What It Checks |
|-----------|------|----------------|
| `classification_correctness` | `classify_question` | factual vs. reasoning label accuracy |
| `factual_tool_selection` | `factual_react_loop` | whether `web_search` is called when expected |
| `reasoning_tool_selection` | `reasoning_react_loop` | whether `scaffold_hint` or `calculator` is called appropriately |
| `distress_detection` | `reasoning_react_loop` | whether the ESCALATE signal fires on distress |
| `resume_recall` | `resume_session` | whether the welcome-back message references the prior concept |

```bash
# Requires OPENAI_API_KEY and `phoenix serve` running
python tests/evals/run_phoenix_evals.py
```

## Design Decisions

| Decision | Why |
|----------|-----|
| **Separate ReAct loops for factual vs. reasoning** | Factual needs `web_search`; reasoning needs `scaffold_hint` + `calculator`. A shared loop bloats the prompt and reduces tool selection precision. |
| **Pure routing functions (no LLM calls)** | Deterministic routing is independently unit-testable. Routing bugs are easy to reproduce and fix; routing bugs inside LLM calls are not. |
| **Defense-in-depth guardrails** | Each layer catches a different attack surface: regex/fuzzy matching catches injection patterns; NLI topic classifier catches off-topic questions; OpenAI Moderation catches harmful content. No single layer is sufficient. |
| **Strategy tracking via state reducer** | `strategies_tried` uses `operator.add` and grows across turns. `scaffold_hint` raises `ValueError` when all 5 are exhausted. This keeps the tool and the state management concerns cleanly separated. |
| **Checkpointer abstraction** | `build_graph()` accepts a checkpointer parameter — `MemorySaver` in dev, `RedisSaver` in production. Graph code is unchanged between environments. |
| **Tool execution via `.invoke()`** | Tools could be called directly, but `.invoke()` is required for Phoenix to auto-instrument them as child spans. The tracing benefit justifies the minor indirection. |

## Lessons Learned

**Orchestration pattern determines flexibility vs. reliability.** The initial graph was a rigid state machine: it expected the student to follow a predictable path, and when they didn't, the tutor got stuck. Reconsidering the choice between ReAct (flexible, tool-driven), plan-and-execute, and code-based orchestration is what led to the current dual-loop design. ReAct lets the reasoning loop adapt — it can try a different tool, read from the conversation history, and decide on the next move without being pre-wired for every case.

**Set up evals early and iterate on hard examples.** Getting an MVP working and then setting up evals immediately and iterating using an evaluation-driven development approach is more valuable than polishing the MVP first. The critical insight was that eval scores can be misleading: a 90% pass rate on easy examples doesn't mean the system works. Spending time on hard examples — ambiguous questions, edge cases, adversarial inputs — and manually reviewing LLM-as-a-judge results is what makes an eval suite trustworthy.

**Observability is how you diagnose unexpected agent behavior.** Adding Phoenix tracing early was what made it possible to see that the initial graph was too rigid. A span tree showing every LLM call, tool invocation, and routing decision gave a clear picture of what was actually happening vs. what was expected. Without that visibility, diagnosing why the tutor wasn't adapting would have been guesswork.

**LLM calls are the main latency bottleneck.** Each LLM-based guardrail check adds a full API round-trip. Auditing every call and replacing LLM checks with regex, rule-based detection, or ML classifiers (zero-shot NLI for topic scoping) where possible roughly halved response time. The principle: use the smallest sufficient tool for each check.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Orchestration | LangGraph — StateGraph, conditional edges, checkpointing |
| LLM | OpenAI gpt-4o — classification, ReAct reasoning, scaffold hints |
| Web search | Tavily |
| Topic safety | HuggingFace Transformers + PyTorch — zero-shot NLI topic classifier |
| Content moderation | OpenAI Moderation API — input + output |
| Observability | Arize Phoenix, OpenInference, OpenTelemetry SDK |
| API | FastAPI + uvicorn — SSE streaming backend |
| Session persistence | MemorySaver (dev) / Redis (prod) |
| Testing | pytest — unit, integration, graph-level |

## Project Structure

```
ai-tutor/
├── tutor/
│   ├── state.py              # TutorState TypedDict + reducers
│   ├── graph.py              # StateGraph, routing functions, build_graph()
│   ├── nodes.py              # All 8 node implementations
│   ├── tools.py              # calculator (AST-safe), web_search, scaffold_hint
│   ├── guardrails.py         # Input/output safety: regex, NLI, moderation
│   ├── instrumentation.py    # Phoenix/OpenTelemetry setup
│   └── cli.py                # Interactive CLI with --resume support
├── api/
│   ├── main.py               # FastAPI app factory, Redis/MemorySaver checkpointer
│   ├── router.py             # POST /chat (SSE), GET /health
│   ├── schemas.py            # Pydantic request/response models
│   └── streaming.py          # Async SSE token generator
├── tests/
│   ├── conftest.py           # Shared fixtures (mocked clients)
│   ├── test_routing.py       # Pure routing function tests
│   ├── test_classifier.py    # Classification accuracy + topic change
│   ├── test_tools.py         # Tool unit tests
│   ├── test_guardrails.py    # Guardrail layer tests
│   ├── test_exhaustion.py    # Strategy exhaustion + REVIEW signal
│   ├── test_graph_guardrails.py  # Graph-level guardrail integration
│   ├── test_guardrail_nodes.py   # Guard node return values
│   ├── test_api.py           # FastAPI endpoint tests (health, SSE streaming)
│   └── evals/
│       └── run_phoenix_evals.py  # Phoenix experiment eval suite
├── requirements.txt
├── docker-compose.yml
└── .env
```

## Setup

### Requirements

Python 3.11 or 3.12. Python 3.14+ is not supported.

```bash
git clone <repo-url>
cd ai-tutor

# Windows
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS / Linux
python3.11 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...

# Phoenix observability
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006
PHOENIX_PROJECT_NAME=ai-tutor

# Optional: Redis session persistence (defaults to in-memory)
REDIS_URL=redis://localhost:6379

# Optional: FastAPI CORS
CORS_ORIGINS=http://localhost:3000
```

## Usage

### CLI — New Session

```bash
python -m tutor.cli
```

The thread ID is printed at session start. Save it to resume later.

### CLI — Resume a Paused Session

```bash
python -m tutor.cli --resume <thread_id>
```

### REST API

```bash
uvicorn api.main:app --reload
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "abc123", "message": "What is 3 + 4?"}'
```

Responses stream as Server-Sent Events.

### Tests

```bash
# Unit and integration tests (no API keys needed)
pytest tests/

# Phoenix experiment evals (requires OPENAI_API_KEY and phoenix serve)
python tests/evals/run_phoenix_evals.py
```
