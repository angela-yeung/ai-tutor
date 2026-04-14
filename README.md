# Personal AI Tutoring Assistant

An AI tutor for Grade 1 students (age 6) built with LangGraph and the OpenAI API. It handles two types of questions with separate ReAct loops: factual questions answered via web search, and reasoning questions guided through adaptive Socratic scaffolding. Phoenix observability is built in.

## Features

- **Dual ReAct loops** — factual questions use web search (Tavily); reasoning questions use adaptive Socratic scaffolding
- **5-strategy scaffolding** — guiding question, analogy, concrete example, sub-problem, number line; strategies are never repeated within a session
- **Strategy exhaustion** — if all 5 strategies are tried without a breakthrough, the tutor gives a direct answer warmly and flags the concept for adult review
- **Distress detection** — session pauses gracefully and preserves state; resume any time with `--resume`
- **Session review** — prints `concepts_needing_review` at session end for parent/teacher follow-up
- **Phoenix observability** — every CLI turn is traced; tool calls appear as child spans

## Setup

### Requirements

Python 3.11 or 3.12 required. Python 3.14+ is not supported.

```bash
git clone <repo-url>
cd ai-tutor

# Windows
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS / Linux
python3.11 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Environment variables

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006
PHOENIX_PROJECT_NAME=ai-tutor
```

## Usage

### Start a new session

```bash
python -m tutor.cli
```

### Resume a paused session

If a session was paused (e.g. the student appeared distressed), resume it with:

```bash
python -m tutor.cli --resume <thread_id>
```

The thread ID is printed when a session starts. Save it to resume later.

### Exit

Type `quit`, `exit`, or `q` at any prompt.

## Architecture

The assistant is a LangGraph `StateGraph` compiled with a `MemorySaver` checkpointer. Each CLI turn is one `invoke()` call.

| Node | Purpose |
|---|---|
| `classify_question` | Binary factual/reasoning classification + concept extraction |
| `factual_react_loop` | ReAct loop with `web_search` tool |
| `reasoning_react_loop` | ReAct loop with `calculator` + `scaffold_hint` tools |
| `format_response` | Grade 1 language post-processing applied to both loop outputs |
| `escalate` | Distress handler — sets `session_paused=True` |
| `resume_session` | Welcome back message after pause; clears `session_paused` |

Graph flow:

```
START
  │
  ├─ (session_paused)  → resume_session → END
  └─ (otherwise)       → classify_question
                              │
                              ├─ factual   → factual_react_loop → format_response → END
                              └─ reasoning → reasoning_react_loop
                                                  │
                                                  ├─ (distressed) → escalate → END
                                                  └─ (otherwise)  → format_response → END
```

## Observability (Phoenix)

Install and start Phoenix:

```bash
pip install arize-phoenix
phoenix serve
```

Open `http://localhost:6006` to view traces.

Each CLI turn creates a trace. Tool calls (`web_search`, `calculator`, `scaffold_hint`) appear as child spans within the loop spans, enabling step-by-step inspection of the ReAct reasoning.

## Tests

Run all unit and structural tests (no API keys needed):

```bash
pytest tests/
```

Run Phoenix experiment evals (requires `OPENAI_API_KEY` and `phoenix serve` running):

```bash
python tests/evals/run_phoenix_evals.py
```

## Project Structure

```
tutor/
  state.py              # TutorState TypedDict
  nodes.py              # Node functions and ReAct loops
  graph.py              # StateGraph, routing logic, compiled tutor_app
  cli.py                # Interactive CLI with --resume support
  instrumentation.py    # Phoenix tracing setup
tests/
  test_tools.py         # Unit tests for tools and nodes
  conftest.py           # Shared fixtures
  evals/
    run_phoenix_evals.py  # Phoenix experiment eval suite (8 evaluators, 5 nodes)
requirements.txt
README.md
```
