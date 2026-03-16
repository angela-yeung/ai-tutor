# Personal AI Tutoring Assistant

A conversational tutoring agent built with LangGraph and the OpenAI API. It guides students aged 5–6 through problems using Socratic questioning rather than giving direct answers.

## Features

- **Socratic hint progression** — guiding question → analogy → step-by-step breakdown
- **6-way understanding classifier** — routes based on `got_it`, `progressing`, `stuck`, `incorrect`, `frustrated`, or `distressed`
- **Pause-and-resume** — sessions pause gracefully on distress and can be resumed later by thread ID
- **Stateful sessions** — full conversation history maintained via LangGraph `MemorySaver`

## Setup

### 1. Clone and create a virtual environment

> **Python version:** Use Python 3.11 or 3.12. Python 3.14+ is not supported — `langchain-core` relies on Pydantic V1 compatibility which breaks on 3.14, causing silent API failures.

```bash
git clone <repo-url>
cd ai-tutor

# Windows (use py launcher to target the correct version)
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS / Linux
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your OpenAI API key

Get your API key from platform.openai.com → API Keys, then add it to a `.env` file in the project root:

```
OPENAI_API_KEY=sk-proj-...
```

Or export it in your shell:

```bash
# Windows (PowerShell)
$env:OPENAI_API_KEY = "sk-proj-..."

# macOS / Linux
export OPENAI_API_KEY="sk-proj-..."
```

## Usage

### Start a new session

```bash
python -m tutor.cli
```

```
=== New session started ===
Session ID: 3f2a1c9e-...
(Save this ID to resume later with --resume <SESSION_ID>)

Hi! I'm your tutor. What would you like to learn today?

You: why is the sky blue?

Tutor: What colour do you see when you look up outside?
```

### Resume a paused session

If a session was paused (e.g. the student appeared distressed), resume it with:

```bash
python -m tutor.cli --resume 3f2a1c9e-...
```

### Exit

Type `quit` or press `Ctrl+C` at any time.

## Running Tests

```bash
pytest tests/
```

All tests are unit tests with no LLM calls required.

## Project Structure

```
tutor/
  state.py        # TutorState TypedDict
  nodes.py        # 7 node functions + llm_call() helper
  graph.py        # StateGraph, routing logic, compiled tutor_app
  cli.py          # Interactive CLI with --resume support
tests/
  test_routing.py # Routing logic tests for all 6 understanding states
requirements.txt
README.md
```

## Architecture

The assistant is a LangGraph `StateGraph` compiled with a `MemorySaver` checkpointer:

```
START
  │
  ├─ (no concept yet)   → assess_question → scaffold_hint → END
  ├─ (session paused)   → resume_session  → scaffold_hint → END
  └─ (concept known)    → check_understanding
                              │
                              ├─ got_it      → reinforce_concept → END
                              ├─ progressing → scaffold_hint → END
                              ├─ stuck       → scaffold_hint → END
                              ├─ incorrect   → scaffold_hint → END  (gentle correction if > 2 attempts)
                              ├─ frustrated  → encourage → scaffold_hint → END
                              └─ distressed  → escalate → END  (session_paused=True)
```
