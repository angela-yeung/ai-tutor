# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest tests/

# Run a single test
pytest tests/test_routing.py::test_route_after_check -v

# Start a new tutoring session
python -m tutor.cli

# Resume a paused session
python -m tutor.cli --resume <thread_id>
```

Requires `OPENAI_API_KEY` env var. Model is hardcoded to `gpt-4o-mini` in `tutor/nodes.py`.

## Architecture

A LangGraph `StateGraph` compiled with `MemorySaver`. Each CLI turn is one `invoke()` call — the graph runs from `START` to `END` on every message, with state persisted between calls via the thread-scoped checkpointer.

**Entry routing** (`graph.py: entry_router`) determines the first node each turn:
- No `concept` in state → `assess_question` (new question)
- `session_paused=True` → `resume_session` (returning after distress)
- Otherwise → `check_understanding` (student is responding to a hint)

**Post-classification routing** (`graph.py: route_after_check`) branches on `understanding_level`:
- `got_it` → `reinforce_concept` → END (`session_complete=True`)
- `frustrated` → `encourage` → `scaffold_hint` → END
- `distressed` → `escalate` → END (`session_paused=True`)
- Everything else → `scaffold_hint` → END

**`scaffold_hint`** has two behavioural modes: if `incorrect_attempts > 2`, it gives gentle direct correction instead of a Socratic hint. Hint depth is driven by `hints_given`: 0/1 = guiding question, 2 = analogy, 3+ = step-by-step breakdown.

**`check_understanding`** uses constrained single-word output — the prompt instructs the model to reply with exactly one of: `got_it`, `progressing`, `stuck`, `incorrect`, `frustrated`, `distressed`. Parsing logic falls back to `stuck` if the output is unrecognised.

**`route_after_check`** is a pure function (no LLM calls) intentionally — this is what makes routing independently testable in `tests/test_routing.py`.

## State fields

Defined in `tutor/state.py` as a `TypedDict`. Key fields to know:
- `concept` — set by `assess_question`; empty string signals a new/unstarted question
- `hints_given` — drives hint depth in `scaffold_hint`
- `incorrect_attempts` — incremented in `check_understanding` on `incorrect`; triggers direct correction in `scaffold_hint` when > 2
- `session_paused` — set `True` by `escalate`; triggers `resume_session` path on next invoke
- `session_complete` — set `True` by `reinforce_concept`; CLI uses this to exit the loop

## Prompt constraints

All system prompts in `nodes.py` include `_AGE_RULE` (sentences ≤10 words, kindergarten vocabulary, analogies from toys/food/animals/home objects only). Do not remove or weaken this rule — it is a core product requirement for the 5–6 year old target audience.
