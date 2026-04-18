# Guardrails Design Spec
**Date:** 2026-04-14

---

## Context

The AI tutor is a LangGraph-based application targeting Grade 1 students (age 6). Currently there is zero content filtering — raw student input flows directly to GPT-4o and LLM responses are printed directly to the child without validation. Adding guardrails is required before any frontend deployment to protect children from harmful content, off-topic interactions, and prompt injection attacks.

---

## Guardrails in scope

| # | Guardrail | Type | Where |
|---|---|---|---|
| 1 | OpenAI Moderation API on input | Hard block | `input_guard` node |
| 2 | NLI topic scope check | Soft redirect | `input_guard` node |
| 3 | Input length cap (500 chars) | Hard block | `input_guard` node |
| 4 | Prompt injection sanitisation (OWASP) | Hard block | `guardrails.sanitise_input()` |
| 5 | Structured prompt with section separation (OWASP) | Prevention | All nodes using `student_input` |
| 6 | OpenAI Moderation API on output | Hard block | `output_guard` node |
| 7 | Web search result filtering | Hard block | `tools.web_search()` |

**Out of scope:** PII detection, age-appropriateness vocabulary check, per-node intermediate response moderation.

---

## Architecture

### Graph structure

```
START
  │
  ▼
input_guard ──(blocked)──────────────────────────────────┐
  │                                                       │
  │ (clean)                                               │
  ▼                                                       │
entry_router                                              │
  ├─(session_paused)──► resume_session ──────────────────┤
  └─(otherwise)──────► classify_question                 │
                              │                           │
                    ┌─────────┴──────────┐               │
                    ▼                    ▼               │
           factual_react_loop   reasoning_react_loop     │
                    │                    │               │
                    │           ┌────────┴────────┐      │
                    │           ▼                 ▼      │
                    │        escalate         (normal)   │
                    │           │                 │      │
                    └───────────┴─────────────────┘      │
                                │                        │
                                ▼                        │
                          output_guard ◄─────────────────┘
                                │
                                ▼
                               END
```

### New files
- `tutor/guardrails.py` — all guardrail logic as pure functions

### Modified files
- `tutor/state.py` — add `input_blocked: bool`
- `tutor/nodes.py` — add `input_guard`, `output_guard` nodes; apply structured prompts
- `tutor/graph.py` — rewire all END edges through `output_guard`; add `input_guard` at START
- `tutor/tools.py` — call `filter_search_results()` inside `web_search`
- `requirements.txt` — add `transformers`, `torch`

---

## Module: tutor/guardrails.py

### Key constants
- `_MAX_INPUT_LENGTH = 500`
- `_CHILD_DEFLECTION` — child-friendly block message
- `_OUTPUT_FALLBACK` — safe fallback for blocked output
- `_EDUCATIONAL_LABELS` — NLI candidate labels for topic scope check
- `_OFF_TOPIC_THRESHOLD = 0.85`
- `_INJECTION_PATTERNS` — regex patterns for prompt injection detection
- `_FUZZY_TARGETS` — words to check for typoglycemia variants

### Key functions
- `sanitise_input(text)` — OWASP injection detection, raises ValueError
- `check_input(text)` — cheapest-first: sanitise → length → NLI → moderation
- `check_output(text)` — OpenAI moderation on LLM response
- `filter_search_results(results)` — batch moderation of web search results
- `wrap_student_input(text)` — wraps in USER_DATA section
- `SECURITY_RULES` — constant string appended to all LLM system prompts

### NLI model
- `cross-encoder/nli-MiniLM2-L6-H768` (HuggingFace, ~85MB)
- Lazy-initialised via `_get_classifier()`

---

## Known issues (out of scope)

**`strategies_tried` is session-wide, not concept-scoped.**
`strategies_tried` accumulates across every reasoning turn regardless of topic. A student who switches topics mid-session may hit the "all strategies exhausted" path prematurely on an unrelated concept.
Suggested fix (separate ticket): add `current_concept` field; reset `strategies_tried` in `classify_question` when concept changes.
