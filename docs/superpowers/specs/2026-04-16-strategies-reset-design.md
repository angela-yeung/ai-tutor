# Design: Per-Concept Reset of `strategies_tried`

**Date**: 2026-04-16  
**Status**: Approved

## Problem

`strategies_tried` accumulates session-wide. When a student switches to a new topic, the pool of strategies is already partially (or fully) depleted from the previous topic, causing the tutor to behave as if strategies have already been attempted for the new concept.

## Goal

Reset `strategies_tried` to `[]` whenever the student switches to a semantically different topic. A fresh start is always used — strategies are not preserved if the student returns to an old topic.

## Approach: Extend `classify_question` with topic-change detection (LLM)

`classify_question` already runs every turn and has access to the full state. We extend it to also compare the new concept against the previous concept and output a `same_topic` flag.

## Changes

### 1. `tutor/state.py` — Remove `operator.add` from `strategies_tried`

Change `strategies_tried` from a LangGraph append-reducer field to a plain `list`. This allows any node to overwrite it entirely.

```python
# Before
strategies_tried: Annotated[list, operator.add]

# After
strategies_tried: list
```

### 2. `tutor/nodes.py` — `reasoning_react_loop` manual accumulation

Since the reducer is removed, the node must now return the full accumulated list instead of only new strategies.

```python
existing = state.get("strategies_tried", [])
new_strategies: list = []
# ... loop: append to new_strategies when scaffold_hint returns a strategy ...
return {"strategies_tried": existing + new_strategies, ...}
```

### 3. `tutor/nodes.py` — `classify_question` extended prompt and output parsing

**Initial turn** (`state.get("concept")` is None or empty): no comparison performed. Return the existing 2-field format. `strategies_tried` is already `[]`.

**Subsequent turns**: inject the previous concept into the system prompt and request a 3-field output:

```
<type>|<concept>|<same_topic>
```

Prompt addition:
```
The student's previous topic was: "{prev_concept}".
Also determine whether the new question is about the same topic.
same_topic = true if semantically the same, false if the topic has changed.
Output format: <type>|<concept>|<same_topic>
Examples: reasoning|simple addition|true, factual|moon distance|false
```

**Parsing logic**:
- 3 parts, `same_topic == "false"` → return `{"concept": concept, "question_type": ..., "strategies_tried": []}`
- 3 parts, `same_topic == "true"` → return `{"concept": concept, "question_type": ...}` (no change to `strategies_tried`)
- 2 parts (initial turn or parse failure) → return `{"concept": concept, "question_type": ...}` (no change to `strategies_tried`)

## Data Flow

```
Turn N:
  classify_question
    reads: state["concept"] (previous concept)
    new concept extracted from LLM
    if prev_concept exists:
      LLM also outputs same_topic flag
      if same_topic=false → return strategies_tried: []
    → routes to reasoning_react_loop

  reasoning_react_loop
    reads: state["strategies_tried"] ([] if reset, or accumulated)
    accumulates new strategies manually
    returns: strategies_tried = existing + new_strategies
```

## Testing

Three new test cases in the existing test suite:

1. **Same concept, no reset**: Two reasoning turns on `"simple addition"` — `strategies_tried` accumulates across turns.
2. **Different concept, reset**: Reasoning turn on `"simple addition"`, then reasoning turn on `"word problems"` — `strategies_tried` is `[]` at start of second loop.
3. **Initial turn safety**: First turn with no previous concept — no crash, no reset attempted, `strategies_tried` stays `[]`.

Mock the `_classifier_llm` response in tests to return the 3-part format and verify `classify_question` correctly resets or preserves `strategies_tried`.

## Files to Modify

- `tutor/state.py` — remove `operator.add` from `strategies_tried`
- `tutor/nodes.py` — extend `classify_question` prompt + parsing; update `reasoning_react_loop` accumulation
- `tests/` — add 3 new test cases
