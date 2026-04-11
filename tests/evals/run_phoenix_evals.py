"""Phoenix experiment evals for the AI Tutor.

Run with: python tests/evals/run_phoenix_evals.py

Requires OPENAI_API_KEY and a running Phoenix server (phoenix serve).
Each run uploads a fresh dataset snapshot and appends a new experiment
run — compare runs in the Phoenix UI for regression testing.

Evaluators:
  1. classification_correctness  — classify_question node
  2. factual_tool_selection       — factual_react_loop tool calls
  3. reasoning_tool_selection     — reasoning_react_loop tool calls
  4. distress_detection           — reasoning_react_loop escalation routing
  5. resume_recall                — resume_session picks up prior concept
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path so `tutor` is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import nest_asyncio
import pandas as pd
import phoenix as px
from dotenv import load_dotenv, find_dotenv
from phoenix.evals import OpenAIModel, llm_classify, TOOL_CALLING_PROMPT_TEMPLATE
from phoenix.experiments import run_experiment
from phoenix.experiments.types import Example, EvaluationResult
from phoenix.experiments.evaluators import create_evaluator

nest_asyncio.apply()
load_dotenv(find_dotenv())

eval_model = OpenAIModel(model="gpt-4o")

# ---------------------------------------------------------------------------
# Tool schemas for TOOL_CALLING_PROMPT_TEMPLATE
# ---------------------------------------------------------------------------

FACTUAL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Searches the web for factual information.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
]

REASONING_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluates a simple arithmetic expression (+-*/^%).",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scaffold_hint",
            "description": "Selects the best Socratic hint strategy not already tried.",
            "parameters": {
                "type": "object",
                "properties": {
                    "concept": {"type": "string"},
                    "strategies_tried": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["concept", "strategies_tried"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# LLM judge prompts
# ---------------------------------------------------------------------------

RESUME_RECALL_PROMPT = """
In this task, you will evaluate whether a tutor's welcome-back message correctly acknowledges
what a student was previously working on.

[BEGIN DATA]
The student was previously working on: {concept}
Tutor's welcome-back message: {response}
[END DATA]

Check: Does the tutor's message clearly mention or reference the concept "{concept}"?

EXPLANATION: Look for direct or indirect mention of the concept.
LABEL: "correct" if the concept is clearly referenced, "incorrect" if absent or wrong.
"""

# ---------------------------------------------------------------------------
# Dataset rows (29 examples)
# ---------------------------------------------------------------------------

DATASET_ROWS = [
    # ── classify (10) ──────────────────────────────────────────────────────
    {
        "eval_type": "classify", "format_focus": "",
        "student_input": "What is the capital of Australia?",
        "concept": "", "strategies_tried": [], "conversation_history": [], "current_response": "",
        "expected_question_type": "factual", "expected_session_paused": None,
    },
    {
        "eval_type": "classify", "format_focus": "",
        "student_input": "How many legs does a spider have?",
        "concept": "", "strategies_tried": [], "conversation_history": [], "current_response": "",
        "expected_question_type": "factual", "expected_session_paused": None,
    },
    {
        "eval_type": "classify", "format_focus": "",
        "student_input": "Who invented the telephone?",
        "concept": "", "strategies_tried": [], "conversation_history": [], "current_response": "",
        "expected_question_type": "factual", "expected_session_paused": None,
    },
    {
        "eval_type": "classify", "format_focus": "",
        "student_input": "What colour is a flamingo?",
        "concept": "", "strategies_tried": [], "conversation_history": [], "current_response": "",
        "expected_question_type": "factual", "expected_session_paused": None,
    },
    {
        "eval_type": "classify", "format_focus": "",
        "student_input": "What is the biggest animal in the ocean?",
        "concept": "", "strategies_tried": [], "conversation_history": [], "current_response": "",
        "expected_question_type": "factual", "expected_session_paused": None,
    },
    {
        "eval_type": "classify", "format_focus": "",
        "student_input": "What is 9 + 6?",
        "concept": "", "strategies_tried": [], "conversation_history": [], "current_response": "",
        "expected_question_type": "reasoning", "expected_session_paused": None,
    },
    {
        "eval_type": "classify", "format_focus": "",
        "student_input": "If I have 12 cookies and eat 4, how many are left?",
        "concept": "", "strategies_tried": [], "conversation_history": [], "current_response": "",
        "expected_question_type": "reasoning", "expected_session_paused": None,
    },
    {
        "eval_type": "classify", "format_focus": "",
        "student_input": "What is 15 divided by 3?",
        "concept": "", "strategies_tried": [], "conversation_history": [], "current_response": "",
        "expected_question_type": "reasoning", "expected_session_paused": None,
    },
    {
        "eval_type": "classify", "format_focus": "",
        "student_input": "How do I count by 2s to 20?",
        "concept": "", "strategies_tried": [], "conversation_history": [], "current_response": "",
        "expected_question_type": "reasoning", "expected_session_paused": None,
    },
    {
        "eval_type": "classify", "format_focus": "",
        "student_input": "If each box has 5 crayons and I have 3 boxes, how many crayons do I have?",
        "concept": "", "strategies_tried": [], "conversation_history": [], "current_response": "",
        "expected_question_type": "reasoning", "expected_session_paused": None,
    },

    # ── factual (6) ────────────────────────────────────────────────────────
    {
        "eval_type": "factual", "format_focus": "",
        "student_input": "Who is the current Prime Minister of Australia?",
        "concept": "current PM of Australia", "strategies_tried": [], "conversation_history": [], "current_response": "",
        "expected_question_type": None, "expected_session_paused": None,
    },
    {
        "eval_type": "factual", "format_focus": "",
        "student_input": "Who won the most recent FIFA World Cup?",
        "concept": "FIFA World Cup", "strategies_tried": [], "conversation_history": [], "current_response": "",
        "expected_question_type": None, "expected_session_paused": None,
    },
    {
        "eval_type": "factual", "format_focus": "",
        "student_input": "What is the tallest building in the world right now?",
        "concept": "tallest building", "strategies_tried": [], "conversation_history": [], "current_response": "",
        "expected_question_type": None, "expected_session_paused": None,
    },
    {
        "eval_type": "factual", "format_focus": "",
        "student_input": "What sound does a dog make?",
        "concept": "animal sounds", "strategies_tried": [], "conversation_history": [], "current_response": "",
        "expected_question_type": None, "expected_session_paused": None,
    },
    {
        "eval_type": "factual", "format_focus": "",
        "student_input": "How many days are in a week?",
        "concept": "days of the week", "strategies_tried": [], "conversation_history": [], "current_response": "",
        "expected_question_type": None, "expected_session_paused": None,
    },
    {
        "eval_type": "factual", "format_focus": "",
        "student_input": "What is the biggest planet in our solar system?",
        "concept": "planets", "strategies_tried": [], "conversation_history": [], "current_response": "",
        "expected_question_type": None, "expected_session_paused": None,
    },

    # ── reasoning (8) ──────────────────────────────────────────────────────
    {
        "eval_type": "reasoning", "format_focus": "",
        "student_input": "What is 8 + 5?",
        "concept": "addition", "strategies_tried": [], "conversation_history": [], "current_response": "",
        "expected_question_type": None, "expected_session_paused": False,
    },
    {
        "eval_type": "reasoning", "format_focus": "",
        "student_input": "I still don't understand addition",
        "concept": "addition", "strategies_tried": ["guiding_question"],
        "conversation_history": [
            {"role": "user", "content": "What is 8 + 5?"},
            {"role": "assistant", "content": "What do you get if you start at 8 and count 5 more?"},
        ],
        "current_response": "",
        "expected_question_type": None, "expected_session_paused": False,
    },
    {
        "eval_type": "reasoning", "format_focus": "",
        "student_input": "Is 6 times 7 the same as 42?",
        "concept": "multiplication", "strategies_tried": [], "conversation_history": [], "current_response": "",
        "expected_question_type": None, "expected_session_paused": False,
    },
    {
        "eval_type": "reasoning", "format_focus": "",
        "student_input": "Is 13 + 9 equal to 21 or 22?",
        "concept": "addition", "strategies_tried": [], "conversation_history": [], "current_response": "",
        "expected_question_type": None, "expected_session_paused": False,
    },
    {
        "eval_type": "reasoning", "format_focus": "",
        "student_input": "I hate maths! I want to give up!",
        "concept": "subtraction", "strategies_tried": ["guiding_question"], "conversation_history": [], "current_response": "",
        "expected_question_type": None, "expected_session_paused": True,
    },
    {
        "eval_type": "reasoning", "format_focus": "",
        "student_input": "I am crying. This is too hard for me.",
        "concept": "addition", "strategies_tried": ["guiding_question", "analogy"], "conversation_history": [], "current_response": "",
        "expected_question_type": None, "expected_session_paused": True,
    },
    {
        "eval_type": "reasoning", "format_focus": "",
        "student_input": "I can't do this. I want to quit right now.",
        "concept": "counting", "strategies_tried": ["guiding_question"], "conversation_history": [], "current_response": "",
        "expected_question_type": None, "expected_session_paused": True,
    },
    {
        "eval_type": "reasoning", "format_focus": "",
        "student_input": "I still don't get it, can you help me again?",
        "concept": "subtraction", "strategies_tried": ["guiding_question"], "conversation_history": [], "current_response": "",
        "expected_question_type": None, "expected_session_paused": False,
    },

    # ── resume (5) ─────────────────────────────────────────────────────────
    {
        "eval_type": "resume", "format_focus": "",
        "student_input": "I am back", "concept": "addition", "strategies_tried": [],
        "conversation_history": [
            {"role": "user", "content": "What is 7 + 8?"},
            {"role": "assistant", "content": "How many do you get if you count 7 then 8 more?"},
        ],
        "current_response": "",
        "expected_question_type": None, "expected_session_paused": None,
    },
    {
        "eval_type": "resume", "format_focus": "",
        "student_input": "I am back", "concept": "fractions", "strategies_tried": [],
        "conversation_history": [
            {"role": "user", "content": "What is half of 6?"},
            {"role": "assistant", "content": "If you split 6 cookies into 2 equal groups, how many in each group?"},
        ],
        "current_response": "",
        "expected_question_type": None, "expected_session_paused": None,
    },
    {
        "eval_type": "resume", "format_focus": "",
        "student_input": "I am back", "concept": "counting by 5s", "strategies_tried": [],
        "conversation_history": [
            {"role": "user", "content": "Help me count by 5s to 30"},
            {"role": "assistant", "content": "Let's start: 5, 10 — what comes next?"},
        ],
        "current_response": "",
        "expected_question_type": None, "expected_session_paused": None,
    },
    {
        "eval_type": "resume", "format_focus": "",
        "student_input": "I am back", "concept": "subtraction", "strategies_tried": [],
        "conversation_history": [
            {"role": "user", "content": "What is 10 minus 4?"},
            {"role": "assistant", "content": "If you had 10 fingers and folded 4 down, how many are still up?"},
        ],
        "current_response": "",
        "expected_question_type": None, "expected_session_paused": None,
    },
    {
        "eval_type": "resume", "format_focus": "",
        "student_input": "I am back", "concept": "shapes and colours", "strategies_tried": [],
        "conversation_history": [
            {"role": "user", "content": "How many sides does a triangle have?"},
            {"role": "assistant", "content": "Think of a slice of pizza — how many edges does it have?"},
        ],
        "current_response": "",
        "expected_question_type": None, "expected_session_paused": None,
    },
]

# ---------------------------------------------------------------------------
# State helper
# ---------------------------------------------------------------------------

def _full_state(**overrides) -> dict:
    base = {
        "student_input": "",
        "concept": "",
        "question_type": "",
        "strategies_tried": [],
        "conversation_history": [],
        "current_response": "",
        "session_paused": False,
        "concepts_needing_review": [],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Task functions
# ---------------------------------------------------------------------------

def task_classify(example: Example) -> dict:
    from tutor.nodes import classify_question
    state = _full_state(student_input=example.input["student_input"])
    return classify_question(state)


def task_factual_loop(example: Example) -> dict:
    from tutor.nodes import factual_react_loop
    state = _full_state(
        student_input=example.input["student_input"],
        concept=example.input.get("concept", ""),
    )
    return factual_react_loop(state)


def task_reasoning_loop(example: Example) -> dict:
    from tutor.nodes import reasoning_react_loop
    state = _full_state(
        student_input=example.input["student_input"],
        concept=example.input.get("concept", ""),
        strategies_tried=example.input.get("strategies_tried", []),
        conversation_history=example.input.get("conversation_history", []),
    )
    return reasoning_react_loop(state)


def task_resume_session(example: Example) -> dict:
    from tutor.nodes import resume_session
    state = _full_state(
        concept=example.input["concept"],
        conversation_history=example.input.get("conversation_history", []),
        session_paused=True,
    )
    return resume_session(state)


def task_route(example: Example) -> dict:
    dispatch = {
        "classify":  task_classify,
        "factual":   task_factual_loop,
        "reasoning": task_reasoning_loop,
        "resume":    task_resume_session,
    }
    return dispatch[example.input["eval_type"]](example)


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------

@create_evaluator(name="classification_correctness")
def eval_classification_correctness(input, output, expected) -> bool | None:
    """Evaluator 1: classify_question node returns correct question type."""
    if input.get("eval_type") != "classify":
        return EvaluationResult(score=None, label="N/A")
    return output.get("question_type") == expected.get("expected_question_type")


@create_evaluator(name="factual_tool_selection")
def eval_factual_tool_selection(input, output):
    """Evaluator 2: factual_react_loop calls the correct tool (TOOL_CALLING_PROMPT_TEMPLATE)."""
    if input.get("eval_type") != "factual":
        return EvaluationResult(score=None, label="N/A")
    tools_called = output.get("tools_called", [])
    if not tools_called:
        return 0.0
    tool_defs = json.dumps(FACTUAL_TOOLS).replace("{", '"').replace("}", '"')
    template = TOOL_CALLING_PROMPT_TEMPLATE.template[0].template.replace(
        "{tool_definitions}", tool_defs
    )
    eval_df = pd.DataFrame({
        "question": [input["student_input"]] * len(tools_called),
        "tool_call": tools_called,
    })
    result = llm_classify(
        data=eval_df,
        template=template,
        rails=["correct", "incorrect"],
        model=eval_model,
        provide_explanation=True,
    )
    result["score"] = result["label"].apply(lambda x: 1.0 if x == "correct" else 0.0)
    return float(result["score"].mean())


@create_evaluator(name="reasoning_tool_selection")
def eval_reasoning_tool_selection(input, output):
    """Evaluator 3: reasoning_react_loop calls the correct tool (TOOL_CALLING_PROMPT_TEMPLATE).

    Returns N/A for distress examples (session_paused=True) because those break
    out of the while loop before any tool call — this is correct behaviour, not a failure.
    """
    if input.get("eval_type") != "reasoning":
        return EvaluationResult(score=None, label="N/A")
    if output.get("session_paused"):
        return EvaluationResult(score=None, label="N/A")  # distress path — no tool calls expected
    tools_called = output.get("tools_called", [])
    if not tools_called:
        return 0.0
    tool_defs = json.dumps(REASONING_TOOLS).replace("{", '"').replace("}", '"')
    template = TOOL_CALLING_PROMPT_TEMPLATE.template[0].template.replace(
        "{tool_definitions}", tool_defs
    )
    eval_df = pd.DataFrame({
        "question": [input["student_input"]] * len(tools_called),
        "tool_call": tools_called,
    })
    result = llm_classify(
        data=eval_df,
        template=template,
        rails=["correct", "incorrect"],
        model=eval_model,
        provide_explanation=True,
    )
    result["score"] = result["label"].apply(lambda x: 1.0 if x == "correct" else 0.0)
    return float(result["score"].mean())


@create_evaluator(name="distress_detection")
def eval_distress_detection(input, output, expected):
    """Evaluator 4: reasoning_react_loop correctly detects distress and sets session_paused."""
    if input.get("eval_type") != "reasoning":
        return EvaluationResult(score=None, label="N/A")
    expected_paused = expected.get("expected_session_paused")
    if expected_paused is None:
        return EvaluationResult(score=None, label="N/A")
    return output.get("session_paused") == expected_paused


@create_evaluator(name="resume_recall")
def eval_resume_recall(input, output, expected):
    """Evaluator 5: resume_session correctly references the concept the student was working on."""
    if input.get("eval_type") != "resume":
        return EvaluationResult(score=None, label="N/A")
    concept = expected.get("expected_concept") or input.get("concept", "")
    df = pd.DataFrame({
        "concept":  [concept],
        "response": [output.get("current_response", "")],
    })
    result = llm_classify(
        data=df,
        template=RESUME_RECALL_PROMPT,
        rails=["correct", "incorrect"],
        model=eval_model,
        provide_explanation=True,
    )
    return result["label"].iloc[0] == "correct"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_experiments() -> None:
    px_client = px.Client()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df = pd.DataFrame(DATASET_ROWS)
    # expected_concept: concept value for resume rows, None for all others
    df["expected_concept"] = df.apply(
        lambda r: r["concept"] if r["eval_type"] == "resume" else None, axis=1
    )

    print(f"Uploading dataset ({len(df)} examples)...")
    dataset = px_client.upload_dataset(
        dataframe=df,
        dataset_name=f"ai_tutor_full_eval-{now}",
        input_keys=[
            "eval_type", "format_focus", "student_input", "concept",
            "strategies_tried", "conversation_history", "current_response",
        ],
        output_keys=["expected_question_type", "expected_session_paused", "expected_concept"],
    )

    print("Running experiment...")
    run_experiment(
        dataset,
        task_route,
        evaluators=[
            eval_classification_correctness,
            eval_factual_tool_selection,
            eval_reasoning_tool_selection,
            eval_distress_detection,
            eval_resume_recall,
        ],
        experiment_name=f"ai_tutor_full_eval-{now}",
        experiment_description=(
            "Full regression eval across all 4 nodes: classify_question, "
            "factual_react_loop, reasoning_react_loop, resume_session."
        ),
    )
    print("Done. Open Phoenix UI to view results.")


if __name__ == "__main__":
    run_all_experiments()
