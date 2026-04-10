"""LLM-as-judge evals for v2 nodes.

Run with:  python tests/evals/run_llm_evals.py

Requires OPENAI_API_KEY. Uses live API calls — NOT safe for CI.
Results are written to tests/evals/results/eval_report_<timestamp>.csv
"""

import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).parent / "results"
ALL_STRATEGIES = ["guiding_question", "analogy", "concrete_example", "sub_problem", "number_line"]


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


def _judge(prompt: str, model: str = "gpt-4o-mini") -> str:
    """Call LLM judge and return its raw text response."""
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model=model, temperature=0)
    response = llm.invoke([{"role": "user", "content": prompt}])
    return response.content.strip()


def _pass_fail(judge_response: str) -> str:
    text = judge_response.lower()
    if text.startswith("pass"):
        return "PASS"
    if text.startswith("fail"):
        return "FAIL"
    # Fallback: look for the word anywhere
    if "pass" in text:
        return "PASS"
    return "FAIL"


# ---------------------------------------------------------------------------
# Eval 1: age_appropriateness
# ---------------------------------------------------------------------------

def eval_age_appropriateness() -> list[dict]:
    """Test that format_response rewrites complex text for Grade 1 reading level."""
    from tutor.nodes import format_response

    test_cases = [
        {
            "id": "age_1",
            "complex_text": (
                "The mathematical operation of addition requires combining two numerical values "
                "to produce a sum, which in this case results in the integer fifteen."
            ),
        },
        {
            "id": "age_2",
            "complex_text": (
                "Photosynthesis is the biological process by which plants utilise chlorophyll "
                "to convert solar radiation into chemical energy stored as glucose."
            ),
        },
        {
            "id": "age_3",
            "complex_text": (
                "Paris is the capital and most populous city of France, serving as the nation's "
                "political, economic, and cultural centre."
            ),
        },
    ]

    rows = []
    for tc in test_cases:
        state = _full_state(current_response=tc["complex_text"])
        result = format_response(state)
        formatted = result.get("current_response", "")

        judge_prompt = (
            f"You are evaluating whether a tutor response is appropriate for a 6-year-old "
            f"Grade 1 student (5–6 years old).\n\n"
            f"Response to evaluate:\n\"{formatted}\"\n\n"
            f"Criteria — the response must:\n"
            f"1. Use sentences of 10 words or fewer.\n"
            f"2. Use only words a 6-year-old would understand.\n"
            f"3. Not use jargon or technical vocabulary.\n\n"
            f"Reply with PASS if all criteria are met, or FAIL followed by a brief reason."
        )
        judge_response = _judge(judge_prompt)
        verdict = _pass_fail(judge_response)
        rows.append({
            "eval": "age_appropriateness",
            "case_id": tc["id"],
            "input": tc["complex_text"][:80],
            "output": formatted[:120],
            "verdict": verdict,
            "judge_notes": judge_response[:200],
        })

    return rows


# ---------------------------------------------------------------------------
# Eval 2: question_classification
# ---------------------------------------------------------------------------

def eval_question_classification() -> list[dict]:
    """Direct comparison — no judge needed. Test 5 factual + 5 reasoning examples."""
    from tutor.nodes import classify_question

    test_cases = [
        # factual
        {"id": "cls_f1", "input": "What is the capital of France?",     "expected": "factual"},
        {"id": "cls_f2", "input": "How many legs does a spider have?",  "expected": "factual"},
        {"id": "cls_f3", "input": "What colour is the sky?",            "expected": "factual"},
        {"id": "cls_f4", "input": "Who was the first person on the moon?", "expected": "factual"},
        {"id": "cls_f5", "input": "What is the biggest planet?",        "expected": "factual"},
        # reasoning
        {"id": "cls_r1", "input": "What is 7 + 8?",                    "expected": "reasoning"},
        {"id": "cls_r2", "input": "If I have 5 apples and eat 2, how many are left?", "expected": "reasoning"},
        {"id": "cls_r3", "input": "What is 12 minus 4?",               "expected": "reasoning"},
        {"id": "cls_r4", "input": "How do I count by 5s to 30?",       "expected": "reasoning"},
        {"id": "cls_r5", "input": "If I share 10 cookies with 2 friends equally, how many each?", "expected": "reasoning"},
    ]

    rows = []
    for tc in test_cases:
        state = _full_state(student_input=tc["input"])
        result = classify_question(state)
        got = result.get("question_type", "")
        verdict = "PASS" if got == tc["expected"] else "FAIL"
        rows.append({
            "eval": "question_classification",
            "case_id": tc["id"],
            "input": tc["input"],
            "output": got,
            "verdict": verdict,
            "judge_notes": f"expected={tc['expected']}",
        })

    return rows


# ---------------------------------------------------------------------------
# Eval 3: strategy_selection
# ---------------------------------------------------------------------------

def eval_strategy_selection() -> list[dict]:
    """Check that the reasoning loop does NOT repeat an already-tried strategy."""
    from tutor.nodes import reasoning_react_loop

    test_cases = [
        {
            "id": "strat_1",
            "concept": "addition",
            "student_input": "I still don't get it",
            "strategies_tried": ["guiding_question"],
            "history": [
                {"role": "user", "content": "What is 7 + 8?"},
                {"role": "assistant", "content": "What do you get if you count 7 then 8 more?"},
            ],
        },
        {
            "id": "strat_2",
            "concept": "subtraction",
            "student_input": "I don't understand",
            "strategies_tried": ["guiding_question", "analogy"],
            "history": [
                {"role": "user", "content": "What is 10 minus 3?"},
                {"role": "assistant", "content": "Imagine you have 10 apples. If you eat 3, how many are left?"},
            ],
        },
    ]

    rows = []
    for tc in test_cases:
        state = _full_state(
            student_input=tc["student_input"],
            concept=tc["concept"],
            question_type="reasoning",
            strategies_tried=tc["strategies_tried"],
            conversation_history=tc["history"],
        )
        result = reasoning_react_loop(state)
        new_strategies = result.get("strategies_tried", [])
        repeated = [s for s in new_strategies if s in tc["strategies_tried"]]
        verdict = "PASS" if not repeated else "FAIL"
        rows.append({
            "eval": "strategy_selection",
            "case_id": tc["id"],
            "input": f"strategies_tried={tc['strategies_tried']}",
            "output": f"new_strategies={new_strategies}",
            "verdict": verdict,
            "judge_notes": f"repeated={repeated}" if repeated else "no repetition",
        })

    return rows


# ---------------------------------------------------------------------------
# Eval 4: exhaustion_direct_answer
# ---------------------------------------------------------------------------

def eval_exhaustion_direct_answer() -> list[dict]:
    """Test that with all 5 strategies exhausted, the response is a direct answer."""
    from tutor.nodes import reasoning_react_loop

    test_cases = [
        {
            "id": "exhaust_1",
            "concept": "addition to 20",
            "student_input": "I still don't know",
            "strategies_tried": ALL_STRATEGIES,
        },
        {
            "id": "exhaust_2",
            "concept": "subtraction within 10",
            "student_input": "I give up",
            "strategies_tried": ALL_STRATEGIES,
        },
    ]

    rows = []
    for tc in test_cases:
        state = _full_state(
            student_input=tc["student_input"],
            concept=tc["concept"],
            question_type="reasoning",
            strategies_tried=tc["strategies_tried"],
        )
        result = reasoning_react_loop(state)
        response = result.get("current_response", "")

        judge_prompt = (
            f"A tutor has exhausted all scaffolding strategies and gave this response to a "
            f"6-year-old who is stuck on '{tc['concept']}':\n\n"
            f"\"{response}\"\n\n"
            f"Does this response give the student the direct answer (rather than asking "
            f"another hint question or refusing to answer)?\n\n"
            f"Reply with PASS if the response directly states the answer, "
            f"or FAIL followed by a brief reason."
        )
        judge_response = _judge(judge_prompt)
        verdict = _pass_fail(judge_response)

        # Also check REVIEW signal was emitted (concepts_needing_review populated)
        review_populated = len(result.get("concepts_needing_review", [])) > 0
        rows.append({
            "eval": "exhaustion_direct_answer",
            "case_id": tc["id"],
            "input": f"concept={tc['concept']}, strategies_tried=all 5",
            "output": response[:120],
            "verdict": verdict,
            "judge_notes": f"{judge_response[:150]} | review_populated={review_populated}",
        })

    return rows


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"eval_report_{timestamp}.csv"

    fieldnames = ["eval", "case_id", "input", "output", "verdict", "judge_notes"]

    all_rows: list[dict] = []

    evals = [
        ("age_appropriateness",      eval_age_appropriateness),
        ("question_classification",  eval_question_classification),
        ("strategy_selection",       eval_strategy_selection),
        ("exhaustion_direct_answer", eval_exhaustion_direct_answer),
    ]

    for name, fn in evals:
        print(f"\n--- Running eval: {name} ---")
        try:
            rows = fn()
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            rows = [{
                "eval": name,
                "case_id": "ERROR",
                "input": "",
                "output": "",
                "verdict": "ERROR",
                "judge_notes": str(exc)[:200],
            }]
        for row in rows:
            verdict = row.get("verdict", "?")
            print(f"  [{verdict}] {row.get('case_id', '?')}")
        all_rows.extend(rows)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    total = len(all_rows)
    passed = sum(1 for r in all_rows if r["verdict"] == "PASS")
    print(f"\n=== Results: {passed}/{total} passed ===")
    print(f"Report written to: {output_path}")


if __name__ == "__main__":
    run_all()
