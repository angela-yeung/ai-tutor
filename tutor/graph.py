from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from tutor.state import TutorState
from tutor.nodes import (
    classify_question,
    factual_react_loop,
    reasoning_react_loop,
    escalate,
    resume_session,
    input_guard,
    output_guard,
)


# ---------------------------------------------------------------------------
# Routing functions (pure — no LLM calls, safe to unit test)
# ---------------------------------------------------------------------------

def route_after_input_guard(state: TutorState) -> str:
    if state.get("input_blocked"):
        return "output_guard"
    return "entry_router"


def entry_router(state: TutorState) -> str:
    if state.get("session_paused"):
        return "resume_session"
    return "classify_question"


def route_after_classify(state: TutorState) -> str:
    if state.get("question_type") == "factual":
        return "factual_react_loop"
    return "reasoning_react_loop"


def route_after_reasoning(state: TutorState) -> str:
    if state.get("session_paused"):
        return "escalate"
    return "output_guard"


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

def build_graph(checkpointer=None):
    """Compile the tutor graph with the given checkpointer.

    Pass a RedisSaver in production; omit (or pass None) to use MemorySaver.
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    builder = StateGraph(TutorState)

    builder.add_node("input_guard", input_guard)
    builder.add_node("entry_router", lambda state: {})      # pass-through; routing via entry_router fn
    builder.add_node("classify_question", classify_question)
    builder.add_node("factual_react_loop", factual_react_loop)
    builder.add_node("reasoning_react_loop", reasoning_react_loop)
    builder.add_node("escalate", escalate)
    builder.add_node("resume_session", resume_session)
    builder.add_node("output_guard", output_guard)

    builder.add_edge(START, "input_guard")
    builder.add_conditional_edges("input_guard", route_after_input_guard)
    builder.add_conditional_edges("entry_router", entry_router)
    builder.add_conditional_edges("classify_question", route_after_classify)
    builder.add_conditional_edges("reasoning_react_loop", route_after_reasoning)
    builder.add_edge("factual_react_loop", "output_guard")
    builder.add_edge("escalate", "output_guard")
    builder.add_edge("resume_session", "output_guard")
    builder.add_edge("output_guard", END)

    return builder.compile(checkpointer=checkpointer)


# Module-level instance — used by cli.py and existing tests
tutor = build_graph()
