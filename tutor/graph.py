from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from tutor.state import TutorState
from tutor.nodes import (
    classify_question,
    factual_react_loop,
    reasoning_react_loop,
    escalate,
    resume_session,
)


# ---------------------------------------------------------------------------
# Routing functions (pure — no LLM calls, safe to unit test)
# ---------------------------------------------------------------------------

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
    return END


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

builder = StateGraph(TutorState)

builder.add_node("classify_question", classify_question)
builder.add_node("factual_react_loop", factual_react_loop)
builder.add_node("reasoning_react_loop", reasoning_react_loop)
builder.add_node("escalate", escalate)
builder.add_node("resume_session", resume_session)

builder.add_conditional_edges(START, entry_router)
builder.add_conditional_edges("classify_question", route_after_classify)
builder.add_conditional_edges("reasoning_react_loop", route_after_reasoning)
builder.add_edge("resume_session", END)
builder.add_edge("factual_react_loop", END)
builder.add_edge("escalate", END)

memory = MemorySaver()
tutor = builder.compile(checkpointer=memory)
