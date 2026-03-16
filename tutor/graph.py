from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from tutor.state import TutorState
from tutor.nodes import (
    assess_question,
    scaffold_hint,
    check_understanding,
    encourage,
    reinforce_concept,
    escalate,
    resume_session,
)


# ---------------------------------------------------------------------------
# Routing functions (pure — no LLM calls, safe to unit test)
# ---------------------------------------------------------------------------

def entry_router(state: TutorState) -> str:
    """Route the first step of each invocation based on session state."""
    if state.get("session_paused"):
        return "resume_session"
    if not state.get("concept"):
        return "assess_question"
    return "check_understanding"


def route_after_check(state: TutorState) -> str:
    """Route after check_understanding based on understanding_level."""
    level = state.get("understanding_level", "stuck")
    return {
        "got_it":      "reinforce_concept",
        "progressing": "scaffold_hint",
        "stuck":       "scaffold_hint",
        "incorrect":   "scaffold_hint",
        "frustrated":  "encourage",
        "distressed":  "escalate",
    }.get(level, "scaffold_hint")


# ---------------------------------------------------------------------------
# Graph definition
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(TutorState)

    # Register all nodes
    graph.add_node("assess_question",    assess_question)
    graph.add_node("scaffold_hint",      scaffold_hint)
    graph.add_node("check_understanding", check_understanding)
    graph.add_node("encourage",          encourage)
    graph.add_node("reinforce_concept",  reinforce_concept)
    graph.add_node("escalate",           escalate)
    graph.add_node("resume_session",     resume_session)

    # Entry: conditional branch from START
    graph.add_conditional_edges(
        START,
        entry_router,
        {
            "assess_question":    "assess_question",
            "check_understanding": "check_understanding",
            "resume_session":     "resume_session",
        },
    )

    # assess_question → scaffold_hint → END (first turn ends after first hint)
    graph.add_edge("assess_question", "scaffold_hint")
    graph.add_edge("scaffold_hint", END)

    # resume_session → scaffold_hint (give next hint after welcome back)
    graph.add_edge("resume_session", "scaffold_hint")

    # check_understanding → conditional routing
    graph.add_conditional_edges(
        "check_understanding",
        route_after_check,
        {
            "reinforce_concept": "reinforce_concept",
            "scaffold_hint":     "scaffold_hint",
            "encourage":         "encourage",
            "escalate":          "escalate",
        },
    )

    # encourage → scaffold_hint → END
    graph.add_edge("encourage", "scaffold_hint")

    # reinforce_concept → END (session complete)
    graph.add_edge("reinforce_concept", END)

    # escalate → END (session paused, state checkpointed)
    graph.add_edge("escalate", END)

    return graph


# ---------------------------------------------------------------------------
# Compiled app (with MemorySaver for pause-and-resume)
# ---------------------------------------------------------------------------

checkpointer = MemorySaver()
tutor_app = build_graph().compile(checkpointer=checkpointer)
