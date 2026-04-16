"""SSE streaming utilities for the tutor API."""

import json
import logging
from typing import AsyncIterator


async def _build_input(graph, thread_id: str, message: str) -> dict:
    """Build the LangGraph input dict, seeding list fields on the first turn."""
    config = {"configurable": {"thread_id": thread_id}}
    state_update = {"student_input": message}
    if not (await graph.aget_state(config)).values:
        state_update.update({
            "strategies_tried": [],
            "concepts_needing_review": [],
            "conversation_history": [],
            "session_paused": False,
        })
    return state_update


def _extract_metadata(state_snapshot) -> dict:
    """Pull session metadata from a LangGraph state snapshot."""
    v = state_snapshot.values
    return {
        "session_paused": v.get("session_paused", False),
        "concept": v.get("concept", ""),
        "concepts_needing_review": v.get("concepts_needing_review", []),
        "conversation_history": v.get("conversation_history", []),
    }


async def stream_chat(graph, thread_id: str, message: str) -> AsyncIterator[str]:
    """Async generator that yields SSE-formatted strings.

    Emits:
      event: token  — one per LLM output token (skips classify_question tokens)
      event: done   — once, with session metadata after the graph finishes
      event: error  — if an exception occurs
    """
    config = {"configurable": {"thread_id": thread_id}}
    input_state = await _build_input(graph, thread_id, message)

    try:
        async for event in graph.astream_events(input_state, config, version="v2"):
            if event["event"] == "on_chat_model_stream":
                node = event.get("metadata", {}).get("langgraph_node", "")
                if node == "classify_question":
                    continue
                chunk = event["data"]["chunk"].content
                if chunk:
                    yield f"event: token\ndata: {json.dumps({'chunk': chunk})}\n\n"

        state = await graph.aget_state(config)
        yield f"event: done\ndata: {json.dumps(_extract_metadata(state))}\n\n"
    except Exception:
        logging.exception("stream_chat error for thread_id=%s", thread_id)
        yield f"event: error\ndata: {json.dumps({'message': 'Something went wrong. Please try again.'})}\n\n"
