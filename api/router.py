"""API route handlers."""

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from api.schemas import ChatRequest
from api.streaming import stream_chat

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/chat")
async def chat(req: ChatRequest, request: Request):
    tutor = request.app.state.tutor
    return StreamingResponse(
        stream_chat(tutor, req.thread_id, req.message),
        media_type="text/event-stream",
    )
