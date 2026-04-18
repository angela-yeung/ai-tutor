# FastAPI Backend — Sub-project 1: API + Redis + Docker

**Date:** 2026-04-16  
**Scope:** Backend only. Auth (Supabase) is Sub-project 2. Frontend (React/Next.js) is Sub-project 3.

---

## Context

The AI tutor currently runs as a CLI-only tool. To serve a React/Next.js web frontend, it needs an HTTP API layer. This spec covers:

1. A FastAPI server exposing the LangGraph tutor graph over HTTP with SSE streaming
2. Redis replacing the in-memory `MemorySaver` checkpointer so sessions persist across server restarts and scale across container instances
3. Docker containerisation for local dev and Railway deployment

The existing `tutor/` package is **not restructured** — the API is a thin adapter layer on top of it.

---

## Architecture

```
ai-tutor/
├── tutor/                  # Unchanged (graph, nodes, state, tools)
│   ├── graph.py            # MODIFIED: build_graph(checkpointer=None) factory
│   ├── state.py
│   ├── nodes.py
│   ├── tools.py
│   └── instrumentation.py
├── api/                    # NEW
│   ├── __init__.py
│   ├── main.py             # FastAPI app, CORS, lifespan (Redis init)
│   ├── router.py           # POST /chat, GET /health
│   ├── schemas.py          # Pydantic: ChatRequest, TokenEvent, DoneEvent
│   └── streaming.py        # astream_events() → SSE generator
├── tests/
│   ├── ... (existing, unchanged)
│   └── test_api.py         # NEW: FastAPI TestClient tests
├── Dockerfile              # NEW
├── docker-compose.yml      # NEW (local dev: api + redis)
└── requirements.txt        # UPDATED: fastapi, uvicorn, langgraph-redis, httpx
```

---

## Endpoints

### `POST /chat`

Request body:
```json
{
  "thread_id": "uuid-v4-string",
  "message": "What is 2 + 2?"
}
```

- `thread_id`: Frontend generates on first page load (`crypto.randomUUID()`), persists in `localStorage`. Maps directly to LangGraph `configurable.thread_id`.
- `message`: The student's input.

Response: `Content-Type: text/event-stream`

Token events (one per LLM token):
```
event: token
data: {"chunk": "Two"}

event: token
data: {"chunk": " plus two"}
```

Done event (after stream ends, carries final state metadata):
```
event: done
data: {
  "session_paused": false,
  "concept": "addition",
  "concepts_needing_review": [],
  "conversation_history": [{"role": "user", "content": "..."}, ...]
}
```

### `GET /health`

Returns `{"status": "ok"}`. Used by Railway health checks.

---

## Session Identity (v1, no auth)

The frontend generates a UUID `thread_id` on first load and stores it in `localStorage`. All `POST /chat` requests include it. The server uses it as-is as the LangGraph `configurable.thread_id`.

**Auth-ready:** When Supabase auth is added (Sub-project 2), the server will extract `user_id` from the JWT and use that as the `thread_id`. The request body shape is unchanged.

---

## Redis — Replacing MemorySaver

`tutor/graph.py` currently compiles with `MemorySaver()`. Change it to accept an injected checkpointer:

```python
# tutor/graph.py
def build_graph(checkpointer=None):
    if checkpointer is None:
        checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
```

In production (`api/main.py` lifespan):
```python
from langgraph.checkpoint.redis import RedisSaver
checkpointer = RedisSaver.from_conn_string(os.getenv("REDIS_URL"))
tutor = build_graph(checkpointer)
```

In existing tests: `build_graph()` uses `MemorySaver` — **no test changes needed.**

**What Redis stores per `thread_id`:** Full `TutorState` — `conversation_history`, `strategies_tried`, `session_paused`, `concepts_needing_review`, `concept`, `current_response`. Sessions survive server restarts and work across multiple API container replicas.

**User name (v1):** Learned from conversation and stored in `conversation_history`. A dedicated user profile field is added in Sub-project 2 (Supabase).

---

## SSE Streaming Implementation

`api/streaming.py` — async generator:

```python
async def stream_chat(tutor, thread_id: str, message: str) -> AsyncIterator[str]:
    config = {"configurable": {"thread_id": thread_id}}
    input_state = _build_input(tutor, thread_id, message)  # mirrors cli.py seeding

    async for event in tutor.astream_events(input_state, config, version="v2"):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"].content
            if chunk:
                yield f"event: token\ndata: {json.dumps({'chunk': chunk})}\n\n"

    state = tutor.get_state(config)
    yield f"event: done\ndata: {json.dumps(_extract_metadata(state))}\n\n"
```

`_build_input()` mirrors the first-turn seeding logic in `cli.py` (initialises `strategies_tried`, `conversation_history`, etc. on the first turn only).

`_extract_metadata()` pulls `session_paused`, `concept`, `concepts_needing_review`, `conversation_history` from the state snapshot.

`api/router.py` — thin endpoint:
```python
@router.post("/chat")
async def chat(req: ChatRequest):
    return StreamingResponse(
        stream_chat(tutor, req.thread_id, req.message),
        media_type="text/event-stream",
    )
```

---

## CORS

Configured in `api/main.py`. Allowed origins read from `CORS_ORIGINS` env var (comma-separated). Defaults to `http://localhost:3000` for local Next.js dev. In Railway, set to the deployed frontend URL.

---

## Docker

**`Dockerfile`:**
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**`docker-compose.yml`** (local dev only):
```yaml
services:
  api:
    build: .
    ports: ["8000:8000"]
    env_file: .env
    environment:
      REDIS_URL: redis://redis:6379
    depends_on: [redis]
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
```

**Railway:**
- FastAPI service: Dockerfile build
- Redis: Railway managed Redis plugin (provides `REDIS_URL`)
- Env vars to set: `OPENAI_API_KEY`, `TAVILY_API_KEY`, `CORS_ORIGINS`

---

## New Dependencies

Add to `requirements.txt`:
```
fastapi>=0.115
uvicorn[standard]>=0.30
langgraph-checkpoint-redis>=0.1
redis>=5.0
httpx>=0.27   # TestClient dependency
```

---

## Testing

`tests/test_api.py` using FastAPI `TestClient`:

1. `GET /health` returns `200 {"status": "ok"}`
2. `POST /chat` with valid body returns `200` with `text/event-stream` content type
3. SSE stream from `POST /chat` contains at least one `token` event and exactly one `done` event
4. `done` event payload includes `session_paused`, `concept`, `concepts_needing_review`, `conversation_history` keys

Tests use `MemorySaver` (no Redis required in CI).

---

## What This Spec Does NOT Cover

- Supabase auth / user login (Sub-project 2)
- React/Next.js frontend (Sub-project 3)
- Stop/interrupt mid-stream (future enhancement)
- Rate limiting (future enhancement)
- Phoenix/OpenTelemetry integration in the API server (future enhancement)
