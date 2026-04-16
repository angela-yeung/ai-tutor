"""FastAPI application factory."""

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.router import router
from tutor.graph import build_graph

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        from langgraph.checkpoint.redis import RedisSaver
        checkpointer = RedisSaver.from_conn_string(redis_url)
        app.state.tutor = build_graph(checkpointer)
    else:
        app.state.tutor = build_graph()  # MemorySaver — local dev / tests
    yield


app = FastAPI(title="AI Tutor API", lifespan=lifespan)

cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
