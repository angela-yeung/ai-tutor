"""
Observability bootstrap for the AI Tutor.

Registers Arize Phoenix (OpenInference) instrumentation for LangChain/LangGraph.

Import ONLY from tutor/cli.py so this never fires during tests.

Usage:
    from tutor.instrumentation import setup_tracing, get_tracer
    setup_tracing()  # call before first tutor.invoke()

Environment variables (set in .env):
    PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006
    PHOENIX_PROJECT_NAME=ai-tutor

Start local Phoenix UI with: phoenix serve
"""
import os


def get_tracer():
    """Return an OpenTelemetry tracer. Call after setup_tracing() has run."""
    from opentelemetry import trace
    return trace.get_tracer("ai-tutor")


def setup_tracing() -> None:
    """Register Phoenix instrumentation.

    Hooks into LangChain's callback dispatch (fired at invoke() time, not at
    graph compile time), so all LLM calls, tool calls, and node transitions
    are captured as spans.
    """
    _setup_phoenix()


def _setup_phoenix() -> None:
    try:
        from phoenix.otel import register
        from openinference.instrumentation.langchain import LangChainInstrumentor

        base = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT")
        if not base:
            return  # tracing is opt-in; set PHOENIX_COLLECTOR_ENDPOINT to enable

        # Phoenix serve listens for OTLP over HTTP — must use the /v1/traces
        # path explicitly, otherwise register() defaults to gRPC on port 4317.
        endpoint = base.rstrip("/") + "/v1/traces"

        tracer_provider = register(
            project_name=os.environ.get("PHOENIX_PROJECT_NAME", "ai-tutor"),
            endpoint=endpoint,
        )
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    except ImportError:
        # Phoenix not installed — skip silently
        pass
    except Exception as e:
        # Don't let instrumentation failures crash the tutor
        print(f"[instrumentation] Phoenix setup skipped: {e}")
