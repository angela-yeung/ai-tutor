"""Interactive CLI for the Personal AI Tutoring Assistant."""

import argparse
import sys
from uuid import uuid4

from dotenv import load_dotenv
load_dotenv()

from tutor.graph import tutor
from tutor.instrumentation import setup_tracing, get_tracer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tutor.cli",
        description="Personal AI Tutoring Assistant",
    )
    parser.add_argument(
        "--resume",
        metavar="THREAD_ID",
        help="Resume a paused session by its thread ID",
    )
    return parser


def _print_response(state: dict) -> None:
    response = state.get("current_response", "")
    if response:
        print(f"\nTutor: {response}\n")


def _print_concepts_needing_review(concepts: list) -> None:
    if concepts:
        print("📋 Topics to revisit with a grown-up: " + ", ".join(concepts))


def run() -> None:
    setup_tracing()
    tracer = get_tracer()
    args = build_parser().parse_args()

    if args.resume:
        thread_id = args.resume
        print(f"\n=== Resuming session {thread_id} ===\n")
        is_resume = True
    else:
        thread_id = str(uuid4())
        print(f"\n=== New session started ===")
        print(f"Session ID: {thread_id}")
        print("(Save this ID to resume later with --resume <SESSION_ID>)\n")
        is_resume = False

    config = {"configurable": {"thread_id": thread_id}}

    print("Hi! I'm your tutor. What would you like to learn today?")
    print("(Type 'quit' to exit)\n")

    # On resume, fire off the resume node before the main loop
    if is_resume:
        try:
            with tracer.start_as_current_span("cli_turn"):
                result = tutor.invoke({"student_input": "I'm back, let's continue."}, config=config)
            _print_response(result)
            if result.get("session_paused"):
                print(f"\nSession paused. Your session ID is: {thread_id}")
                print(f"Type: python -m tutor.cli --resume {thread_id} to continue.")
                state_snapshot = tutor.get_state(config)
                concepts = state_snapshot.values.get("concepts_needing_review", [])
                _print_concepts_needing_review(concepts)
                return
        except Exception as e:
            print("Hmm, something went wrong connecting to the session. Let's try again!")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! See you next time.")
            state_snapshot = tutor.get_state(config)
            concepts = state_snapshot.values.get("concepts_needing_review", [])
            _print_concepts_needing_review(concepts)
            sys.exit(0)

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye! See you next time.")
            state_snapshot = tutor.get_state(config)
            concepts = state_snapshot.values.get("concepts_needing_review", [])
            _print_concepts_needing_review(concepts)
            sys.exit(0)

        if not user_input:
            continue

        # Build state update — seed list fields on first turn of new sessions
        state_update: dict = {"student_input": user_input}
        if not is_resume and not tutor.get_state(config).values:
            state_update.update({
                "strategies_tried": [],
                "concepts_needing_review": [],
                "conversation_history": [],
                "session_paused": False,
            })

        try:
            with tracer.start_as_current_span("cli_turn"):
                result = tutor.invoke(state_update, config=config)
        except Exception:
            print(
                "\nTutor: Hmm, something went wrong. Let's try again!\n"
            )
            continue

        _print_response(result)

        if result.get("session_paused"):
            print(f"\nSession paused. Your session ID is: {thread_id}")
            print(f"Type: python -m tutor.cli --resume {thread_id} to continue.")
            state_snapshot = tutor.get_state(config)
            concepts = state_snapshot.values.get("concepts_needing_review", [])
            _print_concepts_needing_review(concepts)
            return


def main() -> None:
    run()


if __name__ == "__main__":
    run()
