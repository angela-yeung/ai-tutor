"""Interactive CLI for the Personal AI Tutoring Assistant."""

import argparse
import sys
from uuid import uuid4

from dotenv import load_dotenv
load_dotenv()

from tutor.graph import tutor_app


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


def run() -> None:
    args = build_parser().parse_args()

    if args.resume:
        thread_id = args.resume
        print(f"\n=== Resuming session {thread_id} ===\n")
        # Inject a resume trigger so entry_router sees session_paused=True.
        # On resume, the checkpointed state already has session_paused=True,
        # so we just invoke with the thread_id and a placeholder student_input.
        initial_input: dict = {"student_input": "I'm back, let's continue."}
    else:
        thread_id = str(uuid4())
        print(f"\n=== New session started ===")
        print(f"Session ID: {thread_id}")
        print("(Save this ID to resume later with --resume <SESSION_ID>)\n")
        initial_input = None  # first user message will set student_input

    config = {"configurable": {"thread_id": thread_id}}

    print("Hi! I'm your tutor. What would you like to learn today?")
    print("(Type 'quit' to exit)\n")

    # On resume, fire off the resume node before the main loop
    if args.resume:
        try:
            result = tutor_app.invoke(initial_input, config=config)
            _print_response(result)
            if _should_exit(result, thread_id):
                return
        except Exception as e:
            print("Hmm, something went wrong connecting to the session. Let's try again!")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! See you next time.")
            sys.exit(0)

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye! See you next time.")
            sys.exit(0)

        if not user_input:
            continue

        if user_input.lower().rstrip("!.,?") in ("hi", "hello", "hey", "howdy", "hiya", "greetings", "sup", "yo"):
            print("\nTutor: Hi! What would you like to learn today?\n")
            continue

        try:
            result = tutor_app.invoke(
                {"student_input": user_input},
                config=config,
            )
        except Exception:
            print(
                "\nTutor: Hmm, something went wrong. Let's try again!\n"
            )
            continue

        _print_response(result)

        if _should_exit(result, thread_id):
            return


def _print_response(state: dict) -> None:
    response = state.get("current_response", "")
    if response:
        print(f"\nTutor: {response}\n")


def _should_exit(state: dict, thread_id: str) -> bool:
    if state.get("session_paused"):
        print("─" * 50)
        print("Session paused. I hope you feel better soon!")
        print(f"Resume anytime with:\n  python -m tutor.cli --resume {thread_id}")
        print("─" * 50)
        return True

    if state.get("session_complete"):
        print("─" * 50)
        print("Great work today! You did an amazing job.")
        print("Start a new session anytime by running: python -m tutor.cli")
        print("─" * 50)
        return True

    return False


if __name__ == "__main__":
    run()
