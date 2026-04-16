from typing import TypedDict, Annotated
import operator


class TutorState(TypedDict):
    student_input: str
    concept: str                                           # topic extracted by classify_question
    question_type: str                                     # "factual" | "reasoning"
    strategies_tried: list                                  # strategies used so far; returned as full list each turn (no reducer)
    conversation_history: Annotated[list, operator.add]    # {"role": ..., "content": ...} message log
    current_response: str                                  # latest assistant response
    session_paused: bool                                   # True when escalate is triggered
    concepts_needing_review: Annotated[list, operator.add] # concepts flagged for adult follow-up
