from typing import TypedDict, List, Dict


class TutorState(TypedDict):
    student_input: str
    concept: str
    hints_given: int
    understanding_level: str  # got_it | progressing | stuck | incorrect | frustrated | distressed
    session_history: List[Dict[str, str]]
    current_response: str
    incorrect_attempts: int
    session_paused: bool
    session_complete: bool
