import os
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from tutor.state import TutorState

MODEL = "gpt-4o-mini"

_AGE_RULE = (
    "Use sentences of 10 words or fewer. "
    "Only use words a 5-year-old knows. "
    "Use examples only from toys, food, animals, or home objects."
)

_llm = ChatOpenAI(model=MODEL)


def llm_call(
    system_prompt: str,
    user_message: str,
    history: Optional[list] = None,
) -> Optional[str]:
    """Wraps a Claude API call. Returns None on any failure."""
    try:
        messages: list = [SystemMessage(content=system_prompt)]
        for msg in (history or []):
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        messages.append(HumanMessage(content=user_message))
        response = _llm.invoke(messages)
        return response.content
    except Exception as e:
        print(f"\n[API ERROR] {e}\n")
        return None


# ---------------------------------------------------------------------------
# Node: assess_question
# ---------------------------------------------------------------------------

def assess_question(state: TutorState) -> dict:
    system = (
        f"{_AGE_RULE} "
        "You are a kind tutor. The student asked a question. "
        "Identify the key learning concept in one short phrase. "
        "Reply ONLY with that concept phrase — nothing else."
    )
    concept = llm_call(system, state["student_input"])
    if concept is None:
        concept = "your question"

    # Build the opening hint in the same call to give an immediate response
    opening_system = (
        f"{_AGE_RULE} "
        "You are a kind tutor helping a 5-year-old. "
        "Ask ONE short guiding question to help them think. "
        "Do NOT give the answer. Be warm and encouraging."
    )
    prompt = f"The student is learning: {concept}. They asked: {state['student_input']}"
    response = llm_call(opening_system, prompt)
    if response is None:
        response = "Hmm, something went wrong. Let's try again!"

    history = list(state.get("session_history", []))
    history.append({"role": "user", "content": state["student_input"]})
    history.append({"role": "assistant", "content": response})

    return {
        "concept": concept.strip(),
        "hints_given": 1,
        "incorrect_attempts": 0,
        "session_paused": False,
        "session_complete": False,
        "understanding_level": "",
        "current_response": response,
        "session_history": history,
    }


# ---------------------------------------------------------------------------
# Node: scaffold_hint
# ---------------------------------------------------------------------------

def scaffold_hint(state: TutorState) -> dict:
    concept = state.get("concept", "the topic")
    hints_given = state.get("hints_given", 0)
    incorrect_attempts = state.get("incorrect_attempts", 0)

    if incorrect_attempts > 2:
        # Gentle direct correction after repeated incorrect attempts
        system = (
            f"{_AGE_RULE} "
            "You are a kind tutor. The student got it wrong a few times. "
            "Gently tell them the right answer. "
            "Keep it simple and warm. Do not make them feel bad."
        )
        prompt = (
            f"Concept: {concept}. "
            f"Student said: {state.get('student_input', '')}. "
            "Give a short, gentle, correct explanation."
        )
    elif hints_given == 0 or hints_given == 1:
        # Hint 1: guiding question
        system = (
            f"{_AGE_RULE} "
            "You are a kind tutor. Ask ONE short guiding question. "
            "Help the student think. Do NOT give the answer."
        )
        prompt = (
            f"Concept: {concept}. "
            f"Student said: {state.get('student_input', '')}. "
            "Ask a guiding question to help them think it through."
        )
    elif hints_given == 2:
        # Hint 2: analogy
        system = (
            f"{_AGE_RULE} "
            "You are a kind tutor. Give ONE simple analogy. "
            "Use toys, food, animals, or home objects. "
            "Do NOT give the answer directly."
        )
        prompt = (
            f"Concept: {concept}. "
            f"Student said: {state.get('student_input', '')}. "
            "Give a simple analogy to help them understand."
        )
    else:
        # Hint 3+: step-by-step breakdown
        system = (
            f"{_AGE_RULE} "
            "You are a kind tutor. Break the problem into tiny steps. "
            "Number each step. Use very simple words. "
            "Do NOT give the final answer — stop just before it."
        )
        prompt = (
            f"Concept: {concept}. "
            f"Student said: {state.get('student_input', '')}. "
            "Break it into small steps to guide them."
        )

    response = llm_call(system, prompt, state.get("session_history"))
    if response is None:
        response = "Hmm, something went wrong. Let's try again!"

    history = list(state.get("session_history", []))
    history.append({"role": "assistant", "content": response})

    return {
        "hints_given": hints_given + 1,
        "current_response": response,
        "session_history": history,
    }


# ---------------------------------------------------------------------------
# Node: check_understanding
# ---------------------------------------------------------------------------

_VALID_LEVELS = {
    "got_it", "progressing", "stuck", "incorrect", "frustrated", "distressed"
}


def check_understanding(state: TutorState) -> dict:
    system = (
        "You are a tutor evaluating a student's response. "
        "Classify the student's understanding using EXACTLY ONE word from this list: "
        "got_it, progressing, stuck, incorrect, frustrated, distressed. "
        "Rules: "
        "'got_it' — student clearly understands. "
        "'progressing' — student is getting closer but not there yet. "
        "'stuck' — student is confused or not moving forward. "
        "'incorrect' — student gave a factually wrong answer. "
        "'frustrated' — student shows signs of annoyance or giving up. "
        "'distressed' — student shows signs of significant upset or distress. "
        "Reply with ONLY that single word. No punctuation. No explanation."
    )
    prompt = (
        f"Concept being learned: {state.get('concept', '')}. "
        f"Student's latest response: {state.get('student_input', '')}."
    )
    raw = llm_call(system, prompt, state.get("session_history"))
    if raw is None:
        level = "stuck"
    else:
        level = raw.strip().lower()
        if level not in _VALID_LEVELS:
            # Try to find a valid level in the response
            for valid in _VALID_LEVELS:
                if valid in level:
                    level = valid
                    break
            else:
                level = "stuck"

    history = list(state.get("session_history", []))
    history.append({"role": "user", "content": state.get("student_input", "")})

    incorrect_attempts = state.get("incorrect_attempts", 0)
    if level == "incorrect":
        incorrect_attempts += 1
    elif level == "got_it":
        incorrect_attempts = 0

    return {
        "understanding_level": level,
        "incorrect_attempts": incorrect_attempts,
        "session_history": history,
    }


# ---------------------------------------------------------------------------
# Node: encourage
# ---------------------------------------------------------------------------

def encourage(state: TutorState) -> dict:
    system = (
        f"{_AGE_RULE} "
        "You are a warm, kind tutor. The student is feeling frustrated. "
        "Write 2–3 short encouraging sentences. "
        "Reframe the problem as fun. Do NOT give the answer. "
        "End with one gentle prompt to try again."
    )
    prompt = (
        f"Concept: {state.get('concept', '')}. "
        f"Student said: {state.get('student_input', '')}."
    )
    response = llm_call(system, prompt, state.get("session_history"))
    if response is None:
        response = "You're doing great! Let's try again together."

    history = list(state.get("session_history", []))
    history.append({"role": "assistant", "content": response})

    return {
        "current_response": response,
        "session_history": history,
    }


# ---------------------------------------------------------------------------
# Node: reinforce_concept
# ---------------------------------------------------------------------------

def reinforce_concept(state: TutorState) -> dict:
    system = (
        f"{_AGE_RULE} "
        "You are a kind tutor. The student just understood the concept. "
        "First, celebrate warmly in ONE short sentence. "
        "Then, ask ONE slightly harder question about the same idea. "
        "Do NOT give the answer to the new question."
    )
    prompt = (
        f"Concept mastered: {state.get('concept', '')}. "
        "Celebrate and ask a slightly harder follow-up question."
    )
    response = llm_call(system, prompt, state.get("session_history"))
    if response is None:
        response = "Amazing work! You got it! Try this: can you think of another example?"

    history = list(state.get("session_history", []))
    history.append({"role": "assistant", "content": response})

    return {
        "current_response": response,
        "session_history": history,
        "session_complete": True,
    }


# ---------------------------------------------------------------------------
# Node: escalate
# ---------------------------------------------------------------------------

def escalate(state: TutorState) -> dict:
    response = (
        "It's okay. You are safe. "
        "Please talk to a grown-up you trust. "
        "We can come back to this later. "
        "You are doing really well."
    )
    history = list(state.get("session_history", []))
    history.append({"role": "assistant", "content": response})

    return {
        "current_response": response,
        "session_paused": True,
        "session_history": history,
    }


# ---------------------------------------------------------------------------
# Node: resume_session
# ---------------------------------------------------------------------------

def resume_session(state: TutorState) -> dict:
    concept = state.get("concept", "what we were learning")
    system = (
        f"{_AGE_RULE} "
        "You are a warm tutor. The student is coming back after a break. "
        "Welcome them back in ONE short warm sentence. "
        "Remind them what they were learning in ONE short sentence. "
        "Tell them you will help them again."
    )
    prompt = f"The student was learning: {concept}. Welcome them back."
    response = llm_call(system, prompt)
    if response is None:
        response = f"Welcome back! We were learning about {concept}. Let's continue!"

    history = list(state.get("session_history", []))
    history.append({"role": "assistant", "content": response})

    return {
        "session_paused": False,
        "current_response": response,
        "session_history": history,
    }
