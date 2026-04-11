import datetime
import sys

import openai
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage

from tutor.state import TutorState
from tutor.tools import calculator, web_search, scaffold_hint

# ---------------------------------------------------------------------------
# LLM instances
# ---------------------------------------------------------------------------

_classifier_llm = ChatOpenAI(model="gpt-4o", temperature=0)
_factual_llm = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools([web_search])
_reasoning_llm = ChatOpenAI(model="gpt-4o", temperature=0.7).bind_tools([calculator, scaffold_hint])
_format_llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ---------------------------------------------------------------------------
# Age rule
# ---------------------------------------------------------------------------

def _date_context() -> str:
    return f'Today is {datetime.date.today().strftime("%B %d, %Y")}. '


_AGE_RULE = (
    "ALWAYS use sentences of 10 words or fewer. "
    "ONLY use words a 6-year-old Grade 1 student would know. "
    "ONLY use analogies from: toys, food, animals, everyday home objects, playground. "
    "Use a warm, encouraging tone. Never say the student is wrong directly."
)


# ---------------------------------------------------------------------------
# Node 1: classify_question
# ---------------------------------------------------------------------------

def classify_question(state: TutorState) -> dict:
    """Binary classify student input as factual or reasoning, extract concept."""
    system_prompt = (
        f'{_date_context()}'
        "Classify the student's question. "
        "Output EXACTLY in this format: <type>|<concept> "
        "where type is 'factual' or 'reasoning', and concept is the key topic in 3 words or fewer. "
        "'factual' = a fixed fact that can be looked up (capital cities, animal facts, historical events, definitions). "
        "'reasoning' = the student must work through a problem (ANY arithmetic calculation, word problems, sequences, patterns). "
        "IMPORTANT: ALL arithmetic is 'reasoning', even simple sums like 7+4 or 10-3. "
        "Examples: factual|capital of France, reasoning|simple addition, factual|spider legs, "
        "reasoning|7 plus 4, reasoning|word problem subtraction. "
        "Only output the format, nothing else."
    )
    try:
        response = _classifier_llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": state["student_input"]},
        ])
        raw = response.content.strip()
        parts = raw.split("|", 1)
        if len(parts) == 2:
            question_type = parts[0].strip().lower()
            concept = parts[1].strip()
            if question_type not in ("factual", "reasoning"):
                question_type = "reasoning"
        else:
            question_type = "reasoning"
            concept = state["student_input"][:50]
    except (openai.APIError, openai.RateLimitError) as e:
        print(f"[API ERROR] {e}", file=sys.stderr)
        return {"current_response": "Oops! Something went wrong. Let us try again!"}

    return {"question_type": question_type, "concept": concept}


# ---------------------------------------------------------------------------
# Node 2: factual_react_loop
# ---------------------------------------------------------------------------

def factual_react_loop(state: TutorState) -> dict:
    """ReAct loop for factual questions using web_search."""
    concept = state.get("concept", "your question")
    system_prompt = (
        f'{_date_context()}'
        f'You are a friendly tutor for a 6-year-old. '
        f'The student asked a factual question about "{concept}". '
        f'First, reason about whether you need to search for the answer. '
        f'If you need current or specific information, use the web_search tool. '
        f'Then give a warm, enriching answer with fun facts. '
        f'{_AGE_RULE}'
    )

    messages: list = [{"role": "system", "content": system_prompt}]
    for msg in state.get("conversation_history", []):
        messages.append(msg)
    messages.append({"role": "user", "content": state["student_input"]})

    tools_called: list[str] = []  # eval only — discarded by LangGraph state
    try:
        response = None
        while True:
            response = _factual_llm.invoke(messages)
            if not response.tool_calls:
                break
            # Append assistant message with tool calls
            messages.append(response)
            # Execute each tool call
            for tc in response.tool_calls:
                tools_called.append(tc["name"])
                if tc["name"] == "web_search":
                    result = web_search.invoke(tc["args"])
                else:
                    result = ""
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

        final_response = response.content
    except (openai.APIError, openai.RateLimitError) as e:
        print(f"[API ERROR] {e}", file=sys.stderr)
        return {"current_response": "Oops! Something went wrong. Let us try again!"}

    return {
        "current_response": final_response,
        "conversation_history": [
            {"role": "user", "content": state["student_input"]},
            {"role": "assistant", "content": final_response},
        ],
        "tools_called": tools_called,
    }


# ---------------------------------------------------------------------------
# Node 3: reasoning_react_loop
# ---------------------------------------------------------------------------

def reasoning_react_loop(state: TutorState) -> dict:
    """ReAct loop for reasoning/problem-solving questions with strategy selection."""
    concept = state.get("concept", "the problem")
    strategies_tried = state.get("strategies_tried", [])
    strategies_tried_str = ", ".join(strategies_tried) if strategies_tried else "none yet"

    system_prompt = (
        f'{_date_context()}'
        f'You are a Socratic tutor for a 6-year-old Grade 1 student working on: "{concept}".\n\n'
        f'Strategies already tried: {strategies_tried_str}  (NEVER repeat these)\n\n'
        f'Your job each turn:\n'
        f'1. Read the conversation history and assess whether the student is converging on the answer or still stuck.\n'
        f'2. Check for distress signals (crying, "I hate this", "I want to quit", "this is too hard", repeated frustration). '
        f'If detected, output ONLY the word: ESCALATE\n'
        f'3. If the student has clearly demonstrated they understand the answer, respond warmly confirming their understanding. '
        f'Do not call any tool.\n'
        f'4. If all five strategies have been tried (strategies_tried contains all of: guiding_question, analogy, '
        f'concrete_example, sub_problem, number_line) AND the student has not demonstrated understanding: '
        f'give the answer directly and warmly (e.g. "The answer is 23! Let me show you why."), '
        f'ask one simpler confidence-rebuilding question, then output on a new line: REVIEW:{{concept}}\n'
        f'5. Otherwise: choose the single best next strategy not already in strategies_tried, '
        f'then call the scaffold_hint tool OR the calculator tool if arithmetic verification would help.\n\n'
        f'{_AGE_RULE}'
    )

    messages: list = [{"role": "system", "content": system_prompt}]
    for msg in state.get("conversation_history", []):
        messages.append(msg)
    messages.append({"role": "user", "content": state["student_input"]})

    new_strategies: list = []
    final_response = ""
    session_paused = False
    concepts_needing_review: list = []
    tools_called: list[str] = []  # eval only — discarded by LangGraph state

    try:
        while True:
            response = _reasoning_llm.invoke(messages)

            # Check for text signals BEFORE tool calls
            if not response.tool_calls:
                text = response.content.strip()

                if text.rstrip(".! \n") == "ESCALATE":
                    session_paused = True
                    final_response = ""  # escalate node writes the message
                    break

                # Check for REVIEW signal
                if "REVIEW:" in text:
                    parts = text.split("REVIEW:")
                    final_response = parts[0].strip()
                    reviewed_concept = parts[1].strip() if len(parts) > 1 else state.get("concept", "")
                    concepts_needing_review = [reviewed_concept]
                    break

                # Normal response (understanding confirmed or hint delivered via text)
                final_response = text
                break

            # Handle tool calls
            messages.append(response)
            for tc in response.tool_calls:
                tools_called.append(tc["name"])
                try:
                    if tc["name"] == "scaffold_hint":
                        # Pass current strategies_tried including newly added ones this turn
                        args = dict(tc["args"])
                        args["strategies_tried"] = strategies_tried + new_strategies
                        result = scaffold_hint.invoke(args)
                        if isinstance(result, dict):
                            strategy = result.get("strategy", "")
                            if strategy:
                                new_strategies.append(strategy)
                        result_str = result.get("hint", str(result)) if isinstance(result, dict) else str(result)
                    elif tc["name"] == "calculator":
                        result_str = calculator.invoke(tc["args"])
                    else:
                        result_str = ""
                except ValueError as e:
                    # scaffold_hint exhaustion — handle gracefully
                    result_str = f"Error: {e}"

                messages.append(ToolMessage(content=result_str, tool_call_id=tc["id"]))

    except (openai.APIError, openai.RateLimitError) as e:
        print(f"[API ERROR] {e}", file=sys.stderr)
        return {"current_response": "Oops! Something went wrong. Let us try again!"}

    return {
        "current_response": final_response,
        "session_paused": session_paused,
        "strategies_tried": new_strategies,
        "concepts_needing_review": concepts_needing_review,
        "conversation_history": [
            {"role": "user", "content": state["student_input"]},
            {"role": "assistant", "content": final_response},
        ],
        "tools_called": tools_called,
    }


# ---------------------------------------------------------------------------
# Node 4: format_response
# ---------------------------------------------------------------------------

def format_response(state: TutorState) -> dict:
    """Post-process current_response for Grade 1 audience."""
    system_prompt = (
        f'{_date_context()}'
        f"Rewrite the following response for a 6-year-old Grade 1 student. "
        f"{_AGE_RULE} "
        f"CRITICAL: Copy all proper nouns (names of people, places, titles) EXACTLY as written. "
        f"Do NOT substitute, replace, or omit any name or specific fact. "
        f"Preserve the exact meaning. Do not add new information. "
        f"Do not remove any questions asked. Return only the rewritten response."
    )
    try:
        response = _format_llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": state["current_response"]},
        ])
        formatted_response = response.content.strip()
    except (openai.APIError, openai.RateLimitError) as e:
        print(f"[API ERROR] {e}", file=sys.stderr)
        # Fall back to original on failure
        return {"current_response": state["current_response"]}
    except Exception:
        # Any other failure — return original unchanged
        return {"current_response": state["current_response"]}

    return {"current_response": formatted_response}


# ---------------------------------------------------------------------------
# Node 5: escalate
# ---------------------------------------------------------------------------

def escalate(state: TutorState) -> dict:
    """Handle distress — pause session with a warm, safe message."""
    message = (
        "It is okay to feel that way. You are safe. "
        "Please talk to a grown-up you trust. "
        "You are doing really great. "
        "We can come back anytime you are ready."
    )
    return {
        "current_response": message,
        "session_paused": True,
    }


# ---------------------------------------------------------------------------
# Node 6: resume_session
# ---------------------------------------------------------------------------

def resume_session(state: TutorState) -> dict:
    """Welcome student back after a pause."""
    concept = state.get("concept", "what we were working on")
    try:
        system_prompt = (
            f'{_date_context()}'
            f"{_AGE_RULE} "
            f"You are a warm tutor. The student is coming back after a break. "
            f"Welcome them back in one short warm sentence. "
            f"Remind them what they were working on. "
            f"Tell them you are happy to help again."
        )
        response = _format_llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"The student was working on: {concept}. Welcome them back."},
        ])
        message = response.content.strip()
    except (openai.APIError, openai.RateLimitError) as e:
        print(f"[API ERROR] {e}", file=sys.stderr)
        message = f"Welcome back! We were working on {concept}. Let us keep going!"

    return {
        "current_response": message,
        "session_paused": False,
    }
