import math
import os

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from tavily import TavilyClient


_SAFE_BUILTINS = {
    "abs": abs,
    "divmod": divmod,
    "float": float,
    "int": int,
    "max": max,
    "min": min,
    "pow": pow,
    "range": range,
    "round": round,
    "str": str,
    "sum": sum,
    "True": True,
    "False": False,
    "None": None,
}

_VALID_STRATEGIES = [
    "guiding_question",
    "analogy",
    "concrete_example",
    "sub_problem",
    "number_line",
]


@tool
def calculator(expression: str) -> str:
    """Evaluates a simple arithmetic expression."""
    try:
        result = eval(  # noqa: S307
            expression,
            {"__builtins__": _SAFE_BUILTINS, "math": math},
            {},
        )
        return str(result)
    except Exception:
        return ""


@tool
def web_search(query: str) -> str:
    """Searches the web for factual information."""
    try:
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        response = client.search(query, max_results=3)
        parts = []
        for result in response.get("results", []):
            title = result.get("title", "")
            content = result.get("content", result.get("snippet", ""))
            parts.append(f"{title}\n{content}")
        return "\n\n".join(parts)
    except Exception:
        return ""


@tool
def scaffold_hint(concept: str, strategies_tried: list) -> dict:
    """Selects the best Socratic hint strategy not already tried."""
    chosen_strategy = None
    for strategy in _VALID_STRATEGIES:
        if strategy not in strategies_tried:
            chosen_strategy = strategy
            break

    if chosen_strategy is None:
        raise ValueError("All strategies exhausted")

    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    system_prompt = (
        f'You are a Socratic tutor for 6-year-old Grade 1 students. '
        f'Generate a {chosen_strategy} hint for the concept "{concept}". '
        f'Use sentences of 10 words or fewer. '
        f'Use simple Grade 1 vocabulary. '
        f'Use analogies from toys, food, animals, or everyday objects only. '
        f'Return ONLY the hint text, nothing else.'
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Generate a {chosen_strategy} hint for: {concept}"),
    ]
    response = llm.invoke(messages)
    hint_text = response.content.strip()

    return {"strategy": chosen_strategy, "hint": hint_text}
