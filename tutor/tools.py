import ast
import operator as op_module
import os

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from tavily import TavilyClient


_OPS = {
    ast.Add: op_module.add,
    ast.Sub: op_module.sub,
    ast.Mult: op_module.mul,
    ast.Div: op_module.truediv,
    ast.Pow: op_module.pow,
    ast.Mod: op_module.mod,
    ast.FloorDiv: op_module.floordiv,
}
_UNARY_OPS = {
    ast.USub: op_module.neg,
    ast.UAdd: op_module.pos,
}


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


_hint_llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

_VALID_STRATEGIES = [
    "guiding_question",
    "analogy",
    "concrete_example",
    "sub_problem",
    "number_line",
]


@tool
def calculator(expression: str) -> str:
    """Evaluates a simple arithmetic expression (+-*/^%). Returns '' on error."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _eval_node(tree.body)
        if isinstance(result, float) and result == int(result):
            return str(int(result))
        return str(round(result, 6))
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
def scaffold_hint(concept: str, strategies_tried: list[str]) -> dict:
    """Selects the best Socratic hint strategy not already tried.

    Note: the ValueError below is raised in the raw function body. Callers
    using .invoke() should be aware that LangChain may handle (swallow) that
    exception depending on the version — use .func() to reliably test the
    raw ValueError path. This function is intentionally NOT decorated with
    handle_tool_error=True so the exception propagates to graph-level callers.
    """
    chosen_strategy = None
    for strategy in _VALID_STRATEGIES:
        if strategy not in strategies_tried:
            chosen_strategy = strategy
            break

    if chosen_strategy is None:
        raise ValueError("All strategies exhausted")

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
    response = _hint_llm.invoke(messages)
    hint_text = response.content.strip()

    return {"strategy": chosen_strategy, "hint": hint_text}
