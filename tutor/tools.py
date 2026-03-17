import asyncio
import builtins
import math
import os

from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient


async def _mcp_search(query: str) -> str:
    api_key = os.environ.get("TAVILY_API_KEY", "")
    url = f"https://mcp.tavily.com/mcp/?tavilyApiKey={api_key}"
    async with MultiServerMCPClient(
        {
            "tavily": {
                "transport": "streamable_http",
                "url": url,
            }
        }
    ) as client:
        tools = client.get_tools()
        search_tool = next((t for t in tools if t.name == "tavily-search"), None)
        if search_tool is None:
            return ""
        return str(search_tool.invoke({"query": query, "max_results": 3}))


@tool
def web_search(query: str) -> str:
    """Search the web for factual information (e.g. animal facts,
    science facts, geography). Use for real-world knowledge questions."""
    try:
        return asyncio.run(_mcp_search(query))
    except Exception:
        return ""


_SAFE_BUILTINS = {k: getattr(builtins, k) for k in (
    "abs", "divmod", "float", "int", "max", "min",
    "pow", "range", "round", "str", "sum", "True", "False", "None"
)}


@tool
def execute_code(code: str) -> str:
    """Run a Python math calculation. Use for arithmetic that needs
    exact answers: multiplication, division, powers, etc."""
    try:
        namespace = {"__builtins__": _SAFE_BUILTINS, "math": math}
        exec(code, namespace)
        lines = [l.strip() for l in code.strip().splitlines() if l.strip()]
        result = eval(lines[-1], namespace)
        if result is None:
            return ""
        return str(int(result)) if isinstance(result, float) and result.is_integer() else str(round(result, 6))
    except Exception:
        return ""
