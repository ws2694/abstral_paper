"""Real tools for ABSTRAL agent execution.

These tools give agents actual capabilities for GAIA benchmark tasks:
- Web search (DuckDuckGo — no API key needed)
- Calculator (Python eval with safety constraints)
- Wikipedia lookup
"""

from __future__ import annotations

import logging
import math
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo. Use this for factual questions,
    finding current information, looking up specific facts, or researching topics.
    Returns a summary of search results."""
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return f"No results found for: {query}"
        output_parts = []
        for r in results:
            title = r.get("title", "")
            body = r.get("body", "")
            href = r.get("href", "")
            output_parts.append(f"**{title}**\n{body}\nURL: {href}")
        return "\n\n".join(output_parts)
    except ImportError:
        # Fallback to old package name
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))
            if not results:
                return f"No results found for: {query}"
            output_parts = []
            for r in results:
                title = r.get("title", "")
                body = r.get("body", "")
                href = r.get("href", "")
                output_parts.append(f"**{title}**\n{body}\nURL: {href}")
            return "\n\n".join(output_parts)
        except Exception as e:
            logger.warning(f"Web search failed: {e}")
            return f"Search failed: {e}"
    except Exception as e:
        logger.warning(f"Web search failed: {e}")
        return f"Search failed: {e}"


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Supports basic arithmetic (+, -, *, /),
    exponents (**), modulo (%), and math functions (sqrt, log, sin, cos, ceil, floor, abs, round).
    Examples: '2 + 3 * 4', 'sqrt(144)', 'round(1002 * 0.04)', '15 / 7'."""
    # Safe math namespace — no builtins, only math functions
    safe_ns = {
        "__builtins__": {},
        "sqrt": math.sqrt,
        "log": math.log,
        "log10": math.log10,
        "log2": math.log2,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "pi": math.pi,
        "e": math.e,
        "ceil": math.ceil,
        "floor": math.floor,
        "abs": abs,
        "round": round,
        "pow": pow,
        "min": min,
        "max": max,
        "sum": sum,
        "int": int,
        "float": float,
    }
    try:
        # Basic safety: block dangerous constructs
        forbidden = ["import", "exec", "eval", "open", "os.", "sys.", "__"]
        expr_lower = expression.lower()
        for f in forbidden:
            if f in expr_lower:
                return f"Error: '{f}' is not allowed in calculator expressions"
        result = eval(expression, safe_ns)
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}. Expression: {expression}"


@tool
def wikipedia_lookup(topic: str) -> str:
    """Look up a topic on Wikipedia. Returns the first few paragraphs of the article.
    Use this for background information, definitions, or historical facts."""
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(f"site:wikipedia.org {topic}", max_results=3))
        if not results:
            return f"No Wikipedia article found for: {topic}"
        # Return the first result's body
        return f"**{results[0].get('title', '')}**\n{results[0].get('body', '')}\nURL: {results[0].get('href', '')}"
    except Exception as e:
        logger.warning(f"Wikipedia lookup failed: {e}")
        return f"Lookup failed: {e}"


def get_all_tools() -> list:
    """Return all available tools for agent execution."""
    return [web_search, calculator, wikipedia_lookup]


def get_tools_by_names(names: list[str]) -> list:
    """Return tools matching the given names."""
    tool_registry = {
        "web_search": web_search,
        "search": web_search,
        "calculator": calculator,
        "calc": calculator,
        "wikipedia": wikipedia_lookup,
        "wiki": wikipedia_lookup,
    }
    tools = []
    for name in names:
        name_lower = name.lower().replace(" ", "_")
        if name_lower in tool_registry:
            tools.append(tool_registry[name_lower])
    # If no specific tools matched, return all tools
    return tools if tools else get_all_tools()
