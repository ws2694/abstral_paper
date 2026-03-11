"""τ-bench integration adapter for ABSTRAL.

Bridges ABSTRAL's LangGraph agent topology with τ-bench's interactive
environment (tool execution + user simulation + DB-state evaluation).

Key classes:
  - TauEnvManager: Creates and manages τ-bench environments per task
  - wrap_tau_tools(): Converts τ-bench tools to LangChain StructuredTools
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from langchain_core.tools import StructuredTool

logger = logging.getLogger(__name__)

# Add vendor dir to path so we can import tau_bench
_VENDOR_DIR = Path(__file__).parent.parent / "vendor" / "tau_bench"
if str(_VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(_VENDOR_DIR))


class TauEnvManager:
    """Manages τ-bench environment lifecycle for ABSTRAL experiments."""

    def __init__(
        self,
        domain: str = "airline",
        user_model: str = "gpt-4o",
        user_provider: Optional[str] = None,
        task_split: str = "test",
    ):
        self.domain = domain
        self.user_model = user_model
        self.user_provider = user_provider
        self.task_split = task_split

    def create_env(self, task_index: int = 0):
        """Create a fresh τ-bench environment for a specific task.

        Returns (env, initial_user_message).
        """
        from tau_bench.envs import get_env

        env = get_env(
            env_name=self.domain,
            user_strategy="llm",
            user_model=self.user_model,
            task_split=self.task_split,
            user_provider=self.user_provider,
            task_index=task_index,
        )
        reset_response = env.reset(task_index)
        return env, reset_response.observation

    def get_wiki(self) -> str:
        """Return the domain's policy wiki text."""
        if self.domain == "airline":
            from tau_bench.envs.airline.wiki import WIKI
            return WIKI
        elif self.domain == "retail":
            from tau_bench.envs.retail.wiki import WIKI
            return WIKI
        else:
            raise ValueError(f"Unknown domain: {self.domain}")

    def get_task_count(self) -> int:
        """Return the number of tasks in the domain's test split."""
        if self.domain == "airline":
            from tau_bench.envs.airline.tasks_test import TASKS
            return len(TASKS)
        elif self.domain == "retail":
            from tau_bench.envs.retail.tasks_test import TASKS
            return len(TASKS)
        else:
            raise ValueError(f"Unknown domain: {self.domain}")

    def get_tasks(self):
        """Return the task list for the domain."""
        if self.domain == "airline":
            from tau_bench.envs.airline.tasks_test import TASKS
            return TASKS
        elif self.domain == "retail":
            from tau_bench.envs.retail.tasks_test import TASKS
            return TASKS
        else:
            raise ValueError(f"Unknown domain: {self.domain}")


def _openai_schema_to_pydantic(name: str, params: Dict[str, Any]):
    """Convert an OpenAI function parameter schema to a Pydantic model."""
    from pydantic import create_model, Field as PydanticField

    props = params.get("properties", {})
    required = set(params.get("required", []))
    fields = {}

    for pname, pinfo in props.items():
        ptype = pinfo.get("type", "string")
        pdesc = pinfo.get("description", "")

        # Map JSON schema types to Python types
        if ptype == "string":
            py_type = str
        elif ptype == "integer":
            py_type = int
        elif ptype == "number":
            py_type = float
        elif ptype == "boolean":
            py_type = bool
        elif ptype == "array":
            py_type = list
        elif ptype == "object":
            py_type = dict
        else:
            py_type = Any

        if pname in required:
            fields[pname] = (py_type, PydanticField(description=pdesc))
        else:
            fields[pname] = (Optional[py_type], PydanticField(default=None, description=pdesc))

    model_name = f"TauTool_{name}_Args"
    return create_model(model_name, **fields)


def wrap_tau_tools(env) -> List[StructuredTool]:
    """Convert τ-bench environment tools to LangChain StructuredTools.

    Each τ-bench tool has:
    - get_info() → OpenAI function-calling schema
    - invoke(data, **kwargs) → result string

    We wrap each as a LangChain StructuredTool bound to the env's live data dict.
    The 'respond' action is NOT wrapped as a tool — it's handled by the runner
    when the agent produces a text response (no tool calls).
    """
    wrapped = []
    for tool_cls in env.tools_map.values():
        info = tool_cls.get_info()
        func_info = info["function"]
        tool_name = func_info["name"]
        tool_desc = func_info.get("description", tool_name)
        params = func_info.get("parameters", {})

        # Build Pydantic model for args schema
        args_model = _openai_schema_to_pydantic(tool_name, params)

        # Build the wrapper function bound to this tool and env
        def make_fn(tc=tool_cls, e=env):
            def fn(**kwargs) -> str:
                # Filter out None optional args
                clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}
                try:
                    result = tc.invoke(data=e.data, **clean_kwargs)
                    return str(result)
                except Exception as ex:
                    return f"Error: {ex}"
            return fn

        st = StructuredTool.from_function(
            func=make_fn(),
            name=tool_name,
            description=tool_desc,
            args_schema=args_model,
        )
        wrapped.append(st)

    logger.info(f"Wrapped {len(wrapped)} τ-bench tools: {[t.name for t in wrapped]}")
    return wrapped


def make_tau_action(name: str, kwargs: Dict[str, Any]):
    """Create a τ-bench Action object."""
    from tau_bench.types import Action
    return Action(name=name, kwargs=kwargs)
