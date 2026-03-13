"""SOPBench integration adapter for ABSTRAL.

Bridges ABSTRAL's LangGraph agent topology with SOPBench's domain environments
(bank, healthcare) for tool execution and oracle-based evaluation.

SOPBench (Li et al., 2025) evaluates agents on Standard Operating Procedure
compliance with deterministic oracle verifiers — no LLM-as-judge.

Key classes:
  - SOPEnvManager: Creates and manages SOPBench environments per task
  - wrap_sop_tools(): Converts SOPBench domain methods to LangChain StructuredTools
  - evaluate_trajectory(): Post-hoc evaluation via oracle verifiers
"""

from __future__ import annotations

import copy
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool

logger = logging.getLogger(__name__)

# Add vendor dir to path
_VENDOR_DIR = Path(__file__).parent.parent / "vendor" / "sopbench"
if str(_VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(_VENDOR_DIR))


class SOPEnvManager:
    """Manages SOPBench environment lifecycle for ABSTRAL experiments."""

    def __init__(
        self,
        domain: str = "bank",
        mode: str = "prompt",  # "prompt" (agent verifies constraints) or "program" (oracle enforces)
        default_constraint_option: str = "all",
        constraint_descr_format: str = "structured",
    ):
        self.domain = domain
        self.mode = mode
        self.default_constraint_option = default_constraint_option
        self.constraint_descr_format = constraint_descr_format

        # Load task data
        data_path = _VENDOR_DIR / "data" / f"{domain}_tasks.json"
        with open(data_path) as f:
            raw = json.load(f)

        # Flatten tasks with user_goal annotation
        self.tasks = []
        for action_name, task_list in raw.items():
            for task in task_list:
                t = copy.deepcopy(task)
                t["user_goal"] = action_name
                self.tasks.append(t)

        logger.info(f"SOPBench {domain}: loaded {len(self.tasks)} tasks")

        # Pre-compute default dependencies
        from env.task import task_default_dep_full
        self.dep_innate_full, self.default_dep_full, self.default_dep_full_descr = \
            task_default_dep_full(domain, default_constraint_option, constraint_descr_format)

    def create_env(self, task_index: int):
        """Create a SOPBench environment for a specific task.

        Returns (domain_system, user_info, assistant_info, task_info, task_data).
        """
        from env.task import task_initializer

        task = self.tasks[task_index]
        domain_system, user_info, assistant_info, task_info = task_initializer(
            domain_str=self.domain,
            task=task,
            dep_innate_full=self.dep_innate_full,
            default_dep_full=self.default_dep_full,
            default_dep_full_descr=self.default_dep_full_descr,
            included_functions=None,
            mode=self.mode,
            shuffle_func=False,
            constraint_descr_format=self.constraint_descr_format,
        )
        return domain_system, user_info, assistant_info, task_info, task

    def get_task_count(self) -> int:
        return len(self.tasks)

    def get_system_prompt(self, assistant_info: dict) -> str:
        """Extract system prompt from assistant_info."""
        return assistant_info["instructions"]

    def get_tool_schemas(self, assistant_info: dict) -> list:
        """Get OpenAI function-calling tool schemas."""
        return assistant_info.get("tools", [])

    def evaluate(self, task: dict, func_calls: list, final_database: dict) -> dict:
        """Run oracle evaluation on a completed trajectory.

        Args:
            task: The original task dict (with constraints, user_goal, etc.)
            func_calls: List of {"tool_name": str, "arguments": dict, "content": Any}
            final_database: The database state after all tool calls

        Returns:
            Dict with boolean criteria + overall success
        """
        from env.evaluator import evaluator_function_directed_graph

        results = {"final_database": final_database}

        # Build log_msg_fcall (simplified — evaluator only counts messages with "sender")
        log_msg_fcall = [{"sender": "user"}]  # at least 1 sender message
        for fc in func_calls:
            log_msg_fcall.append({"sender": "assistant"})

        eval_result = evaluator_function_directed_graph(
            domain_str=self.domain,
            task=task,
            log_msg_fcall=log_msg_fcall,
            func_calls=func_calls,
            results=results,
            default_constraint_option=self.default_constraint_option,
        )

        # Overall success: all 5 criteria must pass
        success = all([
            eval_result.get("no_tool_call_error", False),
            eval_result.get("constraint_not_violated", False),
            eval_result.get("database_match", False),
            eval_result.get("dirgraph_satisfied", False),
            eval_result.get("action_called_correctly", False),
        ])
        eval_result["success"] = success
        eval_result["reward"] = 1.0 if success else 0.0

        return eval_result


def _openai_schema_to_pydantic(name: str, params: dict):
    """Convert an OpenAI function parameter schema to a Pydantic model."""
    from pydantic import create_model, Field as PydanticField
    from typing import Any as TypAny

    props = params.get("properties", {})
    required = set(params.get("required", []))
    fields = {}

    for pname, pinfo in props.items():
        ptype = pinfo.get("type", "string")
        pdesc = pinfo.get("description", "")

        # Handle union types like ["string", "object"]
        if isinstance(ptype, list):
            py_type = TypAny
        elif ptype == "string":
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
            py_type = TypAny

        if pname in required:
            fields[pname] = (py_type, PydanticField(description=pdesc))
        else:
            fields[pname] = (Optional[py_type], PydanticField(default=None, description=pdesc))

    model_name = f"SOPTool_{name}_Args"
    return create_model(model_name, **fields)


def wrap_sop_tools(domain_system, assistant_info: dict) -> List[StructuredTool]:
    """Convert SOPBench domain system methods to LangChain StructuredTools.

    Each tool in assistant_info["tools"] has OpenAI function-calling schema.
    We bind each to the domain_system instance for direct method invocation.
    """
    wrapped = []
    tools = assistant_info.get("tools", [])

    for tool_entry in tools:
        func_info = tool_entry["function"]
        tool_name = func_info["name"]
        tool_desc = func_info.get("description", tool_name)
        params = func_info.get("parameters", {})

        # Build Pydantic model for args schema
        args_model = _openai_schema_to_pydantic(tool_name, params)

        # Build wrapper bound to domain_system
        # exit_conversation is a standalone signal (not on domain_system);
        # the runner intercepts it, but the LLM needs to see it as a valid tool.
        def make_fn(tn=tool_name, ds=domain_system):
            def fn(**kwargs) -> str:
                if tn == "exit_conversation":
                    return "Conversation ended."
                clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}
                try:
                    method = getattr(ds, tn, None)
                    if method is None:
                        return f"Error: Unknown tool {tn}"
                    result = method(**clean_kwargs)
                    return json.dumps(result) if not isinstance(result, str) else result
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

    logger.info(f"Wrapped {len(wrapped)} SOPBench tools: {[t.name for t in wrapped]}")
    return wrapped
