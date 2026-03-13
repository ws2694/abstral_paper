"""Layer 1 BUILD phase: SKILL.md → LangGraph agent system.

The meta-agent reads ABS_t and constructs a concrete agent system AS_t:
instantiating agent roles from T_t, wiring their topology according to R_t,
and populating their system prompts with K_t. Construction follows P_t.
"""

import json
import logging
from typing import Annotated, Any, Dict, List, Optional, Tuple

from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from abstral.config import ABSTRALConfig, AgentSpec, AgentRole, FunctionalType, TopologyFamily
from abstral.meta_client import create_meta_client
from abstral.skill.document import SkillDocument

# Functional types that perform analysis/routing only (no tool access)
_NON_TOOL_TYPES = {
    FunctionalType.ROUTER,
    FunctionalType.PLANNER,
    FunctionalType.VERIFIER,
    FunctionalType.AGGREGATOR,
}

logger = logging.getLogger(__name__)

BUILD_PROMPT = """\
You are the ABSTRAL meta-agent. Your task is to design a multi-agent system
based on the current SKILL.md document.

Read the skill document carefully and produce an AgentSpec that defines:
1. The topology family to use (from R section reasoning)
2. The agent roles (from T section templates, or create new ones if T is empty)
3. The edges connecting agents (wiring based on R section)
4. The entry point agent
5. The message schema for inter-agent communication

## Current SKILL.md

### K — Domain Knowledge
{domain_knowledge}

### R — Topology Reasoning
{topology_reasoning}

### T — Agent Template Library
{template_library}

### P — Construction Protocol
{construction_protocol}

## Task Description
{task_description}

## CRITICAL Topology Constraints
- **You MUST use the topology family specified in the R section above.** This is mandatory.
- Each topology family has a distinct edge pattern:
  - **pipeline**: Linear chain A→B→C. No fan-out. max_out_degree=1.
  - **ensemble**: Dispatcher fans out to parallel workers, all feed into aggregator. DAG.
  - **debate**: Two+ arguers with bidirectional edges to a judge. MUST have cycles.
  - **hierarchical**: Central manager with BIDIRECTIONAL edges to specialists. MUST have cycles (specialist→manager edges).
  - **dynamic_routing**: Router fans out to 3+ conditional branches, all merge to output. max_out_degree≥3.
  - **single**: One agent, no edges.
- For hierarchical and debate topologies, you MUST include back-edges (cycles). A manager→specialist topology without specialist→manager edges is NOT hierarchical.
- Keep the number of agents between 2 and 6.
- Each agent must have a clear, non-overlapping responsibility.
- If T section is empty, design appropriate roles from scratch based on R and K.
- Entry point must be one of the defined role names.
- Every role name referenced in edges must exist in roles.
- Each agent role MUST have a distinct functional_type from: Router, Planner, Executor, Verifier, Aggregator, Specialist, Oracle. Avoid assigning the same type to all agents.

Return a valid AgentSpec JSON object."""


class AgentBuilder:
    """Builds a LangGraph agent system from a SKILL.md document."""

    def __init__(self, config: ABSTRALConfig):
        self.config = config
        self.client = create_meta_client(config.meta_agent)

    def design_agent_spec(
        self,
        skill_doc: SkillDocument,
        task_description: str,
    ) -> AgentSpec:
        """Use meta-agent to design an agent specification from SKILL.md."""
        prompt = BUILD_PROMPT.format(
            domain_knowledge=skill_doc.K or "(empty)",
            topology_reasoning=skill_doc.R or "(empty)",
            template_library=skill_doc.T or "(empty)",
            construction_protocol=skill_doc.P or "(empty)",
            task_description=task_description,
        )

        spec = self.client.chat.completions.create(
            model=self.config.meta_agent.model,
            max_tokens=self.config.meta_agent.max_tokens,
            messages=[{"role": "user", "content": prompt}],
            response_model=AgentSpec,
        )

        logger.info(
            f"Designed agent spec: {spec.topology_family.value} "
            f"with {len(spec.roles)} roles, {len(spec.edges)} edges"
        )
        return spec

    def build_graph(
        self,
        spec: AgentSpec,
        task_description: str,
        tool_provider: Any = None,
        benchmark_mode: str = "gaia",
        domain_knowledge: str = "",
    ) -> Tuple[StateGraph, Dict[str, Any]]:
        """Build a LangGraph StateGraph from an AgentSpec.

        Both GAIA and τ-bench modes build the full multi-agent graph.
        The key difference is tool assignment:
          - GAIA: all agents get tools and execute them in-graph.
          - τ-bench: only Executor/Specialist/Oracle get tools (can produce
            tool_calls). Router/Planner/Verifier/Aggregator produce text-only
            analysis that flows to downstream agents. The runner handles tool
            execution via env.step().

        This ensures the topology discovered by BUILD is actually instantiated
        at execution time — not collapsed to a single node.
        """
        # Create the LLM for agent execution
        backbone_model = self.config.agent_backbone.model
        if self.config.agent_backbone.provider == "openai":
            llm = ChatOpenAI(
                model=backbone_model,
                temperature=0.0,  # match published τ-bench baseline
                max_tokens=4096,
                max_retries=5,  # retry on 429 rate limits
                request_timeout=120,
            )
        else:
            raise ValueError(f"Unsupported provider: {self.config.agent_backbone.provider}")

        # Define state schema
        from typing import TypedDict

        def _last_value(existing, new):
            """Reducer: keep the last written value."""
            return new

        def _merge_visit_counts(existing, new):
            """Reducer: merge per-node visit counters."""
            if not existing:
                return new or {}
            if not new:
                return existing
            merged = dict(existing)
            for k, v in new.items():
                merged[k] = max(merged.get(k, 0), v)
            return merged

        class AgentState(TypedDict):
            messages: Annotated[list, add_messages]
            task: str
            current_agent: Annotated[str, _last_value]
            result: Annotated[str, _last_value]
            iteration_count: Annotated[int, _last_value]
            routing_context: Annotated[str, _last_value]  # analysis from non-tool agents
            route_to: Annotated[str, _last_value]  # explicit routing directive from non-tool agents
            _visit_counts: Annotated[dict, _merge_visit_counts]  # per-node visit tracking for loop prevention

        # Build the graph — full multi-agent for BOTH modes
        graph = StateGraph(AgentState)

        for role in spec.roles:
            # In τ-bench: only Executor/Specialist/Oracle get tools.
            # Router/Planner/Verifier/Aggregator produce text-only analysis,
            # avoiding the "tool_calls must be followed by tool messages" error
            # that occurs when routing between agents mid-tool-call.
            is_tool_agent = True
            if benchmark_mode == "tau":
                is_tool_agent = role.functional_type not in _NON_TOOL_TYPES

            # Precompute downstream targets for non-tool agents' routing directives
            downstream = [e.target for e in spec.edges if e.source == role.name]

            node_fn = self._make_agent_node(
                role, llm, task_description,
                tool_provider=tool_provider,
                benchmark_mode=benchmark_mode,
                domain_knowledge=domain_knowledge,
                is_tool_agent=is_tool_agent,
                downstream_targets=downstream,
            )
            graph.add_node(role.name, node_fn)

        # Set entry point — conditional for τ-bench so the graph can resume
        # at the last active tool agent instead of re-routing every turn.
        tool_agent_names = {
            r.name for r in spec.roles
            if r.functional_type not in _NON_TOOL_TYPES
        } if benchmark_mode == "tau" else set()

        if benchmark_mode == "tau" and len(spec.roles) > 1:
            all_role_names = {r.name for r in spec.roles}

            def _conditional_entry(state) -> str:
                last_agent = state.get("current_agent", "")
                if last_agent and last_agent in tool_agent_names and last_agent in all_role_names:
                    # Resume at the last active tool agent — skip routing overhead
                    return last_agent
                return spec.entry_point

            entry_map = {r.name: r.name for r in spec.roles}
            graph.set_conditional_entry_point(_conditional_entry, entry_map)
        else:
            graph.set_entry_point(spec.entry_point)

        # Add edges — group by source to detect multi-target nodes
        edge_targets: Dict[str, List[str]] = {}
        for edge in spec.edges:
            edge_targets.setdefault(edge.source, []).append(edge.target)

        processed_sources = set()
        # Max agent-node visits per graph invocation, topology-aware:
        # - SINGLE: just 1 agent, but allow re-entry for multi-turn
        # - PIPELINE/ENSEMBLE: each agent fires once, +2 buffer
        # - HIERARCHICAL: manager↔specialist round-trips need ~2× roles
        # - DEBATE: multiple argumentation rounds need ~3× roles
        # - DYNAMIC_ROUTING: router + one specialist path
        #
        # For τ-bench/SOPBench (benchmark_mode="tau"): cap at 4. The tool_call_guard
        # routes to END when a tool agent produces tool_calls, so the typical path
        # is: entry → (maybe 1 router) → tool agent → END = 2-3 nodes. Higher budgets
        # waste tokens on routing loops when agents produce text-only responses.
        n = len(spec.roles)
        family = spec.topology_family
        if family == TopologyFamily.SINGLE:
            max_agent_steps = 8 if benchmark_mode == "gaia" else 3
        elif family in (TopologyFamily.PIPELINE, TopologyFamily.ENSEMBLE):
            max_agent_steps = max(n + 2, 8) if benchmark_mode == "gaia" else min(n + 1, 4)
        elif family == TopologyFamily.HIERARCHICAL:
            max_agent_steps = max(n * 2, 8) if benchmark_mode == "gaia" else min(n, 4)
        elif family == TopologyFamily.DEBATE:
            max_agent_steps = max(n * 3, 10) if benchmark_mode == "gaia" else min(n + 1, 4)
        elif family == TopologyFamily.DYNAMIC_ROUTING:
            max_agent_steps = max(n + 2, 8) if benchmark_mode == "gaia" else min(n + 1, 4)
        else:
            max_agent_steps = max(n + 2, 8) if benchmark_mode == "gaia" else 4

        logger.info(
            f"Step budget: family={family.value}, n_roles={n}, "
            f"max_agent_steps={max_agent_steps}"
        )

        for source, targets in edge_targets.items():
            if source in processed_sources:
                continue
            processed_sources.add(source)

            # Deduplicate targets
            unique_targets = list(dict.fromkeys(targets))
            route_map = {t: t for t in unique_targets}
            route_map["__end__"] = END

            # Build the base routing function
            if len(unique_targets) > 1:
                if benchmark_mode == "tau":
                    base_router = self._make_content_router(unique_targets, max_agent_steps)
                else:
                    def make_router(tgts: List[str], max_steps: int = max_agent_steps):
                        def router(state) -> str:
                            if state.get("iteration_count", 0) >= max_steps:
                                return "__end__"
                            idx = state.get("iteration_count", 0) % len(tgts)
                            return tgts[idx]
                        return router
                    base_router = make_router(unique_targets)
            else:
                def make_single_router(tgt: str, max_steps: int = max_agent_steps):
                    def router(state) -> str:
                        if state.get("iteration_count", 0) >= max_steps:
                            return "__end__"
                        return tgt
                    return router
                base_router = make_single_router(unique_targets[0])

            # For τ-bench: wrap with tool_call guard. If a tool agent produces
            # tool_calls, route to END so the runner handles execution via
            # env.step(). This prevents "tool_calls must be followed by tool
            # messages" errors when routing between agents mid-tool-call.
            if benchmark_mode == "tau":
                final_router = self._wrap_with_tool_call_guard(base_router)
            else:
                final_router = base_router

            graph.add_conditional_edges(source, final_router, route_map)

        # Add END edges for nodes with no outgoing edges
        all_sources = set(edge_targets.keys())
        all_nodes = {r.name for r in spec.roles}
        terminal_nodes = all_nodes - all_sources
        for node in terminal_nodes:
            graph.add_edge(node, END)

        metadata = {
            "topology_family": spec.topology_family.value,
            "n_roles": len(spec.roles),
            "n_edges": len(spec.edges),
            "role_names": [r.name for r in spec.roles],
        }

        compiled = graph.compile()
        return compiled, metadata

    @staticmethod
    def _make_content_router(targets: List[str], max_steps: int = 5):
        """Create a content-based router for τ-bench multi-agent graphs.

        Routing priority:
        1. Explicit route_to state field (set by non-tool agents)
        2. Exact agent name match in last message text
        3. First target as fallback

        Design-level loop prevention:
        - Global step budget (iteration_count >= max_steps → END)
        - Per-node visit cap: no agent visited more than 2x per invocation
          (prevents A→B→A→B→... ping-pong in hierarchical/debate topologies)
        - route_to consumed after use (cleared to prevent re-triggering)
        """
        def router(state) -> str:
            if state.get("iteration_count", 0) >= max_steps:
                return "__end__"

            # Per-node visit tracking: terminate if any agent visited > 2x
            visit_counts = state.get("_visit_counts", {})
            current_agent = state.get("current_agent", "")
            if current_agent:
                count = visit_counts.get(current_agent, 0)
                if count >= 2:
                    return "__end__"

            # Priority 1: Explicit route_to field from non-tool agent
            route_to = state.get("route_to", "")
            if route_to:
                for target in targets:
                    if target.lower() == route_to.lower():
                        return target

            # Priority 2: Check last message for agent name
            messages = state.get("messages", [])
            if messages:
                last_msg = messages[-1]
                content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
                content_lower = content.lower()
                for target in targets:
                    if target.lower() in content_lower:
                        return target

            return targets[0]  # fallback

        return router

    @staticmethod
    def _wrap_with_tool_call_guard(base_router):
        """Wrap a router with a tool_call guard for τ-bench multi-agent graphs.

        If the source agent produced tool_calls (i.e., it wants to call a
        τ-bench tool), route to END so the runner can execute the tool via
        env.step(). Only continue to the next agent if the output is text-only.

        This prevents the "tool_calls must be followed by tool messages" error
        that occurs when routing an AIMessage with tool_calls to another agent
        node whose LLM would see unpaired tool_calls.
        """
        def guarded_router(state) -> str:
            messages = state.get("messages", [])
            if messages:
                last_msg = messages[-1]
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    return "__end__"
            return base_router(state)

        return guarded_router

    def _make_agent_node(
        self,
        role: AgentRole,
        llm: ChatOpenAI,
        task_description: str,
        tool_provider: Any = None,
        benchmark_mode: str = "gaia",
        domain_knowledge: str = "",
        is_tool_agent: bool = True,
        downstream_targets: List[str] = None,
    ):
        """Create a LangGraph node function for a given agent role.

        For GAIA: all agents get tools and execute them in-graph (max 3 rounds).
        For τ-bench tool agents (Executor/Specialist/Oracle): get τ-bench tools,
            single LLM call, may produce tool_calls handled by the runner.
        For τ-bench non-tool agents (Router/Planner/Verifier/Aggregator): no tools,
            produce text-only analysis passed to downstream agents.
        """
        from abstral.tools import get_all_tools, get_tools_by_names

        # --- System Prompt ---
        if benchmark_mode == "tau":
            if is_tool_agent:
                # Tool-bearing agent: customer service with tool access
                if domain_knowledge:
                    system_prompt = (
                        f"{domain_knowledge}\n\n"
                        f"# Agent Role: {role.name}\n"
                        f"{role.description}\n\n"
                        f"# Instructions\n"
                        f"You are a customer service agent. Follow the above policy strictly.\n"
                        f"- Only call ONE tool at a time.\n"
                        f"- You may receive analysis from other agents on your team — use their insights.\n"
                        f"- Always verify information with tools before making changes.\n"
                        f"- When you have completed the customer's request, respond with a helpful message.\n"
                        f"- Do NOT make up information.\n"
                    )
                else:
                    system_prompt = (
                        f"You are {role.name}: {role.description}\n\n"
                        f"{role.system_prompt}\n\n"
                        f"Task context: {task_description}\n\n"
                        f"IMPORTANT INSTRUCTIONS:\n"
                        f"1. You are a customer service agent. Use your tools to look up and modify data.\n"
                        f"2. Follow the airline/retail policy strictly.\n"
                        f"3. Only call ONE tool at a time.\n"
                        f"4. You may receive analysis from other agents on your team.\n"
                        f"5. When you have completed the customer's request, respond with a helpful message.\n"
                        f"6. Do NOT make up information — always verify with tools.\n"
                    )
            else:
                # Non-tool agent: analysis/routing only, no tool access.
                # Must produce ROUTE_TO: <AgentName> to direct routing.
                # Downstream targets injected so the agent knows valid options.
                targets_str = ", ".join(downstream_targets) if downstream_targets else "(end)"
                system_prompt = (
                    f"You are {role.name} ({role.functional_type.value}): {role.description}\n\n"
                    f"{role.system_prompt}\n\n"
                    f"You are part of a multi-agent customer service team.\n"
                    f"Your role is to analyze the customer's request and route to the right specialist.\n"
                    f"- Do NOT call any tools. You do not have tool access.\n"
                    f"- Analyze what the customer needs in 1-2 sentences.\n"
                    f"- You MUST end your response with a routing directive on its own line:\n"
                    f"  ROUTE_TO: <AgentName>\n"
                    f"- Valid targets: {targets_str}\n"
                    f"- Choose the most appropriate target based on the customer's request.\n"
                )
        else:
            # GAIA mode (unchanged)
            system_prompt = (
                f"You are {role.name}: {role.description}\n\n"
                f"{role.system_prompt}\n\n"
                f"Task context: {task_description}\n\n"
                f"IMPORTANT INSTRUCTIONS:\n"
                f"1. You have access to tools. Use them to find real information.\n"
                f"2. Do NOT make up facts or guess answers.\n"
                f"3. CRITICAL — Your final response MUST end with exactly this format:\n"
                f"   FINAL ANSWER: <answer>\n"
                f"   where <answer> is ONLY the answer itself — a number, name, word, or short phrase.\n"
                f"   Do NOT include explanations, units (unless asked), or extra text in the answer.\n"
                f"   Examples: 'FINAL ANSWER: 42', 'FINAL ANSWER: Paris', 'FINAL ANSWER: 3.14'\n"
            )

        # --- Tool Assignment ---
        if benchmark_mode == "tau" and not is_tool_agent:
            # Non-tool agents: no tools at all → LLM will never produce tool_calls
            tools = []
        elif tool_provider is not None:
            tools = tool_provider() if callable(tool_provider) else tool_provider
        elif role.tools:
            tools = get_tools_by_names(role.tools)
        else:
            tools = get_all_tools()

        # Bind tools to LLM (or use plain LLM if no tools)
        if tools:
            llm_with_tools = llm.bind_tools(tools)
        else:
            llm_with_tools = llm  # plain LLM — guaranteed text-only output

        # τ-bench: tools NOT executed in graph (runner uses env.step).
        # GAIA: tools executed in graph (up to 3 rounds).
        max_tool_rounds = 0 if benchmark_mode == "tau" else 3

        _is_tool_agent = is_tool_agent  # capture for closure

        def agent_node(state):
            from langchain_core.messages import AIMessage, ToolMessage

            # For τ-bench tool agents: inject routing context from upstream
            # non-tool agents as a system-level note, keeping conversation clean.
            effective_prompt = system_prompt
            if benchmark_mode == "tau" and _is_tool_agent:
                routing_ctx = state.get("routing_context", "")
                if routing_ctx:
                    effective_prompt = (
                        f"{system_prompt}\n\n"
                        f"# Team Analysis (from routing agents)\n{routing_ctx}"
                    )

            messages = [SystemMessage(content=effective_prompt)] + state["messages"]
            tool_map = {t.name: t for t in tools}
            current_messages = list(messages)

            # GAIA tool execution loop
            for _ in range(max_tool_rounds):
                response = llm_with_tools.invoke(current_messages)

                if not response.tool_calls:
                    return {
                        "messages": [response],
                        "current_agent": role.name,
                        "iteration_count": state.get("iteration_count", 0) + 1,
                    }

                # Execute tool calls and add results to local context
                current_messages.append(response)
                for tc in response.tool_calls:
                    tool_name = tc["name"]
                    tool_args = tc["args"]
                    if tool_name in tool_map:
                        try:
                            tool_result = tool_map[tool_name].invoke(tool_args)
                        except Exception as e:
                            tool_result = f"Tool error: {e}"
                    else:
                        tool_result = f"Unknown tool: {tool_name}"
                    current_messages.append(
                        ToolMessage(content=str(tool_result), tool_call_id=tc["id"])
                    )

            if max_tool_rounds == 0:
                # τ-bench: single LLM call. For tool agents, response may
                # contain tool_calls (runner handles via env.step). For non-tool
                # agents, response is always text-only (no tools bound).
                response = llm_with_tools.invoke(current_messages)

                if benchmark_mode == "tau" and not _is_tool_agent:
                    # Non-tool agent: store analysis in routing_context and
                    # extract ROUTE_TO directive into route_to field.
                    # Do NOT add response to messages — keeps conversation clean
                    # for the downstream tool agent that talks to the user.
                    import re as _re
                    route_match = _re.search(r"ROUTE_TO:\s*(\S+)", response.content, _re.IGNORECASE)
                    route_target = route_match.group(1) if route_match else ""
                    # Update visit counts for loop prevention
                    visits = dict(state.get("_visit_counts", {}) or {})
                    visits[role.name] = visits.get(role.name, 0) + 1
                    return {
                        "messages": [],  # don't pollute conversation history
                        "current_agent": role.name,
                        "routing_context": response.content,
                        "route_to": route_target,
                        "iteration_count": state.get("iteration_count", 0) + 1,
                        "_visit_counts": visits,
                    }

                # Tool agent or GAIA: update visit counts and clear route_to
                # (consumed — prevents re-triggering on next routing decision)
                visits = dict(state.get("_visit_counts", {}) or {})
                visits[role.name] = visits.get(role.name, 0) + 1
                return {
                    "messages": [response],
                    "current_agent": role.name,
                    "route_to": "",  # consumed — clear to prevent stale routing
                    "iteration_count": state.get("iteration_count", 0) + 1,
                    "_visit_counts": visits,
                }

            # GAIA: exceeded max tool rounds — force a final answer
            current_messages.append(
                HumanMessage(content="Please provide your final answer now based on the information gathered. End with 'FINAL ANSWER: <answer>' where <answer> is ONLY the answer — a number, name, or short phrase.")
            )
            final = llm.invoke(current_messages)  # No tools bound
            return {
                "messages": [final],
                "current_agent": role.name,
                "iteration_count": state.get("iteration_count", 0) + 1,
            }

        agent_node.__name__ = f"agent_{role.name}"
        return agent_node
