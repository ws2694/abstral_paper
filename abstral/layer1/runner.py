"""Layer 1 RUN phase: Execute agent system in sandboxed environment.

AS_t is executed in a sandboxed environment on a stratified sample of task
instances. OpenTelemetry captures execution traces for observability.

Sandbox constraints (§3.2):
  - 2GB RAM limit (enforced via config, not runtime — container-level)
  - Token budget per batch (configured via inner_loop.token_budget)
  - 5-minute wall clock per task (inner_loop.wall_clock_limit_sec=300)
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import Any

from abstral.config import ABSTRALConfig
from abstral.models import TaskInstance, RunResult, BatchRunResult
from abstral.tracing import TraceManager, TraceSet

logger = logging.getLogger(__name__)


class AgentRunner:
    """Executes agent systems on task instances with trace capture and sandbox enforcement."""

    def __init__(self, config: ABSTRALConfig):
        self.config = config
        self.trace_manager = TraceManager(
            project_name=config.tracking.otel_service_name
        )

    def run_batch(
        self,
        compiled_graph: Any,
        tasks: list[TaskInstance],
        run_tag: str = "",
    ) -> BatchRunResult:
        """Execute the agent system on a batch of task instances.

        Enforces sandbox constraints from §3.2:
          - Token budget: stops batch if cumulative tokens exceed budget
          - Wall clock: per-task timeout
        """
        run_id = f"{run_tag}-{uuid.uuid4().hex[:8]}" if run_tag else uuid.uuid4().hex[:8]
        batch_result = BatchRunResult(run_id=run_id)

        # Sandbox constraints (§3.2)
        token_budget = self.config.inner_loop.token_budget  # default 50000
        wall_clock_limit = self.config.inner_loop.wall_clock_limit_sec  # default 300
        cumulative_tokens = 0

        logger.info(
            f"Starting batch run {run_id} with {len(tasks)} tasks "
            f"(token budget: {token_budget}, wall clock: {wall_clock_limit}s/task)"
        )

        batch_start = time.monotonic()

        for i, task in enumerate(tasks):
            # Check token budget
            if cumulative_tokens >= token_budget:
                logger.warning(
                    f"Token budget exhausted ({cumulative_tokens}/{token_budget}). "
                    f"Stopping batch after {len(batch_result.results)}/{len(tasks)} tasks."
                )
                break

            # Pause between tasks to stay within rate limits
            # GPT-4o with gpt-4o user sim: need longer gaps to avoid quota errors
            if i > 0:
                time.sleep(20)

            result = self._run_single(compiled_graph, task, run_id, wall_clock_limit)
            batch_result.results.append(result)
            cumulative_tokens += result.token_count

        # Compute aggregate metrics
        batch_result.metrics = {
            "success_rate": batch_result.success_rate,
            "total_tokens": batch_result.total_tokens,
            "mean_latency_ms": batch_result.mean_latency_ms,
            "n_tasks": len(tasks),
            "n_executed": len(batch_result.results),
            "n_succeeded": sum(1 for r in batch_result.results if r.success),
            "n_failed": sum(1 for r in batch_result.results if not r.success),
            "token_budget_remaining": max(0, token_budget - cumulative_tokens),
        }

        # Build trace set locally from run results (no remote API calls)
        batch_result.trace_set = self.trace_manager.build_trace_set(
            run_id, batch_result.results
        )

        logger.info(
            f"Batch run {run_id} complete: "
            f"{batch_result.metrics['n_succeeded']}/{batch_result.metrics['n_executed']} succeeded "
            f"({batch_result.success_rate:.1%}), "
            f"{cumulative_tokens} tokens used"
        )

        return batch_result

    def _run_single(
        self,
        compiled_graph: Any,
        task: TaskInstance,
        run_id: str,
        wall_clock_limit: int = 300,
    ) -> RunResult:
        """Execute a single task instance with wall clock enforcement."""
        start_time = time.monotonic()
        trace_id = f"{run_id}-{task.id}"

        try:
            config = {
                "metadata": {
                    "run_id": run_id,
                    "task_id": task.id,
                    "task_type": task.task_type,
                },
                "recursion_limit": 50,
            }

            # Execute the graph
            from langchain_core.messages import HumanMessage
            task_with_format = (
                f"{task.input_text}\n\n"
                f"Give ONLY the final answer as a short response (a number, name, or phrase). "
                f"End your response with: FINAL ANSWER: <your answer>"
            )
            initial_state = {
                "messages": [HumanMessage(content=task_with_format)],
                "task": task.input_text,
                "current_agent": "",
                "result": "",
                "iteration_count": 0,
            }

            final_state = compiled_graph.invoke(initial_state, config=config)

            # Check wall clock
            elapsed_s = time.monotonic() - start_time
            if elapsed_s > wall_clock_limit:
                logger.warning(
                    f"Task {task.id} exceeded wall clock limit "
                    f"({elapsed_s:.1f}s > {wall_clock_limit}s)"
                )

            # Extract output and per-agent message flow from final state
            output = ""
            agent_messages = []  # Per-agent message trace for ANALYZE phase
            if final_state.get("messages"):
                last_msg = final_state["messages"][-1]
                raw_output = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

                # Extract answer for GAIA exact-match scoring
                # Try multiple patterns in priority order
                import re as _re
                output = raw_output  # default: full text

                # Pattern 1: "FINAL ANSWER: X"
                fa_match = _re.search(r'FINAL ANSWER:\s*(.+?)(?:\n|$)', raw_output, _re.IGNORECASE)
                if fa_match:
                    output = fa_match.group(1).strip()
                else:
                    # Pattern 2: "the answer is X" or "the final answer is X"
                    ans_match = _re.search(r'(?:the\s+)?(?:final\s+)?answer\s+is[:\s]+(.+?)(?:\.|$)', raw_output, _re.IGNORECASE)
                    if ans_match:
                        output = ans_match.group(1).strip()
                    else:
                        # Pattern 3: last line that's short and looks like an answer
                        lines = [l.strip() for l in raw_output.strip().split('\n') if l.strip()]
                        if lines:
                            last_line = lines[-1]
                            # Strip markdown bold
                            last_line = _re.sub(r'\*\*(.+?)\*\*', r'\1', last_line)
                            if len(last_line) < 100:
                                output = last_line

                # Capture per-agent message flow for rich trace analysis
                for i, msg in enumerate(final_state["messages"]):
                    content = msg.content if hasattr(msg, "content") else str(msg)
                    msg_type = type(msg).__name__  # HumanMessage, AIMessage, SystemMessage
                    agent_name = getattr(msg, "name", None) or ""
                    # AIMessages from agents often have the agent name in metadata
                    if not agent_name and hasattr(msg, "response_metadata"):
                        agent_name = msg.response_metadata.get("agent_name", "")
                    agent_messages.append({
                        "step": i,
                        "type": msg_type,
                        "agent": agent_name or final_state.get("current_agent", "unknown"),
                        "content_preview": content[:300],
                        "length": len(content),
                    })

            # Estimate token count from messages
            token_count = 0
            for msg in final_state.get("messages", []):
                content = msg.content if hasattr(msg, "content") else str(msg)
                # Rough estimate: ~4 chars per token
                token_count += len(content) // 4
                # Use actual usage if available
                if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                    token_count = msg.usage_metadata.get("total_tokens", token_count)

            # Evaluate success
            success = self._evaluate_success(output, task)

            elapsed_ms = (time.monotonic() - start_time) * 1000

            return RunResult(
                task_id=task.id,
                success=success,
                output=output,
                latency_ms=elapsed_ms,
                trace_id=trace_id,
                token_count=token_count,
                agent_messages=agent_messages,
            )

        except Exception as e:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.warning(f"Task {task.id} failed with error: {e}")
            return RunResult(
                task_id=task.id,
                success=False,
                error=str(e),
                latency_ms=elapsed_ms,
                trace_id=trace_id,
            )

    def _evaluate_success(self, output: str, task: TaskInstance) -> bool:
        """Evaluate whether the agent output matches the ground-truth answer.

        Uses the benchmark's own scoring function for exact-match evaluation
        against the known correct answer. NO LLM-as-judge — only ground truth.

        For GAIA: normalized exact match with numeric tolerance.
        For HotPotQA: normalized exact match.
        """
        if not output or not output.strip():
            return False

        if not task.expected_output:
            logger.warning(f"Task {task.id} has no expected_output — cannot evaluate")
            return False

        # Use the benchmark loader's scoring function
        benchmark = task.metadata.get("benchmark", "")
        if benchmark:
            from abstral.benchmarks import get_loader
            try:
                loader = get_loader(benchmark)
                score = loader.score(output, task.expected_output)
                return score >= 0.5
            except ValueError:
                pass

        # Fallback: strict normalized exact match (no LLM-as-judge)
        from abstral.benchmarks import _normalize_answer
        return _normalize_answer(output) == _normalize_answer(task.expected_output)


class TauBenchRunner(AgentRunner):
    """Runner for interactive τ-bench environments.

    Instead of single-shot Q&A (GAIA), this manages a multi-turn conversation:
      1. env.reset(task) → initial user message
      2. LOOP: user_msg → LangGraph → agent action → env.step() → next user_msg
      3. env.calculate_reward() → binary 0/1 (DB hash comparison)

    Tool calls happen INSIDE graph invocations (existing tool loop in builder.py).
    User turns require NEW graph invocations (this runner manages conversation history).

    Token optimization: Uses a sliding window over conversation history to avoid
    O(T²) token growth. Only the last `history_window` messages are sent in full;
    older messages are summarized into a single compact message.
    """

    # Max messages to keep in full (recent). Older messages get summarized.
    HISTORY_WINDOW = 20
    # Max tool call rounds per turn before forcing a text response.
    MAX_TOOL_CALLS_PER_TURN = 8

    def __init__(self, config: ABSTRALConfig):
        super().__init__(config)
        from abstral.tau_adapter import TauEnvManager
        self.env_manager = TauEnvManager(
            domain=config.tau_bench.domain,
            user_model=config.tau_bench.user_model,
            user_provider=config.tau_bench.user_provider,
            task_split=config.tau_bench.task_split,
        )
        self.max_turns = config.tau_bench.max_conversation_turns
        self._invoke_pool = ThreadPoolExecutor(max_workers=1)

    @staticmethod
    def _invoke_with_timeout(graph, state, config, timeout_sec=120):
        """Invoke graph with a hard timeout to prevent indefinite hangs.

        Uses a thread pool so we can enforce a wall-clock deadline even when
        the underlying HTTP call blocks.  Returns the final state or raises
        FuturesTimeoutError.
        """
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            future = pool.submit(graph.invoke, state, config)
            return future.result(timeout=timeout_sec)
        finally:
            pool.shutdown(wait=False)

    @staticmethod
    def _windowed_history(messages, window_size=20):
        """Return a token-efficient version of conversation history.

        Keeps the last `window_size` messages in full. Older messages are
        collapsed into a single HumanMessage summary to avoid O(T²) token growth.

        IMPORTANT: Never splits an AIMessage(tool_calls)/ToolMessage pair.
        If the window boundary would orphan ToolMessages, extends the window
        back to include the preceding AIMessage.
        """
        if len(messages) <= window_size:
            return list(messages)

        from langchain_core.messages import HumanMessage as _HM, AIMessage, ToolMessage

        # Find safe split point: don't split tool_call/tool_result pairs
        split_idx = len(messages) - window_size
        # Walk forward from split_idx to find a safe boundary
        # (not in the middle of an AI(tool_calls)→ToolMessage sequence)
        while split_idx > 0 and split_idx < len(messages):
            msg = messages[split_idx]
            if isinstance(msg, ToolMessage):
                # ToolMessage at boundary → extend window back to include its AIMessage
                split_idx -= 1
            elif isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
                # AIMessage with tool_calls at boundary → include it and its ToolMessages
                split_idx -= 0  # keep this message in recent
                break
            else:
                break

        if split_idx <= 0:
            return list(messages)

        old_msgs = messages[:split_idx]
        recent_msgs = messages[split_idx:]

        # Build compact summary of old messages
        summary_parts = []
        for msg in old_msgs:
            role = type(msg).__name__.replace("Message", "").lower()
            content = msg.content if hasattr(msg, "content") else str(msg)
            preview = content[:80].replace("\n", " ")
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tools = ", ".join(tc["name"] for tc in msg.tool_calls)
                summary_parts.append(f"[{role}: called {tools}]")
            else:
                summary_parts.append(f"[{role}: {preview}]")

        summary_text = (
            f"[CONTEXT — {len(old_msgs)} earlier messages condensed]\n"
            + "\n".join(summary_parts)
        )
        return [_HM(content=summary_text)] + list(recent_msgs)

    @staticmethod
    def _sanitize_history(messages):
        """Ensure message sequence is valid for OpenAI API.

        Rules:
        1. First message must be HumanMessage (SystemMessage is added by agent_node)
        2. Every ToolMessage must be preceded by an AIMessage with matching tool_calls
        3. Remove orphaned ToolMessages that would cause 400 errors
        """
        from langchain_core.messages import AIMessage, ToolMessage

        if not messages:
            return messages

        sanitized = []
        for i, msg in enumerate(messages):
            if isinstance(msg, ToolMessage):
                # Check that the preceding message in sanitized is an AIMessage with tool_calls
                has_preceding_ai = False
                for prev in reversed(sanitized):
                    if isinstance(prev, AIMessage) and hasattr(prev, 'tool_calls') and prev.tool_calls:
                        # Check if any tool_call ID matches
                        tc_ids = {tc["id"] for tc in prev.tool_calls}
                        if msg.tool_call_id in tc_ids:
                            has_preceding_ai = True
                            break
                    elif isinstance(prev, ToolMessage):
                        continue  # Skip past other tool messages from same AI call
                    else:
                        break  # Hit a non-tool, non-AI message — no valid parent
                if has_preceding_ai:
                    sanitized.append(msg)
                else:
                    logger.warning(f"Dropping orphaned ToolMessage (tool_call_id={msg.tool_call_id})")
            else:
                sanitized.append(msg)

        return sanitized

    def _run_single(
        self,
        compiled_graph: Any,
        task: TaskInstance,
        run_id: str,
        wall_clock_limit: int = 300,
    ) -> RunResult:
        """Execute a single τ-bench task as an interactive conversation.

        The graph produces decisions (tool call or text response) but does NOT
        execute tools. ALL actions are routed through env.step() to ensure
        tools operate on the correct per-task env data.

        Flow per turn:
          1. Feed conversation history → graph.invoke() → AIMessage
          2. If AIMessage has tool_calls → env.step(Action(tool_name, args)) → tool result
             → add ToolMessage to history → graph.invoke() again
          3. If AIMessage is text only → env.step(Action("respond", content)) → next user msg or done
        """
        start_time = time.monotonic()
        trace_id = f"{run_id}-{task.id}"
        task_index = task.metadata["task_index"]

        try:
            from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
            from abstral.tau_adapter import make_tau_action

            # Create environment for this task
            env, initial_user_msg = self.env_manager.create_env(task_index)

            # Conversation history (persists across graph invocations)
            conversation_history = []
            agent_messages = []
            total_tokens = 0
            done = False
            current_user_msg = initial_user_msg
            env_response = None

            config = {
                "tags": [f"run_id:{run_id}", f"task_id:{task.id}"],
                "metadata": {"run_id": run_id, "task_id": task.id},
                "recursion_limit": 50,
            }

            # Add initial user message
            conversation_history.append(HumanMessage(content=current_user_msg))

            tool_calls_this_turn = 0
            last_active_agent = ""  # tracks which agent to resume at
            last_routing_context = ""  # analysis from non-tool agents, carried forward
            routing_skips = 0  # turns where conditional entry skipped routing
            routing_full = 0   # turns where full routing chain executed

            for turn in range(self.max_turns):
                if done:
                    break

                # Check wall clock
                elapsed_s = time.monotonic() - start_time
                if elapsed_s > wall_clock_limit:
                    logger.warning(f"Task {task.id} exceeded wall clock at turn {turn}")
                    break

                # Use windowed history to avoid O(T²) token growth
                windowed = self._windowed_history(
                    conversation_history, self.HISTORY_WINDOW
                )
                # Sanitize: ensure no orphaned ToolMessages
                windowed = self._sanitize_history(windowed)

                # Invoke graph with windowed conversation history (with rate limit retry).
                # Pass current_agent from prior turn so the conditional entry point
                # can resume at the last active tool agent (skip routing overhead).
                initial_state = {
                    "messages": windowed,
                    "task": task.input_text,
                    "current_agent": last_active_agent if turn > 0 else "",
                    "result": "",
                    "iteration_count": 0,
                    "routing_context": last_routing_context,
                    "route_to": "",
                    "_visit_counts": {},  # reset per-invocation visit tracking
                }

                # Per-invoke timeout: remaining wall clock, capped at 120s
                remaining_wc = max(30, wall_clock_limit - (time.monotonic() - start_time))
                invoke_timeout = min(remaining_wc, 120)

                final_state = None
                for attempt in range(6):
                    try:
                        final_state = self._invoke_with_timeout(
                            compiled_graph, initial_state, config,
                            timeout_sec=invoke_timeout,
                        )
                        break
                    except FuturesTimeoutError:
                        logger.warning(
                            f"Task {task.id} turn {turn}: graph.invoke timed out "
                            f"after {invoke_timeout:.0f}s (attempt {attempt+1}/6)"
                        )
                        break  # Don't retry timeouts — move on
                    except Exception as e:
                        err_str = str(e)
                        if "429" in err_str or "rate limit" in err_str.lower():
                            wait = min(2 ** attempt * 10, 120)  # 10s, 20s, 40s, 80s, 120s, 120s
                            logger.warning(f"Rate limit hit, waiting {wait}s (attempt {attempt+1}/6)")
                            time.sleep(wait)
                        elif "role 'tool' must be a response" in err_str:
                            # Malformed message history — log and skip this turn
                            logger.warning(
                                f"Task {task.id} turn {turn}: malformed history "
                                f"({len(initial_state['messages'])} msgs), "
                                f"types: {[type(m).__name__ for m in initial_state['messages'][:5]]}"
                            )
                            # Try to recover: force a respond action to move conversation forward
                            from abstral.tau_adapter import make_tau_action
                            action = make_tau_action("respond", {"content": "I apologize, let me help you with that. Could you please repeat your request?"})
                            env_response = env.step(action)
                            done = env_response.done
                            if not done:
                                current_user_msg = env_response.observation
                                # Reset conversation history to just the last user message
                                conversation_history = [HumanMessage(content=current_user_msg)]
                            break
                        else:
                            raise
                if final_state is None and not done:
                    # Timed out or exhausted retries — skip this turn gracefully
                    logger.warning(f"Task {task.id} turn {turn}: no result, skipping turn")
                    break

                # If recovery reset the conversation, skip to next turn
                if final_state is None:
                    continue

                # Track which agent was last active for conditional entry on next turn
                prev_agent = last_active_agent
                last_active_agent = final_state.get("current_agent", "")
                # Track routing efficiency
                if turn > 0 and prev_agent and prev_agent == last_active_agent:
                    routing_skips += 1  # conditional entry hit — skipped routing
                else:
                    routing_full += 1   # full routing chain executed
                # Carry forward routing context so tool agent retains team analysis
                new_ctx = final_state.get("routing_context", "")
                if new_ctx:
                    last_routing_context = new_ctx

                # Extract the AI response from graph execution.
                # The graph returns state with messages = input + [response].
                # Use len check instead of index slicing to handle add_messages
                # deduplication edge cases.
                all_out = final_state["messages"]
                new_msgs = all_out[len(windowed):] if len(all_out) > len(windowed) else []

                # Estimate tokens
                for msg in new_msgs:
                    content = msg.content if hasattr(msg, "content") else str(msg)
                    total_tokens += len(content) // 4

                # Find the last AI message (in new_msgs or fallback to all output)
                last_ai_msg = None
                for msg in reversed(new_msgs):
                    if isinstance(msg, AIMessage):
                        last_ai_msg = msg
                        break
                if last_ai_msg is None:
                    # Fallback: check the last message of all output
                    for msg in reversed(all_out):
                        if isinstance(msg, AIMessage):
                            last_ai_msg = msg
                            break

                if last_ai_msg is None:
                    logger.warning(f"Task {task.id} turn {turn}: no AI message produced")
                    break

                if last_ai_msg.tool_calls:
                    # Agent wants to call tools — route through env.step().
                    # Match published τ-bench baseline: only execute the FIRST tool call.
                    # τ-bench env expects one action at a time, and the published baseline
                    # truncates tool_calls to [:1]. We rewrite the AIMessage to contain
                    # only the first tool call so OpenAI message format stays valid.
                    from langchain_core.messages import AIMessage as _AI
                    if len(last_ai_msg.tool_calls) > 1:
                        last_ai_msg = _AI(
                            content=last_ai_msg.content,
                            tool_calls=[last_ai_msg.tool_calls[0]],
                        )
                    conversation_history.append(last_ai_msg)

                    for tc in last_ai_msg.tool_calls:
                        tool_calls_this_turn += 1
                        action = make_tau_action(tc["name"], tc["args"])
                        env_response = env.step(action)
                        conversation_history.append(
                            ToolMessage(content=str(env_response.observation), tool_call_id=tc["id"])
                        )
                        agent_messages.append({
                            "step": turn,
                            "type": "tool_call",
                            "tool": tc["name"],
                            "agent": final_state.get("current_agent", "unknown"),
                            "content_preview": str(env_response.observation)[:300],
                        })
                        if env_response.done:
                            done = True
                            break

                    if not done and tool_calls_this_turn >= self.MAX_TOOL_CALLS_PER_TURN:
                        # Cap hit: force a respond using the last tool result as context.
                        # Add a nudge message so the next invocation produces text, not more tools.
                        conversation_history.append(
                            HumanMessage(content="Please respond to the customer now based on the information you've gathered.")
                        )
                        logger.info(f"Task {task.id} turn {turn}: tool call cap ({self.MAX_TOOL_CALLS_PER_TURN}) reached, forcing response")

                    # Loop back to get next LLM decision

                else:
                    # Agent produced a text response → "respond" action
                    response_text = last_ai_msg.content
                    action = make_tau_action("respond", {"content": response_text})
                    env_response = env.step(action)
                    done = env_response.done
                    tool_calls_this_turn = 0  # Reset for next user turn

                    conversation_history.append(last_ai_msg)

                    agent_messages.append({
                        "step": turn,
                        "type": "respond",
                        "agent": final_state.get("current_agent", "unknown"),
                        "content_preview": response_text[:300],
                        "length": len(response_text),
                    })

                    if not done:
                        # Add next user message to history
                        current_user_msg = env_response.observation
                        conversation_history.append(HumanMessage(content=current_user_msg))

            # Get reward from environment
            if env_response is not None and done:
                reward = env_response.reward
            else:
                # Conversation didn't finish naturally — calculate reward anyway
                reward_result = env.calculate_reward()
                reward = reward_result.reward

            success = reward == 1.0
            elapsed_ms = (time.monotonic() - start_time) * 1000
            n_turns = turn + 1

            return RunResult(
                task_id=task.id,
                success=success,
                output=f"reward={reward:.1f}, turns={n_turns}",
                latency_ms=elapsed_ms,
                trace_id=trace_id,
                token_count=total_tokens,
                agent_messages=agent_messages,
                metadata={
                    "routing_skips": routing_skips,
                    "routing_full": routing_full,
                    "final_agent": last_active_agent,
                    "n_turns": n_turns,
                },
            )

        except Exception as e:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.warning(f"Task {task.id} failed with error: {e}")
            return RunResult(
                task_id=task.id,
                success=False,
                error=str(e),
                latency_ms=elapsed_ms,
                trace_id=trace_id,
            )


class SOPBenchRunner(AgentRunner):
    """Runner for SOPBench (bank, healthcare) tasks.

    SOPBench tasks follow a simpler pattern than τ-bench:
      1. User sends initial request (with known parameters)
      2. Agent calls tools on the domain system (direct method invocation)
      3. Agent may call exit_conversation when done
      4. Evaluation is post-hoc via oracle verifiers (5 boolean criteria)

    Key difference from τ-bench: no env.step() loop. Tool calls execute directly
    on the domain_system object. User interaction is minimal (static or one-shot).
    """

    HISTORY_WINDOW = 20
    MAX_TOOL_CALLS = 15  # Max total tool calls per task

    def __init__(self, config: ABSTRALConfig):
        super().__init__(config)
        from abstral.sop_adapter import SOPEnvManager
        self.env_manager = SOPEnvManager(
            domain=config.sop_bench.domain,
            mode=config.sop_bench.mode,
        )
        self.max_turns = config.sop_bench.max_turns

    def _run_single(
        self,
        compiled_graph: Any,
        task: TaskInstance,
        run_id: str,
        wall_clock_limit: int = 300,
    ) -> RunResult:
        """Execute a single SOPBench task.

        Flow:
          1. Initialize domain system from task data
          2. Get system prompt + user prompt from task
          3. Feed through LangGraph → collect tool calls
          4. Evaluate trajectory post-hoc via oracle
        """
        start_time = time.monotonic()
        trace_id = f"{run_id}-{task.id}"
        task_index = task.metadata["task_index"]

        try:
            from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

            # Create environment for this task
            domain_system, user_info, assistant_info, task_info, task_data = \
                self.env_manager.create_env(task_index)

            # Collect all tool calls for post-hoc evaluation
            func_calls = []
            conversation_history = []
            agent_messages = []
            total_tokens = 0
            done = False

            # Inject SOPBench assistant instructions (constraint rules) as
            # system context. This contains task-specific dependency rules
            # that the agent MUST follow for dirgraph_satisfied to pass.
            sop_instructions = assistant_info.get("instructions", "")
            if sop_instructions:
                from langchain_core.messages import SystemMessage
                conversation_history.append(SystemMessage(
                    content=f"### SOPBench Operating Instructions ###\n\n{sop_instructions}\n\n"
                    f"IMPORTANT: You must call tools to complete this task. "
                    f"Follow the constraint rules above exactly. "
                    f"Call exit_conversation when done."
                ))

            # Initial user message — use task["user_prompt"] which is the
            # natural language customer request with real data embedded
            # (matches original SOPBench run_simulation.py line 179).
            user_prompt = task_data.get("user_prompt", "")
            if not user_prompt:
                # Fallback: construct from user_instruction + user_known
                user_instruction = task_data.get("user_instruction", "")
                user_known = task_data.get("user_known", {})
                if user_instruction:
                    user_prompt = user_instruction
                    if user_known:
                        user_prompt += " Here is my information: " + ", ".join(
                            f'my {k} is "{v}"' for k, v in user_known.items()
                        )
                else:
                    user_prompt = "Help me with my request."

            # Add exit instruction (matches SOPBench default user message)
            user_prompt += (
                "\n\nPlease directly use the most appropriate tool to solve my request "
                "as quickly as possible and use the `exit_conversation` action to end "
                "our conversation if you have completed my request or cannot assist me."
            )
            conversation_history.append(HumanMessage(content=user_prompt))

            config = {
                "tags": [f"run_id:{run_id}", f"task_id:{task.id}"],
                "metadata": {"run_id": run_id, "task_id": task.id},
                "recursion_limit": 50,
            }

            tool_call_count = 0

            for turn in range(self.max_turns):
                if done:
                    break

                elapsed_s = time.monotonic() - start_time
                if elapsed_s > wall_clock_limit:
                    logger.warning(f"Task {task.id} exceeded wall clock at turn {turn}")
                    break

                # Window history for token efficiency
                windowed = TauBenchRunner._windowed_history(
                    conversation_history, self.HISTORY_WINDOW
                )
                windowed = TauBenchRunner._sanitize_history(windowed)

                initial_state = {
                    "messages": windowed,
                    "task": task.input_text,
                    "current_agent": "",
                    "result": "",
                    "iteration_count": 0,
                    "routing_context": "",
                    "route_to": "",
                    "_visit_counts": {},
                }

                # Invoke graph with retry + timeout
                remaining_wc = max(30, wall_clock_limit - (time.monotonic() - start_time))
                invoke_timeout = min(remaining_wc, 120)

                final_state = None
                for attempt in range(6):
                    try:
                        final_state = TauBenchRunner._invoke_with_timeout(
                            compiled_graph, initial_state, config,
                            timeout_sec=invoke_timeout,
                        )
                        break
                    except FuturesTimeoutError:
                        logger.warning(
                            f"Task {task.id} turn {turn}: graph.invoke timed out "
                            f"after {invoke_timeout:.0f}s"
                        )
                        break
                    except Exception as e:
                        err_str = str(e)
                        if "429" in err_str or "rate limit" in err_str.lower():
                            wait = min(2 ** attempt * 10, 120)
                            logger.warning(f"Rate limit hit, waiting {wait}s (attempt {attempt+1}/6)")
                            time.sleep(wait)
                        elif "role 'tool' must be a response" in err_str:
                            # Malformed history — strip all tool-related messages and retry
                            logger.warning(
                                f"Task {task.id} turn {turn}: malformed history, "
                                f"stripping tool messages and retrying"
                            )
                            from langchain_core.messages import HumanMessage as _HM, AIMessage as _AI
                            clean = []
                            for m in windowed:
                                if isinstance(m, ToolMessage):
                                    continue
                                if isinstance(m, _AI) and hasattr(m, 'tool_calls') and m.tool_calls:
                                    # Convert to plain text AI message
                                    tools_desc = ", ".join(tc["name"] for tc in m.tool_calls)
                                    clean.append(_AI(content=f"[Called tools: {tools_desc}]"))
                                else:
                                    clean.append(m)
                            initial_state["messages"] = clean
                            try:
                                final_state = TauBenchRunner._invoke_with_timeout(
                                    compiled_graph, initial_state, config,
                                    timeout_sec=invoke_timeout,
                                )
                                break
                            except Exception:
                                raise
                        else:
                            raise

                if final_state is None:
                    logger.warning(f"Task {task.id} turn {turn}: no result, skipping turn")
                    continue

                # Extract new messages from graph output
                all_out = final_state["messages"]
                new_msgs = all_out[len(windowed):] if len(all_out) > len(windowed) else []

                for msg in new_msgs:
                    content = msg.content if hasattr(msg, "content") else str(msg)
                    total_tokens += len(content) // 4

                # Find last AI message
                last_ai_msg = None
                for msg in reversed(new_msgs):
                    if isinstance(msg, AIMessage):
                        last_ai_msg = msg
                        break
                if last_ai_msg is None:
                    for msg in reversed(all_out):
                        if isinstance(msg, AIMessage):
                            last_ai_msg = msg
                            break

                if last_ai_msg is None:
                    logger.warning(f"Task {task.id} turn {turn}: no AI message produced")
                    break

                if last_ai_msg.tool_calls:
                    # Limit to first tool call (SOPBench expects sequential calls)
                    from langchain_core.messages import AIMessage as _AI
                    if len(last_ai_msg.tool_calls) > 1:
                        last_ai_msg = _AI(
                            content=last_ai_msg.content,
                            tool_calls=[last_ai_msg.tool_calls[0]],
                        )
                    conversation_history.append(last_ai_msg)

                    for tc in last_ai_msg.tool_calls:
                        tool_name = tc["name"]
                        tool_args = tc["args"]

                        # Check for exit_conversation
                        if tool_name == "exit_conversation":
                            done = True
                            conversation_history.append(
                                ToolMessage(content="Conversation ended.", tool_call_id=tc["id"])
                            )
                            break

                        # Execute tool on domain system
                        tool_call_count += 1
                        try:
                            method = getattr(domain_system, tool_name, None)
                            if method is None:
                                result = f"Error: Unknown tool {tool_name}"
                            else:
                                result = method(**tool_args)
                        except Exception as ex:
                            result = f"Error: {ex}"

                        # Record for evaluation
                        func_calls.append({
                            "tool_name": tool_name,
                            "arguments": tool_args,
                            "content": result,
                        })

                        result_str = str(result) if not isinstance(result, str) else result
                        conversation_history.append(
                            ToolMessage(content=result_str, tool_call_id=tc["id"])
                        )
                        agent_messages.append({
                            "step": turn,
                            "type": "tool_call",
                            "tool": tool_name,
                            "agent": final_state.get("current_agent", "unknown"),
                            "content_preview": result_str[:300],
                        })

                    if tool_call_count >= self.MAX_TOOL_CALLS:
                        done = True

                else:
                    # Text response — in SOPBench this means agent is communicating
                    conversation_history.append(last_ai_msg)
                    agent_messages.append({
                        "step": turn,
                        "type": "respond",
                        "agent": final_state.get("current_agent", "unknown"),
                        "content_preview": last_ai_msg.content[:300],
                    })

                    # Check if agent mentioned exit/done
                    content_lower = last_ai_msg.content.lower()
                    if any(w in content_lower for w in ["goodbye", "have a great day", "is there anything else"]):
                        done = True
                    elif not done:
                        # Agent produced text — send the SOPBench default user
                        # response with known data to keep the conversation going.
                        # This matches run_simulation.py:206-213.
                        user_known = task_data.get("user_known", {})
                        default_reply = "Here is all the information I can provide:\n"
                        default_reply += json.dumps(user_known, indent=2)
                        default_reply += (
                            "\n\nPlease directly use the most appropriate tool to "
                            "solve my request as quickly as possible and use the "
                            "`exit_conversation` action to end our conversation if "
                            "you have completed my request or cannot assist me."
                        )
                        conversation_history.append(HumanMessage(content=default_reply))

            # Post-hoc evaluation via oracle
            final_database = domain_system.evaluation_get_database()
            eval_result = self.env_manager.evaluate(task_data, func_calls, final_database)

            success = eval_result.get("success", False)
            reward = eval_result.get("reward", 0.0)
            elapsed_ms = (time.monotonic() - start_time) * 1000

            return RunResult(
                task_id=task.id,
                success=success,
                output=f"reward={reward:.1f}, turns={turn+1}, tools={tool_call_count}, "
                       f"no_err={eval_result.get('no_tool_call_error', False)}, "
                       f"constraint_ok={eval_result.get('constraint_not_violated', False)}, "
                       f"db_match={eval_result.get('database_match', False)}, "
                       f"dirgraph={eval_result.get('dirgraph_satisfied', False)}, "
                       f"action_ok={eval_result.get('action_called_correctly', False)}, "
                       f"goal={eval_result.get('user_goal', '?')}, "
                       f"should_succeed={eval_result.get('action_should_succeed', '?')}",
                latency_ms=elapsed_ms,
                trace_id=trace_id,
                token_count=total_tokens,
                agent_messages=agent_messages,
            )

        except Exception as e:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.warning(f"Task {task.id} failed with error: {e}")
            return RunResult(
                task_id=task.id,
                success=False,
                error=str(e),
                latency_ms=elapsed_ms,
                trace_id=trace_id,
            )


def load_benchmark_tasks(
    benchmark: str,
    split: str = "val",
    n_instances: int = 50,
    seed: int = 42,
) -> list[TaskInstance]:
    """Load task instances for a given benchmark via the benchmark loader registry."""
    from abstral.benchmarks import get_loader
    loader = get_loader(benchmark)
    return loader.load_tasks(split=split, n_instances=n_instances, seed=seed)
