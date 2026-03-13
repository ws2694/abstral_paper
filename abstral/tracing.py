"""Trace collection, pairing, and analysis utilities.

Uses OpenTelemetry for distributed tracing instrumentation. Traces are
collected locally (in-memory) and optionally exported to any OTel-compatible
backend (Jaeger, Zipkin, OTLP collector, etc.).

No rate limits — all trace data lives locally until explicitly exported.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------- OpenTelemetry setup (lazy, optional export) ----------

_tracer = None


def get_tracer(service_name: str = "abstral"):
    """Get or create a cached OpenTelemetry tracer."""
    global _tracer
    if _tracer is not None:
        return _tracer
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.resources import Resource

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)

        # If OTEL_EXPORTER_OTLP_ENDPOINT is set, add OTLP exporter
        import os
        if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                from opentelemetry.sdk.trace.export import BatchSpanProcessor
                exporter = OTLPSpanExporter()
                provider.add_span_processor(BatchSpanProcessor(exporter))
                logger.info(f"OTel OTLP exporter configured: {os.environ['OTEL_EXPORTER_OTLP_ENDPOINT']}")
            except ImportError:
                logger.debug("OTLP exporter not installed; traces are local-only")

        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("abstral", "0.1.0")
        return _tracer
    except ImportError:
        logger.debug("OpenTelemetry SDK not installed; tracing disabled")
        return None


# ---------- Data classes (unchanged interface) ----------


@dataclass
class TracePair:
    """A paired failed/succeeded trace for contrastive analysis."""
    failed_trace: dict[str, Any]
    succeeded_trace: dict[str, Any]
    task_type: str = ""
    task_id: str = ""


@dataclass
class TraceSet:
    """Collection of traces from a single RUN phase execution."""
    run_id: str
    traces: list[dict[str, Any]] = field(default_factory=list)
    succeeded: list[dict[str, Any]] = field(default_factory=list)
    failed: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


class TraceManager:
    """Manages trace collection and pairing for ANALYZE phase.

    Traces are built locally from RunResults (no remote API calls).
    OpenTelemetry spans are emitted for observability but the ANALYZE
    phase reads from the local TraceSet, not from an external backend.
    """

    def __init__(self, project_name: str = "abstral"):
        self.project_name = project_name
        self._tracer = get_tracer(project_name)

    def build_trace_set(self, run_id: str, results: list[Any]) -> TraceSet:
        """Build a TraceSet from local RunResults.

        This replaces the old LangSmith fetch_traces() — no remote API calls,
        no rate limits. Trace data comes directly from the pipeline's RunResults.
        """
        trace_set = TraceSet(run_id=run_id)

        for r in results:
            agent_flow = ""
            if hasattr(r, "agent_messages") and r.agent_messages:
                flow_parts = []
                for am in r.agent_messages:
                    flow_parts.append(
                        f"[Step {am['step']}] {am['type']} (agent={am['agent']}): "
                        f"{am['content_preview'][:200]}"
                    )
                agent_flow = "\n".join(flow_parts)

            # Count tool calls from agent messages
            n_tool_calls = 0
            if hasattr(r, "agent_messages") and r.agent_messages:
                n_tool_calls = sum(1 for am in r.agent_messages if am.get("type") == "tool_call")

            trace = {
                "id": r.trace_id,
                "status": "success" if r.success else "error",
                "inputs": {"task_id": r.task_id},
                "outputs": {"output": r.output[:500] if r.output else ""},
                "error": r.error,
                "tags": [f"task_type:{r.task_id.split('-')[0]}"] if r.task_id else [],
                "token_count": r.token_count,
                "agent_flow": agent_flow,
                "n_agent_steps": len(r.agent_messages) if hasattr(r, "agent_messages") else 0,
                "n_tool_calls": n_tool_calls,
            }

            trace_set.traces.append(trace)
            if r.success:
                trace_set.succeeded.append(trace)
            else:
                trace_set.failed.append(trace)

        n_total = len(trace_set.traces)
        n_succ = len(trace_set.succeeded)
        n_fail = len(trace_set.failed)
        logger.info(f"Built trace set: {n_total} traces ({n_succ} succeeded, {n_fail} failed)")

        if n_total > 0:
            trace_set.metrics["success_rate"] = n_succ / n_total
            trace_set.metrics["total_tokens"] = sum(
                t.get("token_count", 0) for t in trace_set.traces
            )

        return trace_set

    def fetch_traces(self, run_id: str, limit: int = 200) -> TraceSet:
        """Backward-compatible stub. Returns empty TraceSet.

        The pipeline now builds TraceSet locally from RunResults via
        build_trace_set(). This method exists only to avoid breaking
        callers that still reference it.
        """
        logger.debug(f"fetch_traces called for {run_id} — returning empty (use build_trace_set instead)")
        return TraceSet(run_id=run_id)

    def pair_traces(self, trace_set: TraceSet) -> list[TracePair]:
        """Pair failed traces with succeeded traces for contrastive analysis.

        Pairing strategy: match by task type/category when available,
        otherwise pair sequentially.
        """
        pairs: list[TracePair] = []

        # Group by task type if available
        failed_by_type: dict[str, list] = {}
        succeeded_by_type: dict[str, list] = {}

        for trace in trace_set.failed:
            task_type = self._extract_task_type(trace)
            failed_by_type.setdefault(task_type, []).append(trace)

        for trace in trace_set.succeeded:
            task_type = self._extract_task_type(trace)
            succeeded_by_type.setdefault(task_type, []).append(trace)

        # Pair within task types first
        all_types = set(failed_by_type.keys()) | set(succeeded_by_type.keys())
        for task_type in all_types:
            fails = failed_by_type.get(task_type, [])
            succs = succeeded_by_type.get(task_type, [])
            if not fails or not succs:
                continue

            for i, fail in enumerate(fails):
                succ = succs[i % len(succs)]
                pairs.append(TracePair(
                    failed_trace=fail,
                    succeeded_trace=succ,
                    task_type=task_type,
                    task_id=fail.get("id", ""),
                ))

        # If no typed pairs found, pair sequentially
        if not pairs:
            for i in range(min(len(trace_set.failed), len(trace_set.succeeded))):
                pairs.append(TracePair(
                    failed_trace=trace_set.failed[i],
                    succeeded_trace=trace_set.succeeded[i],
                ))

        logger.info(f"Created {len(pairs)} trace pairs for contrastive analysis")
        return pairs

    def _extract_task_type(self, trace: dict) -> str:
        """Extract task type from trace tags or inputs."""
        for tag in trace.get("tags", []):
            if tag.startswith("task_type:"):
                return tag.split(":", 1)[1]
        # Fall back to input-based classification
        inputs = trace.get("inputs", {})
        return inputs.get("task_type", "unknown")

    def summarize_trace(self, trace: dict, max_length: int = 4000) -> str:
        """Create a detailed summary of a trace for LLM analysis.

        Includes per-agent message flow and topology information to enable
        the analyzer to distinguish between all five evidence classes (EC1-EC5).
        """
        parts = [
            f"Trace ID: {trace.get('id', 'unknown')}",
            f"Status: {trace.get('status', 'unknown')}",
        ]

        if trace.get("error"):
            parts.append(f"Error: {trace['error'][:500]}")

        inputs = trace.get("inputs", {})
        if inputs:
            input_str = str(inputs)[:800]
            parts.append(f"Inputs: {input_str}")

        outputs = trace.get("outputs", {})
        if outputs:
            output_str = str(outputs)[:800]
            parts.append(f"Outputs: {output_str}")

        # Include topology structure so analyzer can diagnose structural issues
        topo_info = trace.get("topology_info", {})
        if topo_info:
            parts.append(f"\nTopology: {topo_info.get('family', 'unknown')} ({topo_info.get('n_roles', '?')} roles)")
            roles = topo_info.get("roles", [])
            if roles:
                role_str = ", ".join(f"{r['name']}({r['type']})" for r in roles)
                parts.append(f"Roles: {role_str}")
            edges = topo_info.get("edges", [])
            if edges:
                edge_str = ", ".join(f"{e['src']}->{e['tgt']}" for e in edges)
                parts.append(f"Edges: {edge_str}")

        # Include per-agent message flow for detailed failure analysis
        agent_flow = trace.get("agent_flow", "")
        n_steps = trace.get("n_agent_steps", 0)
        if agent_flow:
            parts.append(f"\nAgent Interaction Flow ({n_steps} steps):")
            # Truncate flow to fit within max_length
            flow_budget = max_length - len("\n".join(parts)) - 100
            if flow_budget > 200:
                parts.append(agent_flow[:flow_budget])
            else:
                parts.append(f"(flow truncated, {n_steps} agent steps)")

        summary = "\n".join(parts)
        return summary[:max_length]
