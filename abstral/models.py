"""Shared data models that don't depend on heavy external libraries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskInstance:
    """A single task instance for agent evaluation."""
    id: str
    input_text: str
    expected_output: str = ""
    task_type: str = ""
    difficulty: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class RunResult:
    """Result of a single task instance execution."""
    task_id: str
    success: bool
    output: str = ""
    error: str = None
    token_count: int = 0
    latency_ms: float = 0
    trace_id: str = ""
    agent_messages: list = field(default_factory=list)  # Per-agent message flow for trace analysis
    metadata: dict = field(default_factory=dict)  # Routing stats, agent info for appendix


@dataclass
class BatchRunResult:
    """Aggregated result of a batch RUN phase."""
    run_id: str
    results: list = field(default_factory=list)
    trace_set: Any = None
    metrics: dict = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.success) / len(self.results)

    @property
    def total_tokens(self) -> int:
        return sum(r.token_count for r in self.results)

    @property
    def mean_latency_ms(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.latency_ms for r in self.results) / len(self.results)
