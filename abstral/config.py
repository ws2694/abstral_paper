"""Configuration models for ABSTRAL."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Optional, List, Dict

import yaml
from pydantic import BaseModel, Field


class EvidenceClass(str, Enum):
    EC1 = "EC1"  # Reasoning Error → K
    EC2 = "EC2"  # Topology Failure → R
    EC3 = "EC3"  # Missing Specialization → T (add role)
    EC4 = "EC4"  # Interface Failure → P
    EC5 = "EC5"  # Emergent Pattern → T (add heuristic)


EC_SECTION_MAP: dict[EvidenceClass, str] = {
    EvidenceClass.EC1: "K",
    EvidenceClass.EC2: "R",
    EvidenceClass.EC3: "T",
    EvidenceClass.EC4: "P",
    EvidenceClass.EC5: "T",
}

EC_WEIGHTS: dict[str, int] = {
    "C1": 2,  # skill diff empty
    "C2": 2,  # AUC plateau
    "C3": 1,  # EC signal collapse
    "C4": 1,  # complexity penalty (triggers compaction first)
}


class TopologyFamily(str, Enum):
    SINGLE = "single"
    PIPELINE = "pipeline"
    ENSEMBLE = "ensemble"
    DEBATE = "debate"
    HIERARCHICAL = "hierarchical"
    DYNAMIC_ROUTING = "dynamic_routing"


class MetaAgentConfig(BaseModel):
    model: str = "claude-sonnet-4-20250514"
    provider: str = "anthropic"
    max_tokens: int = 8192


class AgentBackboneConfig(BaseModel):
    model: str = "gpt-4o"
    provider: str = "openai"
    max_tokens: int = 50000


class BenchmarkConfig(BaseModel):
    name: str
    metric: str
    val_instances: int = 500
    test_instances: Optional[int] = None


class SandboxConfig(BaseModel):
    memory_mb: int = 2048
    cpus: int = 2


class InnerLoopConfig(BaseModel):
    max_iterations: int = 15
    val_batch_size: int = 50
    convergence_batch_size: int = 500
    token_budget: int = 50000
    wall_clock_limit_sec: int = 300
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)


class ConvergenceConfig(BaseModel):
    skill_diff_threshold: int = 5
    auc_plateau_epsilon: float = 0.005
    auc_plateau_window: int = 3
    ec_collapse_threshold: float = 0.10
    max_rules: int = 200
    max_words: int = 5000
    min_weight_to_terminate: int = 3
    consolidation_interval: int = 5  # scheduled consolidation every k iterations (§3.3.1)
    consolidation_ratio_target: float = 0.65  # target: ≤0.65 (≥35% reduction) (§3.3.1)


class OuterLoopConfig(BaseModel):
    n_outer: int = 6
    min_ged: int = 3  # δ_struct: structural distance threshold (§3.4.1)
    min_semantic_dist: float = 0.25  # δ_sem: semantic cosine distance threshold (§3.4.1)
    embedding_model: str = "text-embedding-3-small"  # for semantic non-synonymy check
    topology_families: list[str] = Field(
        default_factory=lambda: [f.value for f in TopologyFamily]
    )


class TrackingConfig(BaseModel):
    mlflow_tracking_uri: str = "mlruns"
    otel_service_name: str = "abstral"
    experiment_name: str = "abstral-main"


class PathsConfig(BaseModel):
    skill_repo: str = "skill_repo"
    artifacts: str = "artifacts"
    logs: str = "logs"


class TauBenchConfig(BaseModel):
    domain: str = "airline"
    user_model: str = "gpt-4o"  # Match τ-bench default
    user_provider: str = "openai"
    task_split: str = "test"
    max_conversation_turns: int = 30


class SOPBenchConfig(BaseModel):
    domain: str = "bank"
    mode: str = "prompt"  # "prompt" (agent verifies) or "program" (oracle enforces)
    max_turns: int = 25


class ABSTRALConfig(BaseModel):
    meta_agent: MetaAgentConfig = Field(default_factory=MetaAgentConfig)
    agent_backbone: AgentBackboneConfig = Field(default_factory=AgentBackboneConfig)
    benchmarks: list[BenchmarkConfig] = Field(default_factory=list)
    inner_loop: InnerLoopConfig = Field(default_factory=InnerLoopConfig)
    convergence: ConvergenceConfig = Field(default_factory=ConvergenceConfig)
    outer_loop: OuterLoopConfig = Field(default_factory=OuterLoopConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    tau_bench: TauBenchConfig = Field(default_factory=TauBenchConfig)
    sop_bench: SOPBenchConfig = Field(default_factory=SOPBenchConfig)

    @classmethod
    def from_yaml(cls, path: "str | Path") -> "ABSTRALConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def to_yaml(self, path: "str | Path") -> None:
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)


# ---------- Structured output schemas for meta-agent ----------


class EpistemicStance(str, Enum):
    """Primary reasoning mode for discovered agent roles (§3.2.1)."""
    SKEPTIC = "Skeptic"
    VERIFIER = "Verifier"
    EXPLORER = "Explorer"
    SYNTHESIZER = "Synthesizer"
    ENFORCER = "Enforcer"


class FunctionalType(str, Enum):
    """Closed taxonomy for GED canonicalization (§3.4.1)."""
    ROUTER = "Router"
    PLANNER = "Planner"
    EXECUTOR = "Executor"
    VERIFIER = "Verifier"
    AGGREGATOR = "Aggregator"
    SPECIALIST = "Specialist"
    ORACLE = "Oracle"


class AgentRole(BaseModel):
    """A single agent role discovered/defined in the skill document.

    Per §3.2.1, discovered roles have [Adjective][Noun] naming convention
    and include the birth trace that triggered their creation.
    """
    name: str  # [Adjective][Noun] format for discovered roles
    system_prompt: str
    tools: list[str] = Field(default_factory=list)
    description: str = ""
    epistemic_stance: Optional[EpistemicStance] = None
    birth_trace: str = ""  # Trace ID of the EC3 event that created this role
    interface_contract: Dict[str, str] = Field(default_factory=dict)  # required input/output fields
    functional_type: FunctionalType = FunctionalType.SPECIALIST  # for GED canonicalization


class EdgeSpec(BaseModel):
    """An edge in the agent topology graph."""
    source: str
    target: str
    condition: str = ""  # optional conditional routing expression


class AgentSpec(BaseModel):
    """Complete specification for a multi-agent system, produced by BUILD phase."""
    topology_family: TopologyFamily
    roles: list[AgentRole]
    edges: list[EdgeSpec]
    entry_point: str
    message_schema: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""


class TraceEvidence(BaseModel):
    """A single piece of diagnosed evidence from trace analysis."""
    ec_class: EvidenceClass
    trace_id: str
    failed_trace_summary: str
    succeeded_trace_summary: str = ""
    reasoning: str
    suggested_edit: str
    target_section: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class SkillEdit(BaseModel):
    """A targeted edit to apply to a SKILL.md section."""
    section: str  # K, R, T, or P
    action: str  # "add", "modify", "remove"
    content: str
    motivating_trace_ids: list[str]
    ec_class: EvidenceClass
    reasoning: str


class ConvergenceSignal(BaseModel):
    """Status of a single convergence signal."""
    signal_id: str  # C1, C2, C3, C4, T
    name: str
    fired: bool
    weight: int
    value: Any  # float or str
    condition: str


class ConvergenceResult(BaseModel):
    """Overall convergence decision."""
    signals: list[ConvergenceSignal]
    total_weight: int
    should_terminate: bool
    should_compact: bool
    reason: str


class OuterSeedConfig(BaseModel):
    """Configuration for a new outer-loop seed."""
    outer_iteration: int
    preserved_k: str  # domain knowledge carried forward
    cleared_t: bool = True
    repulsion_topologies: list[str]  # prior converged topology descriptions
    target_family: TopologyFamily
    seed_r_constraint: str  # repulsion constraint injected into R
