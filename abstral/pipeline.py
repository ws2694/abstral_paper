"""Real end-to-end pipeline runner for ABSTRAL.

Executes the full three-layer pipeline with real LLM calls:
  - Layer 1: BUILD (meta-agent designs AgentSpec) -> RUN (LangGraph execution) -> ANALYZE (contrastive trace analysis) -> UPDATE (skill edits)
  - Layer 2: Convergence detection (C1-C4 + hard cap)
  - Layer 3: Diversity seeding with topology repulsion

Provides a PipelineState object for real-time dashboard monitoring.

Required environment variables:
  - ANTHROPIC_API_KEY: For meta-agent calls (BUILD, ANALYZE, UPDATE, COMPACT)
  - OPENAI_API_KEY: For agent backbone execution
  - OTEL_EXPORTER_OTLP_ENDPOINT: For OpenTelemetry trace export (optional)
"""

from __future__ import annotations

import logging
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Optional

from abstral.config import (
    ABSTRALConfig,
    AgentSpec,
    TopologyFamily,
)
from abstral.models import RunResult, BatchRunResult
from abstral.skill.document import SkillDocument
from abstral.skill.versioning import SkillRepository
from abstral.layer1.builder import AgentBuilder
from abstral.layer1.runner import AgentRunner, TauBenchRunner, SOPBenchRunner, load_benchmark_tasks
from abstral.layer1.analyzer import TraceAnalyzer
from abstral.layer1.updater import SkillUpdater
from abstral.layer2.convergence import ConvergenceDetector
from abstral.layer3.seeder import DiversitySeeder
from abstral.layer3.topology import (
    spec_to_graph,
    classify_topology,
    compute_ged,
)
from abstral.layer3.landscape import DesignLandscape, LandscapePoint
from abstral.tracing import TraceSet

logger = logging.getLogger(__name__)


class PipelineState:
    """Holds all state for a running pipeline, updated in real-time.

    The dashboard reads from this object to render live metrics.
    """

    def __init__(self):
        self.config: Optional[ABSTRALConfig] = None
        self.status: str = "idle"  # idle, running, complete, error
        self.current_phase: str = ""
        self.outer_iter: int = 0
        self.inner_iter: int = 0
        self.max_outer: int = 0
        self.max_inner: int = 0
        self.benchmark: str = ""

        # Per-iteration data
        self.auc_history: list[dict] = []           # {outer, inner, auc, benchmark, family}
        self.ec_history: list[dict] = []            # {outer, inner, ec_dist}
        self.skill_history: list[dict] = []         # {outer, inner, rules, words, diff}
        self.convergence_history: list[dict] = []   # {outer, inner, signals, terminated}

        # Outer loop data
        self.landscape_points: list[dict] = []
        self.ged_matrix: list[list[float]] = []
        self.umap_coords: list[list[float]] = []
        self.topology_families_seen: list[str] = []

        # Current skill doc text
        self.current_skill_text: str = ""
        self.log_lines: list[str] = []

        # Error tracking
        self.last_error: str = ""

        # ── Rich experiment data for transparency ──
        self.spec_history: list[dict] = []           # Full AgentSpec per iteration
        self.evidence_history: list[dict] = []       # All evidence items per iteration
        self.task_results_history: list[dict] = []   # Per-task success/fail details
        self.timing_history: list[dict] = []         # Phase durations per iteration
        self.cumulative_tokens: int = 0              # Running total of tokens
        self.cumulative_api_calls: int = 0           # Running total of API calls
        self.seed_history: list[dict] = []           # Seed doc info per outer loop
        self.section_diffs: list[dict] = []          # Per-section diff details per iteration

        # ── Appendix-quality diagnostics ──
        self.routing_stats: list[dict] = []           # Per-iteration routing efficiency
        self.k_section_snapshots: list[dict] = []     # K section text at key points
        self.topology_gallery: list[dict] = []        # Best topology per outer loop

    def log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self.log_lines.append(line)
        logger.info(msg)
        # Always print to stderr for real-time visibility
        import sys
        print(line, file=sys.stderr, flush=True)
        # Append to log file if configured
        if hasattr(self, '_log_file') and self._log_file:
            try:
                with open(self._log_file, "a") as f:
                    f.write(line + "\n")
            except Exception:
                pass
        if len(self.log_lines) > 2000:
            self.log_lines = self.log_lines[-1500:]

    def to_landscape_summary(self) -> dict:
        families = {}
        for p in self.landscape_points:
            f = p["family"]
            families[f] = families.get(f, 0) + 1
        best = max(self.landscape_points, key=lambda p: p["auc"]) if self.landscape_points else None
        return {
            "n_points": len(self.landscape_points),
            "n_families": len(families),
            "family_coverage": families,
            "mean_ged": self._mean_ged(),
            "global_optimum_auc": best["auc"] if best else 0,
            "global_optimum_family": best["family"] if best else None,
            "outer_loop_value": (best["auc"] - self.landscape_points[0]["auc"]) if best and self.landscape_points else 0,
            "points": self.landscape_points,
            "ged_matrix": self.ged_matrix,
        }

    def _mean_ged(self) -> float:
        if len(self.ged_matrix) < 2:
            return 0.0
        n = len(self.ged_matrix)
        vals = []
        for i in range(n):
            for j in range(i + 1, n):
                vals.append(self.ged_matrix[i][j])
        return sum(vals) / len(vals) if vals else 0.0


def run_pipeline(
    config: ABSTRALConfig,
    benchmark: str,
    state: PipelineState,
    base_path: Optional[Path] = None,
) -> PipelineState:
    """Run the full three-layer ABSTRAL pipeline with real LLM calls.

    Args:
        config: ABSTRAL configuration
        benchmark: which benchmark to run (alfworld, webarena, gaia, tabular)
        state: mutable state object (for live dashboard updates)
        base_path: directory for skill repos and artifacts
    """
    state.config = config
    state.status = "running"
    state.benchmark = benchmark
    state.max_outer = config.outer_loop.n_outer
    state.max_inner = config.inner_loop.max_iterations

    # ── Validate required API keys ──
    import os
    missing_keys = []
    if not os.environ.get("ANTHROPIC_API_KEY"):
        missing_keys.append("ANTHROPIC_API_KEY (meta-agent: BUILD, ANALYZE, UPDATE)")
    if not os.environ.get("OPENAI_API_KEY"):
        missing_keys.append("OPENAI_API_KEY (agent backbone: RUN)")
    if missing_keys:
        error_msg = "Missing required API keys:\n  - " + "\n  - ".join(missing_keys)
        state.status = "error"
        state.last_error = error_msg
        state.log(f"STARTUP ERROR: {error_msg}")
        return state

    # Initialize OpenTelemetry tracing (local, no rate limits)
    from abstral.tracing import get_tracer
    otel_tracer = get_tracer(config.tracking.otel_service_name)
    if otel_tracer:
        state.log("[STARTUP] OpenTelemetry tracing initialized.")
    else:
        state.log("[STARTUP] OpenTelemetry SDK not available; tracing disabled (non-fatal).")

    if base_path is None:
        base_path = Path(tempfile.mkdtemp(prefix="abstral_"))

    base_path.mkdir(parents=True, exist_ok=True)

    # Initialize real components
    is_tau = benchmark.startswith("tau_")
    is_sop = benchmark.startswith("sop_")
    builder = AgentBuilder(config)
    if is_tau:
        runner = TauBenchRunner(config)
    elif is_sop:
        runner = SOPBenchRunner(config)
    else:
        runner = AgentRunner(config)
    analyzer = TraceAnalyzer(config)
    updater = SkillUpdater(config)
    seeder = DiversitySeeder(config)
    graphs_collected = []

    # For τ-bench: prepare tool provider and wiki
    tau_tool_provider = None
    tau_wiki = ""
    if is_tau:
        from abstral.tau_adapter import TauEnvManager, wrap_tau_tools
        tau_manager = TauEnvManager(
            domain=config.tau_bench.domain,
            user_model=config.tau_bench.user_model,
            user_provider=config.tau_bench.user_provider,
            task_split=config.tau_bench.task_split,
        )
        tau_wiki = tau_manager.get_wiki()
        _ref_env, _ = tau_manager.create_env(0)
        tau_tool_provider = lambda: wrap_tau_tools(_ref_env)

    # For SOPBench: prepare tool provider and system prompt
    sop_tool_provider = None
    sop_system_prompt = ""
    if is_sop:
        from abstral.sop_adapter import SOPEnvManager, wrap_sop_tools
        sop_manager = SOPEnvManager(domain=config.sop_bench.domain)
        # Create ref env for tool schema extraction
        _ref_sys, _ref_user, _ref_asst, _ref_info, _ = sop_manager.create_env(0)
        sop_system_prompt = sop_manager.get_system_prompt(_ref_asst)
        sop_tool_provider = lambda: wrap_sop_tools(_ref_sys, _ref_asst)

    state.log("=" * 80)
    state.log("ABSTRAL PIPELINE — FULL EXPERIMENT LOG")
    state.log("=" * 80)
    state.log(f"Benchmark:          {benchmark}")
    state.log(f"Meta-agent model:   {config.meta_agent.model}")
    state.log(f"Meta-agent tokens:  {config.meta_agent.max_tokens}")
    state.log(f"Backbone model:     {config.agent_backbone.model}")
    state.log(f"Backbone tokens:    {config.agent_backbone.max_tokens}")
    state.log(f"Backbone provider:  {config.agent_backbone.provider}")
    state.log(f"Outer loops (N):    {config.outer_loop.n_outer}")
    state.log(f"Max inner iters:    {config.inner_loop.max_iterations}")
    state.log(f"Val batch size:     {config.inner_loop.val_batch_size}")
    state.log(f"Min GED target:     {config.outer_loop.min_ged}")
    state.log(f"AUC plateau eps:    {config.convergence.auc_plateau_epsilon}")
    state.log(f"Artifacts dir:      {base_path}")
    state.log(f"Start time:         {time.strftime('%Y-%m-%d %H:%M:%S')}")
    state.log("-" * 80)

    try:
        for outer in range(1, config.outer_loop.n_outer + 1):
            if state.status == "stopped":
                state.log("Pipeline stopped by user.")
                break

            state.outer_iter = outer

            # ── Layer 3: Generate seed ──
            state.log("")
            state.log("=" * 80)
            state.log(f"OUTER LOOP {outer}/{config.outer_loop.n_outer}")
            state.log("=" * 80)

            if outer == 1:
                if is_tau:
                    seed_doc = SkillDocument.create_tau_bench_seed(
                        config.tau_bench.domain, "hierarchical", tau_wiki
                    )
                elif is_sop:
                    seed_doc = SkillDocument.create_sop_bench_seed(
                        config.sop_bench.domain, "hierarchical", sop_system_prompt
                    )
                else:
                    seed_doc = SkillDocument.create_seed(benchmark, "hierarchical")
                state.current_phase = f"Outer {outer}: Seeding hierarchical"
                state.log(f"[SEED] Initial seed topology: hierarchical")
                state.log(f"[SEED] Seed K section: {len(seed_doc.K.split())} words")
                state.log(f"[SEED] Seed R section: {len(seed_doc.R.split())} words")
                state.log(f"[SEED] Seed T section: {len(seed_doc.T.split())} words")
                state.log(f"[SEED] Seed P section: {len(seed_doc.P.split())} words")
                state.seed_history.append({
                    "outer": outer, "family": "hierarchical", "source": "initial",
                    "word_count": seed_doc.word_count(), "rule_count": seed_doc.rule_count(),
                })
            else:
                state.current_phase = f"Outer {outer}: Generating diverse seed"
                state.log(f"[SEED] Generating diverse seed via topology repulsion...")
                state.log(f"[SEED] Archive contains {len(seeder.archive)} converged topologies")
                state.log(f"[SEED] Families seen so far: {state.topology_families_seen}")
                seed_doc, seed_config = seeder.generate_seed(outer, benchmark)
                state.log(f"[SEED] Target family: {seed_config.target_family.value}")
                state.log(f"[SEED] Seed word count: {seed_doc.word_count()}, rule count: {seed_doc.rule_count()}")
                state.seed_history.append({
                    "outer": outer, "family": seed_config.target_family.value, "source": "repulsion",
                    "word_count": seed_doc.word_count(), "rule_count": seed_doc.rule_count(),
                })

            # Initialize skill repo for this outer iteration
            repo_path = base_path / f"outer_{outer}"
            repo = SkillRepository(repo_path)
            repo.init(seed_doc)

            skill_doc = seed_doc
            previous_doc = None
            convergence_detector = ConvergenceDetector(config)
            convergence_detector.reset()

            best_spec: Optional[AgentSpec] = None
            best_auc = -1.0
            converged_at = config.inner_loop.max_iterations
            family_name = "unknown"

            for inner in range(1, config.inner_loop.max_iterations + 1):
                if state.status == "stopped":
                    break

                state.inner_iter = inner

                # ── BUILD ──
                iter_start = time.time()
                phase_timings = {}
                state.current_phase = f"O{outer} I{inner}: BUILD"
                state.log("")
                state.log(f"--- Outer {outer}, Inner {inner} ---")
                state.log(f"[BUILD] Meta-agent designing AgentSpec...")
                state.log(f"[BUILD] SKILL.md state: {skill_doc.rule_count()} rules, {skill_doc.word_count()} words")
                build_start = time.time()
                try:
                    if is_tau:
                        task_desc = (
                            f"Benchmark: {benchmark}. Design an optimal multi-agent system "
                            f"for {config.tau_bench.domain} customer service. Agents must interact "
                            f"with customers, query databases, and modify bookings/orders "
                            f"following strict {config.tau_bench.domain} policies."
                        )
                    elif is_sop:
                        task_desc = (
                            f"Benchmark: {benchmark}. Design an optimal multi-agent system "
                            f"for {config.sop_bench.domain} service. Agents must follow Standard "
                            f"Operating Procedures, verify constraints before executing actions, "
                            f"and use domain tools to complete customer requests."
                        )
                    else:
                        task_desc = f"Benchmark: {benchmark}. Design an optimal multi-agent system."
                    spec = builder.design_agent_spec(skill_doc, task_desc)

                    # Validate: does produced topology match target family from R section?
                    produced_family = classify_topology(spec)
                    declared_family = spec.topology_family
                    target_from_r = skill_doc.metadata.get("topology_family", "")

                    # Treat ensemble/dynamic_routing as compatible — structurally
                    # similar (fan-out → workers → fan-in), classifier can't
                    # reliably distinguish them.
                    _COMPATIBLE_FAMILIES = {
                        frozenset({"ensemble", "dynamic_routing"}),
                    }
                    def _families_compatible(a: str, b: str) -> bool:
                        if a == b:
                            return True
                        return any(frozenset({a, b}) <= group for group in _COMPATIBLE_FAMILIES)

                    if (target_from_r
                        and not _families_compatible(produced_family.value, target_from_r)
                        and outer > 1):
                        state.log(f"[BUILD] WARNING: Target family was '{target_from_r}' but produced '{produced_family.value}' (declared: {declared_family.value})")
                        state.log(f"[BUILD] Retrying BUILD with explicit topology enforcement...")
                        # Retry once with stronger constraint
                        retry_desc = f"{task_desc} CRITICAL: You MUST produce a {target_from_r} topology. The R section requires it."
                        try:
                            spec = builder.design_agent_spec(skill_doc, retry_desc)
                            produced_family = classify_topology(spec)
                            state.log(f"[BUILD] Retry produced: {produced_family.value} (declared: {spec.topology_family.value})")
                        except Exception:
                            state.log(f"[BUILD] Retry failed, using original spec")

                    family_name = classify_topology(spec).value
                    build_dur = time.time() - build_start
                    phase_timings["build"] = round(build_dur, 2)
                    state.log(f"[BUILD] Complete in {build_dur:.1f}s")
                    state.log(f"[BUILD]   Topology family: {spec.topology_family.value}")
                    state.log(f"[BUILD]   Entry point: {spec.entry_point}")
                    state.log(f"[BUILD]   Roles ({len(spec.roles)}):")
                    for role in spec.roles:
                        prompt_preview = role.system_prompt[:120].replace("\n", " ") if role.system_prompt else "(empty)"
                        state.log(f"[BUILD]     - {role.name}: {role.description} | prompt: \"{prompt_preview}...\"")
                    state.log(f"[BUILD]   Edges ({len(spec.edges)}):")
                    for edge in spec.edges:
                        cond = f" [if: {edge.condition}]" if hasattr(edge, 'condition') and edge.condition else ""
                        state.log(f"[BUILD]     - {edge.source} -> {edge.target}{cond}")
                    state.spec_history.append({
                        "outer": outer, "inner": inner,
                        "family": family_name,
                        "n_roles": len(spec.roles),
                        "roles": [{"name": r.name, "description": r.description} for r in spec.roles],
                        "n_edges": len(spec.edges),
                        "edges": [{"source": e.source, "target": e.target} for e in spec.edges],
                        "entry_point": spec.entry_point,
                        "build_time_s": round(build_dur, 2),
                    })
                except Exception as e:
                    build_dur = time.time() - build_start
                    phase_timings["build"] = round(build_dur, 2)
                    state.log(f"[BUILD] FAILED after {build_dur:.1f}s: {e}")
                    state.log(f"[BUILD] Full traceback: {traceback.format_exc()}")
                    state.log(f"[BUILD] Falling back to single-agent topology")
                    from abstral.config import AgentRole, EdgeSpec
                    spec = AgentSpec(
                        topology_family=TopologyFamily.SINGLE,
                        roles=[AgentRole(name="agent", system_prompt=f"You are a general-purpose agent for {benchmark} tasks.", description="General agent")],
                        edges=[],
                        entry_point="agent",
                    )
                    family_name = "single"
                    state.spec_history.append({
                        "outer": outer, "inner": inner,
                        "family": "single", "n_roles": 1, "roles": [{"name": "agent", "description": "Fallback"}],
                        "n_edges": 0, "edges": [], "entry_point": "agent",
                        "build_time_s": round(build_dur, 2), "fallback": True, "error": str(e),
                    })

                # ── RUN ──
                state.current_phase = f"O{outer} I{inner}: RUN"
                state.log(f"[RUN] Executing {family_name} topology on {config.inner_loop.val_batch_size} tasks...")
                run_start = time.time()
                try:
                    if is_tau:
                        _tool_prov = tau_tool_provider
                        _bench_mode = "tau"
                        _domain_k = skill_doc.K
                    elif is_sop:
                        _tool_prov = sop_tool_provider
                        _bench_mode = "tau"  # SOPBench uses same multi-agent pattern as τ-bench
                        _domain_k = skill_doc.K
                    else:
                        _tool_prov = None
                        _bench_mode = "gaia"
                        _domain_k = ""
                    compiled_graph, metadata = builder.build_graph(
                        spec, task_desc,
                        tool_provider=_tool_prov,
                        benchmark_mode=_bench_mode,
                        domain_knowledge=_domain_k,
                    )
                    state.log(f"[RUN] LangGraph compiled successfully. Graph nodes: {list(compiled_graph.get_graph().nodes) if hasattr(compiled_graph, 'get_graph') else 'N/A'}")
                    tasks = load_benchmark_tasks(
                        benchmark=benchmark,
                        split="val",
                        n_instances=config.inner_loop.val_batch_size,
                    )
                    state.log(f"[RUN] Loaded {len(tasks)} benchmark tasks")
                    run_tag = f"o{outer}-i{inner}-{benchmark}"
                    run_result = runner.run_batch(compiled_graph, tasks, run_tag=run_tag)
                    auc = run_result.success_rate
                    run_dur = time.time() - run_start
                    phase_timings["run"] = round(run_dur, 2)
                    state.cumulative_tokens += run_result.total_tokens
                    state.cumulative_api_calls += len(run_result.results)
                    state.log(f"[RUN] Complete in {run_dur:.1f}s")
                    state.log(f"[RUN]   AUC / Success Rate: {auc:.4f}")
                    state.log(f"[RUN]   Succeeded: {run_result.metrics.get('n_succeeded', 0)}/{run_result.metrics.get('n_tasks', 0)}")
                    state.log(f"[RUN]   Tokens used (this batch): {run_result.total_tokens}")
                    state.log(f"[RUN]   Cumulative tokens (all iters): {state.cumulative_tokens}")
                    state.log(f"[RUN]   Per-task results:")
                    task_details = []
                    for idx, r in enumerate(run_result.results):
                        status = "SUCCESS" if r.success else "FAIL"
                        output_preview = (r.output[:150].replace("\n", " ") if r.output else "(no output)")
                        error_info = f" | error: {r.error[:100]}" if r.error else ""
                        state.log(f"[RUN]     [{idx+1}] {status} | task={r.task_id} | tokens={r.token_count}{error_info}")
                        state.log(f"[RUN]          output: \"{output_preview}\"")
                        task_details.append({
                            "task_id": r.task_id, "success": r.success,
                            "tokens": r.token_count, "error": r.error,
                            "output_preview": output_preview,
                        })
                    # Aggregate routing stats from individual RunResults
                    batch_routing = {"routing_skips": 0, "routing_full": 0, "tasks_with_routing": 0}
                    for r in run_result.results:
                        meta = getattr(r, "metadata", {}) or {}
                        batch_routing["routing_skips"] += meta.get("routing_skips", 0)
                        batch_routing["routing_full"] += meta.get("routing_full", 0)
                        if meta.get("routing_skips", 0) + meta.get("routing_full", 0) > 0:
                            batch_routing["tasks_with_routing"] += 1
                    total_routing = batch_routing["routing_skips"] + batch_routing["routing_full"]
                    batch_routing["skip_rate"] = round(
                        batch_routing["routing_skips"] / total_routing, 4
                    ) if total_routing > 0 else 0.0

                    state.routing_stats.append({
                        "outer": outer, "inner": inner, "family": family_name,
                        **batch_routing,
                    })

                    state.task_results_history.append({
                        "outer": outer, "inner": inner, "auc": auc,
                        "n_tasks": len(run_result.results),
                        "n_succeeded": run_result.metrics.get("n_succeeded", 0),
                        "total_tokens": run_result.total_tokens,
                        "run_time_s": round(run_dur, 2),
                        "routing": batch_routing,
                        "tasks": task_details,
                    })
                except Exception as e:
                    run_dur = time.time() - run_start
                    phase_timings["run"] = round(run_dur, 2)
                    state.log(f"[RUN] FAILED after {run_dur:.1f}s: {e}")
                    state.log(f"[RUN] Full traceback: {traceback.format_exc()}")
                    auc = 0.0
                    run_result = BatchRunResult(run_id=f"o{outer}-i{inner}-{benchmark}-failed")
                    run_result.results = []
                    run_result.metrics = {"success_rate": 0, "n_tasks": 0, "n_succeeded": 0}
                    state.task_results_history.append({
                        "outer": outer, "inner": inner, "auc": 0.0,
                        "n_tasks": 0, "n_succeeded": 0, "total_tokens": 0,
                        "run_time_s": round(run_dur, 2), "error": str(e), "tasks": [],
                    })

                state.auc_history.append({
                    "outer": outer, "inner": inner, "auc": auc,
                    "benchmark": benchmark, "family": family_name,
                })

                # ── ANALYZE ──
                state.current_phase = f"O{outer} I{inner}: ANALYZE"
                state.log(f"[ANALYZE] Starting contrastive trace analysis...")
                analyze_start = time.time()
                try:
                    # Use trace set built by runner (local, no remote API)
                    trace_set = run_result.trace_set
                    if trace_set is None:
                        trace_set = TraceSet(run_id=run_result.run_id)

                    # Enrich traces with topology info for EC classification
                    topo_info = {
                        "family": family_name,
                        "n_roles": len(spec.roles),
                        "roles": [{"name": rl.name, "type": rl.functional_type.value} for rl in spec.roles],
                        "edges": [{"src": e.source, "tgt": e.target} for e in spec.edges],
                    }
                    for trace in trace_set.traces:
                        trace["topology_info"] = topo_info

                    state.log(f"[ANALYZE] Trace set: {len(trace_set.traces)} total, {len(trace_set.succeeded)} succeeded, {len(trace_set.failed)} failed")

                    if not trace_set.traces:
                        # No traces at all — RUN produced 0 results. Generate synthetic evidence.
                        state.log(f"[ANALYZE] No traces available (RUN produced 0 results). Skipping LLM analysis.")
                        state.log(f"[ANALYZE] Generating synthetic EC2 evidence (topology/execution failure).")
                        from abstral.config import EvidenceClass, TraceEvidence, EC_SECTION_MAP
                        run_error_msg = state.task_results_history[-1].get("error", "Unknown RUN failure") if state.task_results_history else "Unknown"
                        synthetic_ev = TraceEvidence(
                            ec_class=EvidenceClass.EC2,
                            trace_id="synthetic-no-traces",
                            failed_trace_summary=f"RUN phase produced 0 results. Error: {str(run_error_msg)[:300]}",
                            succeeded_trace_summary="",
                            reasoning=f"The agent system failed to execute any tasks. This indicates a topology or construction failure. The RUN phase error was: {str(run_error_msg)[:200]}",
                            suggested_edit="Review the R (Topology Reasoning) section. The current topology may be malformed or incompatible with the benchmark. Consider simplifying to a pipeline or single-agent topology.",
                            target_section=EC_SECTION_MAP[EvidenceClass.EC2],
                            confidence=0.9,
                        )
                        ec_dist = {"EC1": 0, "EC2": 1, "EC3": 0, "EC4": 0, "EC5": 0}

                        class _SyntheticAnalysis:
                            evidence = [synthetic_ev]
                            ec_distribution = ec_dist
                            summary = "Synthetic EC2: RUN produced 0 traces"
                        analysis = _SyntheticAnalysis()

                        analyze_dur = time.time() - analyze_start
                        phase_timings["analyze"] = round(analyze_dur, 2)
                        state.log(f"[ANALYZE] Complete (synthetic) in {analyze_dur:.1f}s — 1 synthetic EC2 evidence item")
                        state.log(f"[ANALYZE]   Reasoning: {synthetic_ev.reasoning[:250]}")
                        state.evidence_history.append({
                            "outer": outer, "inner": inner,
                            "n_evidence": 1, "ec_dist": ec_dist,
                            "analyze_time_s": round(analyze_dur, 2),
                            "evidence": [{
                                "ec_class": "EC2", "target_section": "R",
                                "confidence": 0.9,
                                "reasoning": synthetic_ev.reasoning,
                                "suggested_edit": synthetic_ev.suggested_edit,
                                "failed_trace_summary": synthetic_ev.failed_trace_summary[:200],
                                "trace_id": "synthetic-no-traces",
                            }],
                        })
                    else:
                        # Normal analysis with real traces
                        n_pairs = len(analyzer.trace_manager.pair_traces(trace_set)) if hasattr(analyzer, 'trace_manager') else 0
                        state.log(f"[ANALYZE] Contrastive pairs available: {n_pairs}")
                        if n_pairs == 0:
                            if trace_set.failed and not trace_set.succeeded:
                                state.log(f"[ANALYZE] Mode: single-trace analysis (all tasks failed)")
                            elif trace_set.succeeded and not trace_set.failed:
                                state.log(f"[ANALYZE] Mode: emergent pattern detection (all tasks succeeded)")
                            else:
                                state.log(f"[ANALYZE] Mode: mixed but no contrastive pairs formed")
                        else:
                            state.log(f"[ANALYZE] Mode: contrastive pair analysis")

                        analysis = analyzer.analyze_traces(skill_doc, trace_set)
                        ec_dist = analysis.ec_distribution
                        analyze_dur = time.time() - analyze_start
                        phase_timings["analyze"] = round(analyze_dur, 2)
                        state.log(f"[ANALYZE] Complete in {analyze_dur:.1f}s")
                        state.log(f"[ANALYZE]   Evidence items: {len(analysis.evidence)}")
                        state.log(f"[ANALYZE]   EC distribution: {ec_dist}")
                        state.log(f"[ANALYZE]   Summary: {analysis.summary}")

                        evidence_details = []
                        for idx, ev in enumerate(analysis.evidence):
                            state.log(f"[ANALYZE]   Evidence #{idx+1}:")
                            state.log(f"[ANALYZE]     EC class: {ev.ec_class.value}")
                            state.log(f"[ANALYZE]     Target section: {ev.target_section}")
                            state.log(f"[ANALYZE]     Confidence: {ev.confidence:.2f}")
                            state.log(f"[ANALYZE]     Reasoning: {ev.reasoning[:250]}")
                            state.log(f"[ANALYZE]     Suggested edit: {ev.suggested_edit[:250]}")
                            if ev.failed_trace_summary:
                                state.log(f"[ANALYZE]     Failed trace: {ev.failed_trace_summary[:150]}")
                            if ev.succeeded_trace_summary:
                                state.log(f"[ANALYZE]     Succeeded trace: {ev.succeeded_trace_summary[:150]}")
                            state.log(f"[ANALYZE]     Trace ID: {ev.trace_id}")
                            evidence_details.append({
                                "ec_class": ev.ec_class.value,
                                "target_section": ev.target_section,
                                "confidence": ev.confidence,
                                "reasoning": ev.reasoning,
                                "suggested_edit": ev.suggested_edit,
                                "failed_trace_summary": ev.failed_trace_summary[:200] if ev.failed_trace_summary else "",
                                "trace_id": ev.trace_id,
                            })
                        state.evidence_history.append({
                            "outer": outer, "inner": inner,
                            "n_evidence": len(analysis.evidence),
                            "ec_dist": ec_dist,
                            "analyze_time_s": round(analyze_dur, 2),
                            "evidence": evidence_details,
                        })
                except Exception as e:
                    analyze_dur = time.time() - analyze_start
                    phase_timings["analyze"] = round(analyze_dur, 2)
                    state.log(f"[ANALYZE] FAILED after {analyze_dur:.1f}s: {e}")
                    state.log(f"[ANALYZE] Full traceback: {traceback.format_exc()}")
                    ec_dist = {"EC1": 0, "EC2": 0, "EC3": 0, "EC4": 0, "EC5": 0}

                    class _EmptyAnalysis:
                        evidence = []
                        ec_distribution = ec_dist
                        summary = "Analysis failed"
                    analysis = _EmptyAnalysis()
                    state.evidence_history.append({
                        "outer": outer, "inner": inner,
                        "n_evidence": 0, "ec_dist": ec_dist,
                        "analyze_time_s": round(analyze_dur, 2),
                        "evidence": [], "error": str(e),
                    })

                state.ec_history.append({"outer": outer, "inner": inner, "ec_dist": ec_dist})

                # ── UPDATE ──
                state.current_phase = f"O{outer} I{inner}: UPDATE"
                state.log(f"[UPDATE] Applying {len(analysis.evidence)} evidence items to SKILL.md...")
                update_start = time.time()
                previous_doc = SkillDocument.parse(skill_doc.render())  # deep copy
                prev_rules = previous_doc.rule_count()
                prev_words = previous_doc.word_count()
                try:
                    update_result = updater.apply_updates(
                        skill_doc=skill_doc,
                        evidence=analysis.evidence,
                        repo=repo,
                        iteration=inner,
                    )
                    skill_doc = repo.read()
                    diff = skill_doc.diff_lines(previous_doc)
                    update_dur = time.time() - update_start
                    phase_timings["update"] = round(update_dur, 2)
                    new_rules = skill_doc.rule_count()
                    new_words = skill_doc.word_count()
                    state.log(f"[UPDATE] Complete in {update_dur:.1f}s")
                    state.log(f"[UPDATE]   Rules: {prev_rules} -> {new_rules} (+{update_result.rules_added})")
                    state.log(f"[UPDATE]   Words: {prev_words} -> {new_words} ({new_words - prev_words:+d})")
                    state.log(f"[UPDATE]   Diff lines: {diff}")
                    state.log(f"[UPDATE]   Sections changed: {update_result.sections_changed}")

                    # Per-section diff details
                    section_diff = {}
                    for key in ["K", "R", "T", "P"]:
                        old_sec = previous_doc.get_section(key)
                        new_sec = skill_doc.get_section(key)
                        old_len = len(old_sec.split())
                        new_len = len(new_sec.split())
                        changed = old_sec != new_sec
                        section_diff[key] = {"old_words": old_len, "new_words": new_len, "changed": changed}
                        if changed:
                            state.log(f"[UPDATE]   Section {key}: {old_len} -> {new_len} words (CHANGED)")
                        else:
                            state.log(f"[UPDATE]   Section {key}: {old_len} words (unchanged)")
                    state.section_diffs.append({
                        "outer": outer, "inner": inner, "sections": section_diff,
                    })
                except Exception as e:
                    update_dur = time.time() - update_start
                    phase_timings["update"] = round(update_dur, 2)
                    state.log(f"[UPDATE] FAILED after {update_dur:.1f}s: {e}")
                    state.log(f"[UPDATE] Full traceback: {traceback.format_exc()}")
                    diff = 0

                state.skill_history.append({
                    "outer": outer, "inner": inner,
                    "rules": skill_doc.rule_count(), "words": skill_doc.word_count(), "diff": diff,
                })
                state.current_skill_text = skill_doc.render()

                if auc > best_auc:
                    best_auc = auc
                    best_spec = spec

                # ── CONVERGENCE CHECK ──
                state.current_phase = f"O{outer} I{inner}: CONVERGE?"
                state.log(f"[CONVERGE] Checking convergence signals...")
                conv = convergence_detector.check(
                    iteration=inner,
                    current_doc=skill_doc,
                    previous_doc=previous_doc,
                    auc=auc,
                    ec_distribution=ec_dist,
                )
                signals_summary = {s.signal_id: s.fired for s in conv.signals}
                state.log(f"[CONVERGE] Signal results:")
                for s in conv.signals:
                    status = "FIRED" if s.fired else "inactive"
                    state.log(f"[CONVERGE]   {s.signal_id} (w={s.weight}): {status} — {s.description if hasattr(s, 'description') else ''}")
                state.log(f"[CONVERGE] Total weight: {conv.total_weight} (skill_diff_thresh: {config.convergence.skill_diff_threshold}, auc_plateau_eps: {config.convergence.auc_plateau_epsilon})")
                state.log(f"[CONVERGE] Should terminate: {conv.should_terminate}")
                state.log(f"[CONVERGE] Should compact: {conv.should_compact}")
                if conv.reason:
                    state.log(f"[CONVERGE] Reason: {conv.reason}")

                # Iteration timing summary
                iter_dur = time.time() - iter_start
                phase_timings["total"] = round(iter_dur, 2)
                state.timing_history.append({
                    "outer": outer, "inner": inner, "phases": phase_timings,
                })
                state.log(f"[TIMING] Iteration O{outer}/I{inner} total: {iter_dur:.1f}s | phases: {phase_timings}")

                state.convergence_history.append({
                    "outer": outer, "inner": inner,
                    "signals": signals_summary,
                    "weight": conv.total_weight,
                    "terminated": conv.should_terminate,
                    "reason": conv.reason,
                })

                # Handle compaction — two triggers per §3.3.1:
                # 1. C4 trigger: complexity penalty (conv.should_compact)
                # 2. Scheduled pass every k iterations (consolidation_interval)
                consolidation_interval = config.convergence.consolidation_interval
                scheduled_consolidation = (
                    inner > 0
                    and inner % consolidation_interval == 0
                    and not conv.should_terminate
                )

                if (conv.should_compact or scheduled_consolidation) and not conv.should_terminate:
                    trigger = "C4 complexity" if conv.should_compact else f"scheduled (every {consolidation_interval} iters)"
                    state.log(f"[CONSOLIDATE] Triggering consolidation — {trigger} (rule count: {skill_doc.rule_count()})...")
                    try:
                        from abstral.skill.compactor import compact
                        compact_result = compact(repo, config)
                        skill_doc = repo.read()
                        ratio = compact_result.total_rules_after / max(compact_result.total_rules_before, 1)
                        target_ratio = config.convergence.consolidation_ratio_target
                        state.log(
                            f"[CONSOLIDATE] Complete — "
                            f"{compact_result.total_rules_before} -> {compact_result.total_rules_after} rules "
                            f"({compact_result.reduction_pct:.1f}% reduction, "
                            f"ratio={ratio:.2f}, target≤{target_ratio})"
                        )
                        if compact_result.reduction_pct >= 20:
                            repo.commit_update(
                                doc=skill_doc,
                                iteration=inner,
                                ec_distribution={"compaction": 1},
                                trace_ids=[],
                                message=f"iter-{inner}: consolidation ({trigger}, {compact_result.reduction_pct:.1f}% reduction)",
                            )
                            if conv.should_compact:
                                # C4 consolidation: re-evaluate convergence after compaction
                                state.log(f"[CONSOLIDATE] Re-evaluating convergence after C4 compaction...")
                                continue
                    except Exception as e:
                        state.log(f"[CONSOLIDATE] FAILED: {e}")

                if conv.should_terminate:
                    converged_at = inner
                    state.log(f"[CONVERGE] ** CONVERGED at iteration {inner}: {conv.reason}")
                    state.log(f"[CONVERGE] Best AUC this outer loop: {best_auc:.4f}")
                    break

            # ── Register landscape point ──
            state.log("")
            state.log(f"[LANDSCAPE] Registering outer loop {outer} result...")
            if best_spec is None:
                state.log(f"[LANDSCAPE] No valid spec produced in outer {outer}, skipping")
                continue

            graph = spec_to_graph(best_spec)
            graphs_collected.append(graph)
            family = classify_topology(best_spec)
            state.log(f"[LANDSCAPE] Classified topology: {family.value}")
            state.log(f"[LANDSCAPE] Best AUC: {best_auc:.4f}, converged at iter {converged_at}")
            state.log(f"[LANDSCAPE] Best spec: {len(best_spec.roles)} roles, {len(best_spec.edges)} edges")

            # Register in seeder archive
            seeder.register_converged(
                spec=best_spec,
                doc=skill_doc,
                graph=graph,
                family=family,
                metrics={"auc": best_auc},
            )
            state.log(f"[LANDSCAPE] Registered in seeder archive ({len(seeder.archive)} total)")

            # ── Appendix diagnostics ──
            # Topology gallery: full topology structure for each outer loop's best
            state.topology_gallery.append({
                "outer": outer,
                "family": family.value,
                "n_roles": len(best_spec.roles),
                "n_edges": len(best_spec.edges),
                "roles": [
                    {"name": r.name, "type": r.functional_type.value, "description": r.description}
                    for r in best_spec.roles
                ],
                "edges": [
                    {"source": e.source, "target": e.target, "condition": e.condition}
                    for e in best_spec.edges
                ],
                "entry_point": best_spec.entry_point,
                "best_auc": best_auc,
                "converged_at": converged_at,
            })
            # K section snapshot at convergence
            state.k_section_snapshots.append({
                "outer": outer,
                "k_section": skill_doc.K,
                "word_count": len(skill_doc.K.split()),
                "rule_count": skill_doc.rule_count(),
            })

            # Compute GED matrix
            n = len(graphs_collected)
            ged_matrix = [[0.0] * n for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    g = compute_ged(graphs_collected[i], graphs_collected[j])
                    ged_matrix[i][j] = g
                    ged_matrix[j][i] = g
            state.ged_matrix = ged_matrix

            # Compute UMAP projection (§6.8)
            if n >= 3:
                try:
                    import umap as umap_lib
                    import numpy as np
                    ged_arr = np.array(ged_matrix)
                    reducer = umap_lib.UMAP(
                        n_components=2,
                        metric="precomputed",
                        n_neighbors=min(6, n - 1),
                        min_dist=0.1,
                        random_state=42,
                    )
                    coords = reducer.fit_transform(ged_arr)
                    state.umap_coords = coords.tolist()
                    state.log(f"[LANDSCAPE] UMAP projection computed ({n} points → 2D)")
                except ImportError:
                    state.log("[LANDSCAPE] umap-learn not installed; skipping UMAP")
                except Exception as e:
                    state.log(f"[LANDSCAPE] UMAP failed: {e}")

            if n >= 2:
                state.log(f"[LANDSCAPE] GED matrix ({n}x{n}):")
                for i in range(n):
                    row_str = "  ".join(f"{ged_matrix[i][j]:.1f}" for j in range(n))
                    state.log(f"[LANDSCAPE]   [{row_str}]")
                mean_ged = state._mean_ged()
                state.log(f"[LANDSCAPE] Mean GED: {mean_ged:.2f} (target min: {config.outer_loop.min_ged})")

            point = {
                "outer_iter": outer,
                "family": family.value,
                "auc": best_auc,
                "convergence_iter": converged_at,
                "n_roles": len(best_spec.roles),
            }
            state.landscape_points.append(point)
            state.topology_families_seen.append(family.value)
            state.log(f"[LANDSCAPE] Families explored so far: {list(set(state.topology_families_seen))} ({len(set(state.topology_families_seen))}/6)")

        if state.status != "stopped":
            state.status = "complete"
        state.current_phase = "Complete"
        state.log("")
        state.log("=" * 80)
        state.log("PIPELINE COMPLETE — FINAL SUMMARY")
        state.log("=" * 80)
        state.log(f"End time:              {time.strftime('%Y-%m-%d %H:%M:%S')}")
        state.log(f"Topologies explored:   {len(state.landscape_points)}")
        state.log(f"Families covered:      {list(set(state.topology_families_seen))} ({len(set(state.topology_families_seen))}/6)")
        state.log(f"Total iterations:      {len(state.auc_history)}")
        state.log(f"Cumulative tokens:     {state.cumulative_tokens}")
        state.log(f"Cumulative API calls:  {state.cumulative_api_calls}")
        if state.auc_history:
            best = max(state.auc_history, key=lambda h: h["auc"])
            state.log(f"Global best AUC:       {best['auc']:.4f} (O{best['outer']}/I{best['inner']}, {best['family']})")
        if state.landscape_points:
            for p in state.landscape_points:
                state.log(f"  Outer {p['outer_iter']}: {p['family']} | AUC={p['auc']:.4f} | converged@{p['convergence_iter']} | {p['n_roles']} agents")
        state.log("=" * 80)

    except Exception as e:
        state.status = "error"
        state.last_error = traceback.format_exc()
        state.log(f"PIPELINE ERROR: {e}")
        state.log(state.last_error)

    return state
