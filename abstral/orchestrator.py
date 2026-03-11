"""Prefect-based orchestrator: wires all three layers into a resumable pipeline.

Pipeline structure:
  outer_loop(N=6)
    └── seeder(archive) → seed_config
          └── inner_loop(seed_config, benchmark, max_iter=15)
                └── build → run → analyze → update → check_convergence
  evaluation_harness(archive) → report
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from prefect import flow, task
from prefect.transactions import transaction

from abstral.config import ABSTRALConfig, AgentSpec, TopologyFamily
from abstral.layer1.analyzer import TraceAnalyzer
from abstral.layer1.builder import AgentBuilder
from abstral.layer1.runner import AgentRunner, BatchRunResult, load_benchmark_tasks
from abstral.layer1.updater import SkillUpdater
from abstral.layer2.convergence import ConvergenceDetector
from abstral.layer3.landscape import DesignLandscape, LandscapePoint
from abstral.layer3.seeder import DiversitySeeder
from abstral.layer3.topology import classify_topology, spec_to_graph
from abstral.skill.compactor import compact
from abstral.skill.document import SkillDocument
from abstral.skill.versioning import SkillRepository
from abstral.tracking import ExperimentTracker

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Layer 1 Tasks
# ═══════════════════════════════════════════════════════════════


@task(name="build", retries=2, retry_delay_seconds=10)
def build_agent(
    config: ABSTRALConfig,
    skill_doc: SkillDocument,
    benchmark: str,
) -> AgentSpec:
    """BUILD phase: SKILL.md → AgentSpec."""
    builder = AgentBuilder(config)
    task_desc = f"Benchmark: {benchmark}. Design an optimal multi-agent system."
    return builder.design_agent_spec(skill_doc, task_desc)


@task(name="run", retries=1)
def run_agent(
    config: ABSTRALConfig,
    spec: AgentSpec,
    benchmark: str,
    run_tag: str,
) -> BatchRunResult:
    """RUN phase: Execute agent on benchmark tasks."""
    builder = AgentBuilder(config)
    task_desc = f"Benchmark: {benchmark}"
    compiled_graph, metadata = builder.build_graph(spec, task_desc)

    runner = AgentRunner(config)
    tasks = load_benchmark_tasks(
        benchmark=benchmark,
        split="val",
        n_instances=config.inner_loop.val_batch_size,
    )

    return runner.run_batch(compiled_graph, tasks, run_tag=run_tag)


@task(name="analyze")
def analyze_traces(
    config: ABSTRALConfig,
    skill_doc: SkillDocument,
    run_result: BatchRunResult,
):
    """ANALYZE phase: Contrastive trace analysis → EC classification."""
    analyzer = TraceAnalyzer(config)
    trace_set = run_result.trace_set
    if trace_set is None:
        from abstral.tracing import TraceSet
        trace_set = TraceSet(run_id=run_result.run_id)
        # Build trace set from run results
        for r in run_result.results:
            trace = {
                "id": r.trace_id,
                "status": "success" if r.success else "error",
                "inputs": {"task_id": r.task_id},
                "outputs": {"output": r.output[:500]},
                "error": r.error,
                "tags": [],
                "token_count": r.token_count,
            }
            trace_set.traces.append(trace)
            if r.success:
                trace_set.succeeded.append(trace)
            else:
                trace_set.failed.append(trace)

    return analyzer.analyze_traces(skill_doc, trace_set)


@task(name="update")
def update_skill(
    config: ABSTRALConfig,
    skill_doc: SkillDocument,
    analysis_result,
    repo: SkillRepository,
    iteration: int,
):
    """UPDATE phase: Apply evidence-driven edits to SKILL.md."""
    updater = SkillUpdater(config)
    return updater.apply_updates(
        skill_doc=skill_doc,
        evidence=analysis_result.evidence,
        repo=repo,
        iteration=iteration,
    )


# ═══════════════════════════════════════════════════════════════
# Layer 1+2 Inner Loop Flow
# ═══════════════════════════════════════════════════════════════


@flow(name="inner_loop", log_prints=True)
def inner_loop(
    config: ABSTRALConfig,
    seed_doc: SkillDocument,
    repo: SkillRepository,
    benchmark: str,
    outer_iter: int,
    tracker: ExperimentTracker,
) -> dict[str, Any]:
    """Run the complete inner loop: BUILD → RUN → ANALYZE → UPDATE → CONVERGE?

    Returns a dict with the converged spec, doc, metrics, and iteration count.
    """
    convergence = ConvergenceDetector(config)
    convergence.reset()

    skill_doc = seed_doc
    previous_doc = None
    best_spec: AgentSpec | None = None
    best_auc = -float("inf")
    total_cost = 0.0

    parent_run_id = tracker.start_outer_run(outer_iter, benchmark)

    for iteration in range(1, config.inner_loop.max_iterations + 1):
        run_tag = f"o{outer_iter}-i{iteration}-{benchmark}"
        logger.info(f"═══ Inner loop iteration {iteration}/{config.inner_loop.max_iterations} ═══")

        inner_run_id = tracker.start_inner_run(
            outer_iter, iteration, benchmark, parent_run_id
        )

        # ── BUILD ──
        spec = build_agent(config, skill_doc, benchmark)

        # ── RUN ──
        run_result = run_agent(config, spec, benchmark, run_tag)
        auc = run_result.success_rate

        # ── ANALYZE ──
        analysis = analyze_traces(config, skill_doc, run_result)

        # ── UPDATE ──
        previous_doc = SkillDocument.parse(skill_doc.render())  # deep copy
        update_result = update_skill(
            config, skill_doc, analysis, repo, iteration
        )

        # Track metrics
        tracker.log_iteration_metrics(iteration, {
            "auc": auc,
            "success_rate": run_result.success_rate,
            "total_tokens": run_result.total_tokens,
            "mean_latency_ms": run_result.mean_latency_ms,
            "n_evidence": len(analysis.evidence),
        })
        tracker.log_ec_distribution(iteration, analysis.ec_distribution)
        tracker.log_skill_metrics(
            iteration,
            rule_count=skill_doc.rule_count(),
            word_count=skill_doc.word_count(),
            diff_lines=skill_doc.diff_lines(previous_doc) if previous_doc else 0,
        )

        # Track best
        if auc > best_auc:
            best_auc = auc
            best_spec = spec

        # Re-read skill doc from repo (in case update changed it)
        skill_doc = repo.read()

        # ── CONVERGENCE CHECK ──
        conv_result = convergence.check(
            iteration=iteration,
            current_doc=skill_doc,
            previous_doc=previous_doc,
            auc=auc,
            ec_distribution=analysis.ec_distribution,
        )

        tracker.log_convergence_signals(iteration, {
            s.signal_id: 1.0 if s.fired else 0.0 for s in conv_result.signals
        })

        tracker.end_run()

        # Handle compaction
        if conv_result.should_compact and not conv_result.should_terminate:
            logger.info("Triggering skill compaction...")
            compact_result = compact(repo, config)
            skill_doc = repo.read()

            if compact_result.reduction_pct >= 20:
                logger.info(f"Compaction effective ({compact_result.reduction_pct:.1f}%). Continuing.")
                repo.commit_update(
                    doc=skill_doc,
                    iteration=iteration,
                    ec_distribution={"compaction": 1},
                    trace_ids=[],
                    message=f"iter-{iteration}: compaction ({compact_result.reduction_pct:.1f}% reduction)",
                )
                continue

        if conv_result.should_terminate:
            logger.info(f"Inner loop converged at iteration {iteration}: {conv_result.reason}")
            break

    tracker.end_run()  # end outer run

    return {
        "spec": best_spec,
        "skill_doc": skill_doc,
        "auc": best_auc,
        "convergence_iter": iteration,
        "total_cost": total_cost,
        "benchmark": benchmark,
        "outer_iter": outer_iter,
    }


# ═══════════════════════════════════════════════════════════════
# Layer 3 Outer Loop Flow
# ═══════════════════════════════════════════════════════════════


@flow(name="outer_loop", log_prints=True)
def outer_loop(
    config: ABSTRALConfig,
    benchmark: str,
    base_repo_path: Path,
) -> DesignLandscape:
    """Run the complete outer loop with diversity seeding.

    Returns the design landscape with all locally-optimal topologies.
    """
    seeder = DiversitySeeder(config)
    tracker = ExperimentTracker(config)
    tracker.setup()
    landscape = DesignLandscape(benchmark=benchmark)

    n_outer = config.outer_loop.n_outer

    for outer_iter in range(1, n_outer + 1):
        logger.info(f"══════ Outer loop iteration {outer_iter}/{n_outer} ══════")

        # Generate seed
        if outer_iter == 1:
            seed_doc = SkillDocument.create_seed(benchmark, "hierarchical")
            seed_config = None
        else:
            seed_doc, seed_config = seeder.generate_seed(outer_iter, benchmark)

        # Initialize skill repo for this outer iteration
        repo_path = base_repo_path / f"outer_{outer_iter}"
        repo = SkillRepository(repo_path)
        repo.init(seed_doc)

        # Run inner loop
        result = inner_loop(
            config=config,
            seed_doc=seed_doc,
            repo=repo,
            benchmark=benchmark,
            outer_iter=outer_iter,
            tracker=tracker,
        )

        # Process converged result
        spec = result["spec"]
        if spec is None:
            logger.warning(f"Outer iter {outer_iter}: no valid spec produced, skipping")
            continue

        graph = spec_to_graph(spec)
        family = classify_topology(spec)

        # Register in seeder archive
        seeder.register_converged(
            spec=spec,
            doc=result["skill_doc"],
            graph=graph,
            family=family,
            metrics={"auc": result["auc"]},
        )

        # Add to landscape
        point = LandscapePoint(
            outer_iteration=outer_iter,
            topology_family=family,
            spec=spec,
            skill_doc=result["skill_doc"],
            graph=graph,
            metrics={"auc": result["auc"]},
            convergence_iter=result["convergence_iter"],
            total_cost=result["total_cost"],
        )
        landscape.add_point(point)

        # Log topology diversity
        if landscape.ged_matrix is not None and len(landscape.points) > 1:
            ged_values = landscape.ged_matrix[-1, :-1].tolist()
            tracker.log_topology_metrics(outer_iter, family.value, ged_values)

        logger.info(
            f"Outer iter {outer_iter} complete: "
            f"family={family.value}, AUC={result['auc']:.4f}, "
            f"converged at iter {result['convergence_iter']}"
        )

    # Save landscape
    landscape_path = base_repo_path / "landscape.json"
    landscape.save(landscape_path)

    # Log landscape summary
    summary = landscape.to_summary()
    logger.info(
        f"Landscape complete: {summary['n_points']} points, "
        f"{summary['n_families']} families, "
        f"mean GED={summary['mean_ged']:.2f}, "
        f"outer loop value={summary['outer_loop_value']:.4f}"
    )

    return landscape


# ═══════════════════════════════════════════════════════════════
# Full Pipeline
# ═══════════════════════════════════════════════════════════════


@flow(name="abstral_full", log_prints=True)
def run_full_pipeline(
    config: ABSTRALConfig,
    benchmarks: list[str] | None = None,
    base_path: Path | None = None,
) -> dict[str, DesignLandscape]:
    """Run the complete ABSTRAL pipeline across all benchmarks.

    Returns a dict mapping benchmark name to its design landscape.
    """
    if benchmarks is None:
        benchmarks = [b.name for b in config.benchmarks]
    if base_path is None:
        base_path = Path(config.paths.artifacts)

    base_path.mkdir(parents=True, exist_ok=True)
    landscapes: dict[str, DesignLandscape] = {}

    for benchmark in benchmarks:
        logger.info(f"{'═' * 60}")
        logger.info(f"Starting ABSTRAL for benchmark: {benchmark}")
        logger.info(f"{'═' * 60}")

        benchmark_path = base_path / benchmark
        landscape = outer_loop(config, benchmark, benchmark_path)
        landscapes[benchmark] = landscape

    return landscapes
