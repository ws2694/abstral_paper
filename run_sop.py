#!/usr/bin/env python3
from __future__ import annotations

"""ABSTRAL Experiment Runner — SOPBench (bank / healthcare).

Runs the ABSTRAL pipeline on SOPBench with oracle-based evaluation.
NO synthetic data. NO LLM-as-judge. 5 boolean criteria, all must pass.

Published baselines (Li et al., 2025):
  - GPT-4o (FC): bank 58.96%, healthcare 73.39%
  - Claude-3.7-Sonnet (FC): bank 65.67%, healthcare 70.97%
  - o4-mini-high (FC): bank 76.87%, healthcare 92.74%

Usage:
  python run_sop.py --smoke                     # Quick validation (3 tasks)
  python run_sop.py --baseline                  # Single-agent baseline
  python run_sop.py --main                      # Full ABSTRAL experiment
  python run_sop.py --all                       # Baseline + main + ablations
  python run_sop.py --domain healthcare --smoke # Healthcare domain
  python run_sop.py --tables                    # Print results summary
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

# ── Load .env ──
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                key, val = key.strip(), val.strip()
                if val and not val.startswith("#"):
                    os.environ[key] = val
    print(f"Loaded .env from {env_path}")

from abstral.config import ABSTRALConfig
from abstral.pipeline import PipelineState, run_pipeline

RESULTS_DIR = Path(__file__).parent / "results" / "sop"


def save_results(state: PipelineState, experiment_name: str, output_dir: Path):
    """Save pipeline state to JSON for later analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{experiment_name}.json"

    data = {
        "experiment": experiment_name,
        "benchmark": state.benchmark,
        "status": state.status,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "auc_history": state.auc_history,
        "ec_history": state.ec_history,
        "skill_history": state.skill_history,
        "convergence_history": state.convergence_history,
        "landscape_points": state.landscape_points,
        "ged_matrix": state.ged_matrix,
        "umap_coords": state.umap_coords,
        "topology_families_seen": state.topology_families_seen,
        "spec_history": state.spec_history,
        "evidence_history": state.evidence_history,
        "task_results_history": state.task_results_history,
        "timing_history": state.timing_history,
        "cumulative_tokens": state.cumulative_tokens,
        "cumulative_api_calls": state.cumulative_api_calls,
        "seed_history": state.seed_history,
        "section_diffs": state.section_diffs,
        # Appendix diagnostics
        "routing_stats": getattr(state, "routing_stats", None),
        "k_section_snapshots": getattr(state, "k_section_snapshots", None),
        "topology_gallery": getattr(state, "topology_gallery", None),
        "log_lines": state.log_lines[-500:],
        "last_error": state.last_error,
    }

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Results saved to {out_path}")
    return out_path


def run_benchmark(benchmark: str, config: ABSTRALConfig, output_dir: Path, label: str = "main"):
    """Run full pipeline on a single benchmark with real data."""
    experiment_name = f"{label}_{benchmark}"
    log_msg = (
        f"\n{'='*80}\n"
        f"  EXPERIMENT: {experiment_name}\n"
        f"  Benchmark: {benchmark} (REAL DATA — oracle evaluation)\n"
        f"  Outer loops: {config.outer_loop.n_outer}\n"
        f"  Max inner iters: {config.inner_loop.max_iterations}\n"
        f"  Val batch size: {config.inner_loop.val_batch_size}\n"
        f"{'='*80}\n"
    )
    print(log_msg, flush=True)
    sys.stderr.write(log_msg)
    sys.stderr.flush()

    state = PipelineState()
    base_path = output_dir / "artifacts" / experiment_name

    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{experiment_name}.log"
    state._log_file = str(log_file)

    t0 = time.time()
    state = run_pipeline(config, benchmark, state, base_path=base_path)
    elapsed = time.time() - t0

    summary = f"\n[{experiment_name}] Finished in {elapsed:.0f}s — status={state.status}"
    if state.auc_history:
        best = max(state.auc_history, key=lambda h: h["auc"])
        summary += f"\n[{experiment_name}] Best AUC: {best['auc']:.4f} ({best['family']})"
    if state.last_error:
        summary += f"\n[{experiment_name}] Last error: {state.last_error[:200]}"
    print(summary, flush=True)
    sys.stderr.write(summary + "\n")
    sys.stderr.flush()

    save_results(state, experiment_name, output_dir)
    return state


# ──────────────────────────────────────────────────────────────────────────────
# SOPBench experiments
# ──────────────────────────────────────────────────────────────────────────────

def run_sop_smoke_test(output_dir: Path, domain: str = "bank"):
    """Quick SOPBench smoke test (3 tasks)."""
    print("\n" + "="*80)
    print(f"  SMOKE TEST — SOPBench {domain} (3 tasks)")
    print("="*80)

    config = ABSTRALConfig()
    config.agent_backbone.model = "gpt-4o"
    config.outer_loop.n_outer = 1
    config.inner_loop.max_iterations = 2
    config.inner_loop.val_batch_size = 3
    config.inner_loop.token_budget = 100000
    config.inner_loop.wall_clock_limit_sec = 300
    config.sop_bench.domain = domain

    return run_benchmark(f"sop_{domain}", config, output_dir, label="smoke")


def run_sop_single_agent(output_dir: Path, domain: str = "bank"):
    """SOPBench baseline: single agent, no topology search.

    Compare to published GPT-4o (FC): bank 58.96%, healthcare 73.39%.
    Uses 1 outer × 1 inner with all tasks.
    """
    print("\n" + "="*80)
    print(f"  BASELINE — SOPBench {domain} single-agent (GPT-4o)")
    print(f"  Compare to published GPT-4o (FC): bank 58.96%, healthcare 73.39%")
    print("="*80)

    n_tasks = 134 if domain == "bank" else 124
    config = ABSTRALConfig()
    config.agent_backbone.model = "gpt-4o"
    config.outer_loop.n_outer = 1
    config.inner_loop.max_iterations = 1
    config.inner_loop.val_batch_size = n_tasks
    config.inner_loop.token_budget = 500000
    config.inner_loop.wall_clock_limit_sec = 300
    config.sop_bench.domain = domain

    return run_benchmark(f"sop_{domain}", config, output_dir, label="single_agent")


def run_sop_inner_only(output_dir: Path, domain: str = "bank"):
    """SOPBench ablation: single outer loop (no topology diversity seeding)."""
    config = ABSTRALConfig()
    config.agent_backbone.model = "gpt-4o"
    config.outer_loop.n_outer = 1
    config.inner_loop.max_iterations = 8
    config.inner_loop.val_batch_size = 20
    config.inner_loop.token_budget = 200000
    config.inner_loop.wall_clock_limit_sec = 300
    config.sop_bench.domain = domain

    return run_benchmark(f"sop_{domain}", config, output_dir, label="inner_only")


def run_sop_main(output_dir: Path, domain: str = "bank"):
    """Primary SOPBench experiment: ABSTRAL full pipeline (3 outer × 8 inner).

    20 tasks per batch. Estimated cost: ~$70-100 for bank domain.
    """
    print("\n" + "="*80)
    print(f"  MAIN EXPERIMENT — SOPBench {domain} (3 outer × 8 inner, 20 tasks/batch)")
    print("="*80)

    config = ABSTRALConfig()
    config.agent_backbone.model = "gpt-4o"
    config.outer_loop.n_outer = 3
    config.inner_loop.max_iterations = 8
    config.inner_loop.val_batch_size = 20
    config.inner_loop.token_budget = 200000
    config.inner_loop.wall_clock_limit_sec = 300
    config.sop_bench.domain = domain

    return run_benchmark(f"sop_{domain}", config, output_dir, label="main")


def run_sop_fixed_topology(output_dir: Path, domain: str = "bank"):
    """SOPBench baseline: fixed 3-agent hierarchy, no search."""
    config = ABSTRALConfig()
    config.agent_backbone.model = "gpt-4o"
    config.outer_loop.n_outer = 1
    config.inner_loop.max_iterations = 1
    config.inner_loop.val_batch_size = 50
    config.inner_loop.token_budget = 500000
    config.inner_loop.wall_clock_limit_sec = 300
    config.sop_bench.domain = domain

    return run_benchmark(f"sop_{domain}", config, output_dir, label="fixed_topology")


def run_sop_test_eval(output_dir: Path, domain: str = "bank", source_experiment: str = "main"):
    """Evaluate best topology from a completed experiment on held-out test set.

    Loads the best AgentSpec + SKILL.md from the source experiment,
    rebuilds the graph via BUILD phase, and evaluates on the test split
    (94 tasks for bank, never seen during val-based inner loop).
    """
    benchmark = f"sop_{domain}"
    source_name = f"{source_experiment}_{benchmark}"
    results_path = output_dir / f"{source_name}.json"

    if not results_path.exists():
        print(f"ERROR: No results found at {results_path}")
        print(f"Run --main first, then --test to evaluate on held-out set.")
        sys.exit(1)

    with open(results_path) as f:
        data = json.load(f)

    # Find best iteration
    if not data.get("auc_history"):
        print("ERROR: No AUC history in results.")
        sys.exit(1)

    best_entry = max(data["auc_history"], key=lambda h: h["auc"])
    best_outer = best_entry["outer"]
    best_inner = best_entry["inner"]
    best_auc = best_entry["auc"]
    best_family = best_entry["family"]

    print("\n" + "="*80)
    print(f"  HELD-OUT TEST EVALUATION — SOPBench {domain}")
    print(f"  Source: {source_name} (best val AUC: {best_auc:.4f} at O{best_outer}/I{best_inner})")
    print(f"  Topology: {best_family}")
    print(f"  Test set: {134-40 if domain == 'bank' else 124-40} tasks (never seen during training)")
    print("="*80)

    # Load SKILL.md from best outer loop
    artifact_dir = output_dir / "artifacts" / source_name / f"outer_{best_outer}"
    skill_path = artifact_dir / "SKILL.md"
    if not skill_path.exists():
        print(f"ERROR: SKILL.md not found at {skill_path}")
        sys.exit(1)

    from abstral.skill.document import SkillDocument
    from abstral.skill.versioning import SkillRepository
    skill_repo = SkillRepository(artifact_dir)
    skill_doc = skill_repo.read()

    # Checkout the SKILL.md at the best inner iteration
    import subprocess
    try:
        # Git log to find the right commit
        result = subprocess.run(
            ["git", "log", "--oneline", f"--grep=iter-{best_inner}:"],
            capture_output=True, text=True, cwd=str(artifact_dir)
        )
        commits = result.stdout.strip().split("\n")
        if commits and commits[0]:
            best_commit = commits[0].split()[0]
            subprocess.run(["git", "checkout", best_commit, "--", "SKILL.md"],
                          cwd=str(artifact_dir), capture_output=True)
            skill_doc = skill_repo.read()
            print(f"  Loaded SKILL.md from commit {best_commit} (iter-{best_inner})")
            # Restore HEAD
            subprocess.run(["git", "checkout", "HEAD", "--", "SKILL.md"],
                          cwd=str(artifact_dir), capture_output=True)
        else:
            print(f"  WARNING: Could not find commit for iter-{best_inner}, using latest SKILL.md")
    except Exception as e:
        print(f"  WARNING: Git checkout failed ({e}), using latest SKILL.md")

    print(f"  SKILL.md: {skill_doc.word_count()} words, {skill_doc.rule_count()} rules")

    # Determine turn limit — check if this is a "fair" run
    fair_mode = source_experiment.startswith("fair_")

    # Configure for test evaluation
    config = ABSTRALConfig()
    config.agent_backbone.model = "gpt-4o"
    config.outer_loop.n_outer = 1
    config.inner_loop.max_iterations = 1
    config.inner_loop.val_batch_size = 200  # All test tasks
    config.inner_loop.token_budget = 500000
    config.inner_loop.wall_clock_limit_sec = 300
    config.sop_bench.domain = domain
    if fair_mode:
        config.sop_bench.max_turns = 60

    # Run BUILD phase to reconstruct graph from SKILL.md
    from abstral.layer1.builder import AgentBuilder
    builder = AgentBuilder(config)

    # Get task description for BUILD
    if domain == "bank":
        task_desc = (
            "Design a multi-agent system for banking customer service. "
            "Agents must handle authentication, account operations, transactions, "
            "and credit/loan services following strict banking compliance rules."
        )
    else:
        task_desc = (
            "Design a multi-agent system for healthcare service operations."
        )

    print(f"  Running BUILD phase to reconstruct graph...")
    spec = builder.design_agent_spec(skill_doc, task_desc)
    print(f"  Built: {spec.topology_family.value} with {len(spec.roles)} roles, {len(spec.edges)} edges")

    # Build and compile graph
    from abstral.sop_adapter import SOPEnvManager, wrap_sop_tools
    sop_manager = SOPEnvManager(domain=domain)
    _ref_sys, _ref_user, _ref_asst, _ref_info, _ = sop_manager.create_env(0)
    sop_tool_provider = lambda: wrap_sop_tools(_ref_sys, _ref_asst)

    compiled, meta = builder.build_graph(
        spec, task_desc,
        tool_provider=sop_tool_provider,
        benchmark_mode="tau",
        domain_knowledge=skill_doc.K,
    )
    print(f"  Graph compiled: {meta.get('n_roles', '?')} nodes")

    # Load TEST split
    from abstral.layer1.runner import SOPBenchRunner, load_benchmark_tasks
    runner = SOPBenchRunner(config)
    test_tasks = load_benchmark_tasks(
        benchmark=benchmark,
        split="test",
        n_instances=200,  # All test tasks
    )
    print(f"  Loaded {len(test_tasks)} held-out test tasks")

    # Run evaluation
    t0 = time.time()
    run_result = runner.run_batch(compiled, test_tasks, run_tag=f"test_{source_name}")
    elapsed = time.time() - t0

    test_auc = run_result.success_rate
    n_passed = run_result.metrics.get("n_succeeded", 0)
    n_total = run_result.metrics.get("n_tasks", len(test_tasks))

    print(f"\n  {'='*60}")
    print(f"  TEST SET RESULTS")
    print(f"  {'='*60}")
    print(f"  Val AUC (best iteration):  {best_auc:.4f} ({best_family}, O{best_outer}/I{best_inner})")
    print(f"  Test AUC (held-out):       {test_auc:.4f}")
    print(f"  Passed: {n_passed}/{n_total}")
    print(f"  Time: {elapsed:.0f}s")
    print(f"  Tokens: {run_result.total_tokens}")
    if fair_mode:
        print(f"  Turn limit: 60 (fair multi-agent budget)")
    else:
        print(f"  Turn limit: 20 (matched published protocol)")

    # Per-task breakdown
    should_true_pass = 0
    should_true_total = 0
    should_false_pass = 0
    should_false_total = 0
    for r in run_result.results:
        ss = r.metadata.get("action_should_succeed", True)
        if ss:
            should_true_total += 1
            if r.success:
                should_true_pass += 1
        else:
            should_false_total += 1
            if r.success:
                should_false_pass += 1

    print(f"\n  should_succeed=True:  {should_true_pass}/{should_true_total} ({should_true_pass/max(should_true_total,1):.0%})")
    print(f"  should_succeed=False: {should_false_pass}/{should_false_total} ({should_false_pass/max(should_false_total,1):.0%})")

    # Save test results
    test_data = {
        "experiment": f"test_{source_name}",
        "source_experiment": source_name,
        "best_val_auc": best_auc,
        "best_outer": best_outer,
        "best_inner": best_inner,
        "best_family": best_family,
        "test_auc": test_auc,
        "n_passed": n_passed,
        "n_total": n_total,
        "should_true_pass": should_true_pass,
        "should_true_total": should_true_total,
        "should_false_pass": should_false_pass,
        "should_false_total": should_false_total,
        "elapsed_s": elapsed,
        "tokens": run_result.total_tokens,
        "fair_mode": fair_mode,
        "turn_limit": 60 if fair_mode else 20,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "per_task": [
            {
                "task_id": r.task_id,
                "passed": r.success,
                "turns": r.metadata.get("turns_used", 0),
                "tools": r.metadata.get("tools_called", 0),
                "goal": r.metadata.get("user_goal", ""),
                "should_succeed": r.metadata.get("action_should_succeed", True),
            }
            for r in run_result.results
        ],
    }
    test_path = output_dir / f"test_{source_name}.json"
    with open(test_path, "w") as f:
        json.dump(test_data, f, indent=2, default=str)
    print(f"\n  Results saved to {test_path}")


def run_sop_fair_main(output_dir: Path, domain: str = "bank"):
    """ABSTRAL with fair multi-agent turn budget (max_turns=60).

    Same tool call limit (10) as published baseline, but increased turn
    budget to compensate for multi-agent routing overhead.
    Single-agent baselines use ~100% turn efficiency (every turn = tool call).
    Multi-agent systems average ~26% efficiency, so 60 turns ≈ 15 effective actions.
    """
    print("\n" + "="*80)
    print(f"  FAIR MAIN EXPERIMENT — SOPBench {domain} (3 outer × 8 inner, max_turns=60)")
    print(f"  Same tool budget (10 calls), fair turn budget for multi-agent routing")
    print("="*80)

    config = ABSTRALConfig()
    config.agent_backbone.model = "gpt-4o"
    config.outer_loop.n_outer = 3
    config.inner_loop.max_iterations = 8
    config.inner_loop.val_batch_size = 20
    config.inner_loop.token_budget = 200000
    config.inner_loop.wall_clock_limit_sec = 300
    config.sop_bench.domain = domain
    config.sop_bench.max_turns = 60  # Fair budget for multi-agent

    return run_benchmark(f"sop_{domain}", config, output_dir, label="fair_main")


def format_sop_tables(output_dir: Path, domain: str = "bank"):
    """Print SOPBench results summary."""
    print("\n" + "="*80)
    print(f"  SOPBench {domain.upper()} RESULTS")
    print("="*80)

    experiments = [
        (f"main_sop_{domain}", "ABSTRAL (full, 20 turns)"),
        (f"fair_main_sop_{domain}", "ABSTRAL (full, 60 turns)"),
        (f"inner_only_sop_{domain}", "ABSTRAL (inner-only)"),
        (f"single_agent_sop_{domain}", "Single Agent (GPT-4o)"),
        (f"fixed_topology_sop_{domain}", "Fixed Topology"),
        (f"smoke_sop_{domain}", "Smoke Test"),
    ]

    print(f"\n{'Method':<30} {'Best AUC':>10} {'Family':>18} {'Iters':>8} {'Tokens':>10}")
    print("-" * 80)

    for filename, label in experiments:
        path = output_dir / f"{filename}.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        if data.get("auc_history"):
            best = max(data["auc_history"], key=lambda h: h["auc"])
            n_iters = len(data["auc_history"])
            tokens = data.get("cumulative_tokens", 0)
            print(f"{label:<30} {best['auc']:>10.4f} {best['family']:>18} {n_iters:>8} {tokens:>10}")
        else:
            print(f"{label:<30} {'(no data)':>10}")

    # Test set results
    test_experiments = [
        (f"test_main_sop_{domain}", "ABSTRAL (test, 20 turns)"),
        (f"test_fair_main_sop_{domain}", "ABSTRAL (test, 60 turns)"),
    ]
    has_test = False
    for filename, label in test_experiments:
        path = output_dir / f"{filename}.json"
        if path.exists():
            if not has_test:
                print(f"\n{'Held-out Test Results':<30} {'Test AUC':>10} {'Val AUC':>10} {'Tasks':>8}")
                print("-" * 62)
                has_test = True
            with open(path) as f:
                td = json.load(f)
            print(f"{label:<30} {td['test_auc']:>10.4f} {td['best_val_auc']:>10.4f} {td['n_total']:>8}")

    # Published baselines
    print(f"\n  Published Baselines (Li et al., 2025, FC mode):")
    if domain == "bank":
        baselines = [
            ("o4-mini-high", 76.87),
            ("GPT-5", 71.64),
            ("Claude-3.5-Sonnet", 71.90),
            ("GPT-4.1", 69.40),
            ("Claude-3.7-Sonnet", 65.67),
            ("GPT-4o", 58.96),
            ("GPT-4o-mini", 33.58),
        ]
    else:
        baselines = [
            ("o4-mini-high", 92.74),
            ("GPT-5-mini", 92.74),
            ("Gemini-2.5-Flash", 87.90),
            ("GPT-4.1", 79.03),
            ("GPT-4o", 73.39),
            ("Claude-3.7-Sonnet", 70.97),
        ]
    for model, score in baselines:
        marker = " <-- our backbone" if model == "GPT-4o" else ""
        print(f"  {model:<28} {score:>6.2f}%{marker}")


def main():
    parser = argparse.ArgumentParser(description="ABSTRAL SOPBench Experiment Runner")
    parser.add_argument("--domain", type=str, default="bank",
                        choices=["bank", "healthcare"],
                        help="SOPBench domain (default: bank)")
    parser.add_argument("--smoke", action="store_true", help="Quick validation (3 tasks)")
    parser.add_argument("--baseline", action="store_true", help="Single-agent baseline")
    parser.add_argument("--main", action="store_true", help="Full ABSTRAL experiment")
    parser.add_argument("--ablation", action="store_true", help="Inner-only + fixed topology ablations")
    parser.add_argument("--all", action="store_true", help="Run everything: baseline + main + ablations")
    parser.add_argument("--fair", action="store_true", help="Fair turn budget experiment (max_turns=60)")
    parser.add_argument("--test", action="store_true", help="Evaluate best topology on held-out test set")
    parser.add_argument("--test-source", type=str, default="main",
                        help="Source experiment for --test (default: main, use fair_main for fair)")
    parser.add_argument("--tables", action="store_true", help="Print results summary")
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate API keys
    required_keys = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"]
    missing = [k for k in required_keys if not os.environ.get(k)]
    if missing:
        print(f"ERROR: Missing keys: {', '.join(missing)}")
        print("Add them to .env file.")
        sys.exit(1)

    # Configure logging
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(log_dir / "sop_runner.log"),
        ],
    )

    print(f"Output directory: {output_dir}", flush=True)
    print(f"Domain: {args.domain}", flush=True)

    if args.tables:
        format_sop_tables(output_dir, domain=args.domain)
        return

    if args.smoke:
        run_sop_smoke_test(output_dir, domain=args.domain)
        return

    if args.baseline:
        run_sop_single_agent(output_dir, domain=args.domain)
        return

    if args.main:
        run_sop_main(output_dir, domain=args.domain)
        return

    if args.fair:
        run_sop_fair_main(output_dir, domain=args.domain)
        return

    if args.test:
        run_sop_test_eval(output_dir, domain=args.domain, source_experiment=args.test_source)
        return

    if args.ablation:
        run_sop_inner_only(output_dir, domain=args.domain)
        run_sop_fixed_topology(output_dir, domain=args.domain)
        return

    if args.all:
        print("\n" + "="*80)
        print(f"  ABSTRAL FULL EXPERIMENT SUITE (SOPBench {args.domain})")
        print(f"  Oracle-based evaluation, 5 boolean criteria")
        print("="*80)
        run_sop_single_agent(output_dir, domain=args.domain)
        run_sop_main(output_dir, domain=args.domain)
        run_sop_inner_only(output_dir, domain=args.domain)
        run_sop_fixed_topology(output_dir, domain=args.domain)
        format_sop_tables(output_dir, domain=args.domain)
        return

    # Default: show help
    parser.print_help()


if __name__ == "__main__":
    main()
