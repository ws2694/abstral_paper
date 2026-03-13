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


def format_sop_tables(output_dir: Path, domain: str = "bank"):
    """Print SOPBench results summary."""
    print("\n" + "="*80)
    print(f"  SOPBench {domain.upper()} RESULTS")
    print("="*80)

    experiments = [
        (f"main_sop_{domain}", "ABSTRAL (full)"),
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
