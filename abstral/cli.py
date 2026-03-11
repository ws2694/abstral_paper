"""CLI entry points for ABSTRAL.

Commands:
  abstral init        — Interactive setup, generates configs/run_config.yaml
  abstral run         — Run the full pipeline
  abstral run-inner   — Run a single inner loop
  abstral monitor     — Monitor convergence of a running experiment
  abstral landscape   — Display the design landscape summary
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def cli(verbose: bool) -> None:
    """ABSTRAL: Automated Agent System Design via Skill-Referenced Three-Layer Adaptive Loop."""
    _setup_logging(verbose)


@cli.command()
@click.option("--output", "-o", default="configs/run_config.yaml", help="Output config path")
def init(output: str) -> None:
    """Interactive setup — generates a run configuration."""
    from abstral.config import ABSTRALConfig, BenchmarkConfig

    console.print(Panel.fit(
        "[bold]ABSTRAL[/bold] — Configuration Setup",
        style="cyan",
    ))

    # Benchmark selection
    benchmarks_available = ["alfworld", "webarena", "gaia", "tabular"]
    console.print("\n[bold]Available benchmarks:[/bold]")
    for i, b in enumerate(benchmarks_available, 1):
        console.print(f"  {i}. {b}")
    console.print(f"  {len(benchmarks_available) + 1}. all")

    selection = click.prompt(
        "Select benchmarks (comma-separated numbers or 'all')",
        default="all",
    )

    if selection.strip().lower() == "all" or selection.strip() == str(len(benchmarks_available) + 1):
        selected = benchmarks_available
    else:
        indices = [int(x.strip()) - 1 for x in selection.split(",")]
        selected = [benchmarks_available[i] for i in indices if 0 <= i < len(benchmarks_available)]

    # Agent backbone
    backbone = click.prompt(
        "Agent backbone model",
        default="gpt-4o-mini",
        type=click.Choice(["gpt-4o-mini", "gpt-4o", "gemini-flash", "llama3"]),
    )

    model_map = {
        "gpt-4o-mini": ("gpt-4o-mini", "openai"),
        "gpt-4o": ("gpt-4o", "openai"),
        "gemini-flash": ("gemini-2.0-flash", "google"),
        "llama3": ("llama-3.1-70b", "groq"),
    }

    # Outer loop params
    n_outer = click.prompt("Number of outer loop iterations", default=6, type=int)
    n_iter_max = click.prompt("Max inner loop iterations", default=15, type=int)

    # Build config
    model_name, provider = model_map[backbone]
    benchmark_configs = [
        BenchmarkConfig(
            name=b,
            metric="task_completion_rate" if b == "alfworld" else
                   "task_success_rate" if b == "webarena" else
                   "accuracy" if b == "gaia" else "auroc",
        )
        for b in selected
    ]

    config = ABSTRALConfig(
        benchmarks=benchmark_configs,
        inner_loop={"max_iterations": n_iter_max},
        outer_loop={"n_outer": n_outer},
        agent_backbone={"model": model_name, "provider": provider},
    )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config.to_yaml(output_path)

    console.print(f"\n[green]Configuration saved to {output_path}[/green]")
    console.print(f"  Benchmarks: {', '.join(selected)}")
    console.print(f"  Backbone: {model_name} ({provider})")
    console.print(f"  Outer loops: {n_outer}, Inner max: {n_iter_max}")
    console.print(f"\nRun with: [bold]abstral run --config {output_path}[/bold]")


@cli.command()
@click.option("--config", "-c", default="configs/run_config.yaml", help="Config file path")
@click.option("--benchmark", "-b", default=None, help="Single benchmark to run (overrides config)")
@click.option("--dashboard/--no-dashboard", default=True, help="Launch Streamlit dashboard")
def run(config: str, benchmark: str | None, dashboard: bool) -> None:
    """Run the full ABSTRAL pipeline."""
    from abstral.config import ABSTRALConfig
    from abstral.orchestrator import run_full_pipeline

    console.print(Panel.fit(
        "[bold]ABSTRAL[/bold] — Full Pipeline Run",
        style="cyan",
    ))

    cfg = ABSTRALConfig.from_yaml(config)
    benchmarks = [benchmark] if benchmark else None

    if dashboard:
        _launch_dashboard_background()

    landscapes = run_full_pipeline(cfg, benchmarks=benchmarks)

    # Print summary
    for bench_name, landscape in landscapes.items():
        summary = landscape.to_summary()
        table = Table(title=f"Landscape: {bench_name}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Points", str(summary["n_points"]))
        table.add_row("Families", str(summary["n_families"]))
        table.add_row("Mean GED", f"{summary['mean_ged']:.2f}")
        table.add_row("Best AUC", f"{summary['global_optimum_auc']:.4f}")
        table.add_row("Best Family", str(summary["global_optimum_family"]))
        table.add_row("Outer Loop Value", f"{summary['outer_loop_value']:.4f}")
        console.print(table)


@cli.command("run-inner")
@click.option("--config", "-c", default="configs/run_config.yaml")
@click.option("--benchmark", "-b", required=True)
@click.option("--seed", "-s", default=0, type=int)
@click.option("--n-iter", default=15, type=int)
@click.option("--topology", "-t", default="hierarchical")
@click.option("--dashboard/--no-dashboard", default=True)
def run_inner(config: str, benchmark: str, seed: int, n_iter: int, topology: str, dashboard: bool) -> None:
    """Run a single inner loop from a seed."""
    from abstral.config import ABSTRALConfig
    from abstral.orchestrator import inner_loop
    from abstral.skill.document import SkillDocument
    from abstral.skill.versioning import SkillRepository
    from abstral.tracking import ExperimentTracker

    console.print(Panel.fit(
        f"[bold]ABSTRAL[/bold] — Inner Loop: {benchmark}",
        style="orange1",
    ))

    cfg = ABSTRALConfig.from_yaml(config)
    cfg.inner_loop.max_iterations = n_iter

    seed_doc = SkillDocument.create_seed(benchmark, topology)
    repo_path = Path(cfg.paths.artifacts) / benchmark / f"inner_seed{seed}"
    repo = SkillRepository(repo_path)
    repo.init(seed_doc)

    tracker = ExperimentTracker(cfg)
    tracker.setup()

    if dashboard:
        _launch_dashboard_background()

    result = inner_loop(
        config=cfg,
        seed_doc=seed_doc,
        repo=repo,
        benchmark=benchmark,
        outer_iter=0,
        tracker=tracker,
    )

    console.print(f"\n[green]Inner loop complete.[/green]")
    console.print(f"  Converged at iteration: {result['convergence_iter']}")
    console.print(f"  Best AUC: {result['auc']:.4f}")


@cli.command("monitor-convergence")
@click.option("--run-id", required=True, help="MLflow run ID to monitor")
@click.option("--config", "-c", default="configs/run_config.yaml")
def monitor_convergence(run_id: str, config: str) -> None:
    """Monitor convergence signals for a running experiment."""
    from abstral.config import ABSTRALConfig
    from abstral.tracking import ExperimentTracker

    cfg = ABSTRALConfig.from_yaml(config)
    tracker = ExperimentTracker(cfg)
    tracker.setup()

    signals = ["C1", "C2", "C3", "C4"]
    table = Table(title=f"Convergence Signals — Run {run_id}")
    table.add_column("Signal", style="cyan")
    table.add_column("Latest Value", style="yellow")
    table.add_column("Status", style="green")

    for signal in signals:
        try:
            history = tracker.get_metric_history(run_id, f"convergence/{signal}")
            if history:
                latest = history[-1]
                status = "[green]FIRED" if latest["value"] > 0.5 else "[dim]inactive"
                table.add_row(signal, f"{latest['value']:.2f}", status)
            else:
                table.add_row(signal, "—", "[dim]no data")
        except Exception:
            table.add_row(signal, "—", "[dim]error")

    console.print(table)


@cli.command()
@click.option("--path", "-p", default="artifacts", help="Artifacts directory")
def landscape(path: str) -> None:
    """Display the design landscape summary."""
    import json

    artifacts_dir = Path(path)
    landscape_files = list(artifacts_dir.rglob("landscape.json"))

    if not landscape_files:
        console.print("[yellow]No landscape files found.[/yellow]")
        return

    for lf in landscape_files:
        data = json.loads(lf.read_text())
        table = Table(title=f"Landscape: {data.get('benchmark', lf.parent.name)}")
        table.add_column("Outer Iter", style="cyan")
        table.add_column("Family", style="magenta")
        table.add_column("AUC", style="green")
        table.add_column("Conv. Iter", style="yellow")
        table.add_column("Cost", style="red")

        for point in data.get("points", []):
            is_best = point["auc"] == data.get("global_optimum_auc")
            style = "bold green" if is_best else ""
            table.add_row(
                str(point["outer_iter"]),
                point["family"],
                f"{point['auc']:.4f}",
                str(point["convergence_iter"]),
                f"${point.get('cost', 0):.2f}",
                style=style,
            )

        console.print(table)
        console.print(f"  Mean GED: {data.get('mean_ged', 0):.2f}")
        console.print(f"  Families: {data.get('n_families', 0)}/6")
        console.print(f"  Outer Loop Value: {data.get('outer_loop_value', 0):.4f}")


def _launch_dashboard_background() -> None:
    """Launch the Streamlit dashboard in the background."""
    import subprocess
    import sys

    dashboard_path = Path(__file__).parent.parent / "dashboard" / "app.py"
    if dashboard_path.exists():
        console.print(f"[dim]Launching dashboard at localhost:8501...[/dim]")
        subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", str(dashboard_path),
             "--server.headless", "true"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        console.print(f"[dim]Dashboard not found at {dashboard_path}[/dim]")


if __name__ == "__main__":
    cli()
