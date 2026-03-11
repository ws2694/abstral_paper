"""MLflow experiment tracking for ABSTRAL."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlflow

from abstral.config import ABSTRALConfig

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Wraps MLflow for ABSTRAL experiment tracking."""

    def __init__(self, config: ABSTRALConfig):
        self.config = config
        self._experiment_id: str | None = None

    def setup(self) -> str:
        """Initialize MLflow tracking."""
        mlflow.set_tracking_uri(self.config.tracking.mlflow_tracking_uri)
        experiment = mlflow.set_experiment(self.config.tracking.experiment_name)
        self._experiment_id = experiment.experiment_id
        logger.info(f"MLflow experiment: {experiment.name} (id={experiment.experiment_id})")
        return experiment.experiment_id

    def start_outer_run(self, outer_iter: int, benchmark: str) -> str:
        """Start an outer-loop MLflow run."""
        run = mlflow.start_run(
            run_name=f"outer-{outer_iter}-{benchmark}",
            tags={
                "outer_iteration": str(outer_iter),
                "benchmark": benchmark,
                "layer": "outer",
            },
        )
        return run.info.run_id

    def start_inner_run(
        self,
        outer_iter: int,
        inner_iter: int,
        benchmark: str,
        parent_run_id: str | None = None,
    ) -> str:
        """Start an inner-loop MLflow run (nested under outer)."""
        run = mlflow.start_run(
            run_name=f"inner-{outer_iter}.{inner_iter}-{benchmark}",
            nested=parent_run_id is not None,
            tags={
                "outer_iteration": str(outer_iter),
                "inner_iteration": str(inner_iter),
                "benchmark": benchmark,
                "layer": "inner",
            },
        )
        return run.info.run_id

    def log_iteration_metrics(
        self,
        iteration: int,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        """Log metrics for a single inner-loop iteration."""
        step = step or iteration
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)

    def log_convergence_signals(
        self,
        iteration: int,
        signals: dict[str, Any],
    ) -> None:
        """Log convergence signal values."""
        for signal_id, value in signals.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"convergence/{signal_id}", value, step=iteration)

    def log_ec_distribution(
        self,
        iteration: int,
        ec_dist: dict[str, int],
    ) -> None:
        """Log evidence class distribution."""
        total = sum(ec_dist.values()) or 1
        for ec, count in ec_dist.items():
            mlflow.log_metric(f"ec/{ec}_count", count, step=iteration)
            mlflow.log_metric(f"ec/{ec}_frac", count / total, step=iteration)

    def log_skill_metrics(
        self,
        iteration: int,
        rule_count: int,
        word_count: int,
        diff_lines: int,
    ) -> None:
        """Log skill document metrics."""
        mlflow.log_metric("skill/rule_count", rule_count, step=iteration)
        mlflow.log_metric("skill/word_count", word_count, step=iteration)
        mlflow.log_metric("skill/diff_lines", diff_lines, step=iteration)

    def log_topology_metrics(
        self,
        outer_iter: int,
        topology_family: str,
        ged_values: list[float],
    ) -> None:
        """Log topology diversity metrics for outer loop."""
        mlflow.log_metric("topology/mean_ged", sum(ged_values) / max(len(ged_values), 1))
        mlflow.set_tag(f"topology/family_{outer_iter}", topology_family)

    def log_artifact(self, local_path: str | Path, artifact_path: str = "") -> None:
        """Log a file as an MLflow artifact."""
        mlflow.log_artifact(str(local_path), artifact_path)

    def end_run(self, status: str = "FINISHED") -> None:
        """End the current MLflow run."""
        mlflow.end_run(status=status)

    def get_metric_history(self, run_id: str, metric_key: str) -> list[dict]:
        """Get the history of a metric across steps."""
        client = mlflow.tracking.MlflowClient()
        history = client.get_metric_history(run_id, metric_key)
        return [{"step": m.step, "value": m.value, "timestamp": m.timestamp} for m in history]
