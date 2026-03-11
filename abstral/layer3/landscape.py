"""Design landscape mapping — the primary scientific output of the outer loop.

After N outer-loop iterations, ABSTRAL produces N locally-optimal (AS*, ABS*)
pairs. This module computes the landscape map: a structured record of which
architectures work in which conditions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from scipy.spatial import distance

from abstral.config import AgentSpec, TopologyFamily
from abstral.layer3.topology import (
    compute_ged,
    compute_pairwise_ged,
    classify_topology,
    spec_to_graph,
)
from abstral.skill.document import SkillDocument

logger = logging.getLogger(__name__)


@dataclass
class LandscapePoint:
    """A single point in the agent design landscape."""
    outer_iteration: int
    topology_family: TopologyFamily
    spec: AgentSpec
    skill_doc: SkillDocument
    graph: nx.DiGraph
    metrics: dict[str, float] = field(default_factory=dict)
    convergence_iter: int = 0
    total_cost: float = 0.0


@dataclass
class DesignLandscape:
    """The complete design landscape from all outer-loop iterations."""
    points: list[LandscapePoint] = field(default_factory=list)
    ged_matrix: np.ndarray | None = None
    mds_coords: np.ndarray | None = None
    umap_coords: np.ndarray | None = None
    global_optimum_idx: int = -1
    benchmark: str = ""

    def add_point(self, point: LandscapePoint) -> None:
        self.points.append(point)
        # Recompute GED matrix
        if len(self.points) > 1:
            graphs = [p.graph for p in self.points]
            self.ged_matrix = compute_pairwise_ged(graphs)
            self._compute_mds()
            self._compute_umap()
        self._update_global_optimum()

    def _update_global_optimum(self) -> None:
        """Find the globally best-performing topology."""
        if not self.points:
            return
        best_idx = 0
        best_auc = -float("inf")
        for i, p in enumerate(self.points):
            auc = p.metrics.get("auc", p.metrics.get("success_rate", 0))
            if auc > best_auc:
                best_auc = auc
                best_idx = i
        self.global_optimum_idx = best_idx

    def _compute_mds(self) -> None:
        """Compute 2D MDS projection of the GED matrix."""
        if self.ged_matrix is None or len(self.points) < 2:
            return

        try:
            from sklearn.manifold import MDS
            mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
            self.mds_coords = mds.fit_transform(self.ged_matrix)
        except ImportError:
            # Fallback: simple eigenvalue decomposition
            n = len(self.points)
            D = self.ged_matrix
            H = np.eye(n) - np.ones((n, n)) / n
            B = -0.5 * H @ (D ** 2) @ H
            eigenvalues, eigenvectors = np.linalg.eigh(B)
            idx = np.argsort(eigenvalues)[::-1][:2]
            self.mds_coords = eigenvectors[:, idx] * np.sqrt(np.maximum(eigenvalues[idx], 0))

    def _compute_umap(self) -> None:
        """Compute 2D UMAP projection of the GED matrix (§6.8).

        Uses n_neighbors=6, min_dist=0.1 as specified in the paper to preserve
        both global structure (cluster separation) and local structure.
        """
        if self.ged_matrix is None or len(self.points) < 3:
            return

        try:
            import umap
            reducer = umap.UMAP(
                n_components=2,
                metric="precomputed",
                n_neighbors=min(6, len(self.points) - 1),
                min_dist=0.1,
                random_state=42,
            )
            self.umap_coords = reducer.fit_transform(self.ged_matrix)
        except ImportError:
            logger.warning("umap-learn not installed; skipping UMAP projection")
        except Exception as e:
            logger.warning(f"UMAP projection failed: {e}; falling back to MDS coords")

    @property
    def global_optimum(self) -> LandscapePoint | None:
        if self.global_optimum_idx < 0 or self.global_optimum_idx >= len(self.points):
            return None
        return self.points[self.global_optimum_idx]

    @property
    def mean_ged(self) -> float:
        """Mean pairwise GED across all points."""
        if self.ged_matrix is None or len(self.points) < 2:
            return 0.0
        n = len(self.points)
        upper_triangle = self.ged_matrix[np.triu_indices(n, k=1)]
        return float(np.mean(upper_triangle))

    @property
    def family_coverage(self) -> dict[str, int]:
        """Count of each topology family in the landscape."""
        counts: dict[str, int] = {}
        for p in self.points:
            family = p.topology_family.value
            counts[family] = counts.get(family, 0) + 1
        return counts

    @property
    def n_families(self) -> int:
        """Number of distinct topology families represented."""
        return len(self.family_coverage)

    def outer_loop_value(self) -> float:
        """Performance delta: global optimum vs. first inner-loop run."""
        if len(self.points) < 2:
            return 0.0
        first_auc = self.points[0].metrics.get("auc", 0)
        best_auc = self.global_optimum.metrics.get("auc", 0) if self.global_optimum else 0
        return best_auc - first_auc

    def to_summary(self) -> dict[str, Any]:
        """Generate a summary dict for logging/reporting."""
        return {
            "n_points": len(self.points),
            "benchmark": self.benchmark,
            "global_optimum_family": (
                self.global_optimum.topology_family.value
                if self.global_optimum else None
            ),
            "global_optimum_auc": (
                self.global_optimum.metrics.get("auc", 0)
                if self.global_optimum else None
            ),
            "mean_ged": self.mean_ged,
            "n_families": self.n_families,
            "family_coverage": self.family_coverage,
            "outer_loop_value": self.outer_loop_value(),
            "points": [
                {
                    "outer_iter": p.outer_iteration,
                    "family": p.topology_family.value,
                    "auc": p.metrics.get("auc", 0),
                    "convergence_iter": p.convergence_iter,
                    "cost": p.total_cost,
                }
                for p in self.points
            ],
        }

    def save(self, path: str | Path) -> None:
        """Save the landscape summary to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        summary = self.to_summary()
        if self.ged_matrix is not None:
            summary["ged_matrix"] = self.ged_matrix.tolist()
        if self.mds_coords is not None:
            summary["mds_coords"] = self.mds_coords.tolist()
        if self.umap_coords is not None:
            summary["umap_coords"] = self.umap_coords.tolist()
        path.write_text(json.dumps(summary, indent=2, default=str))
        logger.info(f"Landscape saved to {path}")
