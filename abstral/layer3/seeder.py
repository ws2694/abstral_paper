"""Layer 3: Diversity Seeder.

Constructs new seed SKILL.md documents for each outer-loop iteration,
enforcing topology diversity via repulsion constraints.

Four steps per outer iteration:
1. Preserve K — carry forward domain knowledge
2. Clear T — reset template library
3. Inject repulsion into R — add topology repulsion constraint
4. Sample diverse seed — seed from least-explored family
"""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx

from abstral.config import (
    ABSTRALConfig,
    OuterSeedConfig,
    TopologyFamily,
)
from abstral.layer3.topology import (
    compute_ged,
    compute_semantic_distance,
    is_genuinely_distinct,
    spec_to_graph,
    spec_to_canonical_graph,
    template_to_graph,
    TOPOLOGY_TEMPLATES,
)
from abstral.skill.document import SkillDocument

logger = logging.getLogger(__name__)


class DiversitySeeder:
    """Generates diverse seed SKILL.md documents for outer-loop iterations."""

    def __init__(self, config: ABSTRALConfig):
        self.config = config
        self.archive: list[dict[str, Any]] = []  # converged (spec, doc, graph) from prior runs
        self._family_counts: dict[str, int] = {f.value: 0 for f in TopologyFamily}

    def register_converged(
        self,
        spec: Any,  # AgentSpec
        doc: SkillDocument,
        graph: nx.DiGraph,
        family: TopologyFamily,
        metrics: dict[str, float],
    ) -> None:
        """Register a converged inner-loop result in the archive."""
        self.archive.append({
            "spec": spec,
            "doc": doc,
            "graph": graph,
            "family": family,
            "metrics": metrics,
        })
        self._family_counts[family.value] = self._family_counts.get(family.value, 0) + 1
        logger.info(
            f"Registered converged topology: {family.value} "
            f"(archive size: {len(self.archive)})"
        )

    def generate_seed(
        self,
        outer_iteration: int,
        benchmark: str,
    ) -> tuple[SkillDocument, OuterSeedConfig]:
        """Generate a new seed SKILL.md for the next outer-loop iteration.

        Implements the four-step seeder mechanism:
        1. Preserve K from prior converged documents
        2. Clear T (agent template library)
        3. Inject repulsion constraint into R
        4. Sample from least-explored topology family
        """
        # Step 1: Preserve K — merge domain knowledge from all converged docs
        preserved_k = self._merge_domain_knowledge()

        # Step 4: Select least-explored family
        target_family = self._select_target_family()

        # Step 3: Build repulsion constraint
        repulsion_topologies = [
            entry["family"].value for entry in self.archive
        ]
        repulsion_constraint = self._build_repulsion_constraint(
            target_family, repulsion_topologies
        )

        # Create seed document
        seed_doc = SkillDocument(
            preamble=(
                f"# ABSTRAL SKILL Document\n\n"
                f"Benchmark: {benchmark}\n"
                f"Outer iteration: {outer_iteration}\n"
                f"Target topology: {target_family.value}"
            ),
            K=preserved_k,
            R=repulsion_constraint,
            T="",  # Step 2: Clear T
            P=self._default_protocol(),
            metadata={
                "version": "0",
                "benchmark": benchmark,
                "topology_family": target_family.value,
                "outer_iteration": str(outer_iteration),
                "iteration": "0",
            },
        )

        seed_config = OuterSeedConfig(
            outer_iteration=outer_iteration,
            preserved_k=preserved_k,
            cleared_t=True,
            repulsion_topologies=repulsion_topologies,
            target_family=target_family,
            seed_r_constraint=repulsion_constraint,
        )

        logger.info(
            f"Generated seed for outer iter {outer_iteration}: "
            f"family={target_family.value}, "
            f"K={len(preserved_k)} chars, "
            f"repulsion from {len(repulsion_topologies)} prior topologies"
        )

        return seed_doc, seed_config

    def _merge_domain_knowledge(self) -> str:
        """Merge K sections from all converged documents, de-duplicating."""
        if not self.archive:
            return "No domain knowledge acquired yet."

        all_rules: list[str] = []
        seen: set[str] = set()

        for entry in self.archive:
            doc: SkillDocument = entry["doc"]
            for line in doc.K.splitlines():
                stripped = line.strip()
                if stripped and stripped.startswith(("- ", "* ")):
                    # Simple de-duplication by normalized content
                    normalized = stripped.lower().strip("- *").strip()
                    if normalized not in seen and len(normalized) > 10:
                        seen.add(normalized)
                        all_rules.append(stripped)

        if not all_rules:
            # Carry forward full K from best-performing run
            best = max(self.archive, key=lambda e: e["metrics"].get("auc", 0))
            return best["doc"].K

        return "\n".join(all_rules)

    def _select_target_family(self) -> TopologyFamily:
        """Select the least-explored topology family."""
        available = list(TopologyFamily)

        if not self.archive:
            # First iteration: start with hierarchical (good general-purpose)
            return TopologyFamily.HIERARCHICAL

        # Find least-explored family
        min_count = min(self._family_counts.get(f.value, 0) for f in available)
        candidates = [
            f for f in available
            if self._family_counts.get(f.value, 0) == min_count
        ]

        # Among ties, prefer families with highest expected diversity from archive
        if len(candidates) > 1 and self.archive:
            best_family = None
            best_min_ged = -1.0
            for family in candidates:
                template_graph = template_to_graph(family)
                min_ged = min(
                    compute_ged(template_graph, entry["graph"])
                    for entry in self.archive
                )
                if min_ged > best_min_ged:
                    best_min_ged = min_ged
                    best_family = family
            if best_family:
                return best_family

        return candidates[0]

    def _build_repulsion_constraint(
        self,
        target_family: TopologyFamily,
        prior_families: list[str],
    ) -> str:
        """Build the R section with dual-criteria repulsion constraint (§3.4.1).

        Injects both structural (GED) and semantic (cosine distance) repulsion
        requirements into the R section.
        """
        template = TOPOLOGY_TEMPLATES[target_family]
        min_ged = self.config.outer_loop.min_ged
        min_sem = self.config.outer_loop.min_semantic_dist

        parts = [
            f"Target topology: **{target_family.value}** — {template['description']}",
            "",
            f"### Topology Selection Rules",
            f"- Use a {target_family.value} topology as the starting point.",
            f"- Adapt the topology based on task requirements and trace feedback.",
        ]

        if prior_families:
            parts.append("")
            parts.append("### Dual-Criteria Repulsion Constraint (MANDATORY)")
            parts.append(
                "The following (canonical_graph, embedding_hash) pairs are exhausted. "
                f"Your output must satisfy GED >= {min_ged} AND semantic cosine "
                f"distance >= {min_sem} from all entries in this list."
            )
            for i, family in enumerate(prior_families):
                parts.append(f"  - Run {i + 1}: {family}")
            parts.append(
                f"- Criterion 1 (Structural): GED >= {min_ged} using functional type labels "
                f"from taxonomy {{Router, Planner, Executor, Verifier, Aggregator, Specialist, Oracle}}."
            )
            parts.append(
                f"- Criterion 2 (Semantic): Cosine distance >= {min_sem} between "
                f"role-set embeddings after canonicalization to functional types."
            )
            parts.append(
                f"- The new topology MUST be qualitatively different from: "
                f"{', '.join(set(prior_families))}."
            )

        return "\n".join(parts)

    def _default_protocol(self) -> str:
        return (
            "### Construction Protocol\n\n"
            "1. Read task description and determine task properties.\n"
            "2. Select topology from R section rules.\n"
            "3. Design agent roles appropriate for the topology (add to T).\n"
            "4. Wire agents according to topology.\n"
            "5. Populate system prompts with domain knowledge from K.\n"
            "6. Define message schemas for inter-agent communication.\n"
            "7. Execute and capture traces for analysis."
        )
