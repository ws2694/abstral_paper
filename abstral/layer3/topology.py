"""Topology families, graph representations, Graph Edit Distance, and semantic distance.

Six canonical topology families, seven functional type categories for GED
canonicalization (§3.4.1), and dual-criteria distance (GED + semantic embedding)
for topology repulsion.
"""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx
import numpy as np

from abstral.config import TopologyFamily, AgentSpec, FunctionalType

logger = logging.getLogger(__name__)

# Canonical topology family templates
TOPOLOGY_TEMPLATES: dict[TopologyFamily, dict[str, Any]] = {
    TopologyFamily.SINGLE: {
        "description": "Single agent with full tool access",
        "nodes": ["agent"],
        "edges": [],
    },
    TopologyFamily.PIPELINE: {
        "description": "Linear DAG: A → B → C",
        "nodes": ["stage_1", "stage_2", "stage_3"],
        "edges": [("stage_1", "stage_2"), ("stage_2", "stage_3")],
    },
    TopologyFamily.ENSEMBLE: {
        "description": "Parallel fan-out + aggregator",
        "nodes": ["dispatcher", "worker_1", "worker_2", "worker_3", "aggregator"],
        "edges": [
            ("dispatcher", "worker_1"), ("dispatcher", "worker_2"),
            ("dispatcher", "worker_3"), ("worker_1", "aggregator"),
            ("worker_2", "aggregator"), ("worker_3", "aggregator"),
        ],
    },
    TopologyFamily.DEBATE: {
        "description": "Two agents + judge",
        "nodes": ["proponent", "opponent", "judge"],
        "edges": [
            ("proponent", "judge"), ("opponent", "judge"),
            ("judge", "proponent"), ("judge", "opponent"),
        ],
    },
    TopologyFamily.HIERARCHICAL: {
        "description": "Manager → specialist subagents",
        "nodes": ["manager", "specialist_1", "specialist_2", "specialist_3"],
        "edges": [
            ("manager", "specialist_1"), ("manager", "specialist_2"),
            ("manager", "specialist_3"), ("specialist_1", "manager"),
            ("specialist_2", "manager"), ("specialist_3", "manager"),
        ],
    },
    TopologyFamily.DYNAMIC_ROUTING: {
        "description": "Router node + conditional branches",
        "nodes": ["router", "branch_easy", "branch_medium", "branch_hard", "output"],
        "edges": [
            ("router", "branch_easy"), ("router", "branch_medium"),
            ("router", "branch_hard"), ("branch_easy", "output"),
            ("branch_medium", "output"), ("branch_hard", "output"),
        ],
    },
}


def spec_to_graph(spec: AgentSpec) -> nx.DiGraph:
    """Convert an AgentSpec to a NetworkX directed graph.

    Uses canonical functional type labels (§3.4.1) instead of raw role names
    to prevent synonym-substitution drift.
    """
    G = nx.DiGraph()
    for role in spec.roles:
        G.add_node(
            role.name,
            functional_type=role.functional_type.value,
            role_type=role.description[:50] if role.description else "",
        )
    for edge in spec.edges:
        G.add_edge(edge.source, edge.target, condition=edge.condition)
    return G


def spec_to_canonical_graph(spec: AgentSpec) -> nx.DiGraph:
    """Convert AgentSpec to canonicalized graph for GED comparison.

    Nodes are labeled by functional type (from the 7-category taxonomy),
    not by name, to prevent synonym substitution (§3.4.1).
    """
    G = nx.DiGraph()
    name_to_canonical = {}
    type_counts: dict[str, int] = {}

    for role in spec.roles:
        ft = role.functional_type.value
        type_counts[ft] = type_counts.get(ft, 0) + 1
        canonical_label = f"{ft}_{type_counts[ft]}"
        name_to_canonical[role.name] = canonical_label
        G.add_node(canonical_label, functional_type=ft)

    for edge in spec.edges:
        src = name_to_canonical.get(edge.source, edge.source)
        tgt = name_to_canonical.get(edge.target, edge.target)
        G.add_edge(src, tgt)

    return G


def template_to_graph(family: TopologyFamily) -> nx.DiGraph:
    """Get the canonical template graph for a topology family."""
    template = TOPOLOGY_TEMPLATES[family]
    G = nx.DiGraph()
    G.add_nodes_from(template["nodes"])
    G.add_edges_from(template["edges"])
    return G


def compute_ged(
    g1: nx.DiGraph,
    g2: nx.DiGraph,
    timeout: float = 30.0,
    node_subst_cost: float = 1.0,
    node_del_cost: float = 1.0,
    node_ins_cost: float = 1.0,
    edge_subst_cost: float = 0.5,
) -> float:
    """Compute the Graph Edit Distance between two topologies (§3.4.1).

    Uses uniform costs: node substitution = 1, insertion/deletion = 1,
    edge substitution = 0.5.
    """
    def node_subst(n1, n2):
        # Cost 0 if same functional type, 1 otherwise
        t1 = n1.get("functional_type", "")
        t2 = n2.get("functional_type", "")
        if t1 and t2:
            return 0.0 if t1 == t2 else node_subst_cost
        return node_subst_cost

    def node_cost(n):
        return node_del_cost

    def edge_subst(e1, e2):
        return edge_subst_cost

    def edge_cost(e):
        return edge_subst_cost

    try:
        for ged in nx.optimize_graph_edit_distance(
            g1, g2,
            node_subst_cost=node_subst,
            node_del_cost=node_cost,
            node_ins_cost=node_cost,
            edge_subst_cost=edge_subst,
            edge_del_cost=edge_cost,
            edge_ins_cost=edge_cost,
        ):
            return float(ged)
    except Exception:
        pass

    # Fallback: approximate GED using structural features
    return _approximate_ged(g1, g2)


def compute_semantic_distance(
    spec1: AgentSpec,
    spec2: AgentSpec,
    embedding_model: str = "text-embedding-3-small",
) -> float:
    """Compute semantic cosine distance between two topology role-sets (§3.4.1).

    Role descriptions are canonicalized to functional type labels before
    embedding to prevent synonym-substitution drift.
    """
    # Build canonicalized role-set descriptions
    desc1 = _canonical_role_description(spec1)
    desc2 = _canonical_role_description(spec2)

    try:
        from openai import OpenAI
        client = OpenAI()
        resp = client.embeddings.create(
            model=embedding_model,
            input=[desc1, desc2],
        )
        emb1 = np.array(resp.data[0].embedding)
        emb2 = np.array(resp.data[1].embedding)

        # Cosine distance = 1 - cosine_similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(1.0 - similarity)
    except Exception as e:
        logger.warning(f"Semantic distance computation failed: {e}. Using structural fallback.")
        # Fallback: Jaccard distance on functional type sets
        types1 = {r.functional_type.value for r in spec1.roles}
        types2 = {r.functional_type.value for r in spec2.roles}
        if not types1 and not types2:
            return 0.0
        intersection = len(types1 & types2)
        union = len(types1 | types2)
        return 1.0 - (intersection / union) if union > 0 else 1.0


def _canonical_role_description(spec: AgentSpec) -> str:
    """Build a canonical description of a spec's role-set for embedding.

    Each role is represented by its functional type label, not its name,
    to prevent synonym-substitution drift (§3.4.1).
    """
    parts = []
    for role in sorted(spec.roles, key=lambda r: r.functional_type.value):
        parts.append(
            f"{role.functional_type.value}: {role.description or role.name}"
        )
    return " | ".join(parts)


def is_genuinely_distinct(
    candidate_spec: AgentSpec,
    archive_specs: list[AgentSpec],
    archive_graphs: list[nx.DiGraph],
    min_ged: float = 3.0,
    min_semantic_dist: float = 0.25,
    embedding_model: str = "text-embedding-3-small",
) -> tuple[bool, str]:
    """Check if a candidate topology is genuinely distinct from all archived ones.

    Dual-criteria repulsion rule (§3.4.1):
    T_k is genuinely distinct iff for ALL j < k:
      GED(T_k, T_j*) >= δ_struct AND cosine_dist >= δ_sem

    Returns (is_distinct, reason).
    """
    candidate_graph = spec_to_canonical_graph(candidate_spec)

    for i, (archived_spec, archived_graph) in enumerate(zip(archive_specs, archive_graphs)):
        ged = compute_ged(candidate_graph, archived_graph)
        if ged < min_ged:
            return False, f"GED={ged:.1f} < {min_ged} vs archive entry {i}"

        sem_dist = compute_semantic_distance(
            candidate_spec, archived_spec, embedding_model
        )
        if sem_dist < min_semantic_dist:
            return False, f"Semantic dist={sem_dist:.3f} < {min_semantic_dist} vs archive entry {i}"

    return True, "Distinct from all archived topologies"


def _approximate_ged(g1: nx.DiGraph, g2: nx.DiGraph) -> float:
    """Approximate GED using structural feature differences."""
    features_1 = _graph_features(g1)
    features_2 = _graph_features(g2)

    distance = 0.0
    distance += abs(features_1["n_nodes"] - features_2["n_nodes"]) * 1.0
    distance += abs(features_1["n_edges"] - features_2["n_edges"]) * 0.5
    distance += abs(features_1["max_in_degree"] - features_2["max_in_degree"]) * 0.5
    distance += abs(features_1["max_out_degree"] - features_2["max_out_degree"]) * 0.5
    distance += abs(features_1["n_sources"] - features_2["n_sources"]) * 1.0
    distance += abs(features_1["n_sinks"] - features_2["n_sinks"]) * 1.0
    distance += (0.0 if features_1["is_dag"] == features_2["is_dag"] else 2.0)

    return distance


def _graph_features(G: nx.DiGraph) -> dict[str, Any]:
    """Extract structural features from a graph."""
    in_degrees = [d for _, d in G.in_degree()] if G.nodes else [0]
    out_degrees = [d for _, d in G.out_degree()] if G.nodes else [0]
    return {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "max_in_degree": max(in_degrees),
        "max_out_degree": max(out_degrees),
        "n_sources": sum(1 for _, d in G.in_degree() if d == 0),
        "n_sinks": sum(1 for _, d in G.out_degree() if d == 0),
        "is_dag": nx.is_directed_acyclic_graph(G),
    }


def compute_pairwise_ged(graphs: list[nx.DiGraph]) -> np.ndarray:
    """Compute pairwise GED matrix for a list of graphs."""
    n = len(graphs)
    ged_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            ged = compute_ged(graphs[i], graphs[j])
            ged_matrix[i, j] = ged
            ged_matrix[j, i] = ged
    return ged_matrix


def classify_topology(spec: AgentSpec) -> TopologyFamily:
    """Classify an AgentSpec into the closest canonical topology family.

    Uses a combination of graph structure analysis and GED comparison
    against canonical templates. Falls back to spec's declared family
    when structural analysis is ambiguous.
    """
    G = spec_to_graph(spec)
    features = _graph_features(G)

    n = features["n_nodes"]
    if n <= 1:
        return TopologyFamily.SINGLE

    # Check for cycles (non-DAG) — strong signal for hierarchical or debate
    if not features["is_dag"]:
        if n <= 3 and features["max_in_degree"] >= 2:
            return TopologyFamily.DEBATE
        if features["max_in_degree"] >= 2 and features["max_out_degree"] >= 2:
            return TopologyFamily.HIERARCHICAL
        return TopologyFamily.HIERARCHICAL  # cycles generally = hierarchical coordination

    # DAG analysis — need to distinguish pipeline, ensemble, dynamic_routing
    if features["max_out_degree"] <= 1 and features["max_in_degree"] <= 1:
        return TopologyFamily.PIPELINE

    # Use GED to canonical templates for more precise classification
    # This avoids the "everything is ensemble" problem
    canonical_graph = spec_to_canonical_graph(spec)
    best_family = spec.topology_family  # default to declared
    best_ged = float("inf")

    for family in TopologyFamily:
        if family == TopologyFamily.SINGLE:
            continue
        tmpl = template_to_graph(family)
        ged = compute_ged(canonical_graph, tmpl, timeout=5.0)
        if ged < best_ged:
            best_ged = ged
            best_family = family

    # If GED clearly distinguishes a family (GED < 3), use it
    if best_ged < 3.0:
        return best_family

    # Ambiguous structure: analyze edge patterns more carefully
    # Check for hierarchical pattern: central node with high in+out degree
    degrees = [(node, G.in_degree(node) + G.out_degree(node)) for node in G.nodes]
    max_degree_node, max_degree = max(degrees, key=lambda x: x[1])
    if max_degree >= n - 1:
        # Central coordinator pattern
        return TopologyFamily.HIERARCHICAL

    # Check for ensemble BEFORE dynamic_routing — both have fan-out from a
    # dispatcher, but ensemble has a distinct aggregator sink that receives
    # from multiple workers (high in-degree sink).
    if features["n_sources"] == 1 and features["n_sinks"] == 1:
        sink_nodes = [node for node in G.nodes if G.out_degree(node) == 0]
        if sink_nodes and G.in_degree(sink_nodes[0]) >= 2:
            # Fan-out → parallel workers → single aggregator = ensemble
            return TopologyFamily.ENSEMBLE

    # Check for dynamic routing: single entry fan-out to multiple distinct paths
    # WITHOUT a shared aggregator sink (each branch terminates independently)
    if features["n_sources"] == 1 and features["max_out_degree"] >= 3:
        return TopologyFamily.DYNAMIC_ROUTING

    # Fall back to declared topology
    return spec.topology_family
