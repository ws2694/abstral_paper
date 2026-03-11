# ABSTRAL: Automated Multi-Agent System Design via Skill-Referenced Adaptive Search

This repository contains the full implementation and experiment code for the ABSTRAL framework, submitted to COLM 2026.

ABSTRAL treats multi-agent system (MAS) architecture as an evolving natural-language artifact — a **SKILL.md** document — refined through causal inference over execution traces, rather than scalar fitness optimization.

## Repository Structure

```
abstral/                        # Core framework package
├── config.py                   # ABSTRALConfig, TopologyFamily, EvidenceClass enums
├── models.py                   # TaskInstance, RunResult, BatchRunResult
├── pipeline.py                 # End-to-end pipeline orchestrator (PipelineState)
├── layer1/                     # Layer 1: Inner Trace-Referenced Loop
│   ├── builder.py              #   BUILD: SKILL.md → AgentSpec → LangGraph
│   ├── runner.py               #   RUN: Execute agent graph on task batch
│   ├── analyzer.py             #   ANALYZE: Contrastive trace analysis → EC1–EC5
│   └── updater.py              #   UPDATE: EC-driven targeted edits to SKILL.md
├── layer2/                     # Layer 2: Convergence Detection
│   └── convergence.py          #   C1–C4 signals + hard cap
├── layer3/                     # Layer 3: Diversity Seeding
│   ├── seeder.py               #   Topology repulsion via GED + semantic distance
│   ├── topology.py             #   Graph representations, GED, classifier
│   └── landscape.py            #   Design landscape (GED matrix, MDS, UMAP)
├── skill/                      # SKILL.md document management
│   ├── document.py             #   Parse/render K/R/T/P sections
│   ├── versioning.py           #   Git-based versioning with trace citations
│   └── compactor.py            #   Anti-drift consolidation (C4 trigger)
├── tau_adapter.py              # τ-bench integration
├── sop_adapter.py              # SOPBench integration
├── benchmarks.py               # GAIA, ALFWorld, WebArena, Tabular loaders
├── meta_client.py              # Provider-aware LLM client factory
├── tools.py                    # Agent tool definitions
├── tracing.py                  # OpenTelemetry trace collection
└── tracking.py                 # MLflow experiment tracking

run_experiment.py               # Primary entry point: τ-bench + GAIA experiments
run_sop.py                      # SOPBench experiments (bank, healthcare)
configs/default.yaml            # Default configuration
vendor/tau_bench/               # Vendored τ-bench source (ToolCallingAgent baseline)
vendor/sopbench/                # Vendored SOPBench source

tests/test_core.py              # Unit tests
smoketest.py                    # Component smoke tests
test_e2e.py                     # End-to-end integration tests
dashboard/app.py                # Streamlit real-time monitoring UI
```

## Setup

**Requirements:** Python >= 3.11

```bash
# Install dependencies
pip install -e .

# Copy and fill in API keys
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY, OPENAI_API_KEY, and HF_TOKEN
```

| Key | Used by | Purpose |
|-----|---------|---------|
| `ANTHROPIC_API_KEY` | Meta-agent (Claude Sonnet 4) | BUILD, ANALYZE, UPDATE phases |
| `OPENAI_API_KEY` | Agent backbone (GPT-4o) | RUN phase — task execution |
| `HF_TOKEN` | GAIA loader | Access gated HuggingFace dataset |

## Running Experiments

### τ-bench (primary benchmark)

```bash
# Smoke test (1 outer, 2 inner, 3 tasks — ~2 min)
python run_experiment.py --tau --smoke

# Full experiment (3 outer × 8 inner, 15 tasks/batch)
python run_experiment.py --tau

# Reproduce published single-agent baseline (ToolCallingAgent, GPT-4o)
python run_experiment.py --tau --published-baseline

# Ablation studies (inner-only, single-agent, fixed-topology)
python run_experiment.py --tau --ablation

# Final evaluation (best topology, 50 tasks × k trials)
python run_experiment.py --tau --final-eval

# Print results tables
python run_experiment.py --tau --tables
```

### SOPBench

```bash
python run_sop.py --domain bank --main
python run_sop.py --domain healthcare --main
python run_sop.py --tables
```

### Unit tests

```bash
pytest tests/test_core.py -v
python smoketest.py
```

## How It Works

ABSTRAL operates through three nested loops:

### Layer 1 — Inner Trace-Referenced Loop (§3.2)

Each iteration executes four phases on a batch of tasks:

1. **BUILD**: A meta-agent (Claude Sonnet) reads the current SKILL.md and produces an `AgentSpec` — topology family, agent roles, edges, and entry point. This spec is compiled into a LangGraph.
2. **RUN**: The compiled graph executes on a task batch under sandbox constraints (50K token budget, 5-min wall clock per task). Each agent node uses GPT-4o as its backbone.
3. **ANALYZE**: Failed and succeeded traces are paired contrastively and classified into five Evidence Classes:
   - **EC1** (Reasoning Error) → update K (domain knowledge)
   - **EC2** (Topology Failure) → update R (topology rules)
   - **EC3** (Missing Specialization) → update T (new specialist role via causal inference)
   - **EC4** (Interface Failure) → update P (message schemas)
   - **EC5** (Emergent Pattern) → update T (encode heuristic)
4. **UPDATE**: Targeted edits to the corresponding SKILL.md section, committed via GitPython with trace ID citations.

### Layer 2 — Convergence Detection (§3.3)

Four independent signals determine when the inner loop should stop:

| Signal | Condition | Weight |
|--------|-----------|--------|
| C1 (Skill Diff Collapse) | diff_lines < 5 | 2 |
| C2 (AUC Plateau) | \|AUC_t - AUC_{t-1}\| < 0.005 for 3 consecutive iterations | 2 |
| C3 (EC Signal Collapse) | (EC1 + EC2) / total < 10% | 1 |
| C4 (Complexity Penalty) | rules > 200 or words > 5000 → triggers compaction | 1 |

Termination fires when total weight >= 3.

### Layer 3 — Diversity Seeding (§3.4)

Each outer iteration seeds a new SKILL.md targeting the least-explored topology family, subject to dual-criteria repulsion:
- **Structural**: GED >= 3 from all prior converged topologies
- **Semantic**: Cosine distance >= 0.25 between role-set embeddings

Six canonical families: `single`, `pipeline`, `ensemble`, `debate`, `hierarchical`, `dynamic_routing`.

## Key Design Decisions

**SKILL.md as central artifact.** All design knowledge lives in a single Markdown document with four sections (K/R/T/P). Every edit is committed with trace citations, creating a full audit trail.

**No LLM-as-judge.** τ-bench uses DB-state SHA-256 hash comparison. SOPBench uses oracle-based 5-boolean criteria. GAIA uses exact-match accuracy.

**Topology-aware step budgets.** Each topology family gets a step budget calibrated to its coordination pattern: pipelines get n+2, hierarchical gets 2n, debate gets 3n (floored at 8).

**Conditional entry points.** For multi-turn benchmarks (τ-bench), the graph resumes at the last active tool agent rather than re-routing from the entry point every turn.

## Results

Results are saved to `results/` as JSON files:

```
results/
├── main_tau_airline.json              # Full experiment state
├── published_baseline_tau_airline.json # Reproduced baseline
├── final_eval_tau_airline.json        # Best topology evaluation
├── artifacts/                         # Converged SKILL.md per outer loop
│   └── main_tau_airline/outer_{1,2,3}/SKILL.md
└── logs/                              # Per-experiment log files
```

Each JSON contains: `auc_history`, `ec_history`, `convergence_history`, `topology_gallery`, `routing_stats`, `k_section_snapshots`, `ged_matrix`, `umap_coords`, `timing_history`, and full task-level results.

### Monitoring

```bash
streamlit run dashboard/app.py
```

The dashboard provides real-time AUC curves, EC distributions, convergence signal gauges, GED heatmaps, and UMAP projections of the design landscape.

## Configuration

Default config is in [configs/default.yaml](configs/default.yaml). Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `meta_agent.model` | `claude-sonnet-4-20250514` | Meta-agent for BUILD/ANALYZE/UPDATE |
| `agent_backbone.model` | `gpt-4o` | Agent backbone for RUN |
| `inner_loop.max_iterations` | 15 | Max inner loop iterations before hard cap |
| `inner_loop.val_batch_size` | 50 | Tasks per RUN batch |
| `inner_loop.token_budget` | 50000 | Token budget per batch |
| `inner_loop.wall_clock_limit_sec` | 300 | Per-task wall clock (seconds) |
| `outer_loop.n_outer` | 6 | Number of outer loop iterations |
| `outer_loop.min_ged` | 3 | Minimum GED for topology repulsion |

## Appendix Diagnostics

The experiment automatically captures data for paper appendices:

- **Topology Gallery**: Best topology per outer loop — roles, edges, entry point, AUC, convergence iteration
- **Routing Stats**: Per-iteration routing efficiency — skip rate (conditional entry reuse), full routing counts
- **K Section Snapshots**: Domain knowledge evolution — word count, rule count, full text at each outer loop
- **Per-task traces**: Agent messages, tool calls, routing decisions for representative examples

## Citation

```bibtex
@inproceedings{song2026abstral,
  title={ABSTRAL: Automated Multi-Agent System Design via Skill-Referenced Adaptive Search},
  author={Song, Weijia and Yue, Jiashu and Pang, Zhe},
  booktitle={Conference on Language Modeling (COLM)},
  year={2026}
}
```
