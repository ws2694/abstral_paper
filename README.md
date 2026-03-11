# ABSTRAL: Automated Multi-Agent System Design via Skill-Referenced Adaptive Search

This repository contains the full implementation and experiment code for the ABSTRAL framework.

ABSTRAL treats multi-agent system (MAS) architecture as an evolving natural-language artifact — a **SKILL.md** document — refined through causal inference over execution traces, rather than scalar fitness optimization.

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

