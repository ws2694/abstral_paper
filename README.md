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

---

## Reproducing SOPBench Results (Reviewer Guide)

This section provides step-by-step instructions to reproduce the SOPBench results reported in the paper.

### Prerequisites

1. **Python >= 3.11** and a working `pip` install.
2. **API keys** — two are required:

   | Key | Provider | Purpose |
   |-----|----------|---------|
   | `ANTHROPIC_API_KEY` | Anthropic | Meta-agent (Claude Sonnet 4) — BUILD / ANALYZE / UPDATE phases |
   | `OPENAI_API_KEY` | OpenAI | Agent backbone (GPT-4o) — RUN phase |

3. **SOPBench data** is bundled via the `vendor/sopbench` submodule. After cloning, run:
   ```bash
   git submodule update --init --recursive
   ```

### 1. Install & configure

```bash
pip install -e .
cp .env.example .env
# Edit .env — fill in ANTHROPIC_API_KEY and OPENAI_API_KEY
```

### 2. Smoke test (~2 min, < $0.50)

Validates that the full pipeline (BUILD → RUN → ANALYZE → UPDATE) works end-to-end on 3 tasks before committing to a full run.

```bash
python run_sop.py --domain bank --smoke
```

If this completes without error you are ready to proceed.

### 3. Single-agent baseline

Runs GPT-4o as a single agent with no topology search. Compare to the published GPT-4o (FC) baseline: **bank 58.96%**, **healthcare 73.39%** (Li et al., 2025).

```bash
# Bank (134 tasks, ~$5-10)
python run_sop.py --domain bank --baseline

# Healthcare (124 tasks, ~$5-10)
python run_sop.py --domain healthcare --baseline
```

### 4. Full ABSTRAL experiment (3 outer × 8 inner)

The main experiment that performs topology search and SKILL.md refinement. Uses 20 tasks per inner-loop batch.

```bash
# Bank (~$70-100, ~2-4 hours)
python run_sop.py --domain bank --main

# Healthcare (~$60-90, ~2-4 hours)
python run_sop.py --domain healthcare --main
```

### 5. Ablations

```bash
# Inner-only (no outer-loop topology diversity)
python run_sop.py --domain bank --ablation

# Fixed topology baseline (3-agent hierarchy, no search)
# (included in --ablation above)
```

### 6. Held-out test evaluation

After the main experiment finishes, evaluate the best topology on the held-out test split (tasks never seen during the inner loop).

```bash
python run_sop.py --domain bank --test
```

### 7. View results

```bash
# Print comparison table (val AUC, topology family, published baselines)
python run_sop.py --domain bank --tables
python run_sop.py --domain healthcare --tables
```

### Quick-reference: run everything in sequence

```bash
python run_sop.py --domain bank --all
python run_sop.py --domain healthcare --all
```

`--all` runs: single-agent baseline → full ABSTRAL → inner-only ablation → fixed-topology ablation, then prints the results table.

### Cost summary

| Experiment | Est. cost (bank) | Est. time |
|------------|----------------:|----------:|
| `--smoke` | < $0.50 | ~2 min |
| `--baseline` | $5–10 | ~20 min |
| `--main` | $70–100 | 2–4 hrs |
| `--ablation` | $30–50 | 1–2 hrs |
| `--test` | $5–10 | ~20 min |

### Output structure

```
results/sop/
├── main_sop_bank.json                # Full experiment (val)
├── single_agent_sop_bank.json        # Single-agent baseline
├── inner_only_sop_bank.json          # Inner-only ablation
├── fixed_topology_sop_bank.json      # Fixed topology ablation
├── test_main_sop_bank.json           # Held-out test evaluation
├── smoke_sop_bank.json               # Smoke test
├── artifacts/                        # Converged SKILL.md per outer loop
│   └── main_sop_bank/outer_{1,2,3}/SKILL.md
└── logs/                             # Per-experiment logs
    └── main_sop_bank.log
```

Each JSON contains: `auc_history`, `ec_history`, `convergence_history`, `topology_families_seen`, `routing_stats`, `k_section_snapshots`, `timing_history`, `cumulative_tokens`, and full task-level results.

### Published baselines (Li et al., 2025)

| Model | Bank | Healthcare |
|-------|-----:|-----------:|
| o4-mini-high (FC) | 76.87% | 92.74% |
| GPT-5 (FC) | 71.64% | — |
| Claude-3.5-Sonnet (FC) | 71.90% | — |
| Claude-3.7-Sonnet (FC) | 65.67% | 70.97% |
| **GPT-4o (FC)** ← our backbone | **58.96%** | **73.39%** |
| GPT-4o-mini (FC) | 33.58% | — |

---

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

