# CLAUDE.md — ABSTRAL

## Project

ABSTRAL: 3-layer automated multi-agent design framework. Inner loop (BUILD→RUN→ANALYZE→UPDATE) refines SKILL.md via trace analysis. Convergence detector (C1-C4) gates iteration. Outer seeder diversifies topology via GED.

- Paper: `ABSTRAL_Paper_COLM.tex` (local only, not pushed)
- Entry: `run_experiment.py`, `run_sop.py`
- Config: `configs/default.yaml`, `abstral/config.py`
- EC mapping: EC1→K, EC2→R, EC3→T, EC4→P, EC5→T (`config.py:EC_SECTION_MAP`)

## Cost Awareness

**Be paranoid about token waste.**
- Use `--smoke` or 3-task subset before full runs. Never run 50×4 trials to test a code change.
- Meta-agent (Claude Sonnet): ~$0.01-0.03/call. Agent backbone (GPT-4o): $2-5 per 15-task batch. Full 50-task eval: ~$30-50.
- If pass^1 stagnates after 3 iterations, **stop and diagnose**.
- Validate `.env` keys BEFORE launching. Don't retry 429s in tight loops.

## Known Gotchas

1. **Wiki injection**: τ-bench agent MUST have airline wiki as system prompt. Verify in `tau_adapter.py` / `layer1/builder.py`.
2. **Baseline gap**: Reproduced GPT-4o = ~30% pass^1 vs published ~46%. Expected — different eval protocol.
3. **Consolidation at iter 5**: Compactor fires every k=5 or on C4. Rule/word count drops are by design.
4. **Outer loop resets T**: Seeder preserves K but clears T between outer loops. Intentional.
5. **GED slow for >8 nodes**: `networkx.optimize_graph_edit_distance()` is exact.
6. **umap-learn**: Use `--prefer-binary` or conda. Needs Python 3.12+.
7. **Token budget**: 50K tokens, 5-min wall clock per RUN phase (`configs/default.yaml`).
