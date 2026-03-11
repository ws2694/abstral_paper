---
version: 0
benchmark: generic
topology_family: hierarchical
iteration: 0
---

# ABSTRAL SKILL Document

This is a seed skill document for the ABSTRAL automated agent design system.
All sections below will be iteratively refined through trace analysis.

## K — Domain Knowledge

No domain knowledge acquired yet. This section will be populated with
task-specific facts discovered from execution trace analysis.

## R — Topology Reasoning

Default topology reasoning. Start with a hierarchical architecture:
- Use a manager agent to decompose tasks and delegate to specialists.
- Adjust topology based on observed trace failures.
- If tasks are homogeneous, consider simplifying to a pipeline.
- If tasks require verification, consider adding a debate or ensemble layer.

## T — Agent Template Library

No agent templates defined yet. Templates will emerge from trace analysis.
Each template defines: name, system prompt, tools, and interface contract.

## P — Construction Protocol

### Default Construction Protocol

1. Read task description and identify key requirements.
2. Select topology from R section based on task properties.
3. Instantiate agent roles from T section (or design new ones if T is empty).
4. Wire agents according to the selected topology.
5. Populate system prompts with domain knowledge from K section.
6. Define message schemas for inter-agent communication.
7. Execute and capture full traces via OpenTelemetry.
