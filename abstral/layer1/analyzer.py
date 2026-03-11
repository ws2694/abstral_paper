"""Layer 1 ANALYZE phase: Contrastive trace analysis and EC classification.

The meta-agent performs contrastive trace analysis: comparing failed traces
against succeeded traces to identify structural failure modes. Each failure
is classified into one of five evidence classes (EC1–EC5).
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from abstral.config import (
    ABSTRALConfig,
    EvidenceClass,
    EpistemicStance,
    FunctionalType,
    EC_SECTION_MAP,
    TraceEvidence,
)
from abstral.meta_client import create_meta_client
from abstral.skill.document import SkillDocument
from abstral.tracing import TracePair, TraceManager, TraceSet

logger = logging.getLogger(__name__)


ANALYZE_PROMPT = """\
You are the ABSTRAL trace analyst. Your job is to diagnose WHY an agent system
failed on specific tasks by comparing failed traces against succeeded traces.

## Evidence Class Taxonomy

You MUST consider ALL FIVE classes and use the most specific one that applies:

- EC1 (Reasoning Error): An agent reaches the WRONG CONCLUSION despite having correct
  information available. Look for: incorrect logic, hallucinated facts, wrong domain
  knowledge application. The agent had the right info but reasoned incorrectly.
  → Target: K section (Domain Knowledge) — add missing domain rules

- EC2 (Topology Failure): The agent GRAPH STRUCTURE caused the failure — wrong routing,
  sequential bottleneck where parallelism was needed, or missing connections. The individual
  agents may be fine but the wiring between them is wrong.
  → Target: R section (Topology Reasoning) — fix routing/wiring rules

- EC3 (Missing Specialization): A SINGLE AGENT handled incompatible sub-tasks that required
  different expertise. Look for: one agent switching between unrelated sub-problems, or one
  agent failing because it lacks a specific skill that a dedicated specialist would have.
  → Target: T section (Template Library — add new specialist role)

- EC4 (Interface Failure): Message format issues — malformed schema, missing required field,
  type mismatch between agents, or agents not understanding each other's outputs.
  → Target: P section (Construction Protocol) — fix message contracts

- EC5 (Emergent Pattern): A heuristic or strategy that CONSISTENTLY WORKS in succeeded traces
  but is not yet documented in SKILL.md. Look for: successful patterns in the succeeded trace
  that could be codified as reusable rules.
  → Target: T section (Template Library — codify pattern)

## Current SKILL.md (for context)

### K — Domain Knowledge
{domain_knowledge}

### R — Topology Reasoning
{topology_reasoning}

### T — Agent Template Library
{template_library}

### P — Construction Protocol
{construction_protocol}

## Trace Pair to Analyze

### FAILED Trace
{failed_trace}

### SUCCEEDED Trace (same task type)
{succeeded_trace}

## Instructions

1. Compare the two traces carefully, especially the per-agent interaction flow.
2. Identify the structural failure mode in the failed trace.
3. Classify it into exactly ONE evidence class (EC1-EC5).
   - IMPORTANT: Do NOT default to EC2 or EC4. Carefully distinguish:
     * Is the agent's REASONING wrong? → EC1
     * Is the GRAPH WIRING wrong? → EC2
     * Does one agent need to be SPLIT into specialists? → EC3
     * Is the MESSAGE FORMAT between agents broken? → EC4
     * Is a successful PATTERN worth documenting? → EC5
4. Explain your reasoning with specific citations from the traces.
5. Suggest a concrete edit to the appropriate SKILL.md section.

Return 1-3 evidence items. Each item MUST use the most specific EC class that applies."""


ANALYZE_SINGLE_PROMPT = """\
You are the ABSTRAL trace analyst. Analyze the following agent execution traces
to identify failure modes and improvement opportunities.

## Evidence Class Taxonomy

You MUST consider ALL FIVE classes and use the most specific one:

- EC1 (Reasoning Error): Agent reaches wrong conclusion despite correct info.
  Look for: incorrect logic, hallucinated facts, wrong domain knowledge.
  → Target: K section (Domain Knowledge)

- EC2 (Topology Failure): Graph structure failure — wrong routing, bottleneck.
  → Target: R section (Topology Reasoning)

- EC3 (Missing Specialization): One agent handles incompatible sub-tasks that
  need different expertise. Look for: agent switching between unrelated problems.
  → Target: T section (Template Library — add specialist role)

- EC4 (Interface Failure): Message format issues between agents.
  → Target: P section (Construction Protocol)

- EC5 (Emergent Pattern): Successful heuristic not yet documented.
  → Target: T section (Template Library — codify pattern)

IMPORTANT: Do NOT default to EC2/EC4. Use the most specific class.

## Current SKILL.md (for context)

### K — Domain Knowledge
{domain_knowledge}

### R — Topology Reasoning
{topology_reasoning}

### T — Agent Template Library
{template_library}

### P — Construction Protocol
{construction_protocol}

## Traces to Analyze ({trace_status})

{traces}

## Instructions

1. Analyze the traces for structural failure modes or improvement opportunities.
2. For each issue found, classify it into the MOST SPECIFIC evidence class (EC1-EC5).
3. Explain your reasoning with specific citations from the traces.
4. Suggest a concrete edit to the appropriate SKILL.md section.
5. Return 2-5 evidence items. Aim for diverse EC classes — not all EC2/EC4.

Be specific and actionable. Do not suggest vague improvements."""


class AnalysisResult(BaseModel):
    """Structured result of analyzing a single trace pair."""
    evidence: list[TraceEvidence]
    summary: str
    ec_distribution: dict[str, int] = Field(default_factory=dict)


class TraceAnalyzer:
    """Performs contrastive trace analysis using the meta-agent."""

    def __init__(self, config: ABSTRALConfig):
        self.config = config
        self.client = create_meta_client(config.meta_agent)
        self.trace_manager = TraceManager(
            project_name=config.tracking.otel_service_name
        )

    def analyze_traces(
        self,
        skill_doc: SkillDocument,
        trace_set: TraceSet,
        max_pairs: int = 5,
    ) -> AnalysisResult:
        """Analyze a set of traces by pairing failed/succeeded and classifying failures.

        Falls back to single-trace analysis when contrastive pairs are unavailable
        (e.g., all tasks failed or all succeeded).

        Token optimization: Instead of one meta-agent call per pair, we batch
        up to `max_pairs` pairs into a SINGLE call. This sends the SKILL.md
        context only once instead of N times.

        Returns diagnosed evidence with trace citations.
        """
        all_evidence: list[TraceEvidence] = []
        ec_dist: dict[str, int] = {ec.value: 0 for ec in EvidenceClass}

        # Try contrastive pair analysis first
        pairs = self.trace_manager.pair_traces(trace_set)
        if pairs:
            pairs = pairs[:max_pairs]
            logger.info(f"Analyzing {len(pairs)} contrastive trace pairs (batched)")

            # Batch all pairs into a single meta-agent call
            evidence_items = self._analyze_pairs_batched(skill_doc, pairs)
            for ev in evidence_items:
                all_evidence.append(ev)
                ec_dist[ev.ec_class.value] = ec_dist.get(ev.ec_class.value, 0) + 1
        else:
            # Fallback: analyze traces without contrastive pairs
            if trace_set.failed:
                logger.info(f"No pairs available. Analyzing {min(len(trace_set.failed), 8)} failed traces directly.")
                evidence_items = self._analyze_traces_batch(
                    skill_doc, trace_set.failed[:8], "all failed — no succeeded traces for contrast"
                )
                for ev in evidence_items:
                    all_evidence.append(ev)
                    ec_dist[ev.ec_class.value] = ec_dist.get(ev.ec_class.value, 0) + 1
            elif trace_set.succeeded:
                logger.info(f"All tasks succeeded. Analyzing {min(len(trace_set.succeeded), 5)} traces for emergent patterns.")
                evidence_items = self._analyze_traces_batch(
                    skill_doc, trace_set.succeeded[:5], "all succeeded — looking for emergent patterns (EC5)"
                )
                for ev in evidence_items:
                    all_evidence.append(ev)
                    ec_dist[ev.ec_class.value] = ec_dist.get(ev.ec_class.value, 0) + 1
            else:
                logger.warning("No traces available for analysis at all")
                return AnalysisResult(
                    evidence=[], summary="No traces to analyze.",
                    ec_distribution=ec_dist,
                )

        summary_parts = []
        for ec, count in ec_dist.items():
            if count > 0:
                summary_parts.append(f"{ec}: {count}")
        n_analyzed = len(pairs) if pairs else len(trace_set.failed or trace_set.succeeded or [])
        mode = "pairs" if pairs else "single-trace"
        summary = f"Analyzed {n_analyzed} {mode} → {len(all_evidence)} evidence items. Distribution: {', '.join(summary_parts)}"

        logger.info(summary)

        return AnalysisResult(
            evidence=all_evidence,
            summary=summary,
            ec_distribution=ec_dist,
        )

    def _analyze_pairs_batched(
        self,
        skill_doc: SkillDocument,
        pairs: list[TracePair],
    ) -> list[TraceEvidence]:
        """Analyze multiple trace pairs in a SINGLE meta-agent call.

        This sends the SKILL.md context once instead of N times, saving
        ~(N-1) × SKILL.md tokens worth of input. Uses ANALYZE_SINGLE_PROMPT
        which has a generic {traces} slot, avoiding the misuse of
        {failed_trace}/{succeeded_trace} slots for batched content.
        """
        pair_sections = []
        for i, pair in enumerate(pairs):
            failed_summary = self.trace_manager.summarize_trace(pair.failed_trace)
            succeeded_summary = self.trace_manager.summarize_trace(pair.succeeded_trace)
            pair_sections.append(
                f"### Contrastive Pair {i + 1}\n\n"
                f"**FAILED Trace:**\n{failed_summary}\n\n"
                f"**SUCCEEDED Trace (same task type):**\n{succeeded_summary}"
            )

        prompt = ANALYZE_SINGLE_PROMPT.format(
            domain_knowledge=skill_doc.K or "(empty)",
            topology_reasoning=skill_doc.R or "(empty)",
            template_library=skill_doc.T or "(empty)",
            construction_protocol=skill_doc.P or "(empty)",
            trace_status=f"{len(pairs)} contrastive pairs (failed vs succeeded)",
            traces="\n\n---\n\n".join(pair_sections),
        )

        try:
            result = self.client.chat.completions.create(
                model=self.config.meta_agent.model,
                max_tokens=self.config.meta_agent.max_tokens,
                messages=[{"role": "user", "content": prompt}],
                response_model=list[TraceEvidence],
            )
            for ev in result:
                ev.target_section = EC_SECTION_MAP[ev.ec_class]
            return result
        except Exception as e:
            logger.error(f"Failed to analyze batched pairs: {e}")
            # Fallback: try individual pair analysis
            all_evidence = []
            for pair in pairs[:3]:  # Only retry first 3 to limit cost
                all_evidence.extend(self._analyze_pair(skill_doc, pair))
            return all_evidence

    def _analyze_pair(
        self,
        skill_doc: SkillDocument,
        pair: TracePair,
    ) -> list[TraceEvidence]:
        """Analyze a single trace pair and return evidence items."""
        failed_summary = self.trace_manager.summarize_trace(pair.failed_trace)
        succeeded_summary = self.trace_manager.summarize_trace(pair.succeeded_trace)

        prompt = ANALYZE_PROMPT.format(
            domain_knowledge=skill_doc.K or "(empty)",
            topology_reasoning=skill_doc.R or "(empty)",
            template_library=skill_doc.T or "(empty)",
            construction_protocol=skill_doc.P or "(empty)",
            failed_trace=failed_summary,
            succeeded_trace=succeeded_summary,
        )

        try:
            result = self.client.chat.completions.create(
                model=self.config.meta_agent.model,
                max_tokens=self.config.meta_agent.max_tokens,
                messages=[{"role": "user", "content": prompt}],
                response_model=list[TraceEvidence],
            )
            # Ensure target sections are correctly set
            for ev in result:
                ev.target_section = EC_SECTION_MAP[ev.ec_class]
            return result

        except Exception as e:
            logger.error(f"Failed to analyze trace pair: {e}")
            return []

    def _analyze_traces_batch(
        self,
        skill_doc: SkillDocument,
        traces: list[dict],
        trace_status: str,
    ) -> list[TraceEvidence]:
        """Analyze a batch of traces (all failed or all succeeded) without contrastive pairs."""
        trace_summaries = []
        for i, trace in enumerate(traces[:10]):
            summary = self.trace_manager.summarize_trace(trace)
            trace_summaries.append(f"### Trace {i + 1}\n{summary}")

        prompt = ANALYZE_SINGLE_PROMPT.format(
            domain_knowledge=skill_doc.K or "(empty)",
            topology_reasoning=skill_doc.R or "(empty)",
            template_library=skill_doc.T or "(empty)",
            construction_protocol=skill_doc.P or "(empty)",
            trace_status=trace_status,
            traces="\n\n".join(trace_summaries),
        )

        try:
            result = self.client.chat.completions.create(
                model=self.config.meta_agent.model,
                max_tokens=self.config.meta_agent.max_tokens,
                messages=[{"role": "user", "content": prompt}],
                response_model=list[TraceEvidence],
            )
            for ev in result:
                ev.target_section = EC_SECTION_MAP[ev.ec_class]
            return result
        except Exception as e:
            logger.error(f"Failed to analyze traces batch: {e}")
            return []

    def discover_specialist_role(
        self,
        ec3_evidence: TraceEvidence,
    ) -> dict:
        """EC3 Specialist Role Discovery via causal inference (§3.2.1).

        When EC3 fires, the meta-agent answers: 'What specific capability,
        if present as a dedicated specialist, would have prevented this
        failure class from occurring?'

        Returns a role specification dict with required fields:
        role_name, epistemic_stance, system_prompt, tool_access,
        birth_trace, interface_contract.
        """
        EC3_ROLE_PROMPT = """\
You are the ABSTRAL role discovery agent. An EC3 (Missing Specialization) event
has been detected: a single agent handled structurally incompatible sub-tasks.

## The Failure Event

Trace ID: {trace_id}
Failed trace: {failed_trace}
Reasoning: {reasoning}

## Causal Inference Question

What specific capability, if present as a DEDICATED specialist agent, would
have prevented this failure class from occurring?

## Requirements for the New Role

1. role_name: Use [Adjective][Noun] format (e.g., "Unit-Consistency Enforcer").
   Generic names like "Analyzer" or "Agent_2" are REJECTED.
2. epistemic_stance: One of: Skeptic, Verifier, Explorer, Synthesizer, Enforcer
3. system_prompt: Include:
   - The failure mode this role was created to prevent
   - Relevant domain knowledge
   - Escalation criteria
4. tool_access: Specific tools granted (principle of least privilege)
5. interface_contract: Required input/output fields with types
6. functional_type: One of: Router, Planner, Executor, Verifier, Aggregator, Specialist, Oracle

Return a JSON object with these fields."""

        prompt = EC3_ROLE_PROMPT.format(
            trace_id=ec3_evidence.trace_id,
            failed_trace=ec3_evidence.failed_trace_summary[:500],
            reasoning=ec3_evidence.reasoning,
        )

        try:
            from pydantic import BaseModel as _BM

            class RoleSpec(_BM):
                role_name: str
                epistemic_stance: str
                system_prompt: str
                tool_access: list[str]
                interface_contract: dict[str, str]
                functional_type: str

            result = self.client.chat.completions.create(
                model=self.config.meta_agent.model,
                max_tokens=self.config.meta_agent.max_tokens,
                messages=[{"role": "user", "content": prompt}],
                response_model=RoleSpec,
            )

            role_dict = {
                "role_name": result.role_name,
                "epistemic_stance": result.epistemic_stance,
                "system_prompt": result.system_prompt,
                "tool_access": result.tool_access,
                "birth_trace": ec3_evidence.trace_id,
                "interface_contract": result.interface_contract,
                "functional_type": result.functional_type,
            }

            logger.info(
                f"EC3 Role Discovery: {result.role_name} "
                f"(stance={result.epistemic_stance}, type={result.functional_type})"
            )
            return role_dict

        except Exception as e:
            logger.error(f"EC3 role discovery failed: {e}")
            return {}

    def compute_ec_fractions(
        self,
        ec_distribution: dict[str, int],
    ) -> dict[str, float]:
        """Compute EC class fractions for convergence monitoring."""
        total = sum(ec_distribution.values())
        if total == 0:
            return {ec: 0.0 for ec in ec_distribution}
        return {ec: count / total for ec, count in ec_distribution.items()}

    def ec1_ec2_fraction(self, ec_distribution: dict[str, int]) -> float:
        """Compute the combined EC1+EC2 fraction (used by C3 convergence signal)."""
        total = sum(ec_distribution.values())
        if total == 0:
            return 0.0
        actionable = ec_distribution.get("EC1", 0) + ec_distribution.get("EC2", 0)
        return actionable / total
