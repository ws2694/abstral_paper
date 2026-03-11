"""Layer 1 UPDATE phase: Targeted skill document edits.

EC-classified evidence drives targeted section edits to SKILL.md.
Every edit records the motivating trace ID. GitPython commits
with structured tag: EC distribution, rules added, citations.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from abstral.config import (
    ABSTRALConfig,
    EvidenceClass,
    EC_SECTION_MAP,
    SkillEdit,
    TraceEvidence,
)
from abstral.skill.document import SkillDocument
from abstral.skill.versioning import SkillRepository

logger = logging.getLogger(__name__)


UPDATE_PROMPT = """\
You are the ABSTRAL skill updater. Your job is to apply targeted edits to
the SKILL.md document based on diagnosed evidence from trace analysis.

## Rules for Editing

1. Each edit must target the correct section based on the evidence class:
   - EC1 (Reasoning Error) → K section (Domain Knowledge)
   - EC2 (Topology Failure) → R section (Topology Reasoning)
   - EC3 (Missing Specialization) → T section (Template Library)
   - EC4 (Interface Failure) → P section (Construction Protocol)
   - EC5 (Emergent Pattern) → T section (Template Library)

2. Every edit must cite the motivating trace ID(s).
3. Prefer ADDING new rules over modifying existing ones.
4. Only REMOVE rules when they are directly contradicted by new evidence.
5. Keep rules concise and actionable (one rule per bullet point).
6. Include the trace ID citation inline: `[trace: <id>]`

## Current Section Content: {section_key}

{section_content}

## Evidence to Incorporate

{evidence_items}

## Instructions

Generate the updated section content that incorporates the evidence.
Add new rules as bullet points. Preserve all existing rules unless
they are directly contradicted. Return ONLY the section content
(not the header)."""


class UpdateResult(BaseModel):
    """Result of applying updates to the skill document."""
    edits_applied: list[SkillEdit]
    rules_added: int
    rules_modified: int
    rules_removed: int
    sections_changed: list[str]


class SkillUpdater:
    """Applies evidence-driven updates to the skill document."""

    def __init__(self, config: ABSTRALConfig):
        self.config = config
        from abstral.meta_client import create_raw_client
        self.client = create_raw_client(config.meta_agent)
        self._provider = config.meta_agent.provider.lower()

    def apply_updates(
        self,
        skill_doc: SkillDocument,
        evidence: list[TraceEvidence],
        repo: SkillRepository,
        iteration: int,
    ) -> UpdateResult:
        """Apply all evidence items as targeted edits to the skill document.

        Groups evidence by target section, applies updates via meta-agent,
        then commits the result with structured metadata.
        """
        if not evidence:
            logger.info("No evidence to apply.")
            return UpdateResult(
                edits_applied=[],
                rules_added=0,
                rules_modified=0,
                rules_removed=0,
                sections_changed=[],
            )

        # Group evidence by target section
        by_section: dict[str, list[TraceEvidence]] = {}
        for ev in evidence:
            section = EC_SECTION_MAP[ev.ec_class]
            by_section.setdefault(section, []).append(ev)

        all_edits: list[SkillEdit] = []
        total_rules_before = skill_doc.rule_count()
        sections_changed: list[str] = []

        for section_key, section_evidence in by_section.items():
            # EC3 evidence gets special handling: specialist role discovery (§3.2.1)
            ec3_items = [ev for ev in section_evidence if ev.ec_class == EvidenceClass.EC3]
            other_items = [ev for ev in section_evidence if ev.ec_class != EvidenceClass.EC3]

            # Handle EC3: format as structured role template entries in T section
            if ec3_items and section_key == "T":
                role_entries = []
                for ev in ec3_items:
                    role_entry = self._format_ec3_role_entry(ev)
                    if role_entry:
                        role_entries.append(role_entry)
                        all_edits.append(SkillEdit(
                            section="T",
                            action="add",
                            content=role_entry,
                            motivating_trace_ids=[ev.trace_id],
                            ec_class=ev.ec_class,
                            reasoning=ev.reasoning,
                        ))
                if role_entries:
                    current_t = skill_doc.get_section("T") or ""
                    new_t = current_t + "\n\n" + "\n\n".join(role_entries)
                    skill_doc.set_section("T", new_t.strip())
                    if "T" not in sections_changed:
                        sections_changed.append("T")

            # Handle all other evidence classes via meta-agent
            if other_items:
                new_content = self._update_section(
                    skill_doc, section_key, other_items
                )
                if new_content is not None:
                    skill_doc.set_section(section_key, new_content)
                    if section_key not in sections_changed:
                        sections_changed.append(section_key)

                    for ev in other_items:
                        all_edits.append(SkillEdit(
                            section=section_key,
                            action="add",
                            content=ev.suggested_edit,
                            motivating_trace_ids=[ev.trace_id],
                            ec_class=ev.ec_class,
                            reasoning=ev.reasoning,
                        ))

        total_rules_after = skill_doc.rule_count()
        rules_added = max(0, total_rules_after - total_rules_before)

        # Update metadata
        skill_doc.metadata["iteration"] = str(iteration)

        # Compute EC distribution for commit tag
        ec_dist: dict[str, int] = {}
        for ev in evidence:
            ec_dist[ev.ec_class.value] = ec_dist.get(ev.ec_class.value, 0) + 1

        trace_ids = [ev.trace_id for ev in evidence]

        # Commit
        repo.commit_update(
            doc=skill_doc,
            iteration=iteration,
            ec_distribution=ec_dist,
            trace_ids=trace_ids,
            rules_added=rules_added,
        )

        logger.info(
            f"Applied {len(all_edits)} edits to {len(sections_changed)} sections. "
            f"+{rules_added} rules."
        )

        return UpdateResult(
            edits_applied=all_edits,
            rules_added=rules_added,
            rules_modified=0,
            rules_removed=0,
            sections_changed=sections_changed,
        )

    def _format_ec3_role_entry(self, ev: TraceEvidence) -> str:
        """Format an EC3 evidence item as a structured role template entry for the T section.

        Per §3.2.1, each discovered role entry has:
        role_name, epistemic_stance, system_prompt, tool_access, birth_trace, interface_contract.
        """
        return (
            f"#### Discovered Role: {ev.suggested_edit.split(':')[0] if ':' in ev.suggested_edit else 'Specialist'}\n"
            f"- **Birth trace**: `{ev.trace_id}`\n"
            f"- **Reasoning**: {ev.reasoning}\n"
            f"- **Specification**: {ev.suggested_edit}\n"
            f"- **Failure prevented**: {ev.failed_trace_summary[:200] if ev.failed_trace_summary else 'N/A'}"
        )

    def _update_section(
        self,
        skill_doc: SkillDocument,
        section_key: str,
        evidence: list[TraceEvidence],
    ) -> str | None:
        """Use the meta-agent to update a single section with evidence."""
        current_content = skill_doc.get_section(section_key)

        evidence_text = ""
        for i, ev in enumerate(evidence, 1):
            evidence_text += (
                f"\n### Evidence {i}\n"
                f"- Class: {ev.ec_class.value}\n"
                f"- Trace ID: {ev.trace_id}\n"
                f"- Failed trace: {ev.failed_trace_summary[:300]}\n"
                f"- Reasoning: {ev.reasoning}\n"
                f"- Suggested edit: {ev.suggested_edit}\n"
            )

        prompt = UPDATE_PROMPT.format(
            section_key=section_key,
            section_content=current_content or "(empty — no rules yet)",
            evidence_items=evidence_text,
        )

        try:
            if self._provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.config.meta_agent.model,
                    max_tokens=self.config.meta_agent.max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                new_content = response.choices[0].message.content.strip()
            else:
                response = self.client.messages.create(
                    model=self.config.meta_agent.model,
                    max_tokens=self.config.meta_agent.max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                new_content = response.content[0].text.strip()
            return new_content

        except Exception as e:
            logger.error(f"Failed to update section {section_key}: {e}")
            return None
