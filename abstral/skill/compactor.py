"""Skill document compaction — triggered by Layer 2 complexity penalty (C4).

When the skill document exceeds the complexity threshold, this module
re-distills it using the meta-agent into a clean, non-redundant version.
"""

from __future__ import annotations

import logging

import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field

from abstral.config import ABSTRALConfig
from abstral.skill.document import SkillDocument
from abstral.skill.versioning import SkillRepository

logger = logging.getLogger(__name__)


class CompactedSection(BaseModel):
    """A compacted section of the skill document."""
    section_key: str
    content: str
    rules_before: int
    rules_after: int
    removed_rules: list[str] = Field(default_factory=list)


class CompactionResult(BaseModel):
    """Result of skill document compaction."""
    sections: list[CompactedSection]
    total_rules_before: int
    total_rules_after: int
    reduction_pct: float


COMPACTION_PROMPT = """\
You are a skill document compactor for the ABSTRAL system.

Your task is to distill the following SKILL.md section into a cleaner, more concise
version that preserves all genuinely useful design knowledge while removing:
- Redundant rules (rules that say the same thing differently)
- Contradictory rules (keep the more recent / better-supported one)
- Overly specific rules that can be generalized
- Rules that have been superseded by later edits

IMPORTANT: Do NOT remove rules that encode unique, actionable design knowledge.
The goal is compression without information loss.

Section: ## {section_key}
Current content ({rule_count} rules, {word_count} words):

{content}

Target: Reduce to ~{target_rules} rules while preserving all unique knowledge.

Return the compacted section content as clean Markdown. Every preserved rule should
be a bullet point starting with "- "."""


def compact(
    repo: SkillRepository,
    config: ABSTRALConfig,
    target_rules: int = 80,
    max_history: int = 20,
) -> CompactionResult:
    """Compact the current SKILL.md by re-distilling via meta-agent.

    Returns the compaction result with before/after metrics.
    Does NOT commit — caller decides whether to commit.
    """
    doc = repo.read()
    client = instructor.from_anthropic(Anthropic())

    sections: list[CompactedSection] = []
    total_before = 0
    total_after = 0

    for key in ["K", "R", "T", "P"]:
        content = doc.get_section(key)
        if not content.strip():
            continue

        rule_count = sum(
            1 for line in content.splitlines()
            if line.strip().startswith(("- ", "* "))
        )
        word_count = len(content.split())

        if rule_count < 10:
            # Not worth compacting small sections
            sections.append(CompactedSection(
                section_key=key,
                content=content,
                rules_before=rule_count,
                rules_after=rule_count,
            ))
            total_before += rule_count
            total_after += rule_count
            continue

        section_target = max(int(rule_count * 0.6), target_rules // 4)

        prompt = COMPACTION_PROMPT.format(
            section_key=key,
            content=content,
            rule_count=rule_count,
            word_count=word_count,
            target_rules=section_target,
        )

        response = client.chat.completions.create(
            model=config.meta_agent.model,
            max_tokens=config.meta_agent.max_tokens,
            messages=[{"role": "user", "content": prompt}],
            response_model=CompactedSection,
            context={"section_key": key},
        )

        # Ensure the section key is correct
        response.section_key = key
        response.rules_before = rule_count

        new_rule_count = sum(
            1 for line in response.content.splitlines()
            if line.strip().startswith(("- ", "* "))
        )
        response.rules_after = new_rule_count

        doc.set_section(key, response.content)
        sections.append(response)
        total_before += rule_count
        total_after += new_rule_count

    # Write the compacted document back (but don't commit yet)
    doc.write(repo.skill_path)

    reduction = (1 - total_after / max(total_before, 1)) * 100

    logger.info(
        f"Compaction: {total_before} → {total_after} rules ({reduction:.1f}% reduction)"
    )

    return CompactionResult(
        sections=sections,
        total_rules_before=total_before,
        total_rules_after=total_after,
        reduction_pct=reduction,
    )
