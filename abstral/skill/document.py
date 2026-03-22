"""SKILL.md document parser/writer with K/R/T/P sections.

The skill document ABS_t = (K_t, R_t, T_t, P_t) is the persistent memory
of the entire ABSTRAL system. Each section encodes a different type of
design knowledge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


_SECTION_HEADERS = {
    "K": "## K — Domain Knowledge",
    "R": "## R — Topology Reasoning",
    "T": "## T — Agent Template Library",
    "P": "## P — Construction Protocol",
}

_SECTION_ORDER = ["K", "R", "T", "P"]

_HEADER_RE = re.compile(r"^##\s+([KRTP])\s*[—–-]", re.MULTILINE)


@dataclass
class SkillDocument:
    """Structured representation of a SKILL.md file."""

    K: str = ""  # Domain Knowledge
    R: str = ""  # Topology Reasoning
    T: str = ""  # Agent Template Library
    P: str = ""  # Construction Protocol
    preamble: str = ""  # content before first section
    metadata: dict[str, str] = field(default_factory=dict)

    # ---- Parsing ----

    @classmethod
    def parse(cls, text: str) -> SkillDocument:
        """Parse a SKILL.md string into structured sections."""
        doc = cls()
        # Extract metadata from YAML frontmatter if present
        if text.startswith("---"):
            end = text.find("---", 3)
            if end != -1:
                frontmatter = text[3:end].strip()
                for line in frontmatter.splitlines():
                    if ":" in line:
                        key, _, val = line.partition(":")
                        doc.metadata[key.strip()] = val.strip()
                text = text[end + 3:].strip()

        # Split into sections by ## K/R/T/P headers
        splits = list(_HEADER_RE.finditer(text))
        if not splits:
            doc.preamble = text.strip()
            return doc

        doc.preamble = text[: splits[0].start()].strip()

        for i, match in enumerate(splits):
            section_key = match.group(1)
            start = match.end()
            end = splits[i + 1].start() if i + 1 < len(splits) else len(text)
            content = text[start:end].strip()
            # Strip the rest of the header line
            content = re.sub(r"^[^\n]*\n?", "", content, count=0)
            # Actually, the header line is already consumed by the match boundary
            # Just take everything after the header line
            header_end = text.find("\n", match.start())
            if header_end == -1:
                header_end = len(text)
            section_start = header_end + 1
            section_end = splits[i + 1].start() if i + 1 < len(splits) else len(text)
            setattr(doc, section_key, text[section_start:section_end].strip())

        return doc

    @classmethod
    def from_file(cls, path: str | Path) -> SkillDocument:
        return cls.parse(Path(path).read_text(encoding="utf-8"))

    # ---- Rendering ----

    def render(self) -> str:
        """Render the skill document back to Markdown."""
        parts: list[str] = []

        # Frontmatter
        if self.metadata:
            parts.append("---")
            for k, v in self.metadata.items():
                parts.append(f"{k}: {v}")
            parts.append("---")
            parts.append("")

        # Preamble
        if self.preamble:
            parts.append(self.preamble)
            parts.append("")

        # Sections in canonical order
        for key in _SECTION_ORDER:
            content = getattr(self, key, "")
            parts.append(_SECTION_HEADERS[key])
            parts.append("")
            if content:
                parts.append(content)
            parts.append("")

        return "\n".join(parts).rstrip() + "\n"

    def write(self, path: str | Path) -> None:
        Path(path).write_text(self.render(), encoding="utf-8")

    # ---- Section access ----

    def get_section(self, key: str) -> str:
        assert key in _SECTION_ORDER, f"Invalid section key: {key}"
        return getattr(self, key)

    def set_section(self, key: str, content: str) -> None:
        assert key in _SECTION_ORDER, f"Invalid section key: {key}"
        setattr(self, key, content)

    def append_to_section(self, key: str, content: str) -> None:
        current = self.get_section(key)
        if current:
            self.set_section(key, current + "\n\n" + content)
        else:
            self.set_section(key, content)

    # ---- Metrics ----

    def rule_count(self) -> int:
        """Count the number of rules (lines starting with - or numbered) across all sections."""
        count = 0
        for key in _SECTION_ORDER:
            for line in self.get_section(key).splitlines():
                stripped = line.strip()
                if stripped.startswith(("- ", "* ", "1.", "2.", "3.", "4.", "5.",
                                        "6.", "7.", "8.", "9.")):
                    count += 1
        return count

    def word_count(self) -> int:
        """Total word count across all sections."""
        total = ""
        for key in _SECTION_ORDER:
            total += " " + self.get_section(key)
        return len(total.split())

    def diff_lines(self, other: SkillDocument) -> int:
        """Count the number of differing lines between two skill documents."""
        lines_self = self.render().splitlines()
        lines_other = other.render().splitlines()
        # Simple line-level diff count
        max_len = max(len(lines_self), len(lines_other))
        diff = 0
        for i in range(max_len):
            a = lines_self[i] if i < len(lines_self) else ""
            b = lines_other[i] if i < len(lines_other) else ""
            if a != b:
                diff += 1
        return diff

    # ---- Seed templates ----

    @classmethod
    def create_seed(cls, benchmark: str, topology_family: str) -> SkillDocument:
        """Create a minimal seed SKILL.md for a given benchmark and topology."""
        return cls(
            preamble=f"# ABSTRAL SKILL Document\n\nBenchmark: {benchmark}\nSeed topology: {topology_family}",
            K=f"No domain knowledge acquired yet. Benchmark: {benchmark}.",
            R=(
                f"Default topology reasoning for {topology_family} architecture.\n\n"
                f"- Start with a {topology_family} topology.\n"
                f"- Adjust based on observed trace failures."
            ),
            T="No agent templates defined yet. Templates will emerge from trace analysis.",
            P=(
                "### Default Construction Protocol\n\n"
                "1. Read task description.\n"
                "2. Select topology from R section.\n"
                "3. Instantiate agent roles from T section.\n"
                "4. Wire agents according to topology.\n"
                "5. Populate system prompts with K section knowledge.\n"
                "6. Execute and capture traces."
            ),
            metadata={
                "version": "0",
                "benchmark": benchmark,
                "topology_family": topology_family,
                "iteration": "0",
            },
        )

    @classmethod
    def create_tau_bench_seed(
        cls, domain: str, topology_family: str, wiki_text: str
    ) -> "SkillDocument":
        """Create a seed SKILL.md pre-populated with τ-bench domain wiki.

        Unlike GAIA where K starts empty, τ-bench seeds have a rich K section
        containing the full policy document. This means the meta-agent's
        improvements focus on topology (R) and specialization (T).
        """
        return cls(
            preamble=(
                f"# ABSTRAL SKILL Document\n\n"
                f"Benchmark: tau_{domain}\n"
                f"Seed topology: {topology_family}"
            ),
            K=(
                f"## Domain Policy Knowledge\n\n"
                f"The following is the complete {domain} service policy that agents MUST follow:\n\n"
                f"{wiki_text}\n\n"
                f"## Mandatory Execution Workflow\n\n"
                f"Every customer interaction MUST follow this tool-calling sequence:\n"
                f"1. **LOGIN / IDENTIFY**: Call `find_user_id_by_name_zip` or equivalent to locate the customer.\n"
                f"2. **VERIFY IDENTITY**: Confirm the customer's identity by checking personal details against the database.\n"
                f"3. **CHECK CONSTRAINTS**: Before any modification, verify policy constraints "
                f"(e.g., booking class, change fees, cancellation windows) using the relevant lookup tools.\n"
                f"4. **EXECUTE ACTION**: Call the appropriate tool to make the requested change "
                f"(e.g., `update_reservation_flights`, `cancel_reservation`, `update_reservation_passengers`).\n"
                f"5. **CONFIRM & RESPOND**: Summarize what was done and ask if the customer needs anything else.\n\n"
                f"CRITICAL: You MUST actually call tools. Do NOT just describe what you would do. "
                f"Every successful task requires at least 2-3 tool calls."
            ),
            R=(
                f"Default topology reasoning for {topology_family} architecture.\n\n"
                f"- Start with a {topology_family} topology.\n"
                f"- The task domain is interactive customer service ({domain}).\n"
                f"- Agents must handle multi-turn conversations with tool calls.\n"
                f"- Adjust based on observed trace failures."
            ),
            T="No agent templates defined yet. Templates will emerge from trace analysis.",
            P=(
                f"### Construction Protocol for τ-bench {domain.title()} Tasks\n\n"
                f"1. Only ONE tool call at a time (no parallel tool calling).\n"
                f"2. Always verify customer identity before making changes.\n"
                f"3. Follow the policy rules in K section strictly.\n"
                f"4. Respond to the customer conversationally after completing actions.\n"
                f"5. Use the 'think' tool for intermediate reasoning.\n"
                f"6. Escalate to human agents via 'transfer_to_human_agents' when policy requires it."
            ),
            metadata={
                "version": "0",
                "benchmark": f"tau_{domain}",
                "topology_family": topology_family,
                "iteration": "0",
            },
        )

    @classmethod
    def create_sop_bench_seed(
        cls, domain: str, topology_family: str, system_prompt: str
    ) -> "SkillDocument":
        """Create a seed SKILL.md for SOPBench domain.

        SOPBench tasks require agents to follow Standard Operating Procedures
        with constraint verification. The system prompt contains the full
        role description and constraint rules.
        """
        return cls(
            preamble=(
                f"# ABSTRAL SKILL Document\n\n"
                f"Benchmark: sop_{domain}\n"
                f"Seed topology: {topology_family}"
            ),
            K=(
                f"## Domain Knowledge — {domain.title()} Service\n\n"
                f"{system_prompt}\n\n"
                f"## Operating Notes\n\n"
                f"- You MUST call tools to complete tasks. Do not just describe what you would do.\n"
                f"- Follow the constraint rules listed above for each action.\n"
                f"- Call `exit_conversation` when the task is complete or cannot be fulfilled."
            ),
            R=(
                f"Default topology reasoning for {topology_family} architecture.\n\n"
                f"- Start with a {topology_family} topology.\n"
                f"- SOPBench tasks follow sequential tool-calling workflows "
                f"(check prerequisites → execute action → exit).\n"
                f"- Tool agents handle customer interactions and execute tool calls.\n"
                f"- Non-tool agents (routers, verifiers) analyze requests and direct workflow.\n"
                f"- Adjust based on observed trace failures."
            ),
            T="No agent templates defined yet. Templates will emerge from trace analysis.",
            P=(
                f"### Construction Protocol for SOPBench {domain.title()} Tasks\n\n"
                f"1. Follow the constraint rules in the system prompt exactly.\n"
                f"2. Only ONE tool call at a time.\n"
                f"3. If a constraint check fails, decline the action and explain why.\n"
                f"4. Call exit_conversation when the request is completed or cannot be fulfilled.\n"
                f"5. Prefer FEWER agents (2-3). More agents = more routing overhead = fewer tool calls."
            ),
            metadata={
                "version": "0",
                "benchmark": f"sop_{domain}",
                "topology_family": topology_family,
                "iteration": "0",
            },
        )
