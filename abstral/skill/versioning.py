"""Git-based SKILL.md versioning with structured trace-cited commits."""

from __future__ import annotations

import json
from pathlib import Path

from git import Repo

from abstral.config import EvidenceClass
from abstral.skill.document import SkillDocument


class SkillRepository:
    """Manages SKILL.md versioning via GitPython."""

    SKILL_FILENAME = "SKILL.md"

    def __init__(self, repo_path: str | Path):
        self.repo_path = Path(repo_path)
        self.skill_path = self.repo_path / self.SKILL_FILENAME
        self._repo: Repo | None = None

    @property
    def repo(self) -> Repo:
        if self._repo is None:
            raise RuntimeError("Repository not initialized. Call init() or open().")
        return self._repo

    def init(self, seed: SkillDocument) -> Repo:
        """Initialize a new git repository with a seed SKILL.md."""
        self.repo_path.mkdir(parents=True, exist_ok=True)
        self._repo = Repo.init(self.repo_path)
        seed.write(self.skill_path)
        self.repo.index.add([self.SKILL_FILENAME])
        self.repo.index.commit("Initialize SKILL.md seed document")
        return self.repo

    def open(self) -> Repo:
        """Open an existing skill repository."""
        self._repo = Repo(self.repo_path)
        return self._repo

    def read(self) -> SkillDocument:
        """Read the current SKILL.md."""
        return SkillDocument.from_file(self.skill_path)

    def commit_update(
        self,
        doc: SkillDocument,
        iteration: int,
        ec_distribution: dict[str, int],
        trace_ids: list[str],
        rules_added: int = 0,
        message: str | None = None,
    ) -> str:
        """Commit an updated SKILL.md with structured metadata tag."""
        doc.metadata["iteration"] = str(iteration)
        doc.write(self.skill_path)

        self.repo.index.add([self.SKILL_FILENAME])

        if message is None:
            ec_summary = ", ".join(f"{k}={v}" for k, v in ec_distribution.items() if v > 0)
            message = f"iter-{iteration}: {ec_summary} | +{rules_added} rules"

        commit = self.repo.index.commit(message)

        # Tag with structured metadata
        tag_data = json.dumps({
            "iteration": iteration,
            "ec_distribution": ec_distribution,
            "rules_added": rules_added,
            "trace_ids": trace_ids[:20],  # cap at 20 for tag size
        })
        tag_name = f"iter-{iteration}"
        if tag_name in [t.name for t in self.repo.tags]:
            self.repo.delete_tag(tag_name)
        self.repo.create_tag(tag_name, message=tag_data)

        return str(commit.hexsha)

    def diff_stat(self) -> int:
        """Get the number of changed lines in the working tree vs HEAD."""
        if self.repo.head.is_valid():
            diff = self.repo.head.commit.diff(None, create_patch=True)
            total = 0
            for d in diff:
                if d.diff:
                    total += len(d.diff.decode("utf-8", errors="replace").splitlines())
            return total
        return 0

    def diff_between_iterations(self, iter_a: int, iter_b: int) -> int:
        """Count diff lines between two iteration tags."""
        tag_a = f"iter-{iter_a}"
        tag_b = f"iter-{iter_b}"
        tags = {t.name: t for t in self.repo.tags}
        if tag_a not in tags or tag_b not in tags:
            return -1
        diff = tags[tag_a].commit.diff(tags[tag_b].commit, create_patch=True)
        total = 0
        for d in diff:
            if d.diff:
                total += len(d.diff.decode("utf-8", errors="replace").splitlines())
        return total

    def get_history(self, max_count: int = 50) -> list[dict]:
        """Get commit history with parsed tag metadata."""
        history = []
        for commit in self.repo.iter_commits(max_count=max_count):
            entry = {
                "sha": str(commit.hexsha)[:8],
                "message": commit.message.strip(),
                "timestamp": commit.committed_datetime.isoformat(),
            }
            # Try to find matching tag
            for tag in self.repo.tags:
                if tag.commit == commit and tag.tag:
                    try:
                        entry["tag_data"] = json.loads(tag.tag.message)
                    except (json.JSONDecodeError, AttributeError):
                        pass
            history.append(entry)
        return history
