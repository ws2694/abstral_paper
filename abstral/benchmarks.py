"""Benchmark task loaders for ABSTRAL.

Each benchmark provides:
- Task instances loaded from real datasets (NOT synthetic)
- Ground-truth scoring (NOT LLM-as-judge)
- Deterministic splits with fixed seeds

Supported benchmarks:
  - GAIA: General AI reasoning (exact-match accuracy)
         Real dataset from HuggingFace: gaia-benchmark/GAIA
  - HotPotQA: Multi-hop question answering (exact-match + F1)
         Real dataset from HuggingFace: hotpot_qa
"""

from __future__ import annotations

import logging
import os
import random
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any

from abstral.models import TaskInstance

logger = logging.getLogger(__name__)

# Registry of benchmark loaders
_LOADERS: dict[str, type[BenchmarkLoader]] = {}


class BenchmarkLoader:
    """Base class for benchmark task loaders."""

    name: str = ""
    metric: str = ""

    def __init__(self, data_dir: str | Path | None = None):
        self.data_dir = Path(data_dir) if data_dir else None

    def load_tasks(
        self,
        split: str = "val",
        n_instances: int = 50,
        seed: int = 42,
    ) -> list[TaskInstance]:
        raise NotImplementedError

    def score(self, output: str, expected: str) -> float:
        """Score a single prediction against ground truth.
        Returns 0.0 or 1.0. Subclasses override for benchmark-specific matching."""
        if not output or not expected:
            return 0.0
        return 1.0 if _normalize_answer(output) == _normalize_answer(expected) else 0.0

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.name:
            _LOADERS[cls.name] = cls


def _normalize_answer(s: str) -> str:
    """Normalize answer string for comparison.
    Lowercases, strips whitespace/punctuation, collapses whitespace."""
    s = s.strip().lower()
    # Remove punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    s = " ".join(s.split())
    return s


def _normalize_number(s: str) -> str | None:
    """Try to parse a string as a number for numeric comparison."""
    s = s.strip().replace(",", "").replace("$", "").replace("%", "")
    try:
        return str(float(s))
    except ValueError:
        return None


class GAIALoader(BenchmarkLoader):
    """GAIA: General AI Assistants benchmark.

    Loads real tasks from HuggingFace: gaia-benchmark/GAIA (2023_all config).
    Each task has a ground-truth 'Final answer' for exact-match evaluation.
    Requires HF_TOKEN environment variable for gated dataset access.
    """
    name = "gaia"
    metric = "accuracy"

    def load_tasks(self, split="val", n_instances=50, seed=42) -> list[TaskInstance]:
        from datasets import load_dataset

        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise RuntimeError(
                "HF_TOKEN environment variable required for GAIA dataset access. "
                "Get a token at https://huggingface.co/settings/tokens"
            )

        hf_split = "validation" if split == "val" else split
        logger.info(f"Loading GAIA dataset (split={hf_split}) from HuggingFace...")

        ds = load_dataset(
            "gaia-benchmark/GAIA",
            "2023_all",
            split=hf_split,
            token=hf_token,
        )
        logger.info(f"Loaded {len(ds)} GAIA tasks")

        tasks = []
        for row in ds:
            level = row["Level"]
            # Skip tasks with file attachments for now — agents lack file access
            if row.get("file_name") and row["file_name"]:
                continue

            task_id = row["task_id"]
            question = row["Question"]
            answer = row["Final answer"]

            tasks.append(TaskInstance(
                id=task_id,
                input_text=question,
                expected_output=answer,
                task_type=f"level{level}",
                difficulty=f"level{level}",
                metadata={
                    "benchmark": "gaia",
                    "split": split,
                    "level": level,
                    "has_file": False,
                    "annotator_steps": row.get("Annotator Metadata", {}).get("Number of steps", ""),
                    "annotator_tools": row.get("Annotator Metadata", {}).get("Tools", ""),
                },
            ))

        logger.info(f"GAIA tasks without file attachments: {len(tasks)}")

        # Subsample if requested
        if n_instances and n_instances < len(tasks):
            rng = random.Random(seed)
            tasks = rng.sample(tasks, n_instances)
            logger.info(f"Subsampled to {len(tasks)} tasks (seed={seed})")

        # Log level distribution
        level_dist = Counter(t.difficulty for t in tasks)
        logger.info(f"Level distribution: {dict(sorted(level_dist.items()))}")

        return tasks

    def score(self, output: str, expected: str) -> float:
        """GAIA scoring: exact match after normalization, with numeric tolerance."""
        if not output or not expected:
            return 0.0

        # Normalized string match
        if _normalize_answer(output) == _normalize_answer(expected):
            return 1.0

        # Numeric comparison with tolerance
        out_num = _normalize_number(output)
        exp_num = _normalize_number(expected)
        if out_num is not None and exp_num is not None:
            try:
                if abs(float(out_num) - float(exp_num)) < 0.01:
                    return 1.0
            except (ValueError, OverflowError):
                pass

        return 0.0


class HotPotQALoader(BenchmarkLoader):
    """HotPotQA: Multi-hop question answering benchmark.

    Loads real tasks from HuggingFace: hotpot_qa (fullwiki config).
    Each task has a ground-truth answer for exact-match + F1 evaluation.
    Multi-hop reasoning tasks that benefit from multi-agent topologies.
    """
    name = "hotpotqa"
    metric = "exact_match"

    def load_tasks(self, split="val", n_instances=50, seed=42) -> list[TaskInstance]:
        from datasets import load_dataset

        hf_split = "validation" if split == "val" else split
        logger.info(f"Loading HotPotQA dataset (split={hf_split}) from HuggingFace...")

        ds = load_dataset("hotpot_qa", "fullwiki", split=hf_split)
        logger.info(f"Loaded {len(ds)} HotPotQA tasks")

        tasks = []
        for row in ds:
            # Include supporting context paragraphs
            context_parts = []
            if row.get("context"):
                titles = row["context"].get("title", [])
                sentences_list = row["context"].get("sentences", [])
                for title, sents in zip(titles, sentences_list):
                    context_parts.append(f"[{title}]: {''.join(sents)}")

            context_str = "\n\n".join(context_parts[:5])  # Limit context length

            tasks.append(TaskInstance(
                id=row["id"],
                input_text=(
                    f"Answer this question based on the provided context. "
                    f"Give only the answer, no explanation.\n\n"
                    f"Context:\n{context_str}\n\n"
                    f"Question: {row['question']}"
                ),
                expected_output=row["answer"],
                task_type=row.get("type", "unknown"),
                difficulty=row.get("level", "unknown"),
                metadata={
                    "benchmark": "hotpotqa",
                    "split": split,
                    "type": row.get("type", ""),
                    "level": row.get("level", ""),
                },
            ))

        # Subsample if requested
        if n_instances and n_instances < len(tasks):
            rng = random.Random(seed)
            tasks = rng.sample(tasks, n_instances)
            logger.info(f"Subsampled to {len(tasks)} tasks (seed={seed})")

        # Log type distribution
        type_dist = Counter(t.task_type for t in tasks)
        logger.info(f"Type distribution: {dict(sorted(type_dist.items()))}")

        return tasks

    def score(self, output: str, expected: str) -> float:
        """HotPotQA scoring: exact match after normalization."""
        if not output or not expected:
            return 0.0
        return 1.0 if _normalize_answer(output) == _normalize_answer(expected) else 0.0

    def f1_score(self, output: str, expected: str) -> float:
        """Token-level F1 score for HotPotQA."""
        pred_tokens = _normalize_answer(output).split()
        gold_tokens = _normalize_answer(expected).split()
        if not pred_tokens or not gold_tokens:
            return 0.0
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_common = sum(common.values())
        if num_common == 0:
            return 0.0
        precision = num_common / len(pred_tokens)
        recall = num_common / len(gold_tokens)
        return 2 * precision * recall / (precision + recall)


class TauBenchLoader(BenchmarkLoader):
    """τ-bench: Tool-Agent-User interaction benchmark (Sierra Research).

    Interactive customer-service tasks with ground-truth evaluation via
    database-state hash comparison. NO LLM judge.

    Domains: airline (50 test tasks), retail (train/dev/test splits).
    Evaluation is handled by env.calculate_reward() — not by string matching.
    """
    name = "tau_airline"
    metric = "task_success_rate"

    def __init__(self, data_dir=None, domain="airline"):
        super().__init__(data_dir)
        self.domain = domain

    def load_tasks(self, split="val", n_instances=50, seed=42) -> list[TaskInstance]:
        from abstral.tau_adapter import TauEnvManager

        manager = TauEnvManager(
            domain=self.domain,
            task_split="test",  # τ-bench airline only has test split
        )
        tau_tasks = manager.get_tasks()
        logger.info(f"Loaded {len(tau_tasks)} τ-bench {self.domain} tasks")

        tasks = []
        for i, task in enumerate(tau_tasks):
            tasks.append(TaskInstance(
                id=f"tau_{self.domain}_{i}",
                input_text=task.instruction,
                expected_output="",  # Scoring via DB hash, not string match
                task_type=self.domain,
                difficulty="standard",
                metadata={
                    "benchmark": f"tau_{self.domain}",
                    "task_index": i,
                    "domain": self.domain,
                    "user_id": task.user_id,
                    "n_ground_truth_actions": len(task.actions),
                    "has_outputs": len(task.outputs) > 0,
                },
            ))

        # Subsample if requested
        if n_instances and n_instances < len(tasks):
            rng = random.Random(seed)
            tasks = rng.sample(tasks, n_instances)
            logger.info(f"Subsampled to {len(tasks)} tasks (seed={seed})")

        return tasks

    def score(self, output: str, expected: str) -> float:
        """NOT used for τ-bench. Scoring goes through env.calculate_reward()."""
        raise NotImplementedError(
            "τ-bench uses env.calculate_reward() for DB-state evaluation, "
            "not string matching. Use TauBenchRunner instead."
        )


class TauBenchRetailLoader(TauBenchLoader):
    """τ-bench retail domain loader."""
    name = "tau_retail"

    def __init__(self, data_dir=None):
        super().__init__(data_dir, domain="retail")


class SOPBenchBankLoader(BenchmarkLoader):
    """SOPBench banking domain loader.

    134 tasks across 14 banking actions (apply_credit_card, transfer_funds, etc.)
    with oracle-based evaluation: 5 boolean criteria, all must pass.
    """
    name = "sop_bank"
    metric = "task_success_rate"

    def load_tasks(self, split="val", n_instances=50, seed=42) -> list[TaskInstance]:
        from abstral.sop_adapter import SOPEnvManager

        manager = SOPEnvManager(domain="bank")
        all_tasks = []
        for i, task_data in enumerate(manager.tasks):
            all_tasks.append(TaskInstance(
                id=f"sop_bank_{i}",
                input_text=task_data.get("user_prompt", task_data.get("user_instruction", "")),
                expected_output="",  # Scoring via oracle verifier
                task_type=task_data.get("user_goal", "unknown"),
                difficulty=f"constraints_{len(task_data.get('constraints', []))}",
                metadata={
                    "benchmark": "sop_bank",
                    "task_index": i,
                    "domain": "bank",
                    "user_goal": task_data.get("user_goal", ""),
                    "action_should_succeed": task_data.get("action_should_succeed", True),
                },
            ))

        # Train/test split: first 40 tasks (seed=42 shuffle) = val, rest = test
        if split in ("val", "test"):
            rng = random.Random(42)  # Fixed seed for reproducible split
            indices = list(range(len(all_tasks)))
            rng.shuffle(indices)
            val_indices = set(indices[:40])
            if split == "val":
                tasks = [all_tasks[i] for i in indices[:40]]
                logger.info(f"SOPBench bank val split: {len(tasks)} tasks")
            else:  # test
                tasks = [all_tasks[i] for i in indices[40:]]
                logger.info(f"SOPBench bank test split: {len(tasks)} tasks")
        else:
            tasks = all_tasks

        if n_instances and n_instances < len(tasks):
            rng = random.Random(seed)
            tasks = rng.sample(tasks, n_instances)
            logger.info(f"Subsampled to {len(tasks)} SOPBench bank tasks (seed={seed})")

        return tasks

    def score(self, output: str, expected: str) -> float:
        raise NotImplementedError("SOPBench uses oracle evaluation, not string matching.")


class SOPBenchHealthcareLoader(BenchmarkLoader):
    """SOPBench healthcare domain loader. 124 tasks across 10 healthcare actions."""
    name = "sop_healthcare"
    metric = "task_success_rate"

    def load_tasks(self, split="val", n_instances=50, seed=42) -> list[TaskInstance]:
        from abstral.sop_adapter import SOPEnvManager

        manager = SOPEnvManager(domain="healthcare")
        tasks = []
        for i, task_data in enumerate(manager.tasks):
            tasks.append(TaskInstance(
                id=f"sop_healthcare_{i}",
                input_text=task_data.get("user_prompt", task_data.get("user_instruction", "")),
                expected_output="",
                task_type=task_data.get("user_goal", "unknown"),
                difficulty=f"constraints_{len(task_data.get('constraints', []))}",
                metadata={
                    "benchmark": "sop_healthcare",
                    "task_index": i,
                    "domain": "healthcare",
                    "user_goal": task_data.get("user_goal", ""),
                    "action_should_succeed": task_data.get("action_should_succeed", True),
                },
            ))

        if n_instances and n_instances < len(tasks):
            rng = random.Random(seed)
            tasks = rng.sample(tasks, n_instances)
            logger.info(f"Subsampled to {len(tasks)} SOPBench healthcare tasks (seed={seed})")

        return tasks

    def score(self, output: str, expected: str) -> float:
        raise NotImplementedError("SOPBench uses oracle evaluation, not string matching.")


def get_loader(benchmark: str, data_dir: str | Path | None = None) -> BenchmarkLoader:
    """Get the benchmark loader for a given benchmark name."""
    if benchmark not in _LOADERS:
        available = ", ".join(_LOADERS.keys())
        raise ValueError(f"Unknown benchmark: {benchmark}. Available: {available}")
    return _LOADERS[benchmark](data_dir=data_dir)


def load_benchmark_tasks(
    benchmark: str,
    split: str = "val",
    n_instances: int = 50,
    seed: int = 42,
    data_dir: str | Path | None = None,
) -> list[TaskInstance]:
    """Convenience function to load tasks for a benchmark."""
    loader = get_loader(benchmark, data_dir)
    return loader.load_tasks(split=split, n_instances=n_instances, seed=seed)
