"""Layer 2: Convergence Detector.

Four independent convergence signals, plus a hard cap. Two signals (weight >= 3)
must fire to trigger termination. C4 triggers compaction first, then re-evaluates.

Signals:
  C1 (weight=2): Skill diff empty — diff lines < threshold
  C2 (weight=2): AUC plateau — |AUC_t - AUC_{t-1}| < epsilon for N consecutive iters
  C3 (weight=1): EC signal collapse — EC1+EC2 fraction < threshold
  C4 (weight=1): Complexity penalty — SKILL.md > max_rules or > max_words
  T  (override): Hard cap — iter == T_max
"""

from __future__ import annotations

import logging
from typing import Any

from abstral.config import (
    ABSTRALConfig,
    ConvergenceConfig,
    ConvergenceResult,
    ConvergenceSignal,
    EC_WEIGHTS,
)
from abstral.skill.document import SkillDocument

logger = logging.getLogger(__name__)


class ConvergenceDetector:
    """Monitors four convergence signals and decides when to stop the inner loop."""

    def __init__(self, config: ABSTRALConfig):
        self.config = config.convergence
        self.max_iter = config.inner_loop.max_iterations
        # History for plateau detection
        self._auc_history: list[float] = []
        self._ec_history: list[dict[str, int]] = []

    def reset(self) -> None:
        """Reset state for a new inner loop run."""
        self._auc_history.clear()
        self._ec_history.clear()

    def check(
        self,
        iteration: int,
        current_doc: SkillDocument,
        previous_doc: SkillDocument | None,
        auc: float,
        ec_distribution: dict[str, int],
    ) -> ConvergenceResult:
        """Check all convergence signals and return a decision.

        Args:
            iteration: Current inner-loop iteration number.
            current_doc: The current SKILL.md after UPDATE.
            previous_doc: The SKILL.md from the previous iteration (None if iter 0).
            auc: Current iteration's primary metric (AUC / success rate).
            ec_distribution: EC class counts from this iteration's ANALYZE.

        Returns:
            ConvergenceResult with signal statuses and decision.
        """
        self._auc_history.append(auc)
        self._ec_history.append(ec_distribution)

        signals: list[ConvergenceSignal] = []

        # ── C1: Skill diff empty ──
        c1_value = 0
        if previous_doc is not None:
            c1_value = current_doc.diff_lines(previous_doc)
        c1_fired = previous_doc is not None and c1_value < self.config.skill_diff_threshold
        signals.append(ConvergenceSignal(
            signal_id="C1",
            name="Skill diff empty",
            fired=c1_fired,
            weight=EC_WEIGHTS["C1"],
            value=c1_value,
            condition=f"diff_lines < {self.config.skill_diff_threshold}",
        ))

        # ── C2: AUC plateau ──
        c2_fired = False
        c2_value = "insufficient data"
        window = self.config.auc_plateau_window
        if len(self._auc_history) >= window + 1:
            recent_deltas = [
                abs(self._auc_history[-(i + 1)] - self._auc_history[-(i + 2)])
                for i in range(window)
            ]
            c2_fired = all(d < self.config.auc_plateau_epsilon for d in recent_deltas)
            c2_value = f"max_delta={max(recent_deltas):.4f}"
        signals.append(ConvergenceSignal(
            signal_id="C2",
            name="AUC plateau",
            fired=c2_fired,
            weight=EC_WEIGHTS["C2"],
            value=str(c2_value),
            condition=f"|AUC_t - AUC_{{t-1}}| < {self.config.auc_plateau_epsilon} for {window} iters",
        ))

        # ── C3: EC signal collapse ──
        total_ec = sum(ec_distribution.values())
        ec12_count = ec_distribution.get("EC1", 0) + ec_distribution.get("EC2", 0)
        ec12_frac = ec12_count / max(total_ec, 1)
        c3_fired = total_ec > 0 and ec12_frac < self.config.ec_collapse_threshold
        signals.append(ConvergenceSignal(
            signal_id="C3",
            name="EC signal collapse",
            fired=c3_fired,
            weight=EC_WEIGHTS["C3"],
            value=f"{ec12_frac:.2%}",
            condition=f"EC1+EC2 fraction < {self.config.ec_collapse_threshold:.0%}",
        ))

        # ── C4: Complexity penalty ──
        rule_count = current_doc.rule_count()
        word_count = current_doc.word_count()
        c4_fired = (rule_count > self.config.max_rules) or (word_count > self.config.max_words)
        signals.append(ConvergenceSignal(
            signal_id="C4",
            name="Complexity penalty",
            fired=c4_fired,
            weight=EC_WEIGHTS["C4"],
            value=f"rules={rule_count}, words={word_count}",
            condition=f"rules > {self.config.max_rules} OR words > {self.config.max_words}",
        ))

        # ── T: Hard cap ──
        t_fired = iteration >= self.max_iter
        signals.append(ConvergenceSignal(
            signal_id="T",
            name="Hard cap",
            fired=t_fired,
            weight=0,  # override, not weighted
            value=f"iter={iteration}/{self.max_iter}",
            condition=f"iter >= {self.max_iter}",
        ))

        # ── Decision logic ──
        total_weight = sum(s.weight for s in signals if s.fired and s.signal_id != "T")
        should_compact = c4_fired
        should_terminate = False
        reason = ""

        if t_fired:
            should_terminate = True
            reason = f"Hard cap reached: iteration {iteration} >= T_max={self.max_iter}"
        elif should_compact:
            # C4 triggers compaction first, not immediate termination
            # After compaction, caller should re-check C1 and C2
            reason = (
                f"Complexity penalty fired (rules={rule_count}, words={word_count}). "
                f"Compaction triggered. Will re-evaluate after compaction."
            )
            # Only terminate if other signals also support it
            non_c4_weight = sum(
                s.weight for s in signals
                if s.fired and s.signal_id not in ("T", "C4")
            )
            if non_c4_weight >= self.config.min_weight_to_terminate:
                should_terminate = True
                reason = f"Converged: weight={total_weight} (>= {self.config.min_weight_to_terminate}) + complexity penalty"
        elif total_weight >= self.config.min_weight_to_terminate:
            should_terminate = True
            fired_names = [s.signal_id for s in signals if s.fired]
            reason = f"Converged: total weight={total_weight} from signals {fired_names}"
        else:
            reason = f"Continuing: total weight={total_weight} < {self.config.min_weight_to_terminate}"

        logger.info(
            f"Convergence check iter={iteration}: "
            f"weight={total_weight}, terminate={should_terminate}, "
            f"compact={should_compact}. {reason}"
        )

        return ConvergenceResult(
            signals=signals,
            total_weight=total_weight,
            should_terminate=should_terminate,
            should_compact=should_compact,
            reason=reason,
        )
