from __future__ import annotations

from dataclasses import dataclass


class Policy:
    def decide_replenish_amount(self, cash: float, max_capacity: float) -> float:
        raise NotImplementedError


@dataclass(frozen=True)
class ReorderPointPolicy(Policy):
    """
    If cash <= reorder_point: refill up to target_level (or max_capacity if target_level is None).
    """
    reorder_point: float
    target_level: float | None = None

    def decide_replenish_amount(self, cash: float, max_capacity: float) -> float:
        cash = float(cash)
        max_capacity = float(max_capacity)
        if cash <= float(self.reorder_point):
            tgt = max_capacity if self.target_level is None else float(self.target_level)
            tgt = min(tgt, max_capacity)
            return max(tgt - cash, 0.0)
        return 0.0


@dataclass(frozen=True)
class TargetFillPolicy(Policy):
    """
    Refill to (target_frac * max_capacity) when cash <= reorder_point.
    """
    reorder_point: float
    target_frac: float

    def decide_replenish_amount(self, cash: float, max_capacity: float) -> float:
        cash = float(cash)
        max_capacity = float(max_capacity)
        if cash <= float(self.reorder_point):
            tgt = float(self.target_frac) * max_capacity
            tgt = min(tgt, max_capacity)
            return max(tgt - cash, 0.0)
        return 0.0


def reorder_point_policy(reorder_point: float, target_level: float | None = None) -> ReorderPointPolicy:
    return ReorderPointPolicy(float(reorder_point), target_level)


def target_fill_policy(reorder_point: float, target_frac: float) -> TargetFillPolicy:
    return TargetFillPolicy(float(reorder_point), float(target_frac))