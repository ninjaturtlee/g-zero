from dataclasses import dataclass


@dataclass(frozen=True)
class ReorderPointPolicy:
    reorder_point: float
    target_level: float

    def decide_replenish_amount(self, cash_level: float, max_capacity: float) -> float:
        if cash_level <= self.reorder_point:
            return max(0.0, min(self.target_level, max_capacity) - cash_level)
        return 0.0