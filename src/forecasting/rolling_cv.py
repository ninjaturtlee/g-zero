from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RollingCVConfig:
    n_splits: int = 3
    val_size: int = 60           # days per validation fold
    min_train_size: int = 180    # minimum training days before first fold


def rolling_time_splits(n: int, cfg: RollingCVConfig):
    """
    Expanding-window rolling splits.
    Returns list of (train_idx, val_idx) for arrays of length n.
    """
    if n < cfg.min_train_size + cfg.val_size:
        raise ValueError(f"Not enough rows for rolling CV: n={n} < {cfg.min_train_size}+{cfg.val_size}")

    splits = []
    # place folds near the end to mimic real deployment
    # last fold validates on the last val_size samples
    end = n
    for k in range(cfg.n_splits):
        val_end = end - k * cfg.val_size
        val_start = val_end - cfg.val_size
        train_end = val_start
        train_start = 0

        if train_end < cfg.min_train_size:
            break

        tr = np.arange(train_start, train_end)
        va = np.arange(val_start, val_end)
        splits.append((tr, va))

    splits = list(reversed(splits))  # chronological order
    if len(splits) == 0:
        raise ValueError("No valid splits produced. Reduce min_train_size or n_splits.")
    return splits
    