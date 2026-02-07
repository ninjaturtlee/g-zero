from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class BankRealGeneratorConfig:
    start_date: str = "2024-01-01"
    days: int = 365
    seed: int = 42

    # Base demand + weekly seasonality
    base: float = 8000.0
    weekly_amp: float = 1800.0

    # Noise: heteroscedastic (noise grows with demand level)
    noise_std_base: float = 500.0
    noise_std_scale: float = 0.06  # multiplied by demand level

    # Payday effects
    payday_days: tuple[int, ...] = (15, 30, 31)
    payday_boost: float = 0.35  # +35%

    # Holiday effects (synthetic list; can swap with real TR holidays later)
    # We model pre-holiday rush and post-holiday lull
    holiday_boost_pre: float = 0.55  # +55% before holiday
    holiday_lull_post: float = -0.20  # -20% after holiday
    pre_holiday_window: int = 2
    post_holiday_window: int = 1

    # Regime shifts (structural change)
    regime_shift_day: int = 220
    regime_multiplier: float = 1.20  # +20% after shift

    # Rare extreme events (festivals, outages, etc.)
    extreme_event_prob: float = 0.03
    extreme_event_min: float = 4000.0
    extreme_event_max: float = 12000.0


def default_holidays(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Synthetic holiday anchors (approximate) to create realistic pre-holiday spikes.
    Swap with real holiday calendars later if you want.
    """
    years = sorted(set(index.year))
    anchors = []
    for y in years:
        anchors += [
            f"{y}-01-01",  # New Year
            f"{y}-04-23",  # TR National Sovereignty/Children's Day
            f"{y}-05-01",  # Labor Day
            f"{y}-05-19",  # Atatürk Memorial/Youth & Sports Day
            f"{y}-07-15",  # Democracy and National Unity Day
            f"{y}-08-30",  # Victory Day
            f"{y}-10-29",  # Republic Day
        ]
    return pd.to_datetime(anchors)


def make_calendar_features(idx: pd.DatetimeIndex, holiday_idx: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.DataFrame(index=idx)
    df["dow"] = idx.dayofweek
    df["dom"] = idx.day
    df["month"] = idx.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    # Payday flags
    df["is_payday"] = df["dom"].isin([15, 30, 31]).astype(int)

    # Holiday proximity features
    holiday_set = set(pd.to_datetime(holiday_idx).date)
    df["is_holiday"] = [1 if d.date() in holiday_set else 0 for d in idx]

    # days_to_next_holiday
    holiday_dates = pd.to_datetime(sorted(list(holiday_set)))
    holiday_dates = pd.DatetimeIndex(holiday_dates)
    # For each day, find next holiday (simple loop; OK for 365 days)
    days_to_next = []
    for d in idx:
        future = holiday_dates[holiday_dates >= d.normalize()]
        if len(future) == 0:
            days_to_next.append(999)
        else:
            days_to_next.append(int((future[0] - d.normalize()).days))
    df["days_to_next_holiday"] = days_to_next

    return df


def generate_bank_real_series(cfg: BankRealGeneratorConfig) -> pd.Series:
    rng = np.random.default_rng(cfg.seed)
    idx = pd.date_range(cfg.start_date, periods=cfg.days, freq="D")

    t = np.arange(cfg.days)

    # Base + weekly seasonality
    weekly = cfg.weekly_amp * np.sin(2 * np.pi * t / 7.0)
    demand = cfg.base + weekly

    # Regime shift
    if 0 <= cfg.regime_shift_day < cfg.days:
        demand[cfg.regime_shift_day:] *= cfg.regime_multiplier

    # Calendar features
    hol = default_holidays(idx)
    cal = make_calendar_features(idx, hol)

    # Payday boost
    demand *= (1.0 + cfg.payday_boost * cal["is_payday"].values)

    # Holiday effect: spike before holiday, lull after
    is_hol = cal["is_holiday"].values.astype(bool)
    hol_positions = np.where(is_hol)[0]

    for h in hol_positions:
        # pre-holiday window (h-1, h-2, ...)
        for k in range(1, cfg.pre_holiday_window + 1):
            if h - k >= 0:
                demand[h - k] *= (1.0 + cfg.holiday_boost_pre * (1.0 - (k - 1) / cfg.pre_holiday_window))
        # post-holiday window (h+1, ...)
        for k in range(1, cfg.post_holiday_window + 1):
            if h + k < cfg.days:
                demand[h + k] *= (1.0 + cfg.holiday_lull_post)

    # Rare extreme events (additive spikes)
    event_mask = rng.random(cfg.days) < cfg.extreme_event_prob
    demand[event_mask] += rng.uniform(cfg.extreme_event_min, cfg.extreme_event_max, size=int(event_mask.sum()))

    # Heteroscedastic noise
    noise_std = cfg.noise_std_base + cfg.noise_std_scale * demand
    noise = rng.normal(0.0, noise_std, size=cfg.days)
    demand = demand + noise

    demand = np.clip(demand, 0.0, None)
    return pd.Series(demand, index=idx, name="demand")