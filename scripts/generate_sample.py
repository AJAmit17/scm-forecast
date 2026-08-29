"""Generate a synthetic monthly demand history matching the real ERP export schema
exactly: Item (SKU), YYMM (4-digit year-month code), YYQQ (derived quarter code,
unused by the pipeline but kept for fidelity), Actuals (quantity). Spans all four
SBC demand patterns (smooth, erratic, intermittent, lumpy) plus a strongly
trending+seasonal SKU, each with genuinely exploitable structure so the backtest
has real signal to find - not pure i.i.d. noise that is (correctly) unforecastable
beyond its flat average rate.

Each series generator takes its own independently-seeded RNG rather than sharing
one mutable stream: with a shared stream, editing an earlier generator silently
perturbs the random draws every later series consumes, changing their numeric
character (and therefore which model wins the backtest) with no code change to
that series itself. Independent seeds make each archetype reproducible in isolation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

N_MONTHS = 36  # 3 years of monthly history
START_YEAR, START_MONTH = 2023, 1


def _yymm_codes(n: int) -> list[str]:
    codes = []
    y, m = START_YEAR, START_MONTH
    for _ in range(n):
        codes.append(f"{y % 100:02d}{m:02d}")
        m += 1
        if m == 13:
            m = 1
            y += 1
    return codes


def _yyqq_codes(yymm_codes: list[str]) -> list[str]:
    out = []
    for code in yymm_codes:
        yy, mm = code[:2], int(code[2:])
        q = (mm - 1) // 3 + 1
        out.append(f"{yy}Q{q}")
    return out


YYMM_CODES = _yymm_codes(N_MONTHS)
YYQQ_CODES = _yyqq_codes(YYMM_CODES)


def trending_seasonal_series(seed: int) -> np.ndarray:
    """Clear upward trend + 12-month seasonal cycle: the case a flat forecast
    would be visibly wrong for, and what AutoETS/AutoARIMA should win on MAPE."""
    rng = np.random.default_rng(seed)
    trend = np.linspace(100, 300, N_MONTHS)
    season = 40 * np.sin(2 * np.pi * np.arange(N_MONTHS) / 12)
    noise = rng.normal(0, 10, N_MONTHS)
    return np.clip(trend + season + noise, 0, None).round()


def smooth_series(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    trend = np.linspace(200, 260, N_MONTHS)
    season = 30 * np.sin(2 * np.pi * np.arange(N_MONTHS) / 12)
    noise = rng.normal(0, 6, N_MONTHS)
    return np.clip(trend + season + noise, 0, None).round()


def erratic_series(seed: int) -> np.ndarray:
    """Erratic (ADI < 1.32, CV2 >= 0.49): active most periods with high size
    variability, but a STABLE base level - no secular trend. A trending base
    (e.g. reusing smooth_series) would let AutoETS correctly-but-unhelpfully
    extrapolate that trend unboundedly across an 18-period horizon; real
    erratic demand (highly variable magnitude around a roughly constant
    level) shouldn't secretly encode one.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(60, 8, N_MONTHS)
    spikes = rng.choice([0, 0, 0, 1], size=N_MONTHS) * rng.normal(150, 40, N_MONTHS)
    return np.clip(base + spikes, 0, None).round()


def intermittent_series(seed: int, seasonal_months: list[int], base_prob: float, size_lambda: float) -> np.ndarray:
    """Intermittent (ADI >= 1.32, CV2 < 0.49): mostly zero, with demand
    concentrated in a recurring seasonal window (e.g. year-end maintenance
    surge) plus sparse low-probability demand elsewhere. The seasonal window
    repeats every 12 months by construction, giving AutoETS/AutoARIMA
    (season_length=12) real exploitable signal via backtest - unlike pure
    i.i.d. random timing, which no model can forecast beyond its flat rate.
    """
    rng = np.random.default_rng(seed)
    y = np.zeros(N_MONTHS)
    for m in seasonal_months:
        y[m] = rng.poisson(size_lambda) + 1
    extra_hits = rng.random(N_MONTHS) < base_prob
    for i in range(N_MONTHS):
        if i not in seasonal_months and extra_hits[i]:
            y[i] = rng.poisson(size_lambda) + 1
    return y


def lumpy_series(
    seed: int,
    small_months: list[int],
    big_months: list[int],
    small_range: tuple[int, int],
    big_range: tuple[int, int],
) -> np.ndarray:
    """Lumpy (ADI >= 1.32, CV2 >= 0.49): recurring small routine orders plus
    rare, much larger orders (e.g. a periodic overhaul) - the size gap between
    the two clusters is what actually produces CV2 above the SBC cutoff. A
    uniform size range does NOT (CV2 ~ 0.15-0.25 even with sparse timing);
    genuinely bimodal magnitudes are what makes a series "lumpy" rather than
    just "intermittent". Both month lists repeat every 12 months, so this is
    exploitable seasonal signal, not unforecastable noise.
    """
    rng = np.random.default_rng(seed)
    y = np.zeros(N_MONTHS)
    for m in small_months:
        y[m] = rng.integers(*small_range)
    for m in big_months:
        y[m] = rng.integers(*big_range)
    return y


SKUS = [
    ("123", trending_seasonal_series(seed=1)),
    ("456", smooth_series(seed=2) * 0.6),
    ("789", erratic_series(seed=3)),
    ("321", intermittent_series(seed=4, seasonal_months=[10, 11, 22, 23, 34, 35], base_prob=0.08, size_lambda=12.0)),
    ("654", intermittent_series(seed=5, seasonal_months=[1, 13, 25], base_prob=0.12, size_lambda=22.0)),
    ("987", lumpy_series(seed=6, small_months=[2, 8, 14, 20, 26, 32], big_months=[5, 17, 29], small_range=(15, 40), big_range=(400, 800))),
    ("135", lumpy_series(seed=7, small_months=[4, 10, 16, 22, 28, 34], big_months=[7, 19, 31], small_range=(20, 60), big_range=(500, 1000))),
]


def main() -> None:
    rows = []
    for item, series in SKUS:
        for yymm, yyqq, qty in zip(YYMM_CODES, YYQQ_CODES, series):
            rows.append({"Item": item, "YYMM": yymm, "YYQQ": yyqq, "Actuals": round(float(qty), 2)})
    df = pd.DataFrame(rows)
    out = Path("data/sample_demand.xlsx")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out, index=False)
    print(f"wrote {out} ({len(df)} rows, {len(SKUS)} skus, {N_MONTHS} months each)")


if __name__ == "__main__":
    main()
