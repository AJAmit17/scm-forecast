"""Generate a synthetic monthly demand history matching the real ERP export schema
exactly: Item (SKU), YYMM (4-digit year-month code), YYQQ (derived quarter code,
unused by the pipeline but kept for fidelity), Actuals (quantity). Spans all four
SBC demand patterns plus a strongly trending+seasonal SKU to exercise non-flat
forecasting.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
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


def trending_seasonal_series() -> np.ndarray:
    """Clear upward trend + 12-month seasonal cycle: the case a flat forecast
    would be visibly wrong for, and what AutoETS/AutoARIMA should win on MAPE."""
    trend = np.linspace(100, 300, N_MONTHS)
    season = 40 * np.sin(2 * np.pi * np.arange(N_MONTHS) / 12)
    noise = rng.normal(0, 10, N_MONTHS)
    return np.clip(trend + season + noise, 0, None).round()


def smooth_series() -> np.ndarray:
    trend = np.linspace(200, 260, N_MONTHS)
    season = 20 * np.sin(2 * np.pi * np.arange(N_MONTHS) / 12)
    noise = rng.normal(0, 8, N_MONTHS)
    return np.clip(trend + season + noise, 0, None).round()


def erratic_series() -> np.ndarray:
    base = smooth_series() * 0.3
    spikes = rng.choice([0, 0, 0, 1], size=N_MONTHS) * rng.normal(150, 40, N_MONTHS)
    return np.clip(base + spikes, 0, None).round()


def intermittent_series(demand_prob: float = 0.35, size_lambda: float = 15.0) -> np.ndarray:
    demand_months = rng.random(N_MONTHS) < demand_prob
    sizes = rng.poisson(size_lambda, N_MONTHS) + 1
    return np.where(demand_months, sizes, 0).astype(float)


def lumpy_series(demand_prob: float = 0.2, size_low: int = 50, size_high: int = 400) -> np.ndarray:
    demand_months = rng.random(N_MONTHS) < demand_prob
    sizes = rng.integers(size_low, size_high, N_MONTHS)
    return np.where(demand_months, sizes, 0).astype(float)


SKUS = [
    ("123", trending_seasonal_series()),
    ("456", smooth_series() * 0.6),
    ("789", erratic_series()),
    ("321", intermittent_series()),
    ("654", intermittent_series(0.45, 25.0)),
    ("987", lumpy_series()),
    ("135", lumpy_series(0.12, 100, 600)),
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
