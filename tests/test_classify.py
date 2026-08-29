import numpy as np
import pandas as pd

from scm_forecast.classify import compute_sku_stats


def _series(uid: str, values: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=len(values), freq="D")
    return pd.DataFrame({"unique_id": uid, "ds": dates, "y": values})


def test_smooth_series_classified_smooth():
    rng = np.random.default_rng(0)
    values = list(np.clip(50 + rng.normal(0, 2, 120), 0, None))
    df = _series("SMOOTH", values)
    stats = compute_sku_stats(df)
    row = stats.iloc[0]
    assert row["category"] == "smooth"
    assert row["group"] == "regular"


def test_lumpy_series_classified_lumpy():
    rng = np.random.default_rng(1)
    n = 120
    demand_days = rng.random(n) < 0.05
    # Bimodal sizes (rare tiny orders vs. rare huge orders) give CV2 well above
    # the 0.49 lumpy cutoff; a uniform range does not (CV2 ~ 0.15 for U(20,100)).
    sizes = np.where(rng.random(n) < 0.5, 10.0, 400.0)
    values = list(np.where(demand_days, sizes, 0.0))
    df = _series("LUMPY", values)
    stats = compute_sku_stats(df)
    row = stats.iloc[0]
    assert row["category"] == "lumpy"
    assert row["group"] == "intermittent"


def test_no_demand_series():
    df = _series("DEAD", [0.0] * 60)
    stats = compute_sku_stats(df)
    row = stats.iloc[0]
    assert row["category"] == "no_demand"
    assert row["group"] == "no_demand"


def test_multiple_skus_grouped_independently():
    smooth = _series("A", list(np.full(60, 10.0)))
    dead = _series("B", [0.0] * 60)
    df = pd.concat([smooth, dead], ignore_index=True)
    stats = compute_sku_stats(df).set_index("unique_id")
    assert stats.loc["A", "group"] == "regular"
    assert stats.loc["B", "group"] == "no_demand"
