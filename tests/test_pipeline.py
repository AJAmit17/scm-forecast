import numpy as np
import pandas as pd

from scm_forecast.pipeline import run_pipeline
from scm_forecast.schema import ColumnMapping, PipelineConfig


def _build_raw() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2025-01-01", periods=90, freq="D")

    smooth = np.clip(30 + 3 * np.sin(np.arange(90) / 7) + rng.normal(0, 1, 90), 0, None)

    demand_days = rng.random(90) < 0.15
    sizes = rng.poisson(5, 90) + 1
    lumpy = np.where(demand_days, sizes, 0).astype(float)

    rows = []
    for d, q in zip(dates, smooth):
        rows.append({"sku": "REG-1", "date": d, "qty": q, "lead_time_days": 10, "unit_cost": 5.0, "current_stock": 50})
    for d, q in zip(dates, lumpy):
        rows.append({"sku": "INT-1", "date": d, "qty": q, "lead_time_days": 20, "unit_cost": 40.0, "current_stock": 5})
    return pd.DataFrame(rows)


def _mapping_with_attrs() -> ColumnMapping:
    return ColumnMapping(
        sku="sku", date="date", qty="qty",
        lead_time="lead_time_days", unit_cost="unit_cost", current_stock="current_stock",
    )


def test_pipeline_service_level_mode_end_to_end():
    raw = _build_raw()
    config = PipelineConfig(freq="D", horizon=14, service_level=0.9, mode="service_level")

    outputs = run_pipeline(raw, _mapping_with_attrs(), config)
    inventory = outputs.inventory

    assert set(inventory["unique_id"]) == {"REG-1", "INT-1"}
    assert (inventory["recommended_stock"] >= 0).all()
    assert (inventory["achieved_fill_rate"] >= 0.9 - 1e-6).all()
    # forecast table has one row per SKU per forecast period
    assert set(outputs.forecast["unique_id"]) == {"REG-1", "INT-1"}
    assert (outputs.forecast.groupby("unique_id").size() == 14).all()
    assert "backtest_mape" in outputs.forecast.columns


def test_pipeline_budget_mode_respects_budget():
    raw = _build_raw()
    config = PipelineConfig(freq="D", horizon=14, mode="budget", budget=500.0)

    outputs = run_pipeline(raw, _mapping_with_attrs(), config)
    spend = (outputs.inventory["recommended_stock"] * outputs.inventory["unit_cost"]).sum()
    assert spend <= 500.0 + 1e-6


def test_pipeline_without_optional_attribute_columns_uses_defaults():
    raw = _build_raw().drop(columns=["lead_time_days", "unit_cost", "current_stock"])
    mapping = ColumnMapping(sku="sku", date="date", qty="qty")
    config = PipelineConfig(freq="D", horizon=14, lead_time_default=21.0, unit_cost_default=2.0)

    outputs = run_pipeline(raw, mapping, config)
    inventory = outputs.inventory

    assert set(inventory["unique_id"]) == {"REG-1", "INT-1"}
    assert (inventory["lead_time_days"] == 21.0).all()
    assert (inventory["unit_cost"] == 2.0).all()
    assert (inventory["current_stock"] == 0.0).all()


def test_pipeline_does_not_flatten_a_genuinely_trending_seasonal_sku():
    """Regression guard for the flat-forecast bug: a SKU with a clear trend and
    seasonal cycle must come back with a forecast curve that actually varies
    across the horizon, not the same value repeated for every period."""
    n = 36
    dates = pd.date_range("2023-01-01", periods=n, freq="MS")
    trend = np.linspace(100, 300, n)
    season = 40 * np.sin(2 * np.pi * np.arange(n) / 12)
    rng = np.random.default_rng(3)
    values = np.clip(trend + season + rng.normal(0, 5, n), 0, None)
    raw = pd.DataFrame({"sku": ["TREND-1"] * n, "date": dates, "qty": values})

    mapping = ColumnMapping(sku="sku", date="date", qty="qty")
    config = PipelineConfig(freq="MS", horizon=18)

    outputs = run_pipeline(raw, mapping, config)
    curve = outputs.forecast.sort_values("ds")["forecast_qty"].to_numpy()

    assert len(curve) == 18
    assert curve.std() > 1.0  # not a flat repeated constant
    assert curve.max() - curve.min() > 10.0


def test_backtest_mape_reported_for_pipeline_output():
    raw = _build_raw()
    config = PipelineConfig(freq="D", horizon=14)

    outputs = run_pipeline(raw, _mapping_with_attrs(), config)
    # 90 daily periods is enough history for the backtest holdout to run
    assert outputs.inventory["backtest_mape"].notna().any()


def test_sparse_seasonal_lumpy_sku_is_classified_lumpy_and_not_flattened():
    """Regression guard: a SKU with a genuine recurring annual spike (e.g. a
    once-a-year overhaul) amid mostly-zero months, with bimodal order sizes
    (small routine orders vs. a rare huge one), must (a) actually land in the
    SBC 'lumpy' bucket - a uniform size range does NOT (CV2 stays too low) -
    and (b) not get flattened by an over-wide backtest holdout that starves
    training data below what AutoETS/AutoARIMA need to detect the pattern."""
    n = 36
    dates = pd.date_range("2023-01-01", periods=n, freq="MS")
    rng = np.random.default_rng(11)
    y = np.zeros(n)
    for m in [2, 8, 14, 20, 26, 32]:  # small recurring routine orders
        y[m] = rng.integers(15, 40)
    for m in [5, 17, 29]:  # rare large overhaul orders, annual cycle
        y[m] = rng.integers(400, 800)
    raw = pd.DataFrame({"sku": ["LUMPY-SEASONAL-1"] * n, "date": dates, "qty": y})

    mapping = ColumnMapping(sku="sku", date="date", qty="qty")
    config = PipelineConfig(freq="MS", horizon=18)

    outputs = run_pipeline(raw, mapping, config)

    assert outputs.inventory["category"].iloc[0] == "lumpy"
    curve = outputs.forecast.sort_values("ds")["forecast_qty"].to_numpy()
    assert curve.std() > 1.0, "forecast was flattened despite genuine recurring seasonal signal"
