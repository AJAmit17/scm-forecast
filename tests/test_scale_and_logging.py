"""Coverage for the two other issues reported against the deployed app:

1. Vague/noisy failures - statsforecast's internal benign warnings (numpy
   divide-by-zero on short series, ARIMA convergence UserWarnings) must never
   leak to the console; real problems must be visible as structured log
   messages instead of a wall of framework internals.
2. Larger inputs (dozens-hundreds of SKUs, not just the tiny fixtures used
   elsewhere) must complete without raising and without re-parsing/re-ingesting
   the same data twice.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from scm_forecast.pipeline import run_pipeline
from scm_forecast.schema import ColumnMapping, PipelineConfig


def _build_multi_sku_raw(n_skus: int, n_periods: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n_periods, freq="MS")
    patterns = ["smooth", "erratic", "intermittent", "lumpy"]
    rows = []
    for i in range(n_skus):
        pattern = patterns[i % 4]
        if pattern == "smooth":
            vals = np.clip(np.linspace(50, 100, n_periods) + rng.normal(0, 5, n_periods), 0, None)
        elif pattern == "erratic":
            vals = np.clip(rng.normal(40, 30, n_periods), 0, None)
        elif pattern == "intermittent":
            mask = rng.random(n_periods) < 0.3
            vals = np.where(mask, rng.poisson(10, n_periods) + 1, 0).astype(float)
        else:
            mask = rng.random(n_periods) < 0.1
            vals = np.where(mask, rng.integers(50, 300, n_periods), 0).astype(float)
        for d, v in zip(dates, vals):
            rows.append({"sku": f"SKU-{i:04d}", "date": d, "qty": float(v)})
    return pd.DataFrame(rows)


def test_no_warnings_leak_out_of_model_fitting():
    """statsforecast's internal RuntimeWarning/UserWarning noise must be fully
    suppressed, in-process (n_jobs=1, the default everywhere in this app)."""
    raw = _build_multi_sku_raw(n_skus=20, n_periods=30, seed=1)
    mapping = ColumnMapping(sku="sku", date="date", qty="qty")
    config = PipelineConfig(freq="MS", horizon=6, n_jobs=1)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_pipeline(raw, mapping, config)

    leaked = [str(w.message) for w in caught if issubclass(w.category, (RuntimeWarning, UserWarning))]
    assert leaked == [], f"statsforecast warnings leaked out: {leaked[:5]}"


def test_pipeline_logs_structured_milestones(caplog):
    """Real diagnostic signal (row/SKU counts, stage timing, model selection)
    must be emitted as logging records, not just visible via print/stderr."""
    raw = _build_multi_sku_raw(n_skus=15, n_periods=24, seed=2)
    mapping = ColumnMapping(sku="sku", date="date", qty="qty")
    config = PipelineConfig(freq="MS", horizon=6, n_jobs=1)

    with caplog.at_level(logging.INFO, logger="scm_forecast"):
        run_pipeline(raw, mapping, config)

    messages = "\n".join(r.message for r in caplog.records)
    assert "raw row(s) read" in messages
    assert "distinct SKU(s)" in messages
    assert "Backtest completed in" in messages
    assert "Pipeline finished" in messages


def test_moderate_multi_sku_batch_completes_without_error():
    """60 SKUs x 30 months (1800 rows) exercising every candidate model across
    a real batch - the shape of input most likely to trigger the crash/warning
    issues that small single-SKU fixtures don't exercise."""
    raw = _build_multi_sku_raw(n_skus=60, n_periods=30, seed=3)
    mapping = ColumnMapping(sku="sku", date="date", qty="qty")
    config = PipelineConfig(freq="MS", horizon=12, n_jobs=1)

    outputs = run_pipeline(raw, mapping, config)

    assert outputs.inventory["unique_id"].nunique() == 60
    assert (outputs.forecast.groupby("unique_id").size() == 12).all()
    assert np.isfinite(outputs.forecast["forecast_qty"]).all()
    assert np.isfinite(outputs.inventory["recommended_stock"]).all()


def test_pipeline_outputs_expose_long_df_and_stats_without_reingesting():
    """PipelineOutputs must carry long_df/stats so callers (the Streamlit UI)
    never need to call prepare_long_frame/compute_sku_stats a second time."""
    raw = _build_multi_sku_raw(n_skus=5, n_periods=24, seed=4)
    mapping = ColumnMapping(sku="sku", date="date", qty="qty")
    config = PipelineConfig(freq="MS", horizon=6)

    outputs = run_pipeline(raw, mapping, config)

    assert set(outputs.long_df["unique_id"].unique()) == set(outputs.inventory["unique_id"])
    assert set(outputs.stats["unique_id"]) == set(outputs.inventory["unique_id"])
    assert "category" in outputs.stats.columns
