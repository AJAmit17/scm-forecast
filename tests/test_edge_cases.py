"""Regression coverage for pathological/real-world-messy inputs the pipeline must
survive without crashing: arbitrary-format SKU ids (alphanumeric, numeric-looking,
whitespace/special characters, empty strings), pathologically short per-SKU
history (down to a single data point, which historically crashed AutoETS with an
IndexError inside statsforecast), constant/degenerate demand, negative quantities,
and mixed SKU-column dtypes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scm_forecast.pipeline import run_pipeline
from scm_forecast.schema import ColumnMapping, PipelineConfig


def test_single_history_point_does_not_crash():
    """Regression: a SKU with exactly one recorded period used to crash the
    entire pipeline (statsforecast AutoETS IndexError), not just that SKU."""
    raw = pd.DataFrame({"sku": ["ONLY-1"], "date": [pd.Timestamp("2023-01-01")], "qty": [10.0]})
    mapping = ColumnMapping(sku="sku", date="date", qty="qty")
    config = PipelineConfig(freq="MS", horizon=6)

    outputs = run_pipeline(raw, mapping, config)

    assert len(outputs.forecast) == 6
    assert (outputs.forecast["forecast_qty"] >= 0).all()
    assert outputs.forecast["model_used"].iloc[0] != "AutoETS"  # never the crash-prone fallback


def test_alphanumeric_sku_formats_survive_end_to_end():
    dates = pd.date_range("2023-01-01", periods=24, freq="MS")
    rng = np.random.default_rng(11)
    skus = ["ABC-123", "SKU_007", "X.Y.Z-99", "0001", "  PAD-1  ", ""]
    expected_ids = {"ABC-123", "SKU_007", "X.Y.Z-99", "0001", "PAD-1", ""}  # whitespace stripped
    rows = []
    for sku in skus:
        for d in dates:
            rows.append({"sku": sku, "date": d, "qty": float(rng.integers(0, 40))})
    raw = pd.DataFrame(rows)

    outputs = run_pipeline(
        raw, ColumnMapping(sku="sku", date="date", qty="qty"), PipelineConfig(freq="MS", horizon=6)
    )
    assert set(outputs.forecast["unique_id"]) == expected_ids
    assert outputs.forecast["forecast_qty"].notna().all()
    assert np.isfinite(outputs.forecast["forecast_qty"]).all()


def test_numeric_looking_sku_column_read_as_int_by_pandas():
    """A pure-numeric Item column (e.g. all SKUs are "1234"-style codes) gets
    read by pandas as int64, not string - must still resolve to clean string ids."""
    dates = pd.date_range("2023-01-01", periods=24, freq="MS")
    rng = np.random.default_rng(4)
    raw = pd.DataFrame(
        {
            "sku": [1234] * 24 + [5678] * 24,
            "date": list(dates) * 2,
            "qty": list(rng.integers(0, 50, 24).astype(float)) * 2,
        }
    )
    outputs = run_pipeline(
        raw, ColumnMapping(sku="sku", date="date", qty="qty"), PipelineConfig(freq="MS", horizon=6)
    )
    assert set(outputs.forecast["unique_id"]) == {"1234", "5678"}


def test_mixed_dtype_sku_column_numeric_and_alphanumeric():
    dates = pd.date_range("2023-01-01", periods=20, freq="MS")
    rng = np.random.default_rng(6)
    raw = pd.DataFrame(
        {
            "sku": ([1234] * 20) + (["ABC-999"] * 20),
            "date": list(dates) * 2,
            "qty": list(rng.integers(0, 30, 20).astype(float)) * 2,
        }
    )
    outputs = run_pipeline(
        raw, ColumnMapping(sku="sku", date="date", qty="qty"), PipelineConfig(freq="MS", horizon=6)
    )
    assert set(outputs.forecast["unique_id"]) == {"1234", "ABC-999"}


def test_constant_zero_variance_demand_does_not_crash():
    dates = pd.date_range("2023-01-01", periods=36, freq="MS")
    raw = pd.DataFrame({"sku": ["FLAT-1"] * 36, "date": dates, "qty": [50.0] * 36})
    outputs = run_pipeline(
        raw, ColumnMapping(sku="sku", date="date", qty="qty"), PipelineConfig(freq="MS", horizon=6)
    )
    assert (outputs.forecast["forecast_qty"] == 50.0).all()


def test_negative_quantities_do_not_crash_ebo():
    """Data-entry corrections/returns can leave negative quantities in real
    exports; the pipeline must degrade gracefully, not raise."""
    dates = pd.date_range("2023-01-01", periods=24, freq="MS")
    rng = np.random.default_rng(5)
    raw = pd.DataFrame({"sku": ["NEG-1"] * 24, "date": dates, "qty": rng.integers(-5, 30, 24).astype(float)})
    outputs = run_pipeline(
        raw, ColumnMapping(sku="sku", date="date", qty="qty"), PipelineConfig(freq="MS", horizon=6)
    )
    assert np.isfinite(outputs.inventory["recommended_stock"]).all()
    assert np.isfinite(outputs.inventory["expected_backorders"]).all()


def test_many_singleton_skus_in_one_batch_isolate_independently():
    """Several SKUs each with only one data point, batched together with SKUs
    that have full history - a crash in one must not affect the others."""
    dates = pd.date_range("2023-01-01", periods=36, freq="MS")
    rng = np.random.default_rng(9)
    rows = []
    for i in range(5):
        rows.append({"sku": f"SOLO-{i}", "date": pd.Timestamp("2023-01-01"), "qty": float(i + 1)})
    for d in dates:
        rows.append({"sku": "FULL-HISTORY", "date": d, "qty": float(rng.integers(0, 50))})
    raw = pd.DataFrame(rows)

    outputs = run_pipeline(
        raw, ColumnMapping(sku="sku", date="date", qty="qty"), PipelineConfig(freq="MS", horizon=6)
    )
    assert len(set(outputs.forecast["unique_id"])) == 6
    assert np.isfinite(outputs.forecast["forecast_qty"]).all()


def test_budget_mode_with_zero_budget_does_not_divide_by_zero():
    dates = pd.date_range("2023-01-01", periods=12, freq="MS")
    raw = pd.DataFrame({"sku": ["ZERO-BUDGET"] * 12, "date": dates, "qty": [10.0] * 12})
    outputs = run_pipeline(
        raw,
        ColumnMapping(sku="sku", date="date", qty="qty"),
        PipelineConfig(freq="MS", horizon=6, mode="budget", budget=0.0),
    )
    assert (outputs.inventory["recommended_stock"] == 0).all()
