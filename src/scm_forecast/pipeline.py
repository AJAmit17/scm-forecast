"""End-to-end orchestration: raw DataFrame -> classified -> backtested model
selection -> per-period forecast -> EBO.

Returns both the granular per-SKU-per-period forecast (the actual "18 months
forecast for all SKUs" deliverable - a real curve, not a flattened repeat of
one averaged number) and the SKU-level EBO/inventory-recommendation table
derived from it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from scm_forecast.backtest import backtest_and_select
from scm_forecast.classify import compute_sku_stats
from scm_forecast.ebo import optimize_stock_budget, recommend_stock_service_level
from scm_forecast.forecast import forecast_selected, summarize_forecast, zero_forecast
from scm_forecast.ingest import prepare_long_frame
from scm_forecast.schema import ColumnMapping, PipelineConfig

Z_80 = float(norm.ppf(0.9))

FORECAST_COLUMNS = [
    "unique_id",
    "ds",
    "category",
    "model_used",
    "forecast_qty",
    "forecast_lower_80",
    "forecast_upper_80",
    "backtest_mape",
    "backtest_mae",
]

INVENTORY_COLUMNS = [
    "unique_id",
    "category",
    "model_used",
    "backtest_mape",
    "forecast_mean_per_period",
    "forecast_std_per_period",
    "lead_time_days",
    "mean_lead_time_demand",
    "std_lead_time_demand",
    "unit_cost",
    "current_stock",
    "recommended_stock",
    "expected_backorders",
    "achieved_fill_rate",
    "delta_vs_current",
]

_DAYS_PER_PERIOD = {"D": 1.0, "W": 7.0, "M": 30.4375, "MS": 30.4375}


def lead_time_periods(lead_time_days: float, freq: str) -> float:
    days_per_period = _DAYS_PER_PERIOD.get(freq.upper(), _DAYS_PER_PERIOD.get(freq[0].upper(), 1.0))
    return max(lead_time_days, 0.0) / days_per_period


@dataclass
class PipelineOutputs:
    forecast: pd.DataFrame  # per SKU x period: the actual forecast curve
    inventory: pd.DataFrame  # per SKU: EBO-based stock recommendation


def run_pipeline(raw: pd.DataFrame, mapping: ColumnMapping, config: PipelineConfig) -> PipelineOutputs:
    long_df, attrs = prepare_long_frame(raw, mapping, config.freq)
    stats_df = compute_sku_stats(long_df)

    no_demand_ids = list(stats_df.loc[stats_df["group"] == "no_demand", "unique_id"])
    active_long_df = long_df[~long_df["unique_id"].isin(no_demand_ids)]

    parts = []
    if not active_long_df.empty:
        selection_df = backtest_and_select(active_long_df, config.freq, config.horizon, stats_df)
        parts.append(forecast_selected(active_long_df, config.freq, config.horizon, stats_df, selection_df))
    else:
        selection_df = pd.DataFrame(
            columns=["unique_id", "selected_model", "backtest_mape", "backtest_mae", "n_backtest_periods"]
        )

    if no_demand_ids:
        last_dates = long_df.groupby("unique_id")["ds"].max()
        parts.append(zero_forecast(no_demand_ids, config.freq, config.horizon, last_dates))

    forecast_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    error_cols = selection_df[["unique_id", "backtest_mape", "backtest_mae"]]

    per_period = forecast_df.merge(stats_df[["unique_id", "category"]], on="unique_id", how="left")
    per_period = per_period.merge(error_cols, on="unique_id", how="left")
    per_period["forecast_qty"] = per_period["forecast_mean"]
    per_period["forecast_lower_80"] = (per_period["forecast_mean"] - Z_80 * per_period["forecast_std"]).clip(lower=0)
    per_period["forecast_upper_80"] = per_period["forecast_mean"] + Z_80 * per_period["forecast_std"]
    forecast_table = (
        per_period[FORECAST_COLUMNS].sort_values(["unique_id", "ds"]).reset_index(drop=True)
    )

    summary = summarize_forecast(forecast_df)
    merged = summary.merge(stats_df[["unique_id", "category"]], on="unique_id", how="left")
    merged = merged.merge(error_cols, on="unique_id", how="left")
    merged = merged.merge(attrs, on="unique_id", how="left")

    if "lead_time_days" not in merged.columns:
        merged["lead_time_days"] = np.nan
    if "unit_cost" not in merged.columns:
        merged["unit_cost"] = np.nan
    if "current_stock" not in merged.columns:
        merged["current_stock"] = np.nan

    merged["lead_time_days"] = merged["lead_time_days"].astype(float).fillna(config.lead_time_default)
    merged["unit_cost"] = merged["unit_cost"].astype(float).fillna(config.unit_cost_default)
    merged["current_stock"] = merged["current_stock"].astype(float).fillna(0.0)

    merged["lt_periods"] = merged["lead_time_days"].apply(lambda d: lead_time_periods(d, config.freq))
    merged["mean_ltd"] = merged["forecast_mean"] * merged["lt_periods"]
    merged["var_ltd"] = (merged["forecast_std"] ** 2) * merged["lt_periods"]

    if config.mode == "budget" and config.budget is not None:
        result = optimize_stock_budget(merged, config.budget)
    else:
        result = recommend_stock_service_level(merged, config.service_level)

    result["delta_vs_current"] = result["recommended_stock"] - result["current_stock"]
    result = result.rename(
        columns={
            "forecast_mean": "forecast_mean_per_period",
            "forecast_std": "forecast_std_per_period",
            "mean_ltd": "mean_lead_time_demand",
            "var_ltd": "std_lead_time_demand",
        }
    )
    result["std_lead_time_demand"] = result["std_lead_time_demand"].apply(lambda v: v**0.5 if v is not None else v)
    inventory_table = result[INVENTORY_COLUMNS].sort_values("unique_id").reset_index(drop=True)

    return PipelineOutputs(forecast=forecast_table, inventory=inventory_table)


def run_pipeline_from_path(input_path: str, mapping: ColumnMapping, config: PipelineConfig) -> PipelineOutputs:
    from scm_forecast.ingest import read_input

    raw = read_input(input_path)
    return run_pipeline(raw, mapping, config)
