"""Per-SKU model selection via holdout backtesting.

Demand-pattern classification (classify.py) only proposes a candidate model
*pool*; it must not pin every SKU with some zero periods to a single flat-rate
model. Some "intermittent"-labelled SKUs still have exploitable trend/
seasonality, and a seasonal model can legitimately beat Croston's flat rate on
held-out error. This module fits every candidate model on a train/holdout
split per SKU, scores each by MAPE (mean absolute percentage error, computed
only over periods with non-zero actuals to avoid division by zero) with MAE
as a fallback when a SKU has no non-zero holdout periods, and returns the
single best-backtesting model per SKU plus its error for transparency.

Each candidate model is fit in its own StatsForecast call, wrapped in
try/except: some models (notably AutoETS) hard-crash instead of skipping a
pathological/degenerate series, and that must never take the other candidate
models' results down with it - a model that fails on this batch is simply
dropped from the comparison for this batch (logged as a warning).

SKUs with too little history for a meaningful holdout fall back to
CrostonOptimized (the only candidate verified safe on any history length,
including a single data point) with `backtest_mape = NaN` to make that
fallback explicit downstream.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsforecast import StatsForecast

from scm_forecast.forecast import (
    _suppress_model_fit_warnings,
    build_candidate_models,
    season_length_for_freq,
)
from scm_forecast.logging_config import get_logger

logger = get_logger(__name__)

MIN_TRAIN_PERIODS = 8
FALLBACK_MODEL = "CrostonOptimized"

SELECTION_COLUMNS = ["unique_id", "selected_model", "backtest_mape", "backtest_mae", "n_backtest_periods"]


def _holdout_length(freq: str, horizon: int) -> int:
    season_length = season_length_for_freq(freq)
    return max(3, min(horizon, season_length, 6))


def _fit_model_safe(train_df: pd.DataFrame, model, freq: str, h_eval: int, n_jobs: int = 1) -> pd.DataFrame | None:
    """Fit one candidate model on the holdout train split; None on any failure."""
    try:
        with _suppress_model_fit_warnings():
            sf = StatsForecast(models=[model], freq=freq, n_jobs=n_jobs)
            return sf.forecast(df=train_df, h=h_eval)
    except Exception as exc:
        logger.warning(
            "%s failed to fit the backtest holdout batch (%s: %s); dropped from this batch's comparison.",
            model.alias, type(exc).__name__, exc,
        )
        return None


def backtest_and_select(long_df: pd.DataFrame, freq: str, horizon: int, n_jobs: int = 1) -> pd.DataFrame:
    """One row per SKU: selected_model, backtest_mape, backtest_mae, n_backtest_periods."""
    if long_df.empty:
        return pd.DataFrame(columns=SELECTION_COLUMNS)

    h_eval = _holdout_length(freq, horizon)
    lengths = long_df.groupby("unique_id")["ds"].count()
    eligible_ids = lengths[lengths >= MIN_TRAIN_PERIODS + h_eval].index.tolist()
    logger.info(
        "Backtest: %d/%d SKU(s) have enough history (>= %d periods) for a %d-period holdout.",
        len(eligible_ids), len(lengths), MIN_TRAIN_PERIODS + h_eval, h_eval,
    )

    records: list[dict] = []
    if eligible_ids:
        elig_df = long_df[long_df["unique_id"].isin(eligible_ids)].sort_values(["unique_id", "ds"])
        train_parts, test_parts = [], []
        for _uid, g in elig_df.groupby("unique_id", sort=False):
            train_parts.append(g.iloc[:-h_eval])
            test_parts.append(g.iloc[-h_eval:])
        train_df = pd.concat(train_parts, ignore_index=True)
        test_df = pd.concat(test_parts, ignore_index=True)

        season_length = season_length_for_freq(freq)
        models = build_candidate_models(season_length)

        fcst = None
        available_names: list[str] = []
        for model in models:
            fcst_i = _fit_model_safe(train_df, model, freq, h_eval, n_jobs)
            if fcst_i is None:
                continue
            available_names.append(model.alias)
            fcst = fcst_i if fcst is None else fcst.merge(fcst_i, on=["unique_id", "ds"], how="outer")
        logger.info("Backtest: %d/%d candidate model(s) fit successfully: %s", len(available_names), len(models), available_names)

        if fcst is not None and available_names:
            scored = fcst.merge(test_df[["unique_id", "ds", "y"]], on=["unique_id", "ds"], how="inner")

            for uid, g in scored.groupby("unique_id", sort=False):
                actual = g["y"].to_numpy()
                nonzero = actual > 0
                mape_by_model: dict[str, float] = {}
                mae_by_model: dict[str, float] = {}
                for name in available_names:
                    pred = g[name].to_numpy()
                    mae_by_model[name] = float(np.mean(np.abs(actual - pred)))
                    if nonzero.any():
                        mape_by_model[name] = float(
                            np.mean(np.abs((actual[nonzero] - pred[nonzero]) / actual[nonzero])) * 100
                        )
                if mape_by_model:
                    best = min(mape_by_model, key=mape_by_model.get)
                    best_mape = mape_by_model[best]
                else:
                    best = min(mae_by_model, key=mae_by_model.get)
                    best_mape = np.nan
                records.append(
                    {
                        "unique_id": uid,
                        "selected_model": best,
                        "backtest_mape": best_mape,
                        "backtest_mae": mae_by_model[best],
                        "n_backtest_periods": h_eval,
                    }
                )

    result = pd.DataFrame(records, columns=SELECTION_COLUMNS)
    covered = set(result["unique_id"]) if not result.empty else set()
    missing = set(long_df["unique_id"].unique()) - covered
    if missing:
        logger.info(
            "Backtest: %d SKU(s) fall back to %s (too little history to backtest).",
            len(missing), FALLBACK_MODEL,
        )
        fallback = [
            {
                "unique_id": uid,
                "selected_model": FALLBACK_MODEL,
                "backtest_mape": np.nan,
                "backtest_mae": np.nan,
                "n_backtest_periods": 0,
            }
            for uid in missing
        ]
        result = pd.concat([result, pd.DataFrame(fallback, columns=SELECTION_COLUMNS)], ignore_index=True)
    return result
