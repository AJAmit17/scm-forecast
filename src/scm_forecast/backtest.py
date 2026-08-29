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

SKUs with too little history for a meaningful holdout fall back to a
demand-pattern heuristic (AutoETS for "regular", CrostonOptimized for
"intermittent") with `backtest_mape = NaN` to make that explicit downstream.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsforecast import StatsForecast

from scm_forecast.forecast import build_candidate_models, season_length_for_freq

MIN_TRAIN_PERIODS = 8

SELECTION_COLUMNS = ["unique_id", "selected_model", "backtest_mape", "backtest_mae", "n_backtest_periods"]


def _holdout_length(freq: str, horizon: int) -> int:
    season_length = season_length_for_freq(freq)
    return max(3, min(horizon, season_length, 6))


def backtest_and_select(long_df: pd.DataFrame, freq: str, horizon: int, stats_df: pd.DataFrame) -> pd.DataFrame:
    """One row per SKU: selected_model, backtest_mape, backtest_mae, n_backtest_periods."""
    if long_df.empty:
        return pd.DataFrame(columns=SELECTION_COLUMNS)

    h_eval = _holdout_length(freq, horizon)
    lengths = long_df.groupby("unique_id")["ds"].count()
    eligible_ids = lengths[lengths >= MIN_TRAIN_PERIODS + h_eval].index.tolist()

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
        model_names = [m.alias for m in models]
        sf = StatsForecast(models=models, freq=freq, n_jobs=1)
        fcst = sf.forecast(df=train_df, h=h_eval)
        scored = fcst.merge(test_df[["unique_id", "ds", "y"]], on=["unique_id", "ds"], how="inner")

        for uid, g in scored.groupby("unique_id", sort=False):
            actual = g["y"].to_numpy()
            nonzero = actual > 0
            mape_by_model: dict[str, float] = {}
            mae_by_model: dict[str, float] = {}
            for name in model_names:
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
        group_lookup = stats_df.set_index("unique_id")["group"]
        fallback = [
            {
                "unique_id": uid,
                "selected_model": "AutoETS" if group_lookup.get(uid) == "regular" else "CrostonOptimized",
                "backtest_mape": np.nan,
                "backtest_mae": np.nan,
                "n_backtest_periods": 0,
            }
            for uid in missing
        ]
        result = pd.concat([result, pd.DataFrame(fallback, columns=SELECTION_COLUMNS)], ignore_index=True)
    return result
