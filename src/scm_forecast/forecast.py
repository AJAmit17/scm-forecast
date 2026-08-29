"""Forecasting layer: fits, per SKU, the single model `backtest.backtest_and_select`
chose for it (on that SKU's full history) and produces the per-period forecast curve.

Trend/seasonality-capable models (AutoETS, AutoARIMA) and intermittent-demand
specialists (CrostonOptimized, TSB, ADIDA) are both in the candidate pool for
every SKU; `backtest.py` decides the winner by held-out MAPE/MAE, not by demand-
pattern label alone. This is what prevents forcing a flat Croston-style rate
onto a SKU that a seasonal model would forecast more accurately.

For interval-capable models (AutoETS, AutoARIMA), the per-period std comes
from statsforecast's native 80% prediction interval. For Croston/TSB/ADIDA,
native intervals are not reliably available without extra history/config, so
variance is estimated analytically from the compound-Bernoulli decomposition
(Y = B * Z) computed in classify.py - this is necessarily a single constant
per SKU (these methods estimate one stationary rate, not a time-varying one).

Output schema (per unique_id x ds over the forecast horizon):
  unique_id, ds, forecast_mean, forecast_std, model_used
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsforecast import StatsForecast
from statsforecast.models import ADIDA, TSB, AutoARIMA, AutoETS, CrostonOptimized

Z_80 = float(norm.ppf(0.9))  # 80% interval -> +/- 1.2816 sigma

INTERVAL_CAPABLE_MODELS = {"AutoETS", "AutoARIMA"}

FORECAST_ROW_COLUMNS = ["unique_id", "ds", "forecast_mean", "forecast_std", "model_used"]


def season_length_for_freq(freq: str) -> int:
    f = freq.upper()
    if f.startswith("D"):
        return 7
    if f.startswith("W"):
        return 52
    if f.startswith("M"):
        return 12
    return 1


def build_candidate_models(season_length: int) -> list:
    """Model pool evaluated per SKU by `backtest.backtest_and_select`."""
    return [
        AutoETS(season_length=season_length, alias="AutoETS"),
        AutoARIMA(season_length=season_length, alias="AutoARIMA"),
        CrostonOptimized(alias="CrostonOptimized"),
        TSB(alpha_d=0.2, alpha_p=0.2, alias="TSB"),
        ADIDA(alias="ADIDA"),
    ]


def _compound_bernoulli_variance(p: pd.Series, mu_z: pd.Series, sigma_z: pd.Series) -> pd.Series:
    """Var(Y) for Y = B*Z, B~Bernoulli(p), Z independent with mean mu_z, std sigma_z."""
    return p * sigma_z**2 + p * (1 - p) * mu_z**2


def forecast_selected(
    long_df: pd.DataFrame,
    freq: str,
    horizon: int,
    stats_df: pd.DataFrame,
    selection_df: pd.DataFrame,
) -> pd.DataFrame:
    """Fit each SKU with the single model `selection_df` chose for it, on its full
    history (not the backtest train split), and return the per-period forecast.
    """
    if long_df.empty:
        return pd.DataFrame(columns=FORECAST_ROW_COLUMNS)

    season_length = season_length_for_freq(freq)
    models_by_name = {m.alias: m for m in build_candidate_models(season_length)}
    variance_lookup = stats_df.set_index("unique_id")[["p", "mu_z", "sigma_z"]]

    parts = []
    for model_name, group in selection_df.groupby("selected_model"):
        subset = long_df[long_df["unique_id"].isin(group["unique_id"])]
        if subset.empty:
            continue
        model = models_by_name[model_name]
        sf = StatsForecast(models=[model], freq=freq, n_jobs=1)

        if model_name in INTERVAL_CAPABLE_MODELS:
            fcst = sf.forecast(df=subset, h=horizon, level=[80])
            fcst["forecast_mean"] = fcst[model_name].clip(lower=0)
            fcst["forecast_std"] = (
                (fcst[f"{model_name}-hi-80"] - fcst[f"{model_name}-lo-80"]) / (2 * Z_80)
            ).clip(lower=0)
        else:
            fcst = sf.forecast(df=subset, h=horizon)
            fcst["forecast_mean"] = fcst[model_name].clip(lower=0)
            fcst = fcst.merge(variance_lookup, on="unique_id", how="left")
            variance = _compound_bernoulli_variance(fcst["p"], fcst["mu_z"], fcst["sigma_z"])
            fcst["forecast_std"] = np.sqrt(variance.clip(lower=0))

        fcst["model_used"] = model_name
        parts.append(fcst[FORECAST_ROW_COLUMNS])

    if not parts:
        return pd.DataFrame(columns=FORECAST_ROW_COLUMNS)
    return pd.concat(parts, ignore_index=True)


def zero_forecast(unique_ids: list[str], freq: str, horizon: int, last_dates: pd.Series) -> pd.DataFrame:
    """Placeholder horizon rows for SKUs with zero historical demand (no_demand group)."""
    rows = []
    for uid in unique_ids:
        start = last_dates.loc[uid] if uid in last_dates.index else pd.Timestamp.today()
        future = pd.date_range(start, periods=horizon + 1, freq=freq)[1:]
        for ds in future:
            rows.append(
                {
                    "unique_id": uid,
                    "ds": ds,
                    "forecast_mean": 0.0,
                    "forecast_std": 0.0,
                    "model_used": "no_demand",
                }
            )
    return pd.DataFrame(rows, columns=FORECAST_ROW_COLUMNS)


def summarize_forecast(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the per-period forecast horizon into one row per SKU (used only for
    the lead-time-demand distribution feeding EBO; the per-period curve itself is
    what should be shown/exported as "the forecast").

    `forecast_std` is averaged in variance space (RMS), consistent with
    treating each forecast period as an independent draw.
    """
    if forecast_df.empty:
        return pd.DataFrame(columns=["unique_id", "forecast_mean", "forecast_std", "model_used"])

    def _agg(g: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "forecast_mean": g["forecast_mean"].mean(),
                "forecast_std": np.sqrt((g["forecast_std"] ** 2).mean()),
                "model_used": g["model_used"].iloc[0],
            }
        )

    return forecast_df.groupby("unique_id", sort=False).apply(_agg, include_groups=False).reset_index()
