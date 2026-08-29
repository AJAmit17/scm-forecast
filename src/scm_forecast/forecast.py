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

import contextlib
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsforecast import StatsForecast
from statsforecast.models import ADIDA, TSB, AutoARIMA, AutoETS, CrostonOptimized

from scm_forecast.logging_config import get_logger

logger = get_logger(__name__)

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
    """Model pool evaluated per SKU by `backtest.backtest_and_select`.

    `AutoARIMA(approximation=True)` uses a faster approximate likelihood for
    order search - the dominant cost in the whole pipeline (~1.4x faster per
    SKU, measured); the resulting order selection is occasionally a notch
    less precise, an accepted tradeoff for batch forecasting at scale.
    """
    return [
        AutoETS(season_length=season_length, alias="AutoETS"),
        AutoARIMA(season_length=season_length, alias="AutoARIMA", approximation=True),
        CrostonOptimized(alias="CrostonOptimized"),
        TSB(alpha_d=0.2, alpha_p=0.2, alias="TSB"),
        ADIDA(alias="ADIDA"),
    ]


def _compound_bernoulli_variance(p: pd.Series, mu_z: pd.Series, sigma_z: pd.Series) -> pd.Series:
    """Var(Y) for Y = B*Z, B~Bernoulli(p), Z independent with mean mu_z, std sigma_z."""
    return p * sigma_z**2 + p * (1 - p) * mu_z**2


def _naive_forecast(long_df: pd.DataFrame, freq: str, horizon: int) -> pd.DataFrame:
    """Bulletproof fallback: per-SKU historical mean/std repeated flat over the
    horizon. No statsforecast dependency, cannot fail on any numeric input
    (including a single data point). Used only when every attempt to fit a
    candidate model to a SKU has failed, so that one pathological/degenerate
    SKU degrades gracefully instead of crashing the whole run.
    """
    if long_df.empty:
        return pd.DataFrame(columns=FORECAST_ROW_COLUMNS)

    agg = long_df.groupby("unique_id")["y"].agg(["mean", "std", "count"]).reset_index()
    agg["std"] = agg["std"].fillna(0.0)
    agg.loc[agg["count"] < 2, "std"] = 0.0
    last_dates = long_df.groupby("unique_id")["ds"].max()

    rows = []
    for row in agg.itertuples(index=False):
        start = last_dates.loc[row.unique_id]
        future = pd.date_range(start, periods=horizon + 1, freq=freq)[1:]
        for ds in future:
            rows.append(
                {
                    "unique_id": row.unique_id,
                    "ds": ds,
                    "forecast_mean": max(row.mean, 0.0),
                    "forecast_std": max(row.std, 0.0),
                    "model_used": "NaiveMean",
                }
            )
    return pd.DataFrame(rows, columns=FORECAST_ROW_COLUMNS)


@contextlib.contextmanager
def _suppress_model_fit_warnings():
    """Suppress statsforecast's internal benign warnings (RuntimeWarning from
    numpy divide-by-zero on short series, UserWarning from ARIMA optimizer
    convergence, etc. - none of them indicate the fit we keep actually failed).

    In-process `warnings.simplefilter`/`np.errstate` only cover `n_jobs=1`.
    When `n_jobs != 1`, statsforecast parallelizes via joblib/loky worker
    processes, which start with their own fresh warnings-filter state and do
    NOT inherit the parent's `warnings.catch_warnings()` context - only the
    inherited `PYTHONWARNINGS` environment variable reaches them, so that is
    set (and restored) here too.
    """
    old_env = os.environ.get("PYTHONWARNINGS")
    os.environ["PYTHONWARNINGS"] = "ignore"
    try:
        with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            warnings.simplefilter("ignore")
            yield
    finally:
        if old_env is None:
            os.environ.pop("PYTHONWARNINGS", None)
        else:
            os.environ["PYTHONWARNINGS"] = old_env


def _fit_group(
    df_slice: pd.DataFrame,
    model,
    model_name: str,
    freq: str,
    horizon: int,
    variance_lookup: pd.DataFrame,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Fit `model` on `df_slice` in one StatsForecast call. Raises on failure -
    callers decide how to degrade (retry smaller batch, or fall back to naive).
    """
    with _suppress_model_fit_warnings():
        sf = StatsForecast(models=[model], freq=freq, n_jobs=n_jobs)
        if model_name in INTERVAL_CAPABLE_MODELS:
            fcst = sf.forecast(df=df_slice, h=horizon, level=[80])
            fcst["forecast_mean"] = fcst[model_name].clip(lower=0)
            fcst["forecast_std"] = (
                (fcst[f"{model_name}-hi-80"] - fcst[f"{model_name}-lo-80"]) / (2 * Z_80)
            ).clip(lower=0)
        else:
            fcst = sf.forecast(df=df_slice, h=horizon)
            fcst["forecast_mean"] = fcst[model_name].clip(lower=0)
            fcst = fcst.merge(variance_lookup, on="unique_id", how="left")
            variance = _compound_bernoulli_variance(fcst["p"], fcst["mu_z"], fcst["sigma_z"])
            fcst["forecast_std"] = np.sqrt(variance.clip(lower=0))
    fcst["model_used"] = model_name
    return fcst[FORECAST_ROW_COLUMNS]


def forecast_selected(
    long_df: pd.DataFrame,
    freq: str,
    horizon: int,
    stats_df: pd.DataFrame,
    selection_df: pd.DataFrame,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Fit each SKU with the single model `selection_df` chose for it, on its full
    history (not the backtest train split), and return the per-period forecast.

    Some statsforecast models (notably AutoETS) hard-crash rather than skip a
    pathologically short/degenerate series within a batch. A crash there must
    never take down forecasting for every other SKU in the file: on failure,
    this retries the failing model per-SKU to isolate exactly which SKU(s)
    broke it, and only those fall back to `_naive_forecast` - every SKU that
    fits fine keeps its real model.
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

        try:
            parts.append(_fit_group(subset, model, model_name, freq, horizon, variance_lookup, n_jobs))
            logger.info("Fit %s for %d SKU(s) in one batch.", model_name, group["unique_id"].nunique())
            continue
        except Exception as exc:
            logger.warning(
                "%s failed to fit as a batch for %d SKU(s) (%s: %s); retrying per-SKU.",
                model_name, group["unique_id"].nunique(), type(exc).__name__, exc,
            )

        naive_ids = []
        for uid in group["unique_id"]:
            one = subset[subset["unique_id"] == uid]
            try:
                parts.append(_fit_group(one, model, model_name, freq, horizon, variance_lookup, n_jobs))
            except Exception as exc:
                logger.warning(
                    "SKU %r: %s failed to fit (%s: %s); falling back to NaiveMean.",
                    uid, model_name, type(exc).__name__, exc,
                )
                naive_ids.append(uid)
        if naive_ids:
            parts.append(_naive_forecast(subset[subset["unique_id"].isin(naive_ids)], freq, horizon))

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
