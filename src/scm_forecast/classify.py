"""Syntetos-Boylan-Croston (SBC) demand-pattern classification.

Splits each SKU into smooth / erratic / intermittent / lumpy using Average
inter-Demand Interval (ADI) and squared coefficient of variation of non-zero
demand sizes (CV2). Reference: Syntetos, Boylan & Croston (2005), "On the
categorization of demand patterns".

`group` routes each SKU to the forecasting family used downstream:
  - "regular"      (smooth, erratic)       -> AutoETS / AutoARIMA style models
  - "intermittent" (intermittent, lumpy)    -> Croston/TSB style models
  - "no_demand"    (never sold in history)  -> zero forecast, skip modeling
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scm_forecast.logging_config import get_logger

logger = get_logger(__name__)

ADI_CUTOFF = 1.32
CV2_CUTOFF = 0.49


def _categorize(adi: float, cv2: float) -> str:
    if not np.isfinite(adi):
        return "no_demand"
    if adi < ADI_CUTOFF and cv2 < CV2_CUTOFF:
        return "smooth"
    if adi >= ADI_CUTOFF and cv2 < CV2_CUTOFF:
        return "intermittent"
    if adi < ADI_CUTOFF and cv2 >= CV2_CUTOFF:
        return "erratic"
    return "lumpy"


def _group_for(category: str) -> str:
    if category in ("smooth", "erratic"):
        return "regular"
    if category == "no_demand":
        return "no_demand"
    return "intermittent"


def compute_sku_stats(long_df: pd.DataFrame) -> pd.DataFrame:
    """Per-SKU ADI/CV2 + compound-Bernoulli demand-size stats + category/group.

    `p`, `mu_z`, `sigma_z` decompose demand as Y = B * Z, B ~ Bernoulli(p) (a
    period has any demand), Z = size of demand given it occurs. This
    decomposition is reused by the intermittent forecaster to estimate demand
    variance without relying on library-specific prediction-interval support.
    """
    records = []
    for uid, g in long_df.groupby("unique_id", sort=False):
        y = g["y"].to_numpy(dtype=float)
        n = len(y)
        nz = y[y > 0]
        n_nonzero = len(nz)
        if n_nonzero == 0:
            adi, cv2, p, mu_z, sigma_z = np.inf, 0.0, 0.0, 0.0, 0.0
        else:
            adi = n / n_nonzero
            p = n_nonzero / n
            mu_z = float(nz.mean())
            sigma_z = float(nz.std(ddof=0))
            cv2 = (sigma_z / mu_z) ** 2 if mu_z > 0 else 0.0
        category = _categorize(adi, cv2)
        records.append(
            {
                "unique_id": uid,
                "n_periods": n,
                "n_nonzero": n_nonzero,
                "adi": adi,
                "cv2": cv2,
                "p": p,
                "mu_z": mu_z,
                "sigma_z": sigma_z,
                "category": category,
                "group": _group_for(category),
            }
        )
    stats_df = pd.DataFrame.from_records(records)
    if not stats_df.empty:
        counts = stats_df["category"].value_counts().to_dict()
        logger.info("Classification: %d SKU(s) -> %s", len(stats_df), counts)
    return stats_df
