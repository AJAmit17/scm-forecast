"""Expected-BackOrders (EBO) engine.

Implements the METRIC/VARI-METRIC core: given a lead-time-demand distribution
per SKU (mean, variance), compute the EBO(s) curve — expected number of units
short at stock level s — via a negative-binomial (over-dispersed, var > mean)
or Poisson (var <= mean) approximation, then either:

  - `recommend_stock_service_level`: pick the smallest s hitting a target
    cycle fill rate per SKU, independently.
  - `optimize_stock_budget`: greedy marginal allocation across all SKUs under
    a total spares budget (classic METRIC "marginal analysis" / Kettelle
    heuristic) — at each step, buy the next unit with the best
    backorder-reduction-per-dollar ratio.

This is a single-echelon, single-indenture simplification of full
VARI-METRIC (no depot/base network, no multi-indenture repair pipeline). It
is intended as a defensible, from-scratch reference implementation; swap in
a full multi-echelon solver (e.g. X-METRIC) if your network topology needs it.
"""

from __future__ import annotations

import heapq

import numpy as np
import pandas as pd
from scipy import stats


def _negbin_params(mean: float, var: float) -> tuple[float, float] | None:
    """Return (n, p) for scipy's nbinom, or None to signal Poisson fallback."""
    if mean <= 0 or var <= mean:
        return None
    p = mean / var
    n = mean * p / (1 - p)
    return n, p


def _pmf(mean: float, var: float, max_x: int) -> tuple[np.ndarray, np.ndarray]:
    xs = np.arange(0, max_x + 1)
    params = _negbin_params(mean, var)
    if params is None:
        pmf = stats.poisson.pmf(xs, mu=max(mean, 1e-9))
    else:
        n, p = params
        pmf = stats.nbinom.pmf(xs, n, p)
    return xs, pmf


def _max_stock_bound(mean: float, var: float, floor: int = 20) -> int:
    spread = np.sqrt(max(var, mean, 1.0))
    return int(mean + 10 * spread + floor)


def expected_backorders_curve(
    mean: float, var: float, max_stock: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (stock_levels, ebo[s], fill_rate[s]) for s = 0..max_stock."""
    if max_stock is None:
        max_stock = _max_stock_bound(mean, var)
    if mean <= 0:
        ss = np.arange(0, max_stock + 1)
        return ss, np.zeros_like(ss, dtype=float), np.ones_like(ss, dtype=float)

    max_x = max(max_stock, _max_stock_bound(mean, var))
    xs, pmf = _pmf(mean, var, max_x)
    cdf = np.cumsum(pmf)

    ss = np.arange(0, max_stock + 1)
    # EBO(s) = E[(X - s)+] = sum_x max(x - s, 0) * pmf(x)
    excess_matrix = np.clip(xs[None, :] - ss[:, None], 0, None)
    ebo = excess_matrix @ pmf
    fill = np.clip(cdf[ss], 0.0, 1.0)
    return ss, ebo, fill


def recommend_stock_service_level(df: pd.DataFrame, target: float) -> pd.DataFrame:
    """Per-SKU smallest stock level `s` with fill_rate(s) >= target."""
    out = []
    for row in df.to_dict("records"):
        mean, var = row["mean_ltd"], row["var_ltd"]
        max_stock = _max_stock_bound(mean, var)
        ss, ebo, fill = expected_backorders_curve(mean, var, max_stock)
        idx = int(np.searchsorted(fill, target))
        idx = min(idx, len(ss) - 1)
        out.append(
            {
                **row,
                "recommended_stock": int(ss[idx]),
                "expected_backorders": float(ebo[idx]),
                "achieved_fill_rate": float(fill[idx]),
            }
        )
    return pd.DataFrame(out)


def optimize_stock_budget(df: pd.DataFrame, budget: float) -> pd.DataFrame:
    """Greedy marginal allocation of stock across SKUs under a $ budget.

    At each step, spend on the single next unit (across all SKUs) that
    delivers the largest backorder reduction per dollar, until the budget is
    exhausted or no further unit reduces EBO.
    """
    curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}
    rows_by_id: dict[str, dict] = {}
    stock: dict[str, int] = {}

    for row in df.to_dict("records"):
        uid = row["unique_id"]
        mean, var, cost = row["mean_ltd"], row["var_ltd"], max(row["unit_cost"], 1e-9)
        max_stock = _max_stock_bound(mean, var)
        ss, ebo, fill = expected_backorders_curve(mean, var, max_stock)
        curves[uid] = (ss, ebo, fill, cost)
        rows_by_id[uid] = row
        stock[uid] = 0

    heap: list[tuple[float, str]] = []

    def push(uid: str) -> None:
        ss, ebo, _fill, cost = curves[uid]
        s = stock[uid]
        if s + 1 >= len(ss):
            return
        benefit = ebo[s] - ebo[s + 1]
        if benefit > 1e-9:
            heapq.heappush(heap, (-benefit / cost, uid))

    for uid in curves:
        push(uid)

    remaining = budget
    while heap:
        _neg_ratio, uid = heapq.heappop(heap)
        _ss, _ebo, _fill, cost = curves[uid]
        if cost > remaining:
            continue
        stock[uid] += 1
        remaining -= cost
        push(uid)

    out = []
    for uid, row in rows_by_id.items():
        ss, ebo, fill, _cost = curves[uid]
        s = stock[uid]
        out.append(
            {
                **row,
                "recommended_stock": int(s),
                "expected_backorders": float(ebo[s]),
                "achieved_fill_rate": float(fill[s]),
            }
        )
    return pd.DataFrame(out)
