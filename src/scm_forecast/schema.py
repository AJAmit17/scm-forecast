"""Shared config/data-contract types for the pipeline.

Internal long-format schema (what every stage after `ingest` consumes) is fixed to
`unique_id, ds, y` to match the Nixtla ecosystem's convention directly - no reshaping
cost when handing frames to statsforecast.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ColumnMapping:
    """Maps the caller's raw Excel/CSV column names onto pipeline roles.

    `sku`, `date`, `qty` are required. The rest are optional per-SKU static
    attributes; when absent, pipeline-level defaults are used instead.

    `date_format`:
      - "date": the date column already holds real dates/timestamps.
      - "yymm": the date column holds a 4-digit year-month code, e.g. "2301"
        or 2301 for January 2023 (2-digit year + 2-digit month). This is the
        common ERP/BI export format for monthly SKU demand history.
    """

    sku: str
    date: str
    qty: str
    date_format: str = "date"  # "date" | "yymm"
    lead_time: str | None = None
    unit_cost: str | None = None
    current_stock: str | None = None


@dataclass(frozen=True)
class PipelineConfig:
    """Tunables for the forecast -> EBO pipeline."""

    freq: str = "MS"  # pandas offset alias: D, W, MS, ...
    horizon: int = 18  # periods (18 months at the default monthly freq)
    lead_time_default: float = 90.0  # days, used when a SKU has no lead_time value
    unit_cost_default: float = 1.0
    mode: str = "service_level"  # "service_level" | "budget"
    service_level: float = 0.95  # target cycle fill rate, used when mode == "service_level"
    budget: float | None = None  # total spares $ budget, used when mode == "budget"
    n_jobs: int = 1  # statsforecast parallelism; keep 1 under Streamlit (see forecast.py/backtest.py)
