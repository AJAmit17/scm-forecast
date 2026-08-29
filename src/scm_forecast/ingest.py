"""Excel/CSV ingestion: raw wide user data -> canonical long frame + SKU attributes.

Zero-demand periods must become *explicit* rows (not absence of rows), because
ADI/CV2 classification and intermittent-demand models depend on the true count of
zero periods between sales.

Handles two common real-world export quirks seen in ERP/BI pivot dumps:
  - the SKU column only carries a value on the first row of each SKU's block
    (merged cells in the source spreadsheet) -> forward-filled.
  - fully-blank spacer rows between blocks -> dropped (rows with no period).
"""

from __future__ import annotations

import pandas as pd

from scm_forecast.schema import ColumnMapping


def read_input(path: str) -> pd.DataFrame:
    """Read an Excel (.xlsx/.xls) or CSV file into a raw DataFrame."""
    if str(path).lower().endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_excel(path)


def _clean_sku_id(series: pd.Series) -> pd.Series:
    """Forward-fill merged-cell SKU values and normalize numeric-looking ids.

    Excel exports frequently render whole-number SKU codes as floats
    (123 -> 123.0) once any NaN forces the column to float dtype; strip that.
    """
    filled = series.ffill()

    def _fmt(v: object) -> str | None:
        if pd.isna(v):
            return None
        if isinstance(v, float):
            return str(int(v)) if v.is_integer() else str(v)
        return str(v).strip()

    return filled.map(_fmt)


def _parse_yymm(series: pd.Series) -> pd.Series:
    """Parse a 4-digit YYMM code (e.g. 2301, "2301") into a month-start Timestamp."""

    def _code(v: object) -> str | None:
        if pd.isna(v):
            return None
        if isinstance(v, float):
            v = int(v)
        return str(v).strip().zfill(4)

    codes = series.map(_code)
    years = codes.str[:2].astype(int) + 2000
    months = codes.str[2:4].astype(int)
    return pd.to_datetime(pd.DataFrame({"year": years, "month": months, "day": 1}))


def prepare_long_frame(
    raw: pd.DataFrame, mapping: ColumnMapping, freq: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert raw data to (long_df[unique_id, ds, y], attrs[unique_id, ...]).

    `long_df` is reindexed to a full, gap-free calendar per SKU at `freq`, with
    missing periods filled to 0 demand.
    """
    required = [mapping.sku, mapping.date, mapping.qty]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns in input: {missing}")

    df = raw[[mapping.sku, mapping.date, mapping.qty]].copy()
    df.columns = ["unique_id", "ds_raw", "y"]

    df["unique_id"] = _clean_sku_id(df["unique_id"])
    df = df.dropna(subset=["ds_raw", "unique_id"])  # drop blank spacer rows

    if mapping.date_format == "yymm":
        df["ds"] = _parse_yymm(df["ds_raw"])
    else:
        df["ds"] = pd.to_datetime(df["ds_raw"])
    df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0.0)
    df = df.drop(columns=["ds_raw"])
    df = df.groupby(["unique_id", "ds"], as_index=False)["y"].sum()

    frames = []
    for uid, g in df.groupby("unique_id", sort=False):
        g = g.set_index("ds").sort_index()
        full_idx = pd.date_range(g.index.min(), g.index.max(), freq=freq)
        y = g["y"].reindex(full_idx, fill_value=0.0)
        frames.append(pd.DataFrame({"unique_id": uid, "ds": full_idx, "y": y.to_numpy()}))

    if not frames:
        raise ValueError("No usable rows found after parsing sku/date/qty columns")

    long_df = pd.concat(frames, ignore_index=True)

    attr_source_cols: dict[str, str] = {}
    if mapping.lead_time and mapping.lead_time in raw.columns:
        attr_source_cols["lead_time_days"] = mapping.lead_time
    if mapping.unit_cost and mapping.unit_cost in raw.columns:
        attr_source_cols["unit_cost"] = mapping.unit_cost
    if mapping.current_stock and mapping.current_stock in raw.columns:
        attr_source_cols["current_stock"] = mapping.current_stock

    if attr_source_cols:
        rename_map = {src: dst for dst, src in attr_source_cols.items()}
        cols = [mapping.sku, *attr_source_cols.values()]
        attrs = raw[cols].rename(columns={mapping.sku: "unique_id", **rename_map})
        attrs["unique_id"] = _clean_sku_id(attrs["unique_id"])
        attrs = attrs.dropna(subset=["unique_id"])
        for dst in attr_source_cols:
            attrs[dst] = pd.to_numeric(attrs[dst], errors="coerce")
            attrs[dst] = attrs.groupby("unique_id")[dst].ffill()
        attrs = attrs.groupby("unique_id", as_index=False).last()
    else:
        attrs = pd.DataFrame({"unique_id": sorted(long_df["unique_id"].unique())})
    return long_df, attrs
