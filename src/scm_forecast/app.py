"""Streamlit UI: upload Excel, map columns, run the forecast+EBO pipeline, inspect and download CSV.

Run with: uv run streamlit run src/scm_forecast/app.py  (or `make app`)
"""

from __future__ import annotations

import io
import logging

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from scm_forecast.logging_config import configure_streamlit_logging
from scm_forecast.pipeline import run_pipeline
from scm_forecast.schema import ColumnMapping, PipelineConfig

st.set_page_config(page_title="Demand Forecast + EBO", layout="wide")
st.title("Supply-Chain Demand Forecasting & EBO Inventory Optimization")
st.caption(
    "Excel/CSV demand history in -> per-SKU model selection verified by holdout MAPE "
    "-> per-period forecast + Expected-Backorders stock recommendation -> CSV out."
)

with st.sidebar:
    st.header("1. Input")
    uploaded = st.file_uploader("Demand history (.xlsx, .xls, .csv)", type=["xlsx", "xls", "csv"])

if uploaded is None:
    st.info("Upload a demand-history file to begin, or run `make sample-data` and upload `data/sample_demand.xlsx`.")
    st.stop()


@st.cache_data(show_spinner="Parsing uploaded file...")
def _load_raw(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Cached by (file content, filename): re-parsing a large Excel/CSV file on
    every widget interaction (Streamlit reruns the whole script each time) is
    wasteful for large uploads - this only re-parses when the file itself changes.
    """
    buf = io.BytesIO(file_bytes)
    if filename.lower().endswith(".csv"):
        return pd.read_csv(buf)
    return pd.read_excel(buf)


raw = _load_raw(uploaded.getvalue(), uploaded.name)
if len(raw) > 50_000:
    st.warning(
        f"Large file: {len(raw):,} rows. Runtime scales with the number of distinct SKUs, "
        "not row count - see the run log below after running for per-stage timing."
    )

st.subheader("Preview")
st.dataframe(raw.head(20), width="stretch")

columns = list(raw.columns)

with st.sidebar:
    st.header("2. Column mapping")

    def _guess_index(candidates: list[str], cols: list[str], fallback: int) -> int:
        lower = [c.lower() for c in cols]
        for cand in candidates:
            if cand.lower() in lower:
                return lower.index(cand.lower())
        return fallback

    sku_col = st.selectbox(
        "SKU / item id", columns, index=_guess_index(["item", "sku"], columns, 0)
    )
    date_col = st.selectbox(
        "Date / period", columns, index=_guess_index(["yymm", "date"], columns, min(1, len(columns) - 1))
    )
    date_format_label = st.radio(
        "Date column format", ["YYMM code (e.g. 2301)", "Standard date"], horizontal=True
    )
    date_format = "yymm" if date_format_label.startswith("YYMM") else "date"
    qty_col = st.selectbox(
        "Demand quantity",
        columns,
        index=_guess_index(["actuals", "actauls", "qty"], columns, min(2, len(columns) - 1)),
    )
    optional_cols = ["(none)"] + columns
    lead_time_col = st.selectbox("Lead time (days) [optional]", optional_cols)
    unit_cost_col = st.selectbox("Unit cost [optional]", optional_cols)
    current_stock_col = st.selectbox("Current stock [optional]", optional_cols)

    st.header("3. Forecast settings")
    freq_label = st.radio("Frequency", ["Daily", "Weekly", "Monthly"], index=2, horizontal=True)
    freq = {"Daily": "D", "Weekly": "W", "Monthly": "MS"}[freq_label]
    horizon = st.slider("Forecast horizon (periods)", 4, 36, 18)
    lead_time_default = st.number_input("Default lead time (days, used if column missing)", value=90.0, min_value=0.0)
    unit_cost_default = st.number_input("Default unit cost (used if column missing)", value=1.0, min_value=0.0)

    st.header("4. Inventory objective")
    mode_label = st.radio("Mode", ["Target service level", "Budget-constrained (minimize EBO)"])
    service_level = 0.95
    budget = None
    if mode_label == "Target service level":
        service_level = st.slider("Target cycle fill rate", 0.50, 0.999, 0.95)
    else:
        budget = st.number_input("Total spares budget ($)", value=10000.0, min_value=0.0)

    run_clicked = st.button("Run pipeline", type="primary")

mapping = ColumnMapping(
    sku=sku_col,
    date=date_col,
    qty=qty_col,
    date_format=date_format,
    lead_time=None if lead_time_col == "(none)" else lead_time_col,
    unit_cost=None if unit_cost_col == "(none)" else unit_cost_col,
    current_stock=None if current_stock_col == "(none)" else current_stock_col,
)
config = PipelineConfig(
    freq=freq,
    horizon=horizon,
    lead_time_default=lead_time_default,
    unit_cost_default=unit_cost_default,
    mode="budget" if budget is not None and mode_label != "Target service level" else "service_level",
    service_level=service_level,
    budget=budget if mode_label != "Target service level" else None,
)

if run_clicked:
    log_handler = configure_streamlit_logging(logging.INFO)
    with st.spinner("Backtesting candidate models per SKU (MAPE) and fitting the winning forecast..."):
        outputs = run_pipeline(raw, mapping, config)
    st.session_state["pipeline_outputs"] = outputs
    st.session_state["log_records"] = log_handler.records
    st.session_state["log_max_level"] = log_handler.max_level

if "pipeline_outputs" not in st.session_state:
    st.info("Configure options in the sidebar, then click **Run pipeline**.")
    st.stop()

outputs = st.session_state["pipeline_outputs"]
long_df = outputs.long_df
stats_df = outputs.stats
forecast_df = outputs.forecast
inventory_df = outputs.inventory
log_records = st.session_state["log_records"]
log_max_level = st.session_state["log_max_level"]

with st.expander("Run log", expanded=log_max_level >= logging.WARNING):
    if log_records:
        st.code("\n".join(log_records), language=None)
    else:
        st.caption("No log entries.")
st.subheader("Demand-pattern classification")
st.caption(
    "Informational only - the model that actually forecasts each SKU is chosen by holdout "
    "MAPE below, not by this label alone."
)
counts = stats_df["category"].value_counts().reindex(
    ["smooth", "erratic", "intermittent", "lumpy", "no_demand"]
).fillna(0)
col_a, col_b = st.columns([1, 2])
with col_a:
    st.dataframe(counts.rename("SKU count"), width="stretch")
with col_b:
    st.bar_chart(counts)

st.subheader("Model selection & forecast accuracy (holdout MAPE)")
accuracy = inventory_df[["unique_id", "category", "model_used", "backtest_mape"]].rename(
    columns={"backtest_mape": "backtest_mape_pct"}
)
st.dataframe(accuracy, width="stretch")
valid_mape = accuracy["backtest_mape_pct"].dropna()
if not valid_mape.empty:
    st.caption(
        f"Median backtest MAPE across {len(valid_mape)} SKU(s) with enough history to backtest: "
        f"{valid_mape.median():.1f}%. MAPE is computed only over held-out periods with non-zero "
        "actuals; SKUs without enough history to backtest show NaN and fall back to a "
        "demand-pattern heuristic model."
    )

st.subheader("Per-SKU history + forecast")
sku_choice = st.selectbox("Select SKU", sorted(long_df["unique_id"].unique()))
history = long_df[long_df["unique_id"] == sku_choice]
sku_forecast = forecast_df[forecast_df["unique_id"] == sku_choice].sort_values("ds")

fig = go.Figure()
fig.add_trace(go.Scatter(x=history["ds"], y=history["y"], name="Actual demand", mode="lines"))
if not sku_forecast.empty:
    fig.add_trace(
        go.Scatter(x=sku_forecast["ds"], y=sku_forecast["forecast_qty"], name="Forecast", mode="lines+markers")
    )
    fig.add_trace(
        go.Scatter(
            x=list(sku_forecast["ds"]) + list(sku_forecast["ds"][::-1]),
            y=list(sku_forecast["forecast_upper_80"]) + list(sku_forecast["forecast_lower_80"][::-1]),
            fill="toself",
            fillcolor="rgba(99,110,250,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="80% interval",
            showlegend=True,
        )
    )
fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig, width="stretch")

if not sku_forecast.empty:
    row = sku_forecast.iloc[0]
    mape_val = row["backtest_mape"]
    st.json(
        {
            "category": row["category"],
            "model_used": row["model_used"],
            "backtest_mape_pct": None if pd.isna(mape_val) else float(mape_val),
            "backtest_mae": None if pd.isna(row["backtest_mae"]) else float(row["backtest_mae"]),
            "forecast_periods": len(sku_forecast),
        }
    )

st.subheader("Forecast (per SKU x period)")
st.dataframe(forecast_df, width="stretch")
forecast_csv = io.StringIO()
forecast_df.to_csv(forecast_csv, index=False)
st.download_button(
    "Download forecast (CSV)",
    data=forecast_csv.getvalue(),
    file_name="forecast.csv",
    mime="text/csv",
)

st.subheader("EBO / inventory recommendation (per SKU)")
st.dataframe(inventory_df, width="stretch")

total_current = inventory_df["current_stock"].sum()
total_recommended = inventory_df["recommended_stock"].sum()
reduction = total_current - total_recommended
m1, m2, m3 = st.columns(3)
m1.metric("Current stock (units)", f"{total_current:,.0f}")
m2.metric("Recommended stock (units)", f"{total_recommended:,.0f}")
m3.metric("Inventory delta", f"{reduction:,.0f}", delta=f"{-reduction:,.0f}")

inventory_csv = io.StringIO()
inventory_df.to_csv(inventory_csv, index=False)
st.download_button(
    "Download EBO / inventory recommendation (CSV)",
    data=inventory_csv.getvalue(),
    file_name="inventory_ebo.csv",
    mime="text/csv",
)
