# AGENTS.md

## What this is
Supply-chain demand forecasting + inventory reduction tool. Excel/CSV demand
history in -> per-SKU holdout backtest picks the best-fitting model (verified
by MAPE, not assumed by demand-pattern label) -> per-period forecast curve +
Expected-BackOrders (EBO) stock recommendation -> two CSVs out. Streamlit UI
on top for interactive upload/inspect/download.

No external services, no API keys, no network calls at runtime. Everything
runs locally against the uploaded file (statsforecast/scipy are pure local
compute). `.env` is not used by this project.

## Deployment (Streamlit Community Cloud)
Streamlit Cloud detects `uv.lock` and deploys via its **uv-sync** path, which
picks the Python version itself (via `.python-version` / `requires-python`) -
it does **not** read `runtime.txt` (that's only honored by the legacy
pip/`requirements.txt` deploy path). The working fix is:
- `.python-version` (`3.12`) + `requires-python = ">=3.11,<3.13"` in
  `pyproject.toml` - pins uv to a Python version statsforecast has
  broadly-tested wheels for. Without this, Cloud's uv-sync picked the newest
  available Python (observed: 3.14), too new for statsforecast's compiled
  ARIMA extension (`_lib.*.so`).
- `packages.txt` (`libstdc++6`) - installs a current libstdc++ via apt before
  pip/uv install. Without it, importing that same compiled extension fails
  with `undefined symbol: _ZTVN10__cxxabiv117__class_type_infoE` even on a
  correctly-pinned Python.
Both are Cloud-image issues, not app bugs - neither reproduces locally under
`uv sync`. If you change deployment target/runtime, re-verify this still
resolves before removing either file. `uv.lock` MUST be regenerated
(`uv sync -p 3.12`) whenever `requires-python`/`.python-version` changes, or
Cloud's uv-sync will reject a stale lock resolved against a different Python.

### Target input schema (real ERP/BI export)
Monthly SKU demand history, one row per SKU x month:

| Item | YYMM | YYQQ | Actuals |
|---|---|---|---|
| 123 | 2301 | 23Q1 | 100 |

- `Item`: SKU id (may be numeric-looking; may only be populated on the first
  row of each SKU's block due to merged cells in the source sheet).
- `YYMM`: 4-digit year-month code (2-digit year + 2-digit month, e.g. `2301`
  = Jan 2023). This is the default `date_format="yymm"` path in `ingest.py`.
- `YYQQ`: derived quarter code — informational only, not consumed by the
  pipeline (redundant with `YYMM`).
- `Actuals`: historical quantity sold/consumed.

Default CLI/UI settings target this schema directly: `sku_col=Item`,
`date_col=YYMM`, `date_format=yymm`, `qty_col=Actuals`, `freq=MS` (monthly),
`horizon=18` (18-month-ahead forecast, one row of results per SKU).
Real dates instead of YYMM codes: pass `--date-format date` (CLI) or pick
"Standard date" in the UI.

## Environment
- Package manager: **uv** (not pip/poetry/conda). Python >= 3.11.
- Setup: `make install` (== `uv sync`). This creates `.venv/` and installs
  from `pyproject.toml` + `uv.lock`.
- No environment variables required.

## Commands
| Command | Effect |
|---|---|
| `make install` | `uv sync` — install/update the environment |
| `make sample-data` | generates `data/sample_demand.xlsx` (7 synthetic SKUs, `Item`/`YYMM`/`YYQQ`/`Actuals` schema, 36 months history, covering smooth/erratic/intermittent/lumpy patterns) |
| `make app` | launches the Streamlit UI (`uv run streamlit run src/scm_forecast/app.py`) |
| `make forecast` | headless CLI run against the sample data -> `output/forecast.csv` (per-SKU per-period, 18-month monthly) + `output/inventory_ebo.csv` (per-SKU EBO stock recommendation) |
| `make test` | `uv run pytest -q` |
| `make lint` / `make fmt` | `ruff check` / `ruff format` |
| `make clean` | removes venv, generated data/output, caches |

CLI usage directly: `uv run scm-forecast --input <file> --output <forecast.csv> --inventory-output <inventory.csv> --sku-col ... --help`.

## Code layout
```
src/scm_forecast/
  schema.py     ColumnMapping (raw column -> pipeline role) + PipelineConfig (tunables)
  ingest.py     Excel/CSV -> canonical long frame (unique_id, ds, y), gap-filled with
                explicit zero-demand periods; extracts per-SKU static attributes
                (lead_time_days, unit_cost, current_stock)
  classify.py   Syntetos-Boylan-Croston (ADI/CV2) demand-pattern classification into
                smooth/erratic/intermittent/lumpy/no_demand. INFORMATIONAL ONLY as of
                the backtest-driven model selection below - it no longer gates which
                model forecasts a SKU. Also computes the compound-Bernoulli
                decomposition (p, mu_z, sigma_z) reused for variance estimation.
  backtest.py   Per-SKU holdout backtest: fits every candidate model
                (`forecast.build_candidate_models`) on a train/holdout split, scores
                each by MAPE (over non-zero-actual periods; MAE fallback when a SKU's
                holdout window is all zero), and picks the single best model per SKU.
                SKUs with too little history to backtest fall back to a demand-pattern
                heuristic (AutoETS for "regular", CrostonOptimized for "intermittent")
                with `backtest_mape = NaN` so that fallback is visible, not silent.
  forecast.py   Candidate pool: AutoETS, AutoARIMA (native 80% interval -> std) and
                CrostonOptimized, TSB, ADIDA (compound-Bernoulli variance, since native
                prediction intervals for these aren't reliably available without extra
                history/config). `forecast_selected` fits each SKU with whichever model
                `backtest.py` chose for it, on that SKU's FULL history, producing the
                actual per-period forecast curve (not a flattened repeated average).
  ebo.py        METRIC-style Expected-BackOrders engine: negative-binomial (var>mean) or
                Poisson (var<=mean) approximation of lead-time demand; EBO(s) curve;
                either per-SKU service-level stock recommendation or budget-constrained
                greedy marginal allocation across SKUs (classic METRIC marginal analysis).
  pipeline.py   Orchestrates ingest -> classify -> backtest+select -> per-period forecast
                -> lead-time scaling -> EBO. Returns `PipelineOutputs(forecast, inventory)`:
                `forecast` = per-SKU-per-period curve (the actual forecast deliverable),
                `inventory` = per-SKU EBO/stock-recommendation table.
  cli.py        Typer CLI wrapping pipeline.run_pipeline_from_path; writes both CSVs.
  app.py        Streamlit UI: upload, column mapping, run, model-selection/MAPE table,
                per-SKU forecast chart (real curve, not a repeated average), two CSV
                downloads (forecast + inventory).
scripts/generate_sample.py   synthetic demand generator (smooth/erratic/intermittent/
                              lumpy/trending-seasonal SKUs, Item/YYMM/YYQQ/Actuals schema)
tests/                       pytest: classify thresholds, EBO math invariants, ingest
                              schema handling, end-to-end pipeline smoke tests, a
                              regression guard that a trending+seasonal SKU's forecast
                              curve is NOT flat
```

## Key modeling assumptions (read before changing math)
- Demand-pattern classification (ADI/CV2, SBC cutoffs 1.32 / 0.49) is informational and
  drives only the *fallback* model when a SKU has too little history to backtest. The
  model that actually forecasts a SKU is chosen by `backtest.py` on held-out MAPE/MAE -
  this is deliberate: forcing every SKU with some zero periods onto a flat Croston-style
  rate produced visibly wrong (flat) forecasts for SKUs that actually have trend/
  seasonality. A flat forecast is still correct output for SKUs where it wins the
  backtest (i.e. demand timing is genuinely unpredictable) - MAPE makes that verifiable
  instead of assumed.
- MAPE is computed only over backtest periods with non-zero actuals (standard practice;
  otherwise it's undefined). `backtest_mae` and `n_backtest_periods` are reported
  alongside it; `backtest_mape = NaN` means the SKU had too little history to backtest
  (falls back to CrostonOptimized - the only candidate model verified safe on any
  history length down to a single data point; see Robustness below).
- Lead-time demand mean/variance scale linearly with lead-time periods
  (`mean_ltd = mean_per_period * lt_periods`, `var_ltd = var_per_period * lt_periods`),
  i.e. iid demand across periods within the lead time — the standard safety-stock
  assumption. If lead time and forecast frequency disagree in unit, `lead_time_periods()`
  in `pipeline.py` converts days -> periods via `_DAYS_PER_PERIOD`.
- EBO is single-echelon/single-indenture (no depot/base network, no repair pipeline).
  This is a from-scratch simplification of Sherbrooke's METRIC/VARI-METRIC, not a full
  multi-echelon solver. If you need multi-echelon, look at X-METRIC
  (https://r-forge.r-project.org/projects/xmetric/, R, early-stage) as an algorithm
  reference rather than a drop-in dependency.
- Budget-constrained allocation is a greedy marginal-analysis heuristic (integer greedy
  on backorder-reduction-per-dollar), not a certified-optimal knapsack solution.

## Robustness against malformed/pathological input
SKU ids are treated as opaque strings everywhere - any format survives
(`"ABC-123"`, `"1234"`, mixed numeric/alphanumeric in the same column,
leading zeros, whitespace-padded, even an empty string). `ingest._clean_sku_id`
forward-fills merged-cell blanks, strips whitespace, and normalizes
Excel's `123` -> `123.0` float rendering back to `"123"`.

Per-SKU history length is NOT assumed to be uniform or "enough". Some
statsforecast models - AutoETS in particular - hard-crash (uncaught
`IndexError`/`NotImplementedError`) instead of gracefully skipping a
pathologically short or degenerate series, and since models fit a whole batch
of SKUs in one call, an uncaught crash for ONE SKU would otherwise take down
forecasting for every other SKU in the file. Two independent layers guard
against this:
1. `backtest.py`'s fallback for SKUs with too little history to backtest
   (`< MIN_TRAIN_PERIODS + h_eval` periods) is always `CrostonOptimized` -
   verified safe on any history length down to n=1 - never AutoETS/AutoARIMA.
2. `forecast.py::forecast_selected` still wraps every batch fit in
   try/except; on failure it retries per-SKU to isolate exactly which SKU(s)
   broke that model, and only those fall back to `_naive_forecast` (historical
   mean/std, pure pandas, cannot fail on any numeric input) - every other SKU
   in the batch keeps its real model and forecast.
`tests/test_edge_cases.py` locks this in: single-data-point SKUs, mixed-dtype
SKU columns, negative quantities, constant/zero-variance demand, zero budget,
and a batch mixing all of the above must all complete without raising.

## Extending
- New candidate model: add it in `forecast.py::build_candidate_models` with an explicit
  `alias`; add it to `INTERVAL_CAPABLE_MODELS` if it supports native `level=` intervals,
  otherwise it gets the compound-Bernoulli variance fallback. `backtest.py` and
  `forecast_selected` both consume this same pool automatically - no other change needed.
- Multi-echelon EBO: add a new function in `ebo.py` next to `optimize_stock_budget`
  rather than modifying the single-echelon path, so both remain usable.

## Testing conventions
- No network access needed for tests (statsforecast fits are local/CPU).
- `tests/test_pipeline.py` fits real candidate models incl. a backtest holdout split;
  keep synthetic fixtures small when adding more end-to-end tests.
