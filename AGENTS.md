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
  schema.py     ColumnMapping (raw column -> pipeline role) + PipelineConfig (tunables,
                incl. n_jobs)
  ingest.py     Excel/CSV -> canonical long frame (unique_id, ds, y), gap-filled with
                explicit zero-demand periods; extracts per-SKU static attributes
                (lead_time_days, unit_cost, current_stock)
  classify.py   Syntetos-Boylan-Croston (ADI/CV2) demand-pattern classification into
                smooth/erratic/intermittent/lumpy/no_demand. INFORMATIONAL ONLY as of
                the backtest-driven model selection below - it no longer gates which
                model forecasts a SKU. Also computes the compound-Bernoulli
                decomposition (p, mu_z, sigma_z) reused for variance estimation.
  backtest.py   Per-SKU holdout backtest: fits every candidate model
                (`forecast.build_candidate_models`) in its own StatsForecast call (isolated
                try/except - one model failing never drops the others), scores each by MAPE
                (over non-zero-actual periods; MAE fallback when a SKU's holdout window is
                all zero), and picks the single best model per SKU. SKUs with too little
                history to backtest fall back to CrostonOptimized (verified safe on any
                history length) with `backtest_mape = NaN` so that fallback is visible.
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
  logging_config.py  Central logging setup: console handler for the CLI, in-memory
                handler for the Streamlit "Run log" panel. Every other module logs via
                `get_logger(__name__)`; this module only decides routing (see Observability
                below).
  pipeline.py   Orchestrates ingest -> classify -> backtest+select -> per-period forecast
                -> lead-time scaling -> EBO, with per-stage timing logged. Returns
                `PipelineOutputs(forecast, inventory, long_df, stats)`: `forecast` = per-SKU-
                per-period curve (the actual forecast deliverable), `inventory` = per-SKU
                EBO/stock-recommendation table, `long_df`/`stats` = reusable ingest/
                classification results so callers (the UI) never re-ingest.
  cli.py        Typer CLI wrapping pipeline.run_pipeline_from_path; writes both CSVs;
                `--verbose`/`-v` for DEBUG logs, `--jobs`/`-j` for statsforecast parallelism.
  app.py        Streamlit UI: upload (cached via `st.cache_data`), column mapping, run,
                model-selection/MAPE table, "Run log" expander, per-SKU forecast chart
                (real curve, not a repeated average), two CSV downloads.
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

## Observability: logging (not print/vague errors)
Every module logs via `logging_config.get_logger(__name__)` under the shared
`scm_forecast` logger tree - never `print`. `logging_config.py` only decides
ROUTING:
- CLI: `configure_console_logging()` (called once, at the top of `cli.run`) -
  plain-text lines to stderr; `--verbose`/`-v` switches INFO -> DEBUG.
- Streamlit: `configure_streamlit_logging()` (called once per script run, in
  `app.py`, right before `run_pipeline`) - an `InMemoryLogHandler` whose
  contents are rendered in a "Run log" `st.expander`, auto-expanded if any
  WARNING+ record was emitted. A Streamlit user may not have the launching
  terminal visible, so this is the primary place they see diagnostics.

What gets logged: `ingest.py` (raw/gap-filled row counts, SKU count, date
range, dropped spacer rows), `classify.py` (category distribution),
`backtest.py`/`forecast.py` (eligible-vs-fallback SKU counts, which candidate
models fit successfully, and - at WARNING, with the SKU id and the real
exception type/message - every time a model fails and a fallback kicks in),
`pipeline.py` (per-stage wall-clock timing and a final summary). This is
intentionally the OPPOSITE of statsforecast's own internal warnings (see
below): those are noise about degenerate-but-harmless internal model-search
states; ours are signal about what this app actually decided and why.

**Never suppress a statsforecast/numpy warning without routing the same
information through our own `logger.warning(...)` first** if it could
plausibly indicate an accuracy/coverage problem - the goal is replacing vague
noise with precise signal, not just going quiet.

## Performance & scalability (large uploads)
Runtime scales with **distinct SKU count**, not row count - 20,000 rows as
500 SKUs x 40 months behaves very differently from 20,000 rows as 50 SKUs x
400 months. Measured on a 10-core M-series machine, single-threaded
(`n_jobs=1`, the default everywhere): 500 SKUs x 40 months (20,000 rows) ->
~80s end to end, dominated by `AutoARIMA` (the only candidate model with a
real per-series optimization cost; the other four are near-instant).

What's already done for this:
- `AutoARIMA(approximation=True)` in `forecast.build_candidate_models` -
  ~1.4x faster order search, no multiprocessing risk, on by default everywhere.
- `PipelineConfig.n_jobs` (default `1`) threads through to every
  `StatsForecast(...)` call. The CLI exposes `--jobs`/`-j` (still defaults to
  `1`): measured, `-1` (all cores) only gave ~12% net improvement on the
  500-SKU benchmark above, because this codebase makes several *separate*
  `StatsForecast` calls (one per candidate model in the backtest, one per
  winning-model group in the final fit) - each pays its own process-pool
  spawn/teardown cost (~seconds), which eats most of the parallel win and can
  make **small** files slower, not faster. Only worth trying `--jobs -1` for
  genuinely large SKU counts (hundreds+); benchmark before relying on it.
- The Streamlit app does NOT expose `n_jobs` and never uses anything but `1`:
  `n_jobs=-1` previously reproduced a `concurrent.futures.process.
  BrokenProcessPool` crash specifically under Streamlit's script-rerun
  execution model (forking/spawning inside its own execution loop is
  unsafe). Do not re-enable multiprocessing in `app.py` without re-verifying
  that failure mode is actually gone.
- `app.py` caches the uploaded file parse with `st.cache_data` (keyed on file
  bytes + name) so Streamlit's "rerun the whole script on every widget
  interaction" behavior doesn't re-parse a large Excel file on every click.
- `PipelineOutputs` carries `long_df`/`stats` so the UI never calls
  `prepare_long_frame`/`compute_sku_stats` a second time after `run_pipeline`
  already did (previously an accidental double-ingest).
- A `> 200,000` raw-row ingest and a `> 50,000` raw-row Streamlit upload both
  log/display an explicit heads-up rather than silently taking a long time.

If real usage needs faster large-batch throughput beyond this, the next lever
is restructuring backtest/forecast to fit ALL candidate models in one shared
`StatsForecast(models=[...])` call (one process pool instead of five) - this
was deliberately NOT done here because it reintroduces the "one bad model
crashes the whole batch" failure mode Robustness above fixes; do not make
that tradeoff without re-adding the same per-model isolation some other way.

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
- `tests/test_edge_cases.py`: malformed/pathological input must never crash the
  pipeline (see Robustness above).
- `tests/test_scale_and_logging.py`: no statsforecast warnings may leak out of
  `run_pipeline` (asserted via `warnings.catch_warnings(record=True)`), structured
  log milestones must be present (via `caplog`), and a moderate multi-SKU batch
  (60 SKUs) must complete cleanly.
- `tests/test_app_session_state.py`: drives `app.py` for real via
  `streamlit.testing.v1.AppTest` (upload sample data, click Run, then select
  every SKU in turn) - catches Streamlit-state bugs unit tests on `pipeline.py`
  alone cannot see (see App state pitfall below). Extend this file rather than
  re-deriving ad hoc repro scripts if a similar "works once, breaks on the next
  click" issue resurfaces.

## Streamlit app-state pitfall (read before touching app.py)
`st.button(...)` only returns `True` in the exact script rerun its click
triggered; every subsequent rerun - including one triggered by an unrelated
widget like the "Select SKU" selectbox - it returns `False` again. Gating the
results section behind `if not run_clicked: st.stop()` therefore wiped the
*entire* results view (chart, tables, downloads) the instant a user selected
any SKU other than whichever was already showing - which looked exactly like
"every category except the default one is broken", even though `run_pipeline`
itself was working correctly for every category. Fixed by persisting
`PipelineOutputs` (+ the log handler's records) in `st.session_state` keyed
independently of `run_clicked`, so any later rerun re-reads the same cached
result instead of re-gating on the button. Any new top-level `st.stop()` or
early-return added to `app.py` MUST be re-checked against this failure mode:
trigger it, then interact with an unrelated widget, and confirm results stay.
