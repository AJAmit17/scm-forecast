# SCM Forecast

Supply-chain demand forecasting and Expected-BackOrders (EBO) inventory optimization tool. Upload Excel/CSV demand history, get per-SKU forecasts and optimal stock recommendations.

## Features

- **Automated Model Selection**: Per-SKU holdout backtest evaluates multiple forecasting models (AutoETS, AutoARIMA, CrostonOptimized, TSB, ADIDA) and selects the best-performing one based on MAPE
- **Demand Classification**: Syntetos-Boylan-Croston pattern analysis (smooth/erratic/intermittent/lumpy/no_demand)
- **EBO Inventory Optimization**: METRIC-style Expected-BackOrders calculation for optimal stock levels, with service-level or budget-constraint modes
- **Dual Interface**: Streamlit web UI for interactive exploration or headless CLI for batch processing
- **Local-First**: No external services, API keys, or network calls required—everything runs locally

## Requirements

- Python >= 3.11, < 3.13
- [uv](https://github.com/astral-sh/uv) package manager

## Installation

```bash
# Install dependencies
make install
# or directly: uv sync
```

## Quick Start

### Option 1: Streamlit Web UI (Interactive)

```bash
# Generate sample data
make sample-data

# Launch the Streamlit app
make app
# or: uv run streamlit run src/scm_forecast/app.py
```

Upload `data/sample_demand.xlsx` (or your own file), map columns, configure pipeline settings, and download the results.

### Option 2: Command-Line Interface (Headless)

```bash
# Run forecast pipeline on sample data
make forecast
# Outputs:
#   - output/forecast.csv        (per-SKU per-period forecast)
#   - output/inventory_ebo.csv   (per-SKU stock recommendations)
```

**Custom CLI usage:**

```bash
uv run scm-forecast \
  --input your_data.xlsx \
  --output output/forecast.csv \
  --inventory-output output/inventory_ebo.csv \
  --sku-col "Item" \
  --date-col "YYMM" \
  --date-format "yymm" \
  --qty-col "Actuals" \
  --horizon 18 \
  --service-level 0.95 \
  --verbose
```

Run `uv run scm-forecast --help` for all options.

## Input Data Format

Default schema (monthly SKU demand history, one row per SKU × month):

| Item | YYMM | Actuals |
|------|------|---------|
| 123  | 2301 | 100     |
| 123  | 2302 | 150     |
| 456  | 2301 | 50      |

- **Item**: SKU identifier
- **YYMM**: 4-digit year-month code (e.g., `2301` = January 2023)
- **Actuals**: Historical quantity sold/consumed

Optional columns:
- **Lead time** (days)
- **Unit cost**
- **Current stock**

For standard dates instead of YYMM codes, use `--date-format date` (CLI) or select "Standard date" in the UI.

## Output

### Forecast CSV
Per-SKU per-period forecast with prediction intervals:

| unique_id | ds         | forecast | lo_80 | hi_80 | model           | pattern      | backtest_mape |
|-----------|------------|----------|-------|-------|-----------------|--------------|---------------|
| 123       | 2025-01-01 | 120.5    | 110.2 | 130.8 | AutoETS         | smooth       | 8.3           |
| 123       | 2025-02-01 | 125.0    | 114.0 | 136.0 | AutoETS         | smooth       | 8.3           |

### Inventory CSV
Per-SKU EBO-based stock recommendations:

| unique_id | reorder_point | demand_mean | demand_std | lead_time_days | unit_cost | current_stock | service_level |
|-----------|---------------|-------------|------------|----------------|-----------|---------------|---------------|
| 123       | 350           | 120.5       | 15.2       | 90             | 10.5      | 200           | 0.95          |

## Pipeline Architecture

1. **Ingest** (`ingest.py`): Parse Excel/CSV → canonical long-form DataFrame, gap-fill missing periods with zeros
2. **Classify** (`classify.py`): Syntetos-Boylan-Croston demand pattern analysis
3. **Backtest** (`backtest.py`): Per-SKU holdout validation to select best model by MAPE
4. **Forecast** (`forecast.py`): Re-fit selected models on full history to generate forecasts
5. **EBO** (`ebo.py`): Calculate optimal inventory levels using Expected-BackOrders logic

## Configuration

Key pipeline parameters (CLI flags or UI widgets):

- `--horizon`: Forecast periods ahead (default: 18)
- `--freq`: Pandas offset alias—`MS` (monthly), `W` (weekly), `D` (daily)
- `--service-level`: Target service level for EBO (default: 0.95)
- `--budget`: Switch to budget-constrained mode (optional)
- `--jobs`: Parallelism for statsforecast (1 = sequential, -1 = all cores; see note below)

**Parallelism note**: Process-pool startup overhead means `-j -1` is only faster for hundreds+ of SKUs. For small files, serial (`-j 1`) is often quicker.

## Development

```bash
# Run tests
make test
# or: uv run pytest -q

# Lint
make lint

# Format code
make fmt

# Clean generated files
make clean
```

## Project Structure

```
scm-forecast/
├── src/scm_forecast/
│   ├── schema.py         # Data models (ColumnMapping, PipelineConfig)
│   ├── ingest.py         # Excel/CSV parsing and canonicalization
│   ├── classify.py       # Demand pattern classification
│   ├── backtest.py       # Model selection via holdout validation
│   ├── forecast.py       # Forecasting engine
│   ├── ebo.py            # Inventory optimization
│   ├── pipeline.py       # Orchestration
│   ├── cli.py            # Typer CLI
│   └── app.py            # Streamlit UI
├── tests/                # Pytest suite
├── scripts/              # Utilities (e.g., sample data generator)
├── data/                 # Sample input files
├── output/               # Generated forecasts and inventory CSVs
├── pyproject.toml        # Project metadata and dependencies
└── Makefile              # Common tasks
```

## Deployment

The app deploys to **Streamlit Community Cloud** with:
- `.python-version` (3.12) + `requires-python = ">=3.11,<3.13"` in `pyproject.toml`
- `packages.txt` (`libstdc++6`) for statsforecast compiled extensions
- `uv.lock` (must be regenerated with `uv sync -p 3.12` if Python version changes)

## License

See repository for license details.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Run tests and linting (`make test && make lint`)
4. Submit a pull request

---

*This README was created by an AI agent (OpenHands) on behalf of the repository maintainer.*
