"""Headless CLI: Excel/CSV in, CSV out. `uv run scm-forecast --help`."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from scm_forecast.logging_config import configure_console_logging
from scm_forecast.pipeline import run_pipeline_from_path
from scm_forecast.schema import ColumnMapping, PipelineConfig

app = typer.Typer(add_completion=False)


@app.command()
def run(
    input: Path = typer.Option(..., "--input", "-i", help="Excel or CSV file with demand history"),
    output: Path = typer.Option(Path("output/forecast.csv"), "--output", "-o", help="per-SKU per-period forecast CSV"),
    inventory_output: Path = typer.Option(
        Path("output/inventory_ebo.csv"), "--inventory-output", help="per-SKU EBO stock recommendation CSV"
    ),
    sku_col: str = typer.Option("Item", "--sku-col"),
    date_col: str = typer.Option("YYMM", "--date-col"),
    date_format: str = typer.Option(
        "yymm", "--date-format", help="'yymm' for 4-digit year-month codes (e.g. 2301), or 'date' for real dates"
    ),
    qty_col: str = typer.Option("Actuals", "--qty-col"),
    lead_time_col: str | None = typer.Option(None, "--lead-time-col"),
    unit_cost_col: str | None = typer.Option(None, "--unit-cost-col"),
    current_stock_col: str | None = typer.Option(None, "--current-stock-col"),
    freq: str = typer.Option("MS", "--freq", help="pandas offset alias: D, W, MS"),
    horizon: int = typer.Option(18, "--horizon", help="periods ahead to forecast (18 months at default freq)"),
    lead_time_default: float = typer.Option(90.0, "--lead-time-default", help="days"),
    unit_cost_default: float = typer.Option(1.0, "--unit-cost-default"),
    service_level: float = typer.Option(0.95, "--service-level"),
    budget: float | None = typer.Option(None, "--budget", help="if set, switches to budget-constrained EBO minimization"),
    jobs: int = typer.Option(
        1, "--jobs", "-j",
        help="statsforecast parallelism (1 = sequential, -1 = all cores). Process-pool "
        "startup overhead (~seconds per model, paid repeatedly) means -1 is only worth it "
        "for large SKU counts (hundreds+); it can be SLOWER for small files. Not exposed "
        "in the Streamlit app - see AGENTS.md.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="DEBUG-level logging (default: INFO)"),
) -> None:
    configure_console_logging(logging.DEBUG if verbose else logging.INFO)

    mapping = ColumnMapping(
        sku=sku_col,
        date=date_col,
        qty=qty_col,
        date_format=date_format,
        lead_time=lead_time_col,
        unit_cost=unit_cost_col,
        current_stock=current_stock_col,
    )
    config = PipelineConfig(
        freq=freq,
        horizon=horizon,
        lead_time_default=lead_time_default,
        unit_cost_default=unit_cost_default,
        mode="budget" if budget is not None else "service_level",
        service_level=service_level,
        budget=budget,
        n_jobs=jobs,
    )

    outputs = run_pipeline_from_path(str(input), mapping, config)

    output.parent.mkdir(parents=True, exist_ok=True)
    inventory_output.parent.mkdir(parents=True, exist_ok=True)
    outputs.forecast.to_csv(output, index=False)
    outputs.inventory.to_csv(inventory_output, index=False)
    typer.echo(
        f"wrote {len(outputs.forecast)} forecast rows to {output}, "
        f"{len(outputs.inventory)} SKU inventory rows to {inventory_output}"
    )


if __name__ == "__main__":
    app()
