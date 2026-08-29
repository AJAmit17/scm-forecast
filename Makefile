.PHONY: install sample-data app forecast test lint fmt clean

install:
	uv sync

sample-data:
	uv run python scripts/generate_sample.py

app:
	uv run streamlit run src/scm_forecast/app.py

forecast: sample-data
	uv run scm-forecast \
		--input data/sample_demand.xlsx \
		--output output/forecast.csv \
		--inventory-output output/inventory_ebo.csv

test:
	uv run pytest -q

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

clean:
	rm -rf .venv output/*.csv data/sample_demand.xlsx .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
