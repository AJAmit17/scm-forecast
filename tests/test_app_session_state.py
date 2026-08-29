"""Regression test for a Streamlit session-state bug: `st.button()` only
returns True in the exact rerun its click triggers. Any other widget
interaction afterwards (e.g. picking a different SKU from the "Select SKU"
selectbox) causes a NEW rerun in which the button reports False again - if
the app gates its results behind `if not run_clicked: st.stop()`, the entire
results view (including every non-default SKU) disappears the instant a user
tries to inspect anything but the SKU shown right after clicking Run. This
made every category except whichever SKU was selected by default look
"broken" to a user just clicking around.

Fix: persist pipeline outputs in `st.session_state`, keyed independently of
the button's per-rerun return value.
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parents[1] / "src" / "scm_forecast" / "app.py")
SAMPLE_XLSX = Path(__file__).resolve().parents[1] / "data" / "sample_demand.xlsx"


def _run_app_with_sample() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    uploader = at.get("file_uploader")[0]
    file_bytes = SAMPLE_XLSX.read_bytes()
    uploader.set_value(
        (SAMPLE_XLSX.name, file_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    )
    at.run()
    return at


def test_results_persist_after_selecting_a_different_sku():
    if not SAMPLE_XLSX.exists():
        import subprocess

        subprocess.run(["uv", "run", "python", "scripts/generate_sample.py"], check=True, cwd=Path(__file__).resolve().parents[1])

    at = _run_app_with_sample()
    assert not at.exception

    run_button = next(b for b in at.button if b.label == "Run pipeline")
    run_button.click().run()
    assert not at.exception, f"Run pipeline raised: {[str(e) for e in at.exception]}"

    selectbox = next(sb for sb in at.selectbox if sb.label == "Select SKU")
    all_skus = list(selectbox.options)
    assert len(all_skus) >= 5  # every SKU in the sample file, not just one

    # Regression: switching the SKU selection must NOT wipe the results view
    # or raise, for every SKU/category in the sample data.
    for sku in all_skus:
        selectbox.set_value(sku).run()
        assert not at.exception, f"SKU {sku!r} raised: {[str(e) for e in at.exception]}"
        subheaders = [h.value for h in at.subheader]
        assert "Per-SKU history + forecast" in subheaders
        assert "EBO / inventory recommendation (per SKU)" in subheaders
