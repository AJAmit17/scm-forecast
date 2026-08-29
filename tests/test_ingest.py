import pandas as pd

from scm_forecast.ingest import prepare_long_frame
from scm_forecast.schema import ColumnMapping


def test_missing_dates_become_explicit_zero_rows():
    raw = pd.DataFrame(
        {
            "sku": ["A", "A", "A"],
            "date": ["2025-01-01", "2025-01-03", "2025-01-05"],
            "qty": [10, 20, 30],
        }
    )
    mapping = ColumnMapping(sku="sku", date="date", qty="qty")
    long_df, _attrs = prepare_long_frame(raw, mapping, freq="D")

    assert len(long_df) == 5  # Jan 1..5 inclusive
    zero_days = long_df[long_df["y"] == 0]
    assert len(zero_days) == 2


def test_optional_attributes_extracted_and_last_wins():
    raw = pd.DataFrame(
        {
            "sku": ["A", "A", "B"],
            "date": ["2025-01-01", "2025-01-02", "2025-01-01"],
            "qty": [1, 2, 3],
            "lead_time_days": [10, 20, 5],
            "unit_cost": [1.5, 1.5, 9.0],
        }
    )
    mapping = ColumnMapping(
        sku="sku", date="date", qty="qty", lead_time="lead_time_days", unit_cost="unit_cost"
    )
    _long_df, attrs = prepare_long_frame(raw, mapping, freq="D")
    attrs = attrs.set_index("unique_id")

    assert attrs.loc["A", "lead_time_days"] == 20  # last row for A
    assert attrs.loc["B", "unit_cost"] == 9.0


def test_missing_required_column_raises():
    raw = pd.DataFrame({"date": ["2025-01-01"], "qty": [1]})
    mapping = ColumnMapping(sku="sku", date="date", qty="qty")
    try:
        prepare_long_frame(raw, mapping, freq="D")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "sku" in str(exc)


def test_yymm_date_format_parses_year_month_codes():
    raw = pd.DataFrame(
        {
            "Item": ["123", "123", "123"],
            "YYMM": ["2301", 2302, "2607"],  # mixed str/int, as Excel may yield
            "Actuals": [100, 110, 400],
        }
    )
    mapping = ColumnMapping(sku="Item", date="YYMM", qty="Actuals", date_format="yymm")
    long_df, _attrs = prepare_long_frame(raw, mapping, freq="MS")

    parsed = set(long_df["ds"])
    assert pd.Timestamp("2023-01-01") in parsed
    assert pd.Timestamp("2023-02-01") in parsed
    assert pd.Timestamp("2026-07-01") in parsed
    # gap between Feb 2023 and Jul 2026 must be explicit zero-filled, not just 3 rows
    assert len(long_df) > 3


def test_merged_cell_sku_column_is_forward_filled():
    # Excel export where Item is only populated on the first row of each block.
    raw = pd.DataFrame(
        {
            "Item": ["123", None, None, "456", None],
            "YYMM": ["2301", "2302", "2303", "2301", "2302"],
            "Actuals": [100, 110, 90, 50, 60],
        }
    )
    mapping = ColumnMapping(sku="Item", date="YYMM", qty="Actuals", date_format="yymm")
    long_df, _attrs = prepare_long_frame(raw, mapping, freq="MS")

    assert set(long_df["unique_id"]) == {"123", "456"}
    assert long_df[long_df["unique_id"] == "123"]["y"].sum() == 300
    assert long_df[long_df["unique_id"] == "456"]["y"].sum() == 110


def test_blank_spacer_rows_between_blocks_are_dropped():
    raw = pd.DataFrame(
        {
            "Item": ["123", None, None, None],
            "YYMM": ["2301", None, None, "2303"],
            "Actuals": [100, None, None, 90],
        }
    )
    mapping = ColumnMapping(sku="Item", date="YYMM", qty="Actuals", date_format="yymm")
    long_df, _attrs = prepare_long_frame(raw, mapping, freq="MS")

    # Jan + Feb (implicit zero, gap-filled) + Mar = 3 rows, all attributed to SKU 123
    assert len(long_df) == 3
    assert set(long_df["unique_id"]) == {"123"}


def test_numeric_sku_ids_are_not_rendered_as_floats():
    raw = pd.DataFrame({"Item": [123.0, 123.0], "YYMM": ["2301", "2302"], "Actuals": [1, 2]})
    mapping = ColumnMapping(sku="Item", date="YYMM", qty="Actuals", date_format="yymm")
    long_df, _attrs = prepare_long_frame(raw, mapping, freq="MS")
    assert set(long_df["unique_id"]) == {"123"}
