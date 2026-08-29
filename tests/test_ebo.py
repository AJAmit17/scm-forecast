import numpy as np
import pandas as pd

from scm_forecast.ebo import (
    expected_backorders_curve,
    optimize_stock_budget,
    recommend_stock_service_level,
)


def test_ebo_curve_monotonically_decreasing_in_stock():
    _ss, ebo, fill = expected_backorders_curve(mean=20.0, var=35.0, max_stock=80)
    assert np.all(np.diff(ebo) <= 1e-9)
    assert np.all(np.diff(fill) >= -1e-9)
    assert fill[-1] > 0.99


def test_ebo_zero_when_mean_is_zero():
    _ss, ebo, fill = expected_backorders_curve(mean=0.0, var=0.0, max_stock=10)
    assert np.all(ebo == 0.0)
    assert np.all(fill == 1.0)


def test_higher_service_level_yields_more_stock():
    df = pd.DataFrame(
        [{"unique_id": "X", "mean_ltd": 15.0, "var_ltd": 40.0, "unit_cost": 5.0, "current_stock": 0.0}]
    )
    low = recommend_stock_service_level(df, target=0.80)
    high = recommend_stock_service_level(df, target=0.98)
    assert high["recommended_stock"].iloc[0] >= low["recommended_stock"].iloc[0]
    assert high["achieved_fill_rate"].iloc[0] >= 0.98 - 1e-6


def test_budget_optimization_prefers_cheap_high_variance_skus_under_tight_budget():
    df = pd.DataFrame(
        [
            {"unique_id": "cheap", "mean_ltd": 10.0, "var_ltd": 30.0, "unit_cost": 1.0, "current_stock": 0.0},
            {"unique_id": "expensive", "mean_ltd": 10.0, "var_ltd": 30.0, "unit_cost": 100.0, "current_stock": 0.0},
        ]
    )
    result = optimize_stock_budget(df, budget=20.0).set_index("unique_id")
    assert result.loc["cheap", "recommended_stock"] > 0
    assert result.loc["expensive", "recommended_stock"] == 0


def test_budget_optimization_never_exceeds_budget_by_construction():
    df = pd.DataFrame(
        [
            {"unique_id": f"sku{i}", "mean_ltd": 5.0, "var_ltd": 8.0, "unit_cost": 3.0, "current_stock": 0.0}
            for i in range(5)
        ]
    )
    budget = 17.0
    result = optimize_stock_budget(df, budget=budget)
    spent = (result["recommended_stock"] * result["unit_cost"]).sum()
    assert spent <= budget + 1e-9
