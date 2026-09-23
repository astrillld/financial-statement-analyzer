"""Unit tests for the financial statement analyzer."""

import pandas as pd
import pytest

from analyzer import calculate_metrics, detect_anomalies, load_data


def _toy() -> pd.DataFrame:
    # One steady company plus one obvious expense spike in 2022.
    return pd.DataFrame({
        "company": ["A"] * 6,
        "year":    [2017, 2018, 2019, 2020, 2021, 2022],
        "revenue": [100, 110, 121, 133, 146, 160],
        "expenses":[ 80,  88,  97, 106, 117, 150],   # spike in 2022
        "profit":  [ 20,  22,  24,  27,  29,  10],   # profit collapses in 2022
    })


def test_metrics_are_correct():
    df = calculate_metrics(_toy())
    row = df[df["year"] == 2018].iloc[0]
    assert row["profit_margin"] == pytest.approx(22 / 110)
    assert row["expense_ratio"] == pytest.approx(88 / 110)
    assert row["revenue_growth"] == pytest.approx((110 - 100) / 100)


def test_first_year_growth_is_nan():
    df = calculate_metrics(_toy())
    first = df[df["year"] == 2017].iloc[0]
    assert pd.isna(first["revenue_growth"])


def test_detects_planted_expense_spike():
    df = calculate_metrics(_toy())
    anomalies = detect_anomalies(df, z_threshold=1.5)
    spikes = anomalies[(anomalies["year"] == 2022)
                       & (anomalies["metric"] == "expense_ratio")]
    assert len(spikes) == 1
    assert spikes.iloc[0]["z_score"] > 0


def test_no_flags_on_flat_data():
    df = calculate_metrics(pd.DataFrame({
        "company": ["B"] * 5,
        "year": [2018, 2019, 2020, 2021, 2022],
        "revenue": [100, 110, 121, 133, 146],   # constant +10% growth
        "expenses": [80, 88, 96.8, 106.4, 116.8],
        "profit": [20, 22, 24.2, 26.6, 29.2],
    }))
    anomalies = detect_anomalies(df, z_threshold=2.0)
    assert anomalies.empty


def test_load_data_rejects_bad_schema(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("company,year,revenue\nA,2020,100\n")
    with pytest.raises(ValueError):
        load_data(bad)
