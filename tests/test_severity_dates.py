"""Regression: severity_v2 must not lose rows or swap day/month when it reads the prepared highway CSV."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, pytest
from v4_pipeline import severity_v2 as sv


@pytest.fixture()
def prepared_csv(tmp_path, monkeypatch):
    rng = np.random.default_rng(0); n = 600
    d = pd.to_datetime("2013-01-01") + pd.to_timedelta(rng.integers(0, 365 * 10, n), unit="D")
    df = pd.DataFrame({"Date": d.strftime("%d-%m-%Y"), "accident_date_parsed": d.strftime("%Y-%m-%d"),
                       "Time_of_Accident": "10:30", "Accident_Severity_C": rng.choice([1, 2, 3, 4], n)})
    p = tmp_path / "highway_accidents_clean.csv"; df.to_csv(p, index=False)
    monkeypatch.setattr(sv, "DATA", p)
    return d


def test_no_rows_lost_and_dates_unchanged(prepared_csv):
    d = prepared_csv
    df, train, val, test = sv.prepare_data()
    assert len(df) == len(d)                                   # old code kept ~39% (rows with day <= 12)
    assert sorted(df["__date"]) == sorted(d)                   # old code swapped day and month
    assert (df["month"] == df["__date"].dt.month).all()
    assert (df["weekday_from_date_iso"].astype(int) == df["__date"].dt.dayofweek).all()


def test_guard_raises_if_many_rows_dropped(tmp_path, monkeypatch):
    df = pd.DataFrame({"Date": ["x"] * 50, "accident_date_parsed": ["not-a-date"] * 50,
                       "Time_of_Accident": "10:30", "Accident_Severity_C": [1, 2] * 25})
    p = tmp_path / "h.csv"; df.to_csv(p, index=False); monkeypatch.setattr(sv, "DATA", p)
    with pytest.raises(ValueError, match="dropped"):
        sv.prepare_data()
