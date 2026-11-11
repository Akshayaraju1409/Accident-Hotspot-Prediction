"""Unit tests for the analysis helpers (no model training needed)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from v4_pipeline import severity_analysis as sa


def test_collapsed_probability_sums_fatal_and_grievous():
    proba = np.array([[.1, .2, .3, .4], [.7, .1, .1, .1]])
    assert np.allclose(sa.severe_prob_from_proba(proba, ["1", "2", "3", "4"]), [.3, .8])
    assert np.allclose(sa.severe_prob_from_proba(proba[:, ::-1], ["4", "3", "2", "1"]), [.3, .8])   # order independent


def test_threshold_tradeoff_counts_are_consistent():
    rng = np.random.default_rng(0); y = rng.integers(0, 2, 500); p = rng.random(500)
    y4 = np.where(y == 1, rng.choice(["1", "2"], 500), rng.choice(["3", "4"], 500))
    t = sa.threshold_tradeoff(y, y4, p)
    assert (t.missed_severe + (t.severe_recall * y.sum()).round()).eq(y.sum()).all()
    assert t.severe_recall.is_monotonic_decreasing and t.flagged_share_of_all_crashes.is_monotonic_decreasing


def test_error_profile_flags_a_concentrated_miss_pattern():
    n = 600; rng = np.random.default_rng(1)
    test = pd.DataFrame({"Accident_Location_A": rng.choice([1, 2], n), "time_minutes": rng.integers(0, 1440, n),
                         "weekday_from_date_iso": rng.integers(0, 7, n).astype(str), "chainage_km_numeric": rng.uniform(0, 700, n),
                         "Vehicle_Type_Involved_J_V1": rng.choice([1, 6], n)})
    severe = np.ones(n, bool)
    fn = (test.Vehicle_Type_Involved_J_V1 == 6).to_numpy() & (rng.random(n) < 0.9)     # misses concentrated on one vehicle type
    prof = sa.error_profile(test, fn, severe)
    row = prof[(prof.feature == "Vehicle_Type_Involved_J_V1") & (prof.value == "Two wheeler")].iloc[0]
    assert row.fn_rate > 0.8 and bool(row.flag_after_bonferroni)


def test_calibration_and_markdown_helpers():
    p = np.linspace(0.01, 0.99, 200); y = (p > 0.5).astype(int)
    c = sa.calibration_deciles(y, p); assert len(c) == 10 and c.observed_severe_rate.is_monotonic_increasing
    md = sa.md_table(pd.DataFrame({"a": [1, 2], "b": [0.5, np.nan], "c": [True, False]}))
    assert "| 1 | 0.500 | True |" in md and "| 2 |  | False |" in md
