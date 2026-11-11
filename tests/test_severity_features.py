"""Feature engineering, scored-model wrapper and chronological CV (no leakage)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, pytest
from sklearn.linear_model import LogisticRegression
from v4_pipeline import severity_features as sf, severity_improve as si, severity_v2 as sv
from v4_pipeline.prepare_highway import CODEBOOK


def _raw(**kw):
    base = {"time_minutes": [30, 12 * 60, 23 * 60 + 30], "weekday_from_date_iso": ["5", "2", "0"],
            sf.V1: [6, 1, 5], sf.V2: [1, 9, 1]}
    base.update(kw); return pd.DataFrame(base)


def test_engineered_features_values():
    out = sf.SeverityFeatures().transform(_raw())
    assert out["is_night"].tolist() == [1.0, 0.0, 1.0] and out["is_weekend"].tolist() == [1.0, 0.0, 0.0]
    assert out["n_vulnerable"].tolist() == [1.0, 1.0, 0.0] and out["n_heavy"].tolist() == [0.0, 0.0, 1.0]
    assert out["heavy_and_vulnerable"].tolist() == [0.0, 0.0, 0.0] and out["vulnerable_at_night"].tolist() == [1.0, 0.0, 0.0]
    assert np.allclose(out["hour_sin"] ** 2 + out["hour_cos"] ** 2, 1)


def test_missing_and_odd_values_do_not_crash():
    X = _raw().astype({sf.V1: object, sf.V2: object, "time_minutes": float}); X.loc[0, sf.V1] = np.nan; X.loc[1, "time_minutes"] = np.nan; X.loc[2, sf.V2] = "x"
    out = sf.SeverityFeatures().transform(X)
    assert out["n_vulnerable"].notna().all() and np.isnan(out.loc[1, "is_night"])
    assert set(sf.engineered_names(["chainage_km_numeric"])) == set()           # nothing derivable -> nothing added


def test_vehicle_groups_match_the_codebook_labels():
    lab = CODEBOOK[sf.V1]
    assert {lab[str(c)] for c in sf.VULNERABLE} == {"Two wheeler", "Cycle", "Pedestrian"}
    assert {lab[str(c)] for c in sf.HEAVY} == {"Bus", "Truck", "Tractor"}


def test_chronological_folds_never_train_on_the_future():
    dates = pd.to_datetime("2013-01-01") + pd.to_timedelta(np.random.default_rng(0).permutation(1000), unit="D")
    for k, tr, va in si.chrono_folds(dates.to_numpy()):
        assert dates.to_numpy()[tr].max() <= dates.to_numpy()[va].min()
        assert len(set(tr) & set(va)) == 0


def test_scored_model_averages_and_calibrates():
    rng = np.random.default_rng(1); X = pd.DataFrame({"a": rng.normal(size=300)}); y = (X.a + rng.normal(size=300) > 0).astype(int)
    pipes = [si.Pipeline([("preprocess", sv.make_preprocessor([], ["a"])), ("classifier", LogisticRegression(C=c))]).fit(X, y) for c in (0.1, 1)]
    m = sf.ScoredModel(pipes)
    assert np.allclose(m.predict_proba(X)[:, 1], (pipes[0].predict_proba(X)[:, 1] + pipes[1].predict_proba(X)[:, 1]) / 2)
    cal = LogisticRegression().fit(sf._logit(m.raw_score(X)).reshape(-1, 1), y)
    mc = sf.ScoredModel(pipes, calibrator=cal); p = mc.predict_proba(X)
    assert np.allclose(p.sum(axis=1), 1) and np.corrcoef(p[:, 1], m.raw_score(X))[0, 1] > 0.99
    assert "preprocess" in mc.named_steps                                          # API reads categories from here
