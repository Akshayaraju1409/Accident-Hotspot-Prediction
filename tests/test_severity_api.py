"""Severity API: schema, prediction, validation errors, missing-model handling. Uses a tiny model built in the test."""
import sys, pathlib, json; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, joblib, pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from api import severity as api
from v4_pipeline import severity_v2 as sv


@pytest.fixture()
def client(tmp_path, monkeypatch):
    rng = np.random.default_rng(0); n = 400
    d = pd.to_datetime("2013-01-01") + pd.to_timedelta(rng.integers(0, 365 * 6, n), unit="D")
    df = pd.DataFrame({"accident_date_parsed": d.strftime("%Y-%m-%d"), "Time_of_Accident": "10:30",
                       "Accident_Location_A": rng.choice([1, 2], n), "Accident_Location_A_Chainage_km": rng.uniform(0, 700, n),
                       "Accident_Location_A_Chainage_km_RoadSide": rng.choice([1, 2], n), "Road_Feature_E": rng.integers(1, 5, n),
                       "Road_Condition_F": rng.integers(1, 9, n), "Weather_Conditions_H": rng.integers(1, 6, n),
                       "Vehicle_Type_Involved_J_V1": rng.integers(1, 16, n), "Vehicle_Type_Involved_J_V2": rng.integers(1, 16, n)})
    X = pd.DataFrame({c: df[c].astype(object) for c in sv.CAT_FEATURES if c in df})
    X["weekday_from_date_iso"] = d.dayofweek.astype(str).astype(object); X["month"] = d.month
    X["time_minutes"] = 630; X["chainage_km_numeric"] = df["Accident_Location_A_Chainage_km"]
    y = (rng.random(n) < 0.45 + 0.2 * (df.Vehicle_Type_Involved_J_V1 == 6)).astype(int)
    cats, nums = [c for c in sv.CAT_FEATURES if c in X], [c for c in sv.NUM_FEATURES if c in X]
    model = Pipeline([("preprocess", sv.make_preprocessor(cats, nums)), ("classifier", LogisticRegression(max_iter=500))]).fit(X[cats + nums], y)
    art, proc = tmp_path / "artifacts_v4", tmp_path / "processed"; art.mkdir(); proc.mkdir()
    joblib.dump({"model": model, "threshold": 0.45, "features": cats + nums}, art / "highway_severity_v2_binary_best.joblib")
    (art / "highway_severity_v2_metrics.json").write_text(json.dumps({"model_selection": {"selected_model": "Logistic Regression"},
        "final_binary_test_metrics": {"roc_auc": 0.58, "balanced_accuracy": 0.55}, "final_binary_test_uncertainty": {"roc_auc_95ci": [0.54, 0.62]}}))
    df.to_csv(proc / "highway_accidents_clean.csv", index=False)
    monkeypatch.setattr(api, "ART", art); monkeypatch.setattr(api, "PROCESSED", proc); api.reset_cache()
    app = FastAPI(); app.include_router(api.router)
    yield TestClient(app); api.reset_cache()


BODY = {"date": "2020-01-13", "time": "21:45", "chainage_km": 120.5, "area_type": 2, "weather": 1, "vehicle_1": 6, "vehicle_2": "9"}


def test_schema_lists_fields_with_labels(client):
    r = client.get("/api/v4/severity/schema"); assert r.status_code == 200; j = r.json()
    assert {f["name"] for f in j["fields"]} >= {"area_type", "weather", "vehicle_1", "vehicle_2", "road_feature"}
    veh = next(f for f in j["fields"] if f["name"] == "vehicle_1")
    assert {"value": "6", "label": "Two wheeler"} in veh["options"]
    assert any("weak" in s for s in j["limitations"])


def test_predict_returns_consistent_result(client):
    j = client.post("/api/v4/severity/predict", json=BODY).json(); p = j["prediction"]
    assert 0 <= p["score"] <= 1 and p["flagged_severe"] == (p["score"] >= p["threshold"])
    assert j["input"]["Vehicle 1"] == "Two wheeler" and "Monday" in j["input"]["Date"]
    eff = [c["effect"] for c in j["explanation"]["contributions"]]
    assert eff and eff == sorted(eff, key=abs, reverse=True)
    assert any("does not predict whether or where" in w for w in j["warnings"])


def test_optional_fields_are_imputed(client):
    r = client.post("/api/v4/severity/predict", json={"date": "2021-06-02", "time": "08:00"}); assert r.status_code == 200


def test_unknown_code_and_bad_time_rejected(client):
    assert client.post("/api/v4/severity/predict", json={**BODY, "vehicle_1": 99}).status_code == 422
    assert client.post("/api/v4/severity/predict", json={**BODY, "time": "25:99"}).status_code == 422
    assert client.post("/api/v4/severity/predict", json={**BODY, "time": "noon"}).status_code == 422


def test_weekday_and_month_come_from_the_date(client):
    b = api._bundle(); req = api.SeverityRequest(**BODY); X = api._frame(req, b)
    assert X.loc[0, "weekday_from_date_iso"] == "0" and X.loc[0, "month"] == 1 and X.loc[0, "time_minutes"] == 21 * 60 + 45


def test_missing_model_gives_503(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "ART", tmp_path); api.reset_cache()
    app = FastAPI(); app.include_router(api.router)
    r = TestClient(app).get("/api/v4/severity/schema"); assert r.status_code == 503 and "severity_v2" in r.json()["detail"]


def test_main_app_mounts_the_routes():
    from api.main import app
    paths = set(app.openapi()["paths"])      # stable across FastAPI versions
    assert {"/api/v4/severity/predict", "/api/v4/severity/schema", "/api/v4/highway/metrics"} <= paths
