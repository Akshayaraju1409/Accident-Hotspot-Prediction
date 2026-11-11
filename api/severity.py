"""RoadWatch V4: severity prediction endpoints (severe vs non-severe AMONG RECORDED highway crashes).

Mounted by api/main.py. This is NOT a crash-occurrence or hotspot model.

  GET  /api/v4/severity/schema    form definition: allowed codes + labels, model info, limitations
  POST /api/v4/severity/predict   score one crash
  GET  /api/v4/severity/metrics   headline test metrics of the saved model
  GET  /api/v4/severity/analysis  ablation / comparison / error analysis (after severity_analysis.py)
"""
from pathlib import Path
import datetime as dt
import json
import re
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from v4_pipeline.prepare_highway import CODEBOOK

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts_v4"
PROCESSED = ROOT / "data" / "v4" / "processed"
router = APIRouter(prefix="/api/v4/severity", tags=["severity"])

# request field -> model column
FIELDS = {
    "area_type": "Accident_Location_A",
    "road_side": "Accident_Location_A_Chainage_km_RoadSide",
    "road_feature": "Road_Feature_E",
    "road_condition": "Road_Condition_F",
    "weather": "Weather_Conditions_H",
    "vehicle_1": "Vehicle_Type_Involved_J_V1",
    "vehicle_2": "Vehicle_Type_Involved_J_V2",
}
FIELD_TITLES = {
    "area_type": "Area type", "road_side": "Road side (source code, unmapped)", "road_feature": "Road feature",
    "road_condition": "Road condition", "weather": "Weather", "vehicle_1": "Vehicle 1", "vehicle_2": "Vehicle 2",
}
# groups used for the "what pushed the score" explanation: label -> model columns
EXPLAIN_GROUPS = {
    "Area type": ["Accident_Location_A"],
    "Location (chainage, road side)": ["chainage_km_numeric", "Accident_Location_A_Chainage_km_RoadSide"],
    "Road feature": ["Road_Feature_E"], "Road condition": ["Road_Condition_F"], "Weather": ["Weather_Conditions_H"],
    "Vehicle 1": ["Vehicle_Type_Involved_J_V1"], "Vehicle 2": ["Vehicle_Type_Involved_J_V2"],
    "Date (month, weekday)": ["month", "weekday_from_date_iso"], "Time of day": ["time_minutes"],
}
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_STATE = {}


def reset_cache():
    _STATE.clear()


def _norm(v):
    """'1', 1, 1.0 -> '1' so user input matches whatever type the training categories have."""
    try:
        return str(int(float(v)))
    except (TypeError, ValueError):
        return str(v).strip()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else None


def _bundle():
    if "b" in _STATE:
        return _STATE["b"]
    p = ART / "highway_severity_v2_binary_best.joblib"
    if not p.exists():
        raise HTTPException(503, "Severity model not trained yet. Run: python -m v4_pipeline.severity_v2")
    try:
        saved = joblib.load(p)
    except Exception as exc:  # e.g. library version mismatch
        raise HTTPException(503, f"Could not load the severity model ({exc}). Retrain with python -m v4_pipeline.severity_v2")
    model = saved["model"]
    categories, cat_cols = {}, []
    for name, trans, cols in model.named_steps["preprocess"].transformers_:
        if name == "cat":
            ohe = trans.named_steps["onehot"]
            for col, cats in zip(cols, ohe.categories_):
                categories[col] = {_norm(c): c for c in cats}
                cat_cols.append(col)
    metrics = _read_json(ART / "highway_severity_v2_metrics.json") or {}
    analysis = _read_json(ART / "analysis" / "severity_analysis.json")
    _STATE["b"] = {"model": model, "threshold": float(saved["threshold"]), "features": list(saved["features"]),
                   "calibrated": bool(saved.get("calibrated", False)), "members": saved.get("members"),
                   "categories": categories, "cat_cols": cat_cols, "metrics": metrics, "analysis": analysis}
    return _STATE["b"]


def _background(features):
    """Reference crashes (from the cleaned training file) for the explanation. None if unavailable."""
    if "bg" in _STATE:
        return _STATE["bg"]
    bg = None
    p = PROCESSED / "highway_accidents_clean.csv"
    if p.exists():
        try:
            df = pd.read_csv(p)
            d = (pd.to_datetime(df["accident_date_parsed"], format="ISO8601", errors="coerce")
                 if "accident_date_parsed" in df else pd.to_datetime(df["Date"].astype(str), dayfirst=True, errors="coerce"))
            t = pd.to_datetime(df["Time_of_Accident"].astype(str).str.strip(), format="%H:%M", errors="coerce")
            frame = pd.DataFrame({c: df[c].astype(object) for c in FIELDS.values() if c in df})
            frame["chainage_km_numeric"] = pd.to_numeric(df["Accident_Location_A_Chainage_km"], errors="coerce")
            frame["time_minutes"] = t.dt.hour * 60 + t.dt.minute
            frame["month"] = d.dt.month
            frame["weekday_from_date_iso"] = d.dt.day_of_week.astype("Int64").astype("string").astype(object)
            frame = frame.dropna(subset=["month"])
            bg = frame.sample(min(250, len(frame)), random_state=0).reset_index(drop=True)[features]
        except Exception:
            bg = None
    _STATE["bg"] = bg
    return bg


def _label(col, code):
    return CODEBOOK.get(col, {}).get(_norm(code), f"Code {_norm(code)}")


class SeverityRequest(BaseModel):
    date: dt.date = Field(..., description="Crash date (YYYY-MM-DD)")
    time: str = Field(..., pattern=r"^\d{1,2}:\d{2}$", description="Crash time, 24h HH:MM")
    chainage_km: Optional[float] = Field(None, ge=0, le=5000)
    area_type: Optional[Union[int, str]] = None
    road_side: Optional[Union[int, str]] = None
    road_feature: Optional[Union[int, str]] = None
    road_condition: Optional[Union[int, str]] = None
    weather: Optional[Union[int, str]] = None
    vehicle_1: Optional[Union[int, str]] = None
    vehicle_2: Optional[Union[int, str]] = None


def _frame(req, b):
    hh, mm = (int(x) for x in req.time.split(":"))
    if hh > 23 or mm > 59:
        raise HTTPException(422, "time must be a valid 24-hour HH:MM")
    row = {}
    for field, col in FIELDS.items():
        v = getattr(req, field)
        if v is None or str(v).strip() == "":
            row[col] = np.nan                      # the model imputes the most frequent training value
            continue
        known = b["categories"].get(col, {})
        if _norm(v) not in known:
            raise HTTPException(422, f"{field}: unknown code {v!r}. Allowed: {sorted(known)}")
        row[col] = known[_norm(v)]
    row["weekday_from_date_iso"] = b["categories"]["weekday_from_date_iso"].get(str(req.date.weekday()), str(req.date.weekday()))
    row["month"] = req.date.month
    row["time_minutes"] = hh * 60 + mm
    row["chainage_km_numeric"] = np.nan if req.chainage_km is None else float(req.chainage_km)
    X = pd.DataFrame([row])[b["features"]]
    for c in b["cat_cols"]:
        X[c] = X[c].astype(object)
    return X


def _contributions(b, X, score):
    bg = _background(b["features"])
    if bg is None:
        return None
    out = []
    for label, cols in EXPLAIN_GROUPS.items():
        cols = [c for c in cols if c in X.columns]
        Xr = pd.concat([X] * len(bg), ignore_index=True)
        for c in cols:
            Xr[c] = bg[c].to_numpy()
        for c in b["cat_cols"]:
            Xr[c] = Xr[c].astype(object)
        typical = float(b["model"].predict_proba(Xr)[:, 1].mean())
        out.append({"factor": label, "effect": round(score - typical, 4)})
    return sorted(out, key=lambda r: -abs(r["effect"]))


def _describe(req):
    d = {"Date": f"{req.date.isoformat()} ({WEEKDAYS[req.date.weekday()]})", "Time": req.time}
    for field, col in FIELDS.items():
        v = getattr(req, field)
        if v is not None and str(v).strip() != "":
            d[FIELD_TITLES[field]] = _label(col, v)
    if req.chainage_km is not None:
        d["Chainage (km)"] = req.chainage_km
    return d


def _model_info(b):
    m = b["metrics"]
    test = m.get("final_binary_test_metrics", {}); unc = m.get("final_binary_test_uncertainty", {})
    return {"name": m.get("model_selection", {}).get("selected_model", "unknown"),
            "task": "Severe (fatal or grievous injury) vs non-severe, among recorded crashes",
            "threshold": b["threshold"], "calibrated": b.get("calibrated", False), "members": b.get("members"),
            "test_roc_auc": test.get("roc_auc"), "test_roc_auc_95ci": unc.get("roc_auc_95ci"),
            "test_balanced_accuracy": test.get("balanced_accuracy"), "test_severe_recall": test.get("severe_recall"),
            "test_severe_precision": test.get("severe_precision"), "rows_used": m.get("rows_used"),
            "split": m.get("split")}


def _nearest_decile(b, score):
    cal = (b["analysis"] or {}).get("calibration_deciles")
    if not cal:
        return None
    d = min(cal, key=lambda r: abs(r["mean_predicted"] - score))
    return {"observed_severe_rate_for_similar_scores": round(d["observed_severe_rate"], 3),
            "from_test_crashes": int(d["rows"]), "score_band_mean": round(d["mean_predicted"], 3)}


@router.get("/schema")
def schema():
    b = _bundle(); info = _model_info(b)
    fields = []
    for field, col in FIELDS.items():
        known = b["categories"].get(col, {})
        fields.append({"name": field, "title": FIELD_TITLES[field], "column": col, "type": "select", "required": False,
                       "options": [{"value": k, "label": _label(col, k) if col in CODEBOOK else f"Code {k}"}
                                   for k in sorted(known, key=lambda x: (len(x), x))]})
    bg = _background(b["features"])
    rng = ([float(bg["chainage_km_numeric"].min()), float(bg["chainage_km_numeric"].max())]
           if bg is not None and bg["chainage_km_numeric"].notna().any() else [0, 800])
    return {"model": info, "fields": fields, "chainage_range_km": rng,
            "limitations": limitations(info), "status": "ready"}


def limitations(info):
    auc = info.get("test_roc_auc"); ci = info.get("test_roc_auc_95ci")
    weak = (f"Test ROC-AUC is {auc:.2f}" + (f" (95% CI {ci[0]:.2f} to {ci[1]:.2f})" if ci else "") +
            ", where 0.5 is chance: discrimination is weak.") if auc else "Discrimination is weak."
    return ["Estimates severity of a crash that has already happened; it does not predict whether or where crashes occur.",
            weak,
            ("The score is calibrated on past recorded crashes, so it reads roughly as the share of similar crashes that were severe. "
             "It is still not a probability of death or injury for any individual crash." if info.get("calibrated") else
             "The score is an uncalibrated model output, not a probability of death or injury."),
            "No speed, traffic volume, road geometry or corridor ID is available to the model.",
            "Research prototype; not for emergency or operational decisions."]


@router.post("/predict")
def predict(req: SeverityRequest):
    b = _bundle()
    X = _frame(req, b)
    score = float(b["model"].predict_proba(X)[:, 1][0])
    flagged = score >= b["threshold"]
    info = _model_info(b)
    contrib = _contributions(b, X, score)
    warnings = list(limitations(info))
    if contrib is None:
        warnings.append("Explanation unavailable: data/v4/processed/highway_accidents_clean.csv not found.")
    return {"input": _describe(req),
            "prediction": {"label": "Severe (fatal or grievous)" if flagged else "Non-severe (minor or no injury)",
                           "flagged_severe": bool(flagged), "score": round(score, 4), "threshold": b["threshold"],
                           "margin_to_threshold": round(score - b["threshold"], 4),
                           "history_for_similar_scores": _nearest_decile(b, score)},
            "explanation": {"meaning": "How much each input moved this score versus a typical recorded crash. "
                                       "Patterns in the training data, not causes.",
                            "contributions": contrib},
            "model": info, "warnings": warnings}


@router.get("/metrics")
def metrics():
    b = _bundle(); m = b["metrics"]
    return {"model": _model_info(b), "four_class_reference": m.get("four_class_reference_test_metrics"),
            "year_profile": m.get("year_profile"), "ablation_without_chainage": m.get("ablation_without_chainage"),
            "limitations": limitations(_model_info(b))}


@router.get("/analysis")
def analysis():
    b = _bundle()
    if not b["analysis"]:
        raise HTTPException(404, "Run: python -m v4_pipeline.severity_analysis")
    return b["analysis"]
