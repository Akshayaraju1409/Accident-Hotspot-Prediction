"""
Feature engineering and scoring wrapper for the highway severity model.

SeverityFeatures adds a few interpretable columns derived ONLY from the raw inputs (no target, no future data),
so it is safe inside a cross-validated sklearn Pipeline and at prediction time in the API.

ScoredModel bundles one or more fitted pipelines (an ensemble) with an optional probability calibrator.
It exposes `named_steps` of its first member so api/severity.py can read the category codes without changes.

Vehicle groups follow the codes documented in prepare_highway.CODEBOOK:
  vulnerable road user: 6 Two wheeler, 8 Cycle, 9 Pedestrian
  heavy vehicle:        3 Bus, 5 Truck, 10 Tractor
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

V1, V2 = "Vehicle_Type_Involved_J_V1", "Vehicle_Type_Involved_J_V2"
VULNERABLE = {6, 8, 9}
HEAVY = {3, 5, 10}
NIGHT_HOURS = set(range(22, 24)) | set(range(0, 6))


def engineered_names(available):
    """Engineered column names that can be built from the raw columns present in `available`."""
    a = set(available); out = []
    if "time_minutes" in a:
        out += ["hour_sin", "hour_cos", "is_night"]
    if "weekday_from_date_iso" in a:
        out += ["is_weekend"]
    if V1 in a or V2 in a:
        out += ["n_vulnerable", "n_heavy", "heavy_and_vulnerable"]
    if "time_minutes" in a and (V1 in a or V2 in a):
        out += ["vulnerable_at_night"]
    return out


def _code(series):
    return pd.to_numeric(series, errors="coerce")


class SeverityFeatures(BaseEstimator, TransformerMixin):
    """Adds engineered numeric columns to a DataFrame of raw features. Stateless; unknown/missing codes count as 0."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "time_minutes" in X:
            minutes = _code(X["time_minutes"])
            hour = (minutes // 60) % 24
            X["hour_sin"] = np.sin(2 * np.pi * minutes / 1440.0)
            X["hour_cos"] = np.cos(2 * np.pi * minutes / 1440.0)
            X["is_night"] = hour.isin(NIGHT_HOURS).astype(float).where(minutes.notna(), np.nan)
        if "weekday_from_date_iso" in X:
            X["is_weekend"] = _code(X["weekday_from_date_iso"]).isin([5, 6]).astype(float)
        vul = pd.Series(0.0, index=X.index); hvy = pd.Series(0.0, index=X.index)
        for col in (V1, V2):
            if col in X:
                c = _code(X[col])
                vul = vul + c.isin(VULNERABLE).astype(float)
                hvy = hvy + c.isin(HEAVY).astype(float)
        if V1 in X or V2 in X:
            X["n_vulnerable"] = vul
            X["n_heavy"] = hvy
            X["heavy_and_vulnerable"] = ((vul > 0) & (hvy > 0)).astype(float)
            if "is_night" in X:
                X["vulnerable_at_night"] = (vul > 0).astype(float) * X["is_night"].fillna(0.0)
        return X


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class ScoredModel:
    """Weighted mean of fitted pipelines' P(severe), optionally mapped through a Platt calibrator.

    Behaves like a fitted binary classifier: predict_proba(X) -> (n, 2). `named_steps` proxies the first member.
    """

    classes_ = np.array([0, 1])

    def __init__(self, members, weights=None, calibrator=None, name="model"):
        self.members = list(members)
        self.weights = np.ones(len(self.members)) / len(self.members) if weights is None else np.asarray(weights, float)
        self.calibrator = calibrator
        self.name = name

    @property
    def named_steps(self):
        return self.members[0].named_steps

    def raw_score(self, X):
        return sum(w * m.predict_proba(X)[:, 1] for w, m in zip(self.weights, self.members))

    def predict_proba(self, X):
        p = self.raw_score(X)
        if self.calibrator is not None:
            p = self.calibrator.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
        return np.column_stack([1 - p, p])
