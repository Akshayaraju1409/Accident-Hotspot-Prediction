"""M5-M7: models, calibration, metrics."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import config as C


def make_model(name, pos_weight=1.0, seed=C.SEED, **kw):
    if name == "LogReg (baseline)":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(max_iter=500, class_weight="balanced"))
    if name == "RandomForest":
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestClassifier(300, min_samples_leaf=20, class_weight="balanced_subsample",
                                                    n_jobs=-1, random_state=seed))
    if name == "XGBoost":
        return XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                             scale_pos_weight=pos_weight, n_jobs=-1, random_state=seed, verbosity=0)
    if name == "LightGBM":
        p = dict(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=50, subsample=0.8,
                 subsample_freq=1, colsample_bytree=0.8, scale_pos_weight=pos_weight, random_state=seed,
                 n_jobs=-1, verbose=-1); p.update(kw)
        return LGBMClassifier(**p)
    if name == "MLP":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             MLPClassifier((32, 16), alpha=1e-3, early_stopping=True, max_iter=200, random_state=seed))
    raise ValueError(name)


def proba(m, X):
    return m.predict_proba(X)[:, 1]


def sample_negatives(y, neg_per_pos, rng):
    pos = np.flatnonzero(y == 1); neg = np.flatnonzero(y == 0)
    take = rng.choice(neg, size=min(len(neg), neg_per_pos * len(pos)), replace=False)
    return np.sort(np.r_[pos, take])


class Calibrator:
    """Isotonic calibration fitted on the (full, unsampled) validation period, which also
    corrects the prior shift introduced by negative undersampling."""
    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)

    def fit(self, score, y):
        self.iso.fit(score, y); return self

    def __call__(self, score):
        return self.iso.predict(score)


def ece(y, p, bins=10):
    q = pd.qcut(p, bins, duplicates="drop") if len(np.unique(p)) > bins else pd.Series(p).astype(str)
    d = pd.DataFrame({"y": y, "p": p, "b": np.asarray(q)}).groupby("b", observed=True).agg(y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
    return float((d.n * (d.y - d.p).abs()).sum() / d.n.sum())


def topk_recall(y, s, frac):
    k = int(len(s) * frac); idx = np.argpartition(-s, k)[:k]
    return float(y[idx].sum() / max(y.sum(), 1))


def best_f1_threshold(y, s):
    qs = np.unique(np.quantile(s, np.linspace(0.5, 0.9995, 80)))
    best = (0, qs[0])
    for t in qs:
        pred = s >= t; tp = (pred & (y == 1)).sum()
        pr = tp / max(pred.sum(), 1); rc = tp / max(y.sum(), 1)
        f = 2 * pr * rc / max(pr + rc, 1e-12)
        if f > best[0]: best = (f, t)
    return float(best[1])


def f1_at(y, s, t):
    pred = s >= t; tp = float((pred & (y == 1)).sum())
    pr = tp / max(pred.sum(), 1); rc = tp / max(y.sum(), 1)
    return dict(precision=pr, recall=rc, f1=2 * pr * rc / max(pr + rc, 1e-12))


def bootstrap_ap(y, s, n=200, neg_frac=0.1, seed=C.SEED):
    """Bootstrap CI for PR-AUC (negatives subsampled and re-weighted for speed)."""
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1); neg = np.flatnonzero(y == 0)
    out = []
    for _ in range(n):
        p = rng.choice(pos, len(pos)); q = rng.choice(neg, int(len(neg) * neg_frac))
        idx = np.r_[p, q]; w = np.r_[np.ones(len(p)), np.full(len(q), 1 / neg_frac)]
        out.append(average_precision_score(y[idx], s[idx], sample_weight=w))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def metrics(y, raw, cal=None, thr=None):
    base = float(y.mean())
    m = dict(prevalence=base, pr_auc=float(average_precision_score(y, raw)),
             roc_auc=float(roc_auc_score(y, raw)))
    m["pr_auc_lift"] = m["pr_auc"] / base
    m["recall_top10pct"] = topk_recall(y, raw, 0.10)
    m["recall_top20pct"] = topk_recall(y, raw, 0.20)
    if thr is not None: m.update(f1_at(y, raw, thr))
    if cal is not None:
        m["brier"] = float(brier_score_loss(y, cal)); m["brier_ref"] = float(brier_score_loss(y, np.full(len(y), base)))
        m["ece"] = ece(y, cal)
    return m
