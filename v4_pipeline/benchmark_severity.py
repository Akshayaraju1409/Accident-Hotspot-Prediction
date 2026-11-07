"""
RoadWatch V4: benchmark six severity-classification algorithms on one fixed
chronological holdout. This predicts severity CONDITIONAL ON A RECORDED CRASH.
It does not predict whether/where a crash will occur.

Run from accident_hotspot_v2:
    python -m v4_pipeline.benchmark_severity

Optional models:
    python -m pip install xgboost catboost
"""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_recall_fscore_support, confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "v4" / "processed" / "highway_accidents_clean.csv"
ART = ROOT / "artifacts_v4"
TARGET = "Accident_Severity_C"
CUTOFF = pd.Timestamp("2018-12-02")
LABELS = ["1", "2", "3", "4"]
RANDOM_STATE = 42

# Causes_D is intentionally excluded: it may be assigned after the crash and
# can leak information unavailable at the point a severity estimate is made.
CAT_FEATURES = [
    "Accident_Location_A",
    "Accident_Location_A_Chainage_km_RoadSide",
    "Road_Feature_E",
    "Road_Condition_F",
    "Weather_Conditions_H",
    "Vehicle_Type_Involved_J_V1",
    "Vehicle_Type_Involved_J_V2",
    "weekday_from_date_iso",
]
NUM_FEATURES = [
    "Accident_Location_A_Chainage_km",
    "time_minutes",
    "month",
]

def make_preprocessor(cat_cols, num_cols):
    transformers = []
    if cat_cols:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("categorical", cat_pipe, cat_cols))
    if num_cols:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])
        transformers.append(("numeric", num_pipe, num_cols))
    # Dense output keeps optional CatBoost/XGBoost handling consistent.
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)

def load_data():
    if not DATA.exists():
        raise FileNotFoundError(
            f"Missing {DATA}. Run: python -m v4_pipeline.prepare_highway"
        )
    df = pd.read_csv(DATA)
    if TARGET not in df:
        raise ValueError(f"Target column {TARGET!r} is missing.")
    date = pd.to_datetime(df["Date"].astype(str).str.strip(),
                          dayfirst=True, errors="coerce")
    time = pd.to_datetime(df["Time_of_Accident"].astype(str).str.strip(),
                          format="%H:%M", errors="coerce")
    df["date_for_split"] = date
    df["weekday_from_date_iso"] = date.dt.dayofweek.astype("Int64").astype("string")
    df["time_minutes"] = time.dt.hour * 60 + time.dt.minute
    df["month"] = date.dt.month
    df[TARGET] = df[TARGET].astype("string").str.strip()
    df = df.dropna(subset=["date_for_split", TARGET]).copy()
    # Only the four documented target codes are retained for this experiment.
    df = df[df[TARGET].isin(LABELS)].copy()
    if df.empty:
        raise ValueError("No rows remain after parsing dates and severity codes.")
    train = df[df["date_for_split"] < CUTOFF].copy()
    test = df[df["date_for_split"] >= CUTOFF].copy()
    if train.empty or test.empty:
        raise ValueError(f"Fixed cutoff {CUTOFF.date()} produced an empty split.")
    return df, train, test

def evaluate(name, estimator, X_train, y_train, X_test, y_test, fit_kwargs=None):
    fit_kwargs = fit_kwargs or {}
    estimator.fit(X_train, y_train, **fit_kwargs)
    pred = estimator.predict(X_test)
    p, r, f, support = precision_recall_fscore_support(
        y_test, pred, labels=LABELS, zero_division=0
    )
    report = classification_report(
        y_test, pred, labels=LABELS, output_dict=True, zero_division=0
    )
    metrics = {
        "model": name,
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, labels=LABELS,
                                   average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, pred, labels=LABELS,
                                      average="weighted", zero_division=0)),
        "fatal_recall_class_1": float(r[0]),
        "fatal_precision_class_1": float(p[0]),
        "per_class": {
            label: {
                "precision": float(p[i]), "recall": float(r[i]),
                "f1": float(f[i]), "support": int(support[i])
            } for i, label in enumerate(LABELS)
        },
        "confusion_matrix_labels_1_2_3_4": confusion_matrix(
            y_test, pred, labels=LABELS
        ).tolist(),
        "classification_report": report,
    }
    return metrics, estimator

def metrics_from_predictions(name, y_test, pred):
    p, r, f, support = precision_recall_fscore_support(y_test, pred, labels=LABELS, zero_division=0)
    report = classification_report(y_test, pred, labels=LABELS, output_dict=True, zero_division=0)
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, labels=LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, pred, labels=LABELS, average="weighted", zero_division=0)),
        "fatal_recall_class_1": float(r[0]), "fatal_precision_class_1": float(p[0]),
        "per_class": {label: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(support[i])} for i, label in enumerate(LABELS)},
        "confusion_matrix_labels_1_2_3_4": confusion_matrix(y_test, pred, labels=LABELS).tolist(),
        "classification_report": report,
    }

def main():
    ART.mkdir(parents=True, exist_ok=True)
    df, train, test = load_data()
    cat_cols = [c for c in CAT_FEATURES if c in df.columns]
    num_cols = [c for c in NUM_FEATURES if c in df.columns]
    feature_cols = cat_cols + num_cols

    # Imputation/encoding are fitted inside each pipeline on training data only.
    X_train = train[feature_cols].copy()
    X_test = test[feature_cols].copy()
    for c in cat_cols:
        X_train[c] = X_train[c].astype(object).where(X_train[c].notna(), np.nan)
        X_test[c] = X_test[c].astype(object).where(X_test[c].notna(), np.nan)
    y_train = train[TARGET].astype(str)
    y_test = test[TARGET].astype(str)

    # Compute balanced weights only for classes actually present in training.
    class_counts = y_train.value_counts()
    present_classes = list(class_counts.index)
    weight_by_class = {
        cls: len(y_train) / (len(present_classes) * int(class_counts[cls]))
        for cls in present_classes
    }
    train_sample_weights = y_train.map(weight_by_class).astype(float).to_numpy()

    def pipe(model):
        return Pipeline([
            ("preprocess", make_preprocessor(cat_cols, num_cols)),
            ("classifier", model),
        ])

    candidates = {
        "Dummy (most frequent)": pipe(DummyClassifier(strategy="most_frequent")),
        "Logistic Regression (balanced)": pipe(LogisticRegression(
            max_iter=3000, C=1.0,
            solver="lbfgs", random_state=RANDOM_STATE
        )),
        "Random Forest (balanced)": pipe(RandomForestClassifier(
            n_estimators=600, min_samples_leaf=3, max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )),
        "Extra Trees (balanced)": pipe(ExtraTreesClassifier(
            n_estimators=600, min_samples_leaf=3, max_features="sqrt",
            random_state=RANDOM_STATE, n_jobs=-1
        )),
    }

    results = []
    fitted = {}
    for name, model in candidates.items():
        print(f"Training {name} ...")
        fit_kwargs = {} if name.startswith("Dummy") else {
            "classifier__sample_weight": train_sample_weights
        }
        metric, fitted_model = evaluate(
            name, model, X_train, y_train, X_test, y_test, fit_kwargs
        )
        results.append(metric)
        fitted[name] = fitted_model

    # Optional libraries use the exact same train-fitted preprocessing matrix.
    # Target labels are encoded 0..3 for XGBoost, then decoded for metrics.
    shared_preprocessor = make_preprocessor(cat_cols, num_cols)
    Xtr_matrix = shared_preprocessor.fit_transform(X_train)
    Xte_matrix = shared_preprocessor.transform(X_test)
    label_to_int = {label: i for i, label in enumerate(LABELS)}
    y_train_int = y_train.map(label_to_int).to_numpy()
    y_test_text = y_test.to_numpy()

    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            objective="multi:softprob", num_class=len(LABELS),
            n_estimators=500, max_depth=4, learning_rate=0.04,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=2.0,
            eval_metric="mlogloss", tree_method="hist",
            random_state=RANDOM_STATE, n_jobs=-1
        )
        print("Training XGBoost ...")
        xgb.fit(Xtr_matrix, y_train_int, sample_weight=train_sample_weights)
        xgb_pred = np.asarray([LABELS[int(i)] for i in xgb.predict(Xte_matrix)])
        metric = metrics_from_predictions("XGBoost (balanced sample weights)", y_test_text, xgb_pred)
        results.append(metric)
        fitted["XGBoost (balanced sample weights)"] = {"preprocessor": shared_preprocessor, "classifier": xgb, "label_mapping": label_to_int}
    except ImportError:
        print("XGBoost not installed; skipped. Install with: python -m pip install xgboost")
    except Exception as exc:
        warnings.warn(f"XGBoost skipped due to runtime issue: {exc}")

    try:
        from catboost import CatBoostClassifier
        print("Training CatBoost ...")
        cat = CatBoostClassifier(
            loss_function="MultiClass", iterations=600, depth=5,
            learning_rate=0.04, l2_leaf_reg=4.0,
            verbose=False, random_seed=RANDOM_STATE, allow_writing_files=False
        )
        cat.fit(Xtr_matrix, y_train_int, sample_weight=train_sample_weights)
        cat_pred_int = cat.predict(Xte_matrix).astype(int).reshape(-1)
        cat_pred = np.asarray([LABELS[int(i)] for i in cat_pred_int])
        metric = metrics_from_predictions("CatBoost (balanced class weights)", y_test_text, cat_pred)
        results.append(metric)
        fitted["CatBoost (balanced class weights)"] = {"preprocessor": shared_preprocessor, "classifier": cat, "label_mapping": label_to_int}
    except ImportError:
        print("CatBoost not installed; skipped. Install with: python -m pip install catboost")
    except Exception as exc:
        warnings.warn(f"CatBoost skipped due to runtime issue: {exc}")

    # Primary selection criterion: macro-F1. Fatal recall is a separately visible
    # safety-critical metric; it is not hidden inside overall accuracy.
    ranked = sorted(results, key=lambda m: (
        m["macro_f1"], m["fatal_recall_class_1"], m["balanced_accuracy"]
    ), reverse=True)
    winner = ranked[0]
    winner_name = winner["model"]
    joblib.dump(fitted[winner_name], ART / "highway_severity_best_benchmark.joblib")

    summary = {
        "task": "severity classification among observed crash records only",
        "not_a_crash_occurrence_model": True,
        "split": "fixed chronological holdout",
        "cutoff_date": str(CUTOFF.date()),
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "train_date_range": [str(train.date_for_split.min().date()),
                             str(train.date_for_split.max().date())],
        "test_date_range": [str(test.date_for_split.min().date()),
                            str(test.date_for_split.max().date())],
        "target": TARGET, "class_labels": {
            "1": "Fatal", "2": "Grievous injury",
            "3": "Minor injury", "4": "No injury"
        },
        "features": feature_cols,
        "excluded_feature": {
            "Causes_D": "Excluded because accident-cause coding may be assigned after the event and could leak information into a severity estimate."
        },
        "selection_rule": "Highest macro-F1; ties broken by fatal-class recall, then balanced accuracy.",
        "selected_model": winner_name,
        "selected_model_metrics": winner,
        "ranking": [
            {k: m[k] for k in ["model", "accuracy", "balanced_accuracy",
                               "macro_f1", "weighted_f1", "fatal_recall_class_1",
                               "fatal_precision_class_1"]}
            for m in ranked
        ],
        "all_model_metrics": results,
        "limitations": [
            "The dataset contains recorded crashes only; this model estimates severity conditional on a recorded crash.",
            "Fatal recall should be interpreted with its test support and precision; improving recall can increase false alarms.",
            "The row-level highway corridor identifier is absent, so geographic generalisation cannot be tested.",
            "This benchmark uses a fixed historical cutoff to match the prior report, but a single holdout is not sufficient evidence of robust generalisation.",
            "Category meanings are based only on mappings documented in the partial codebook; ambiguous cause codes are excluded.",
            "No probability calibration or operational threshold selection has been performed."
        ]
    }
    (ART / "highway_severity_benchmark.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    pd.DataFrame(summary["ranking"]).to_csv(
        ART / "highway_severity_model_comparison.csv", index=False
    )
    print("\\n=== ROADWATCH V4 SEVERITY MODEL COMPARISON ===")
    print(pd.DataFrame(summary["ranking"]).to_string(index=False))
    print(f"\\nSelected model: {winner_name}")
    print(f"Metrics: {ART / 'highway_severity_benchmark.json'}")
    print(f"Comparison: {ART / 'highway_severity_model_comparison.csv'}")
    print("Note: XGBoost/CatBoost are included only if their packages are installed.")

if __name__ == "__main__":
    main()
