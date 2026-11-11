"""
RoadWatch V4 Severity V2
Primary task: severe vs non-severe classification among recorded crashes.
Severe = Fatal (1) OR Grievous injury (2).
Non-severe = Minor injury (3) OR No injury (4).

Also trains a four-class Extra Trees reference model for comparison.
This is NOT an accident-occurrence or geographic hotspot prediction model.

Run from accident_hotspot_v2:
    python -m v4_pipeline.severity_v2
Optional:
    python -m pip install xgboost catboost
"""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
    precision_recall_curve, average_precision_score, roc_auc_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "v4" / "processed" / "highway_accidents_clean.csv"
ART = ROOT / "artifacts_v4"
TARGET = "Accident_Severity_C"
TEST_START = pd.Timestamp("2018-12-02")
VALIDATION_START = pd.Timestamp("2017-01-01")
LABELS_4 = ["1", "2", "3", "4"]
RANDOM_STATE = 42
MIN_VALIDATION_SPECIFICITY = 0.50

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
    "chainage_km_numeric",
    "time_minutes",
    "month",
]

def make_preprocessor(cat_cols, num_cols):
    parts = []
    if cat_cols:
        parts.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_cols))
    if num_cols:
        parts.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), num_cols))
    return ColumnTransformer(parts, remainder="drop", sparse_threshold=0.0)

def prepare_data():
    if not DATA.exists():
        raise FileNotFoundError(
            f"Missing {DATA}. Run `python -m v4_pipeline.prepare_highway` first."
        )
    df = pd.read_csv(DATA)
    required = [TARGET, "Date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    n_source = len(df)
    # accident_date_parsed is written by prepare_highway as ISO (YYYY-MM-DD). Re-parsing ISO text with
    # dayfirst=True makes pandas infer "%Y-%d-%m": rows with day > 12 become NaT (about 60% of the data is
    # silently dropped) and day/month are swapped on the rest (corrupting month and weekday).
    if "accident_date_parsed" in df:
        dates = pd.to_datetime(df["accident_date_parsed"], format="ISO8601", errors="coerce")
    else:  # raw source column is day-first
        dates = pd.to_datetime(df["Date"].astype(str).str.strip(), dayfirst=True, errors="coerce")
    df["__date"] = dates
    # Always derive weekday/month from the correctly parsed date (Causes_D stays excluded: possible leakage).
    df["weekday_from_date_iso"] = dates.dt.day_of_week.astype("Int64").astype("string")
    if "time_minutes" not in df:
        tm = pd.to_datetime(df.get("Time_of_Accident", pd.Series(index=df.index, dtype="object")),
                            format="%H:%M", errors="coerce")
        df["time_minutes"] = tm.dt.hour * 60 + tm.dt.minute
    df["month"] = dates.dt.month
    if "chainage_km_numeric" not in df:
        source = "Accident_Location_A_Chainage_km"
        df["chainage_km_numeric"] = pd.to_numeric(df[source], errors="coerce") if source in df else np.nan

    df[TARGET] = df[TARGET].astype("string").str.strip()
    df = df.dropna(subset=["__date", TARGET]).copy()
    df = df[df[TARGET].isin(LABELS_4)].copy()
    lost = n_source - len(df)
    print(f"Rows: source {n_source}, usable {len(df)}, dropped {lost} ({lost / max(n_source, 1):.1%})")
    if lost / max(n_source, 1) > 0.02:
        raise ValueError(f"{lost} of {n_source} rows were dropped (missing date/target). Check date parsing before modelling.")
    df["severe_binary"] = df[TARGET].isin(["1", "2"]).astype(int)
    df = df.sort_values("__date").reset_index(drop=True)

    train = df[df["__date"] < VALIDATION_START].copy()
    validation = df[(df["__date"] >= VALIDATION_START) & (df["__date"] < TEST_START)].copy()
    test = df[df["__date"] >= TEST_START].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError(
            "Temporal split produced an empty partition. Check source dates and split constants."
        )
    for name, part in [("train", train), ("validation", validation), ("test", test)]:
        if part["severe_binary"].nunique() < 2:
            raise ValueError(f"{name} partition does not contain both binary target classes.")
    return df, train, validation, test

def clean_features(part, cat_cols, num_cols):
    X = part[cat_cols + num_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X

def make_candidates(cat_cols, num_cols):
    def pipe(clf):
        return Pipeline([
            ("preprocess", make_preprocessor(cat_cols, num_cols)),
            ("classifier", clf),
        ])
    models = {
        "Extra Trees": pipe(ExtraTreesClassifier(
            n_estimators=700, min_samples_leaf=2, max_features="sqrt",
            random_state=RANDOM_STATE, n_jobs=-1
        )),
        "Random Forest": pipe(RandomForestClassifier(
            n_estimators=700, min_samples_leaf=2, max_features="sqrt",
            random_state=RANDOM_STATE, n_jobs=-1
        )),
        "Logistic Regression": pipe(LogisticRegression(
            max_iter=4000, C=0.5, solver="lbfgs", random_state=RANDOM_STATE
        )),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = pipe(XGBClassifier(
            objective="binary:logistic", n_estimators=500, max_depth=3,
            learning_rate=0.035, subsample=0.85, colsample_bytree=0.85,
            reg_lambda=3.0, eval_metric="logloss", tree_method="hist",
            random_state=RANDOM_STATE, n_jobs=-1
        ))
    except ImportError:
        print("Optional XGBoost not installed; skipped.")
    try:
        from catboost import CatBoostClassifier
        models["CatBoost"] = pipe(CatBoostClassifier(
            iterations=600, depth=5, learning_rate=0.035, l2_leaf_reg=5,
            loss_function="Logloss", verbose=False, random_seed=RANDOM_STATE,
            allow_writing_files=False
        ))
    except ImportError:
        print("Optional CatBoost not installed; skipped.")
    return models

def fbeta(precision, recall, beta=2.0):
    denom = beta * beta * precision + recall
    return 0.0 if denom == 0 else (1 + beta * beta) * precision * recall / denom

def choose_threshold(y_true, probabilities):
    # Threshold selection is performed ONLY on the validation partition.
    rows = []
    for threshold in np.round(np.arange(0.10, 0.901, 0.01), 2):
        pred = (probabilities >= threshold).astype(int)
        p = precision_score(y_true, pred, zero_division=0)
        r = recall_score(y_true, pred, zero_division=0)
        specificity = recall_score(1 - y_true, 1 - pred, zero_division=0)
        rows.append({
            "threshold": float(threshold),
            "precision": float(p),
            "recall": float(r),
            "specificity": float(specificity),
            "f2": float(fbeta(p, r, beta=2.0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        })
    # Prioritise severe recall via F2, but require at least 50% specificity
    # on validation to prevent the trivial "predict every crash as severe" rule.
    eligible = [r for r in rows if r["specificity"] >= MIN_VALIDATION_SPECIFICITY]
    if eligible:
        eligible.sort(key=lambda x: (x["f2"], x["precision"], x["threshold"]), reverse=True)
        return eligible[0], rows
    # If no threshold meets the floor, choose the highest-specificity option.
    rows.sort(key=lambda x: (x["specificity"], x["f2"], x["precision"]), reverse=True)
    return rows[0], rows

def binary_metrics(y_true, probabilities, threshold):
    pred = (probabilities >= threshold).astype(int)
    p = precision_score(y_true, pred, zero_division=0)
    r = recall_score(y_true, pred, zero_division=0)
    result = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "severe_precision": float(p),
        "severe_recall": float(r),
        "severe_f2": float(fbeta(p, r, 2.0)),
        "non_severe_specificity": float(recall_score(1 - y_true, 1 - pred, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "confusion_matrix_labels_non_severe_0_severe_1": confusion_matrix(
            y_true, pred, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            y_true, pred, labels=[0, 1],
            target_names=["Non-severe", "Severe"], output_dict=True, zero_division=0
        ),
    }
    try:
        result["roc_auc"] = float(roc_auc_score(y_true, probabilities))
    except ValueError:
        result["roc_auc"] = None
    return result

def bootstrap_ci(y_true, probabilities, threshold, n_boot=2000, seed=RANDOM_STATE):
    """95% bootstrap intervals. With a few hundred test rows an AUC of 0.60 is only +/- ~0.06 wide."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true); p = np.asarray(probabilities)
    auc, bal = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        auc.append(roc_auc_score(y[i], p[i]))
        bal.append(balanced_accuracy_score(y[i], (p[i] >= threshold).astype(int)))
    return {"roc_auc_95ci": [float(np.percentile(auc, 2.5)), float(np.percentile(auc, 97.5))],
            "balanced_accuracy_95ci": [float(np.percentile(bal, 2.5)), float(np.percentile(bal, 97.5))],
            "n_boot": int(len(auc))}

def year_profile(parts):
    """Per-year rows / severe rate / median chainage. Large swings across years mean the chronological split
    is also a change of road section (chainage has no corridor id), not just a change of time."""
    rows = []
    for name, part in parts:
        g = part.groupby(part["__date"].dt.year)
        for year, x in g:
            rows.append({"partition": name, "year": int(year), "rows": int(len(x)),
                         "severe_rate": float(x["severe_binary"].mean()),
                         "median_chainage_km": float(pd.to_numeric(x["chainage_km_numeric"], errors="coerce").median())})
    return rows

def ablation_without_chainage(winner_name, cat_cols, num_cols, train, val, test):
    """Does the model rely on chainage (a proxy for 'which road section / which project')?"""
    drop = {"Accident_Location_A_Chainage_km_RoadSide", "chainage_km_numeric"}
    cats = [c for c in cat_cols if c not in drop]; nums = [c for c in num_cols if c not in drop]
    model = make_candidates(cats, nums)[winner_name]
    y_tr = train["severe_binary"].astype(int).to_numpy()
    model.fit(clean_features(train, cats, nums), y_tr, classifier__sample_weight=compute_sample_weight("balanced", y_tr))
    out = {}
    for name, part in [("validation", val), ("test", test)]:
        prob = model.predict_proba(clean_features(part, cats, nums))[:, 1]
        out[f"{name}_roc_auc"] = float(roc_auc_score(part["severe_binary"].astype(int), prob))
    return out

def main():
    ART.mkdir(parents=True, exist_ok=True)
    df, train, val, test = prepare_data()
    cat_cols = [c for c in CAT_FEATURES if c in df.columns]
    num_cols = [c for c in NUM_FEATURES if c in df.columns]
    features = cat_cols + num_cols
    if not features:
        raise ValueError("No supported model features found in the prepared dataset.")

    X_train = clean_features(train, cat_cols, num_cols)
    X_val = clean_features(val, cat_cols, num_cols)
    X_pretest = clean_features(pd.concat([train, val]), cat_cols, num_cols)
    X_test = clean_features(test, cat_cols, num_cols)
    y_train = train["severe_binary"].astype(int).to_numpy()
    y_val = val["severe_binary"].astype(int).to_numpy()
    y_pretest = pd.concat([train, val])["severe_binary"].astype(int).to_numpy()
    y_test = test["severe_binary"].astype(int).to_numpy()

    candidates = make_candidates(cat_cols, num_cols)
    validation_results = []
    validation_models = {}
    for name, model in candidates.items():
        print(f"Fitting {name} on pre-2017 training data ...")
        weights = compute_sample_weight("balanced", y_train)
        try:
            model.fit(X_train, y_train, classifier__sample_weight=weights)
            val_prob = model.predict_proba(X_val)[:, 1]
            chosen, threshold_curve = choose_threshold(y_val, val_prob)
            metrics = binary_metrics(y_val, val_prob, chosen["threshold"])
            validation_results.append({
                "model": name,
                "validation_metrics_at_selected_threshold": metrics,
                "threshold_selection": chosen,
                "validation_rows": int(len(val)),
            })
            validation_models[name] = {
                "model": model, "threshold": chosen["threshold"],
                "threshold_curve": threshold_curve,
            }
            print(f"  validation severe F2={metrics['severe_f2']:.3f}; "
                  f"recall={metrics['severe_recall']:.3f}; threshold={chosen['threshold']:.2f}")
        except Exception as exc:
            warnings.warn(f"{name} skipped because this installed library version is incompatible: {exc}")

    # Choose on validation F2 only; test data is not used for model or threshold selection.
    validation_results.sort(key=lambda x: (
        x["validation_metrics_at_selected_threshold"]["severe_f2"],
        x["validation_metrics_at_selected_threshold"]["severe_precision"]
    ), reverse=True)
    winner_name = validation_results[0]["model"]
    threshold = validation_models[winner_name]["threshold"]

    # Keep the model trained only on the training partition. This ensures the
    # validation threshold is applied to the same fitted model and avoids
    # threshold/model mismatch caused by refitting after threshold selection.
    winner = validation_models[winner_name]["model"]
    test_prob = winner.predict_proba(X_test)[:, 1]
    test_binary = binary_metrics(y_test, test_prob, threshold)
    test_ci = bootstrap_ci(y_test, test_prob, threshold)
    try:
        ablation = ablation_without_chainage(winner_name, cat_cols, num_cols, train, val, test)
    except Exception as exc:  # diagnostic only; never block the main result
        ablation = {"error": str(exc)}
    joblib.dump(
        {"model": winner, "threshold": threshold,
         "positive_class": "Severe (Fatal or Grievous injury)",
         "negative_class": "Non-severe (Minor or No injury)",
         "features": features},
        ART / "highway_severity_v2_binary_best.joblib"
    )

    # Preserve a four-class reference experiment. It is not the prioritised output.
    X4_train = X_train
    y4_train = train[TARGET].astype(str).to_numpy()
    y4_test = test[TARGET].astype(str).to_numpy()
    four_class = Pipeline([
        ("preprocess", make_preprocessor(cat_cols, num_cols)),
        ("classifier", ExtraTreesClassifier(
            n_estimators=700, min_samples_leaf=2, max_features="sqrt",
            random_state=RANDOM_STATE, n_jobs=-1
        ))
    ])
    w4 = compute_sample_weight("balanced", y4_train)
    four_class.fit(X4_train, y4_train, classifier__sample_weight=w4)
    pred4 = four_class.predict(X_test)
    four_class_metrics = {
        "accuracy": float(accuracy_score(y4_test, pred4)),
        "balanced_accuracy": float(balanced_accuracy_score(y4_test, pred4)),
        "macro_f1": float(f1_score(y4_test, pred4, labels=LABELS_4, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y4_test, pred4, labels=LABELS_4, average="weighted", zero_division=0)),
        "class_labels": {"1": "Fatal", "2": "Grievous injury", "3": "Minor injury", "4": "No injury"},
        "confusion_matrix_labels_1_2_3_4": confusion_matrix(y4_test, pred4, labels=LABELS_4).tolist(),
        "classification_report": classification_report(
            y4_test, pred4, labels=LABELS_4, output_dict=True, zero_division=0
        ),
    }
    joblib.dump(four_class, ART / "highway_severity_v2_four_class_reference.joblib")

    split_info = {
        "training": {"rows": int(len(train)), "start": str(train["__date"].min().date()), "end": str(train["__date"].max().date())},
        "validation": {"rows": int(len(val)), "start": str(val["__date"].min().date()), "end": str(val["__date"].max().date())},
        "test": {"rows": int(len(test)), "start": str(test["__date"].min().date()), "end": str(test["__date"].max().date())},
    }
    report = {
        "project": "RoadWatch V4 Severity V2",
        "primary_task": "Binary severe-vs-non-severe classification among observed crash records",
        "severe_definition": {"positive": ["1 Fatal", "2 Grievous injury"], "negative": ["3 Minor injury", "4 No injury"]},
        "not_a_crash_occurrence_model": True,
        "split_method": "Chronological train/validation/test; model and threshold selected on validation only; final test held out.",
        "split_boundaries": {
            "train_before": str(VALIDATION_START.date()),
            "validation_from": str(VALIDATION_START.date()),
            "validation_before": str(TEST_START.date()),
            "test_from": str(TEST_START.date()),
        },
        "split": split_info,
        "features": features,
        "excluded_feature": {
            "Causes_D": "Excluded because it may be coded after the crash and leak outcome information."
        },
        "model_selection": {
             "criterion": "Highest validation severe-class F2 subject to at least 0.50 validation specificity; ties broken by severe precision. This avoids labelling nearly every crash severe.",
            "minimum_validation_specificity": MIN_VALIDATION_SPECIFICITY,
            "selected_model": winner_name,
            "selected_threshold_from_validation": float(threshold),
            "validation_model_results": validation_results,
        },
        "rows_used": int(len(df)),
        "final_binary_test_metrics": test_binary,
        "final_binary_test_uncertainty": test_ci,
        "test_severe_prevalence_(ap_chance_level)": float(y_test.mean()),
        "ablation_without_chainage": ablation,
        "year_profile": year_profile([("train", train), ("validation", val), ("test", test)]),
        "four_class_reference_test_metrics": four_class_metrics,
        "limitations": [
            "The model estimates severity conditional on a recorded crash; it does not predict crash occurrence or location.",
            "Severe combines fatal and grievous injury to prioritise higher-harm outcomes.",
            "The validation and test periods are historical and contain gaps in calendar-year coverage.",
             "Threshold was selected on validation data with a minimum specificity constraint; it must not be retuned against the final test set.",
            "Only 11 contextual features are used; no traffic exposure, speed, road geometry, or reliable corridor identifier is available.",
            "Model probabilities have not been calibrated; scores should not be interpreted as calibrated real-world risk.",
            "This model is a research prototype and is not suitable for emergency or operational safety decisions."
        ],
    }
    (ART / "highway_severity_v2_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame([
        {
            "model": x["model"],
            "validation_severe_precision": x["validation_metrics_at_selected_threshold"]["severe_precision"],
            "validation_severe_recall": x["validation_metrics_at_selected_threshold"]["severe_recall"],
            "validation_severe_f2": x["validation_metrics_at_selected_threshold"]["severe_f2"],
            "validation_balanced_accuracy": x["validation_metrics_at_selected_threshold"]["balanced_accuracy"],
            "selected_threshold": x["threshold_selection"]["threshold"],
        } for x in validation_results
    ]).to_csv(ART / "highway_severity_v2_validation_comparison.csv", index=False)

    print("\n=== ROADWATCH V4 SEVERITY V2 ===")
    print(f"Selected model: {winner_name}")
    print(f"Validation-selected threshold: {threshold:.2f}")
    print("\nFinal untouched test results:")
    for k in ["accuracy", "balanced_accuracy", "severe_precision", "severe_recall", "severe_f2", "non_severe_specificity", "average_precision", "roc_auc"]:
        print(f"{k}: {test_binary.get(k)}")
    print("\nFour-class reference:")
    print(f"macro F1: {four_class_metrics['macro_f1']:.4f}")
    print(f"balanced accuracy: {four_class_metrics['balanced_accuracy']:.4f}")
    print(f"test ROC-AUC 95% CI: {test_ci['roc_auc_95ci']}  (chance 0.5); severe prevalence {y_test.mean():.3f}")
    print(f"ablation without chainage: {ablation}")
    print(f"\nSaved outputs in: {ART}")

if __name__ == "__main__":
    main()
