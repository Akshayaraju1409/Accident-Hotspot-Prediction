"""
RoadWatch V4 severity analysis: feature ablation, model comparison, error analysis.

Run AFTER the patched severity_v2 (it needs the saved models):
    python -m v4_pipeline.severity_v2
    python -m v4_pipeline.severity_analysis

Writes to artifacts_v4/analysis/:
    severity_analysis.json            everything below (also served by the API)
    severity_analysis_report.md       paste-ready tables and plain-language findings
    ablation.csv, model_comparison_binary.csv, model_comparison_fourclass.csv,
    error_by_severity.csv, error_profile.csv, threshold_tradeoff.csv, calibration_deciles.csv,
    false_negatives.csv               every severe crash the model labelled non-severe (test set)
    fig_ablation.png, fig_errors.png

All tasks are severity among RECORDED crashes. None of this predicts whether or where a crash will happen.
"""
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binomtest
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, f1_score,
                             precision_recall_fscore_support, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from . import severity_v2 as sv
from .prepare_highway import CODEBOOK

SEED = sv.RANDOM_STATE
N_BOOT = 1000
MIN_SEVERE_PER_SLICE = 20
LABELS_4 = sv.LABELS_4
SEV_NAMES = {"1": "Fatal", "2": "Grievous injury", "3": "Minor injury", "4": "No injury"}

# Feature groups used for ablation (columns are the model's own feature names).
GROUPS = {
    "Location (chainage + road side)": ["chainage_km_numeric", "Accident_Location_A_Chainage_km_RoadSide"],
    "Area type (urban/rural)": ["Accident_Location_A"],
    "Road (feature + condition)": ["Road_Feature_E", "Road_Condition_F"],
    "Weather": ["Weather_Conditions_H"],
    "Vehicles (V1 + V2)": ["Vehicle_Type_Involved_J_V1", "Vehicle_Type_Involved_J_V2"],
    "Time (hour, month, weekday)": ["time_minutes", "month", "weekday_from_date_iso"],
}


# ------------------------------------------------------------------ helpers
def code_label(col, v):
    try:
        key = str(int(float(v)))
    except (TypeError, ValueError):
        return str(v)
    return CODEBOOK.get(col, {}).get(key, key)


def md_table(df, floatfmt="{:.3f}"):
    if df.empty:
        return "_(none)_\n"
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples(index=False):
        cells = []
        for v in r:
            if isinstance(v, (bool, np.bool_)):
                cells.append(str(bool(v)))
            elif isinstance(v, (float, np.floating)):
                cells.append(floatfmt.format(v) if np.isfinite(v) else "")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def boot_ci(stat_fn, n, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        try:
            vals.append(stat_fn(i))
        except ValueError:
            continue
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def severe_prob_from_proba(proba, classes):
    classes = [str(c) for c in classes]
    return proba[:, classes.index("1")] + proba[:, classes.index("2")]


def eval_binary(name, task, train_rows, p_val, p_test, y_val, y_test, fixed_threshold=None, flag_rate=None):
    """Threshold sources, in order: the model's own deployed threshold; a threshold matching a given flag rate
    (for models whose training included the validation rows, so validation scores are in-sample); otherwise
    chosen on VALIDATION with sv.choose_threshold. The test labels are never used to pick a threshold."""
    if fixed_threshold is not None:
        thr, src = float(fixed_threshold), "deployed threshold"
    elif flag_rate is not None:
        thr, src = float(np.quantile(p_test, 1 - flag_rate)), "matched to the deployed model's flag rate"
    else:
        thr, src = float(sv.choose_threshold(y_val, p_val)[0]["threshold"]), "validation"
    m = sv.binary_metrics(y_test, p_test, thr)
    auc_ci = boot_ci(lambda i: roc_auc_score(y_test[i], p_test[i]), len(y_test))
    return {"model": name, "task": task, "train_rows": int(train_rows),
            "test_roc_auc": m["roc_auc"], "test_roc_auc_95ci": f"{auc_ci[0]:.3f}-{auc_ci[1]:.3f}",
            "test_avg_precision": m["average_precision"], "ap_chance_(prevalence)": float(y_test.mean()),
            "threshold": thr, "threshold_source": src, "severe_recall": m["severe_recall"],
            "severe_precision": m["severe_precision"], "specificity": m["non_severe_specificity"],
            "balanced_accuracy": m["balanced_accuracy"]}


def eval_fourclass(name, train_rows, y4_test, pred4):
    p, r, f, s = precision_recall_fscore_support(y4_test, pred4, labels=LABELS_4, zero_division=0)
    n = len(y4_test)
    mf1_ci = boot_ci(lambda i: f1_score(y4_test[i], pred4[i], labels=LABELS_4, average="macro", zero_division=0), n)
    return {"model": name, "train_rows": int(train_rows), "accuracy": float(accuracy_score(y4_test, pred4)),
            "balanced_accuracy": float(balanced_accuracy_score(y4_test, pred4)),
            "macro_f1": float(f1_score(y4_test, pred4, labels=LABELS_4, average="macro", zero_division=0)),
            "macro_f1_95ci": f"{mf1_ci[0]:.3f}-{mf1_ci[1]:.3f}",
            "fatal_recall": float(r[0]), "fatal_precision": float(p[0])}


def fit_four_class(train, cat_cols, num_cols):
    """Benchmark recipe (benchmark_severity.py): Extra Trees, 600 trees, min_samples_leaf=3, balanced weights."""
    y4 = train[sv.TARGET].astype(str).to_numpy()
    model = Pipeline([("preprocess", sv.make_preprocessor(cat_cols, num_cols)),
                      ("classifier", ExtraTreesClassifier(n_estimators=600, min_samples_leaf=3, max_features="sqrt",
                                                          random_state=SEED, n_jobs=-1))])
    model.fit(sv.clean_features(train, cat_cols, num_cols), y4,
              classifier__sample_weight=compute_sample_weight("balanced", y4))
    return model


# ------------------------------------------------------------------ 1. feature ablation
def _fit_predict(model_name, drop, cat_cols, num_cols, train, parts):
    cats = [c for c in cat_cols if c not in drop]
    nums = [c for c in num_cols if c not in drop]
    candidates = sv.make_candidates(cats, nums)
    model = candidates[model_name]
    y = train["severe_binary"].astype(int).to_numpy()
    model.fit(sv.clean_features(train, cats, nums), y, classifier__sample_weight=compute_sample_weight("balanced", y))
    return {k: model.predict_proba(sv.clean_features(p, cats, nums))[:, 1] for k, p in parts.items()}


def paired_delta(y, p_full, p_alt):
    d = roc_auc_score(y, p_alt) - roc_auc_score(y, p_full)
    ci = boot_ci(lambda i: roc_auc_score(y[i], p_alt[i]) - roc_auc_score(y[i], p_full[i]), len(y), n_boot=500)
    return float(d), ci


def run_ablation(model_names, cat_cols, num_cols, train, val, test):
    y_test = test["severe_binary"].astype(int).to_numpy(); y_val = val["severe_binary"].astype(int).to_numpy()
    all_cols = cat_cols + num_cols
    rows = []
    for model_name in model_names:
        experiments = [("Full model", [])]
        experiments += [(f"Drop: {g}", cols) for g, cols in GROUPS.items()]
        experiments += [(f"Only: {g}", [c for c in all_cols if c not in cols]) for g, cols in GROUPS.items()]
        full = None
        for label, drop in experiments:
            keep = [c for c in all_cols if c not in drop]
            if not keep:
                continue
            pr = _fit_predict(model_name, set(drop), cat_cols, num_cols, train, {"val": val, "test": test})
            if full is None:
                full = pr
            d, dci = paired_delta(y_test, full["test"], pr["test"])
            ci = boot_ci(lambda i: roc_auc_score(y_test[i], pr["test"][i]), len(y_test), n_boot=500)
            rows.append({"model": model_name, "experiment": label, "n_features": len(keep),
                         "val_roc_auc": float(roc_auc_score(y_val, pr["val"])),
                         "test_roc_auc": float(roc_auc_score(y_test, pr["test"])),
                         "test_auc_95ci": f"{ci[0]:.3f}-{ci[1]:.3f}",
                         "delta_vs_full": d, "delta_95ci": f"{dci[0]:+.3f} to {dci[1]:+.3f}",
                         "delta_distinguishable_from_0": bool(dci[0] > 0 or dci[1] < 0)})
    return pd.DataFrame(rows)


def grouped_permutation_importance(model, X_test, y_test, n_repeats=30):
    rng = np.random.default_rng(SEED)
    base = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    rows = []
    for g, cols in GROUPS.items():
        cols = [c for c in cols if c in X_test.columns]
        if not cols:
            continue
        drops = []
        for _ in range(n_repeats):
            Xp = X_test.copy()
            Xp[cols] = X_test[cols].to_numpy()[rng.permutation(len(X_test))]   # shuffle the whole group together
            drops.append(base - roc_auc_score(y_test, model.predict_proba(Xp)[:, 1]))
        rows.append({"feature_group": g, "auc_drop_mean": float(np.mean(drops)),
                     "auc_drop_sd": float(np.std(drops)), "base_test_auc": float(base)})
    return pd.DataFrame(rows).sort_values("auc_drop_mean", ascending=False)


# ------------------------------------------------------------------ 3. error analysis
def time_band(minutes):
    h = (minutes // 60) % 24
    return pd.cut(h, bins=[-1, 5, 9, 15, 21, 24], labels=["Night 00-06", "Morning 06-10", "Midday 10-16",
                                                          "Evening 16-22", "Late night 22-24"]).astype(str)


def error_profile(test, fn, severe):
    slices = {}
    for col in ["Accident_Location_A", "Road_Feature_E", "Road_Condition_F", "Weather_Conditions_H",
                "Vehicle_Type_Involved_J_V1", "Vehicle_Type_Involved_J_V2"]:
        if col in test:
            slices[col] = test[col].map(lambda v, c=col: code_label(c, v))
    slices["time_of_day"] = time_band(pd.to_numeric(test["time_minutes"], errors="coerce").fillna(0))
    slices["time_of_day"].index = test.index
    slices["weekday"] = test["weekday_from_date_iso"].map(
        lambda v: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][int(v)] if pd.notna(v) else "NA")
    ch = pd.to_numeric(test["chainage_km_numeric"], errors="coerce")
    slices["chainage_quartile"] = pd.qcut(ch, 4, duplicates="drop").astype(str)
    overall = fn.sum() / max(severe.sum(), 1)
    rows = []
    for name, s in slices.items():
        for val in s.dropna().unique():
            m = (s == val).to_numpy()
            n_sev = int((severe & m).sum())
            if n_sev < MIN_SEVERE_PER_SLICE:
                continue
            n_fn = int((fn & m).sum())
            rows.append({"feature": name, "value": val, "severe_crashes": n_sev, "false_negatives": n_fn,
                         "fn_rate": n_fn / n_sev, "overall_fn_rate": overall,
                         "p_vs_overall": float(binomtest(n_fn, n_sev, overall).pvalue)})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["flag_after_bonferroni"] = df["p_vs_overall"] < 0.05 / len(df)
        df = df.sort_values("fn_rate", ascending=False)
    return df


def threshold_tradeoff(y_sev, y4, p):
    rows = []
    for t in np.round(np.arange(0.20, 0.801, 0.05), 2):
        pred = (p >= t).astype(int)
        rows.append({"threshold": float(t), "flagged_share_of_all_crashes": float(pred.mean()),
                     "severe_recall": float(recall_score(y_sev, pred, zero_division=0)),
                     "severe_precision": float(precision_score(y_sev, pred, zero_division=0)),
                     "fatal_recall": float(((pred == 1) & (y4 == "1")).sum() / max((y4 == "1").sum(), 1)),
                     "missed_severe": int(((pred == 0) & (y_sev == 1)).sum()),
                     "false_alarms": int(((pred == 1) & (y_sev == 0)).sum())})
    return pd.DataFrame(rows)


def calibration_deciles(y_sev, p):
    d = pd.DataFrame({"p": p, "y": y_sev})
    d["decile"] = pd.qcut(d["p"].rank(method="first"), 10, labels=False) + 1
    return d.groupby("decile").agg(rows=("y", "size"), mean_predicted=("p", "mean"), observed_severe_rate=("y", "mean")).reset_index()


# ------------------------------------------------------------------ main
def main():
    warnings.filterwarnings("ignore")
    art = Path(sv.ART); out = art / "analysis"; out.mkdir(parents=True, exist_ok=True)
    bin_path = art / "highway_severity_v2_binary_best.joblib"
    four_path = art / "highway_severity_v2_four_class_reference.joblib"
    if not bin_path.exists() or not four_path.exists():
        raise FileNotFoundError("Run `python -m v4_pipeline.severity_v2` first (saved models are missing).")
    saved = joblib.load(bin_path); four_saved = joblib.load(four_path)

    df, train, val, test = sv.prepare_data()
    cat_cols = [c for c in sv.CAT_FEATURES if c in df.columns]; num_cols = [c for c in sv.NUM_FEATURES if c in df.columns]
    y_val = val["severe_binary"].astype(int).to_numpy(); y_test = test["severe_binary"].astype(int).to_numpy()
    y4_test = test[sv.TARGET].astype(str).to_numpy()
    X_val = sv.clean_features(val, cat_cols, num_cols); X_test = sv.clean_features(test, cat_cols, num_cols)
    winner_name = json.loads((art / "highway_severity_v2_metrics.json").read_text())["model_selection"]["selected_model"]
    print(f"Rows train/val/test: {len(train)}/{len(val)}/{len(test)}; v2 winner: {winner_name}")

    # ---- 1. ablation -------------------------------------------------------------
    cands = sv.make_candidates(cat_cols, num_cols)
    abl_name = winner_name if winner_name in cands else ("XGBoost" if "XGBoost" in cands else "Logistic Regression")
    names = [abl_name] + (["Logistic Regression"] if abl_name != "Logistic Regression" else [])
    print("Running feature ablation ...")
    ablation = run_ablation(names, cat_cols, num_cols, train, val, test)
    perm = grouped_permutation_importance(saved["model"], X_test, y_test)
    ablation.to_csv(out / "ablation.csv", index=False); perm.to_csv(out / "permutation_importance.csv", index=False)

    # ---- 2. model comparison -----------------------------------------------------
    print("Running model comparison ...")
    p_v2_val = saved["model"].predict_proba(X_val)[:, 1]; p_v2_test = saved["model"].predict_proba(X_test)[:, 1]
    pre_test = pd.concat([train, val])            # everything before the test period (benchmark's training set)
    thr = float(saved["threshold"]); v2_flag_rate = float((p_v2_test >= thr).mean())
    trained_on = len(pre_test) if saved.get("refit_on") else len(train)
    bin_rows = [eval_binary(f"Deployed binary model: {winner_name}", "A: binary", trained_on, p_v2_val, p_v2_test, y_val, y_test,
                            fixed_threshold=thr)]
    for label, tr in [("4-class Extra Trees, benchmark training window (all rows before test), P(fatal)+P(grievous)", pre_test),
                      ("4-class Extra Trees, same training window as V2 (before 2017), P(fatal)+P(grievous)", train)]:
        m4 = fit_four_class(tr, cat_cols, num_cols); cls = m4.named_steps["classifier"].classes_
        in_sample_val = tr is pre_test      # trained on validation rows -> match the deployed model's flag rate instead
        bin_rows.append(eval_binary(label, "B->A: 4-class collapsed", len(tr),
                                    severe_prob_from_proba(m4.predict_proba(X_val), cls),
                                    severe_prob_from_proba(m4.predict_proba(X_test), cls), y_val, y_test,
                                    flag_rate=v2_flag_rate if in_sample_val else None))
    cls = four_saved.named_steps["classifier"].classes_
    bin_rows.append(eval_binary("V2 four-class reference (saved), P(fatal)+P(grievous)", "B->A: 4-class collapsed", len(train),
                                severe_prob_from_proba(four_saved.predict_proba(X_val), cls),
                                severe_prob_from_proba(four_saved.predict_proba(X_test), cls), y_val, y_test))
    comp_bin = pd.DataFrame(bin_rows)
    m4_bench = fit_four_class(pre_test, cat_cols, num_cols); m4_match = fit_four_class(train, cat_cols, num_cols)
    major = pd.Series(train[sv.TARGET].astype(str)).mode()[0]
    comp_four = pd.DataFrame([
        eval_fourclass("4-class Extra Trees, benchmark training window", len(pre_test), y4_test, m4_bench.predict(X_test)),
        eval_fourclass("4-class Extra Trees, same training window as V2", len(train), y4_test, m4_match.predict(X_test)),
        eval_fourclass("V2 four-class reference (saved)", len(train), y4_test, four_saved.predict(X_test)),
        eval_fourclass("Dummy (always majority class)", len(train), y4_test, np.full(len(y4_test), major))])
    comp_bin.to_csv(out / "model_comparison_binary.csv", index=False); comp_four.to_csv(out / "model_comparison_fourclass.csv", index=False)

    # ---- 3. error analysis (saved V2 model, validation-selected threshold) ---------
    pred = (p_v2_test >= thr).astype(int)
    severe = y_test == 1; fn = severe & (pred == 0); tp = severe & (pred == 1)
    by_sev = pd.DataFrame([{"true_severity": f"{k} {SEV_NAMES[k]}", "crashes": int((y4_test == k).sum()),
                            "predicted_non_severe": int(((y4_test == k) & (pred == 0)).sum()),
                            "share_predicted_non_severe": float(((y4_test == k) & (pred == 0)).sum() / max((y4_test == k).sum(), 1)),
                            "mean_model_score": float(p_v2_test[y4_test == k].mean())} for k in LABELS_4])
    prof = error_profile(test, pd.Series(fn, index=test.index).to_numpy(), severe)
    trade = threshold_tradeoff(y_test, y4_test, p_v2_test); calib = calibration_deciles(y_test, p_v2_test)
    fn_rows = test.loc[fn, ["__date", sv.TARGET] + cat_cols + ["chainage_km_numeric", "time_minutes"]].copy()
    fn_rows.insert(2, "model_score", p_v2_test[fn])
    for c in cat_cols:
        if c in CODEBOOK:
            fn_rows[c + "_label"] = fn_rows[c].map(lambda v, c=c: code_label(c, v))
    fn_rows.sort_values("model_score").to_csv(out / "false_negatives.csv", index=False)
    by_sev.to_csv(out / "error_by_severity.csv", index=False); prof.to_csv(out / "error_profile.csv", index=False)
    trade.to_csv(out / "threshold_tradeoff.csv", index=False); calib.to_csv(out / "calibration_deciles.csv", index=False)
    order = np.argsort(-p_v2_test); capture = []
    for share in [0.1, 0.2, 0.3, 0.4, 0.5]:
        k = int(round(share * len(order))); top = np.zeros(len(order), bool); top[order[:k]] = True
        capture.append({"flag_top_share_of_crashes": share, "severe_captured": float((top & severe).sum() / severe.sum()),
                        "fatal_captured": float((top & (y4_test == "1")).sum() / max((y4_test == "1").sum(), 1))})
    capture = pd.DataFrame(capture)

    # ---- figures -----------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    a = ablation[(ablation.model == abl_name) & ablation.experiment.str.startswith("Drop")]
    ax[0].barh(a.experiment.str.replace("Drop: ", ""), a.delta_vs_full, color="#d9822b")
    ax[0].axvline(0, color="k", lw=.8); ax[0].set_xlabel("Change in test ROC-AUC when group removed"); ax[0].set_title(f"Drop-one-group ablation ({abl_name})")
    ax[1].barh(perm.feature_group[::-1], perm.auc_drop_mean[::-1], xerr=perm.auc_drop_sd[::-1], color="#3a7ca5")
    ax[1].set_xlabel("AUC drop when group is shuffled"); ax[1].set_title("Grouped permutation importance (test)")
    plt.tight_layout(); plt.savefig(out / "fig_ablation.png", dpi=150); plt.close()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    ax[0].bar(by_sev.true_severity.str.replace(" ", "\n", 1), by_sev.share_predicted_non_severe, color=["#c0392b", "#e67e22", "#7f8c8d", "#7f8c8d"])
    ax[0].set_ylabel("Share predicted non-severe"); ax[0].set_title("Predicted non-severe by true severity\n(for Fatal/Grievous this is the miss rate)")
    ax[1].plot(trade.flagged_share_of_all_crashes, trade.severe_recall, "o-", label="severe recall")
    ax[1].plot(trade.flagged_share_of_all_crashes, trade.fatal_recall, "s-", label="fatal recall")
    ax[1].plot([0, 1], [0, 1], "k--", lw=.8, label="random"); ax[1].set_xlabel("Share of crashes flagged"); ax[1].legend(); ax[1].set_title("Recall vs flag rate (test)")
    ax[2].plot(calib.mean_predicted, calib.observed_severe_rate, "o-"); ax[2].plot([0, 1], [0, 1], "k--", lw=.8)
    ax[2].set_xlabel("Mean model score (decile)"); ax[2].set_ylabel("Observed severe rate"); ax[2].set_title("Calibration (uncalibrated model)")
    plt.tight_layout(); plt.savefig(out / "fig_errors.png", dpi=150); plt.close()

    # ---- report ------------------------------------------------------------------
    full = ablation[(ablation.model == abl_name) & (ablation.experiment == "Full model")].iloc[0]
    drops = ablation[(ablation.model == abl_name) & ablation.experiment.str.startswith("Drop")]
    only = ablation[(ablation.model == abl_name) & ablation.experiment.str.startswith("Only")].sort_values("test_roc_auc", ascending=False)
    n_sev, n_fn = int(severe.sum()), int(fn.sum())
    fatal_miss = by_sev.iloc[0]; grievous_miss = by_sev.iloc[1]
    flagged = prof[prof.get("flag_after_bonferroni", pd.Series(dtype=bool))] if not prof.empty else prof
    lines = [
        "# RoadWatch V4: severity model analysis\n",
        f"Task: severe (fatal or grievous) vs non-severe among **recorded** highway crashes. Rows: train {len(train)} (before 2017), "
        f"validation {len(val)}, test {len(test)} (from 2018-12-02). Model analysed: **{winner_name}**, threshold {thr:.2f} chosen on validation only. "
        "Nothing here predicts whether or where a crash occurs.\n",
        "## 1. Feature ablation\n",
        f"Full model test ROC-AUC {full.test_roc_auc:.3f} (95% CI {full.test_auc_95ci}). Each row retrains the same model without one feature group (or with only that group)."
        + (f" The deployed model is {winner_name}; ablation retrains a single {abl_name} on the same inputs because an ensemble cannot be refit per feature subset." if abl_name != winner_name else "") + "\n",
        md_table(ablation[ablation.model == abl_name].drop(columns=["model"])),
        "Grouped permutation importance of the saved model on the test set:\n", md_table(perm),
        f"- Largest AUC loss from dropping one group: **{drops.loc[drops.delta_vs_full.idxmin(), 'experiment']}** "
        f"({drops.delta_vs_full.min():+.3f}). {int(drops.delta_distinguishable_from_0.sum())} of {len(drops)} drops are distinguishable from zero (paired bootstrap).",
        f"- Best single group on its own: **{only.iloc[0].experiment}** (test AUC {only.iloc[0].test_roc_auc:.3f}).",
        "- A 95% interval that includes 0 means the data cannot show that group matters; it does not prove the group is useless.\n",
        "## 2. Model comparison (same test rows, different tasks)\n",
        "**Task A is binary (severe vs non-severe). Task B is four-class severity.** A binary model cannot produce four classes, so Task B has no binary entry. "
        "To compare across tasks, the four-class models are collapsed to a binary score P(fatal)+P(grievous) and evaluated as Task A. "
        "Threshold sources are listed per row and never use test labels: the deployed model keeps its own threshold; models trained on the validation rows are matched to its flag rate; the rest choose on validation.\n",
        "### Task A: binary severe vs non-severe\n", md_table(comp_bin.drop(columns=["task"])),
        "### Task B: four-class severity\n", md_table(comp_four),
        "Caveats: `benchmark_severity.py` chose its winner by test-set macro-F1, which is optimistically biased; V2 chose on validation. "
        "Training windows differ (benchmark trains on all pre-test rows, V2 on pre-2017 only), so both are shown. "
        "A model that merely beats the dummy by a few points on ~800 rows is within sampling noise (see the intervals).\n",
        "## 3. Error analysis: severe crashes predicted non-severe (false negatives)\n",
        f"On the test set there are {n_sev} severe crashes; the model missed **{n_fn} ({n_fn / n_sev:.0%})** and caught {int(tp.sum())}. "
        f"Fatal crashes missed: {fatal_miss.predicted_non_severe} of {fatal_miss.crashes} ({fatal_miss.share_predicted_non_severe:.0%}); "
        f"grievous missed: {grievous_miss.predicted_non_severe} of {grievous_miss.crashes} ({grievous_miss.share_predicted_non_severe:.0%}).\n",
        md_table(by_sev),
        "Where misses concentrate (slices with at least %d severe crashes; flag = still different from the overall miss rate after Bonferroni correction):\n" % MIN_SEVERE_PER_SLICE,
        md_table((flagged if len(flagged) else prof).sort_values("fn_rate", ascending=False).head(20) if not prof.empty else prof),
        (f"{len(prof)} slices tested; {len(flagged)} differ from the overall miss rate after Bonferroni correction "
         "(table shows those; fn_rate above the overall rate = missed more often, below = missed less often).\n" if not prof.empty else ""),
        "Cost of changing the threshold (test set, for illustration only; do not tune the threshold on test):\n", md_table(trade),
        "Share of severe/fatal crashes captured by flagging the highest-scoring crashes:\n", md_table(capture),
        "Calibration by score decile (the model is not calibrated):\n", md_table(calib),
        f"`false_negatives.csv` lists all {n_fn} missed severe crashes with decoded labels for manual review.\n",
        "## Limits\n",
        f"- Only {len(test)} test rows: intervals are wide; differences of a few AUC points are not reliable.\n"
        "- No speed, traffic volume, road geometry or corridor ID; chainage may stand in for road section.\n"
        "- Cause code (Causes_D) is excluded because it may be recorded after the crash.\n"
        "- Slice-level miss rates are exploratory (many comparisons).\n"
        "- Research prototype; not for operational or emergency decisions.\n"]
    (out / "severity_analysis_report.md").write_text("\n".join(map(str, lines)), encoding="utf-8")

    # ---- paste-ready README section (so the repo never quotes stale numbers) -------------
    mt = sv.binary_metrics(y_test, p_v2_test, thr); uci = sv.bootstrap_ci(y_test, p_v2_test, thr)
    f1 = 2 * mt["severe_precision"] * mt["severe_recall"] / max(mt["severe_precision"] + mt["severe_recall"], 1e-12)
    ref4 = comp_four[comp_four.model.str.startswith("V2 four-class reference")].iloc[0]
    built = saved.get("members")
    method = ("A chronological split is used. Model selection used time-ordered cross-validation (ROC-AUC) on all rows before "
              "2018-12-02; engineered features and a small ensemble were compared, scores were calibrated on out-of-fold predictions, "
              "the decision threshold was chosen on those out-of-fold scores, and the final model was refit on all pre-test rows. "
              "The test partition (from 2018-12-02) was held out and evaluated once."
              if saved.get("refit_on") else
              "A chronological train-validation-test strategy is used. Model selection and decision-threshold selection are performed "
              "using validation data, while the final test partition is held out for evaluation.")
    readme = [
        method, "",
        f"**Selected model:** {winner_name}" + (f" (average of: {', '.join(built)})" if built else ""), "",
        f"**Selected decision threshold:** {thr:.2f}" + (" (chosen on out-of-fold, calibrated training scores)" if saved.get("calibrated") else " (chosen on validation data)"), "",
        f"#### Final Severity V2 Test Results (n = {len(test)} crashes, {int(severe.sum())} severe)", "",
        "| Metric | Result |", "|---|---:|",
        f"| Accuracy | {mt['accuracy']:.2%} |", f"| Balanced Accuracy | {mt['balanced_accuracy']:.2%} |",
        f"| Severe Precision | {mt['severe_precision']:.2%} |", f"| Severe Recall | {mt['severe_recall']:.2%} |",
        f"| Severe F1-score | {f1:.2%} |", f"| Severe F2-score | {mt['severe_f2']:.2%} |",
        f"| Non-severe Specificity | {mt['non_severe_specificity']:.2%} |",
        f"| ROC-AUC | {mt['roc_auc']:.3f} (95% CI {uci['roc_auc_95ci'][0]:.3f} to {uci['roc_auc_95ci'][1]:.3f}) |", "",
        f"The binary model identified {int(tp.sum())} of the {int(severe.sum())} severe cases in the held-out test dataset "
        f"and missed {n_fn}. A ROC-AUC of 0.5 is chance, so discrimination is weak.", "",
        f"A four-class Extra Trees reference experiment achieved {ref4.accuracy:.2%} accuracy and a macro F1-score of {ref4.macro_f1:.2%} "
        "(four-class chance accuracy is about 25-38% depending on class balance).", "",
        "These results represent a research baseline. The model has limited discriminatory capability and should not be considered "
        "suitable for operational safety decisions.", ""]
    (out / "readme_severity_section.md").write_text("\n".join(readme), encoding="utf-8")

    def recs(d): return json.loads(d.to_json(orient="records"))
    (out / "severity_analysis.json").write_text(json.dumps({
        "winner": winner_name, "threshold": thr, "rows": {"train": len(train), "validation": len(val), "test": len(test)},
        "ablation": recs(ablation), "permutation_importance": recs(perm), "model_comparison_binary": recs(comp_bin),
        "model_comparison_fourclass": recs(comp_four), "error_by_severity": recs(by_sev), "error_profile": recs(prof.head(25)),
        "threshold_tradeoff": recs(trade), "capture_curve": recs(capture), "calibration_deciles": recs(calib),
        "false_negative_count": n_fn, "severe_count": n_sev}, indent=2), encoding="utf-8")
    print(f"Test: {n_sev} severe, {n_fn} missed ({n_fn / n_sev:.0%}); fatal missed {fatal_miss.share_predicted_non_severe:.0%}")
    print(f"Full-model test AUC {full.test_roc_auc:.3f}; analysis written to {out}")


if __name__ == "__main__":
    main()
