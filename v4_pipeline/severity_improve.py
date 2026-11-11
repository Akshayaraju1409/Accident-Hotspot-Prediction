"""
RoadWatch V4: try to improve the severity model WITHOUT touching the test set for any decision.

What it changes versus severity_v2 (and why):
  1. Model selection by rolling-origin (time-ordered) cross-validation on all pre-test rows, scored by ROC-AUC.
     severity_v2 picked the winner by F2 at a threshold on one 573-row validation block, which is noisy.
  2. Engineered features (night, cyclical hour, vulnerable road user, heavy vehicle, interactions; see severity_features.py).
  3. Small ensembles and stronger regularisation, compared on equal footing.
  4. Platt calibration on out-of-fold scores, so the score reads closer to an observed severe rate.
  5. Final model refit on ALL pre-test rows (train + validation) instead of pre-2017 only. Threshold comes from
     out-of-fold scores, so no test data and no in-sample predictions are used for it.

The test set (from 2018-12-02) is evaluated ONCE, against the saved V2 model, with a paired bootstrap.

    python -m v4_pipeline.severity_improve            # experiment + report only, nothing overwritten
    python -m v4_pipeline.severity_improve --adopt    # also replace the saved model/metrics (old one is backed up)
    python -m v4_pipeline.severity_analysis           # rerun afterwards so ablation/errors describe the new model
"""
from pathlib import Path
import argparse
import json
import shutil
import warnings

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from . import severity_v2 as sv
from .severity_features import ScoredModel, SeverityFeatures, engineered_names, _logit

SEED = sv.RANDOM_STATE
N_BLOCKS = 5          # chronological blocks; folds validate on blocks 1..4 using only earlier blocks for training
SIMPLICITY = {"LR": 0, "ExtraTrees": 1, "XGBoost": 2, "Ensemble": 3}
OLD_BACKUP = "highway_severity_v2_binary_best.before_improve.joblib"


# ------------------------------------------------------------------ candidates
def build_pipeline(clf, cat_cols, num_cols, engineered):
    steps = []
    eng = []
    if engineered:
        steps.append(("features", SeverityFeatures()))
        eng = engineered_names(cat_cols + num_cols)
    steps += [("preprocess", sv.make_preprocessor(cat_cols, num_cols + eng)), ("classifier", clf)]
    return Pipeline(steps)


def candidate_specs():
    """(name, family, engineered?, classifier factory). Includes the exact V2 recipes as the baseline."""
    specs = []
    xgb_ok = True
    try:
        from xgboost import XGBClassifier
    except ImportError:
        xgb_ok = False
        print("XGBoost not installed; XGBoost candidates skipped.")

    def xgb(depth, n, lr, lam, mcw):
        return lambda: XGBClassifier(objective="binary:logistic", n_estimators=n, max_depth=depth, learning_rate=lr,
                                     subsample=0.85, colsample_bytree=0.85, reg_lambda=lam, min_child_weight=mcw,
                                     eval_metric="logloss", tree_method="hist", random_state=SEED, n_jobs=-1)
    for eng in (False, True):
        tag = "+engineered" if eng else "raw"
        for C in (0.05, 0.3, 1.0):
            specs.append((f"LR C={C} [{tag}]", "LR", eng,
                          (lambda C=C: LogisticRegression(max_iter=4000, C=C, random_state=SEED))))
        specs.append((f"ExtraTrees leaf=8 [{tag}]", "ExtraTrees", eng,
                      lambda: ExtraTreesClassifier(n_estimators=500, min_samples_leaf=8, max_features="sqrt",
                                                   random_state=SEED, n_jobs=-1)))
        if xgb_ok:
            specs.append((f"XGBoost d2 [{tag}]", "XGBoost", eng, xgb(2, 300, 0.04, 5.0, 5)))
            specs.append((f"XGBoost d3 [{tag}]", "XGBoost", eng, xgb(3, 400, 0.03, 5.0, 5)))
    # the exact recipes severity_v2 used, for an equal-footing baseline
    specs.append(("V2 recipe: ExtraTrees leaf=2 [raw]", "ExtraTrees", False,
                  lambda: ExtraTreesClassifier(n_estimators=700, min_samples_leaf=2, max_features="sqrt",
                                               random_state=SEED, n_jobs=-1)))
    if xgb_ok:
        specs.append(("V2 recipe: XGBoost d3 [raw]", "XGBoost", False, xgb(3, 500, 0.035, 3.0, 1)))
    return specs


# ------------------------------------------------------------------ cross-validation
def chrono_folds(dates, n_blocks=N_BLOCKS):
    order = np.argsort(np.asarray(dates), kind="stable")
    blocks = np.array_split(order, n_blocks)
    for k in range(1, n_blocks):
        yield k, np.concatenate(blocks[:k]), blocks[k]


def cv_candidate(spec, X, y, dates, cat_cols, num_cols):
    name, family, eng, factory = spec
    oof = np.full(len(y), np.nan); aucs = []
    for k, tr, va in chrono_folds(dates):
        model = build_pipeline(factory(), cat_cols, num_cols, eng)
        model.fit(X.iloc[tr], y[tr], classifier__sample_weight=compute_sample_weight("balanced", y[tr]))
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
        aucs.append(roc_auc_score(y[va], oof[va]))
    return {"name": name, "family": family, "engineered": eng, "fold_aucs": aucs,
            "cv_auc_mean": float(np.mean(aucs)), "cv_auc_sd": float(np.std(aucs, ddof=1)), "oof": oof}


def ensemble_oof(members, y, dates):
    oof = np.nanmean(np.vstack([m["oof"] for m in members]), axis=0)
    aucs = [roc_auc_score(y[va], oof[va]) for _, _, va in chrono_folds(dates)]
    return oof, aucs


# ------------------------------------------------------------------ metrics
def expected_calibration_error(y, p, bins=10):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1)); edges[-1] += 1e-9
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean()) for b in range(bins) if (idx == b).any()))


def paired_auc_delta(y, p_new, p_old, n_boot=2000, seed=SEED):
    rng = np.random.default_rng(seed); d = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        d.append(roc_auc_score(y[i], p_new[i]) - roc_auc_score(y[i], p_old[i]))
    return {"delta_auc": float(roc_auc_score(y, p_new) - roc_auc_score(y, p_old)),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "p_new_better": float(np.mean(np.asarray(d) > 0))}


def md(df, fmt="{:.3f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------ main
def main(adopt=False):
    warnings.filterwarnings("ignore")
    ART = Path(sv.ART); out = ART / "improve"; out.mkdir(parents=True, exist_ok=True)
    df, train, val, test = sv.prepare_data()
    cat_cols = [c for c in sv.CAT_FEATURES if c in df.columns]; num_cols = [c for c in sv.NUM_FEATURES if c in df.columns]
    features = cat_cols + num_cols
    pre = pd.concat([train, val]).reset_index(drop=True)
    X = sv.clean_features(pre, cat_cols, num_cols); y = pre["severe_binary"].astype(int).to_numpy(); dates = pre["__date"].to_numpy()
    X_test = sv.clean_features(test, cat_cols, num_cols); y_test = test["severe_binary"].astype(int).to_numpy()
    print(f"Pre-test rows {len(pre)} (train+validation), test rows {len(test)}; CV folds validate on {N_BLOCKS - 1} later blocks")

    # ---- 1. cross-validate every candidate (test set untouched) -------------------
    results = []
    for spec in candidate_specs():
        r = cv_candidate(spec, X, y, dates, cat_cols, num_cols); results.append(r)
        print(f"  {r['name']:<40} CV AUC {r['cv_auc_mean']:.3f} +/- {r['cv_auc_sd']:.3f}")
    by_family = {}
    for r in sorted(results, key=lambda r: -r["cv_auc_mean"]):
        by_family.setdefault(r["family"], r)
    top3 = sorted(by_family.values(), key=lambda r: -r["cv_auc_mean"])[:3]
    if len(top3) >= 2:
        oof_e, aucs_e = ensemble_oof(top3, y, dates)
        results.append({"name": "Ensemble: mean of " + " + ".join(m["name"] for m in top3), "family": "Ensemble",
                        "engineered": any(m["engineered"] for m in top3), "fold_aucs": aucs_e,
                        "cv_auc_mean": float(np.mean(aucs_e)), "cv_auc_sd": float(np.std(aucs_e, ddof=1)), "oof": oof_e,
                        "members": top3})
    results.sort(key=lambda r: (-round(r["cv_auc_mean"], 4), SIMPLICITY[r["family"]]))
    best = results[0]
    base_recipes = [r for r in results if r["name"].startswith("V2 recipe")]
    base_cv = max((r["cv_auc_mean"] for r in base_recipes), default=float("nan"))
    cv_table = pd.DataFrame([{"candidate": r["name"], "cv_auc_mean": r["cv_auc_mean"], "cv_auc_sd": r["cv_auc_sd"],
                              **{f"fold{i + 1}": a for i, a in enumerate(r["fold_aucs"])}} for r in results])
    cv_table.to_csv(out / "cv_results.csv", index=False)
    print(f"\nBest by CV: {best['name']}  (CV AUC {best['cv_auc_mean']:.3f}; V2 recipes best {base_cv:.3f})")

    # ---- 2. final fit on all pre-test rows, calibrate + threshold from out-of-fold scores
    member_specs = [m for m in best["members"]] if "members" in best else [best]
    label = f"Ensemble ({len(member_specs)} models)" if "members" in best else best["name"]
    composition = [m["name"] for m in member_specs]
    spec_by_name = {s[0]: s for s in candidate_specs()}
    fitted = []
    for m in member_specs:
        _, family, eng, factory = spec_by_name[m["name"]]
        mod = build_pipeline(factory(), cat_cols, num_cols, eng)
        mod.fit(X, y, classifier__sample_weight=compute_sample_weight("balanced", y)); fitted.append(mod)
    seen = ~np.isnan(best["oof"])
    calibrator = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(best["oof"][seen]).reshape(-1, 1), y[seen])
    new_model = ScoredModel(fitted, calibrator=calibrator, name=label)
    oof_cal = calibrator.predict_proba(_logit(best["oof"][seen]).reshape(-1, 1))[:, 1]
    chosen, _ = sv.choose_threshold(y[seen], oof_cal); thr = float(chosen["threshold"])
    print(f"Threshold {thr:.2f} chosen on out-of-fold calibrated scores (specificity >= {sv.MIN_VALIDATION_SPECIFICITY})")

    # ---- 3. single look at the test set, vs the saved V2 model ---------------------
    base_path = ART / OLD_BACKUP if (ART / OLD_BACKUP).exists() else ART / "highway_severity_v2_binary_best.joblib"
    old = joblib.load(base_path)
    p_old = old["model"].predict_proba(X_test)[:, 1]; p_new = new_model.predict_proba(X_test)[:, 1]
    p_new_uncal = new_model.raw_score(X_test)
    delta = paired_auc_delta(y_test, p_new, p_old)
    m_new = sv.binary_metrics(y_test, p_new, thr); m_old = sv.binary_metrics(y_test, p_old, float(old["threshold"]))
    ci_new = sv.bootstrap_ci(y_test, p_new, thr)
    rows = []
    for row_label, p, m, t in [("Saved V2 model (baseline)", p_old, m_old, float(old["threshold"])), ("Improved model", p_new, m_new, thr)]:
        rows.append({"model": row_label, "test_roc_auc": m["roc_auc"], "avg_precision": m["average_precision"],
                     "balanced_acc": m["balanced_accuracy"], "severe_recall": m["severe_recall"], "severe_precision": m["severe_precision"],
                     "threshold": t, "brier": float(brier_score_loss(y_test, p)), "calibration_error_ece": expected_calibration_error(y_test, p)})
    rows.append({"model": "Improved model before calibration", "test_roc_auc": float(roc_auc_score(y_test, p_new_uncal)),
                 "avg_precision": float(average_precision_score(y_test, p_new_uncal)), "balanced_acc": np.nan, "severe_recall": np.nan,
                 "severe_precision": np.nan, "threshold": np.nan, "brier": float(brier_score_loss(y_test, p_new_uncal)),
                 "calibration_error_ece": expected_calibration_error(y_test, p_new_uncal)})
    cmp_df = pd.DataFrame(rows); cmp_df.to_csv(out / "test_comparison.csv", index=False)
    clear = delta["ci95"][0] > 0
    verdict = ("The AUC gain is distinguishable from noise (paired 95% interval excludes 0)." if clear else
               "The AUC change is NOT distinguishable from noise (paired 95% interval includes 0). "
               "Report the accuracy as essentially unchanged; the refit still uses more data and gives calibrated scores.")
    report = [
        "# Severity model improvement experiment\n",
        f"Selection used only pre-test rows ({len(pre)}) with {N_BLOCKS - 1} time-ordered folds. Test rows ({len(test)}, from {sv.TEST_START.date()}) were evaluated once.\n",
        "## Cross-validated ROC-AUC (higher is better; folds are later time blocks)\n", md(cv_table),
        f"Best by CV: **{best['name']}** (mean {best['cv_auc_mean']:.3f}). Best exact V2 recipe: {base_cv:.3f}.\n",
        "## Test set (single look)\n", md(cmp_df),
        f"Paired AUC difference (improved minus V2): {delta['delta_auc']:+.3f}, 95% CI {delta['ci95'][0]:+.3f} to {delta['ci95'][1]:+.3f}; "
        f"P(improved better) = {delta['p_new_better']:.2f}.\n", verdict + "\n",
        f"Improved model at threshold {thr:.2f}: test AUC {m_new['roc_auc']:.3f} (95% CI {ci_new['roc_auc_95ci'][0]:.3f} to {ci_new['roc_auc_95ci'][1]:.3f}).\n",
        "Brier score and ECE: lower is better. Calibration changes how the score reads, not how well it ranks crashes.\n",
        "## Limits\n- Only 11 raw inputs; no speed, traffic volume or road geometry. Gains from re-modelling these are small by nature.\n"
        "- Many candidates were compared; the CV winner is slightly optimistic, which is why the test set is the final check.\n"]
    (out / "improve_report.md").write_text("\n".join(report), encoding="utf-8")
    summary = {"best_candidate": best["name"], "deployed_label": label, "members": composition, "cv_auc_mean": best["cv_auc_mean"], "v2_recipe_cv_auc": base_cv, "threshold": thr,
               "test_delta_vs_v2": delta, "improved_test_metrics": m_new, "improved_test_auc_95ci": ci_new["roc_auc_95ci"],
               "baseline_test_auc": m_old["roc_auc"], "adopted": False}

    # ---- 4. optional adoption ---------------------------------------------------------
    if adopt:
        live = ART / "highway_severity_v2_binary_best.joblib"
        if not (ART / OLD_BACKUP).exists():
            shutil.copy2(live, ART / OLD_BACKUP)
        joblib.dump({"model": new_model, "threshold": thr, "features": features, "calibrated": True, "name": label, "members": composition,
                     "refit_on": "all pre-test rows (train + validation)"}, live)
        mpath = ART / "highway_severity_v2_metrics.json"; metrics = json.loads(mpath.read_text(encoding="utf-8"))
        metrics.setdefault("model_selection", {}).update({
            "selected_model": label, "members": composition, "selected_threshold_from_validation": thr,
            "criterion": "Time-ordered cross-validation ROC-AUC on pre-test rows; threshold from out-of-fold calibrated scores.",
            "adopted_from": "severity_improve", "refit_on_rows": int(len(pre))})
        metrics["final_binary_test_metrics"] = m_new; metrics["final_binary_test_uncertainty"] = ci_new
        metrics["improvement_vs_previous"] = {"previous_test_auc": m_old["roc_auc"], **delta}
        mpath.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        summary["adopted"] = True
        print(f"Adopted. Previous model backed up to {ART / OLD_BACKUP}. Restart the API and rerun severity_analysis.")
    (out / "improve_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nTest ROC-AUC: V2 {m_old['roc_auc']:.3f} -> improved {m_new['roc_auc']:.3f} "
          f"(paired delta {delta['delta_auc']:+.3f}, 95% CI {delta['ci95'][0]:+.3f} to {delta['ci95'][1]:+.3f})")
    print(verdict)
    print(f"Report: {out / 'improve_report.md'}" + ("" if adopt else "   (nothing overwritten; add --adopt to use the new model)"))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--adopt", action="store_true", help="replace the saved model and metrics")
    main(adopt=ap.parse_args().adopt)
