"""Training pipeline: M1-M7 end to end. Run:  python train.py"""
import json, time, warnings
import joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score, r2_score, mean_absolute_error
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier, LGBMRegressor
import config as C, data_prep as dp, hotspots as hs, features as ft, modeling as md

warnings.filterwarnings("ignore")
rng = np.random.default_rng(C.SEED)
C.ART.mkdir(exist_ok=True)
log = {}
T0 = time.time()
say = lambda *a: print(f"[{time.time()-T0:5.0f}s]", *a, flush=True)

# ---- M1/M2 ---------------------------------------------------------------------------------
df, rep = dp.load_clean(); log["cleaning"] = rep
cells, spec = dp.build_grid(df)
df["cell_id"] = dp.assign_cells(df.latitude, df.longitude, df.city, spec)
start = df.date.min(); ndays = (df.date.max() - start).days + 1
ctx = dp.Context(df, cells, start, ndays)
log["data_range"] = [str(df.date.min().date()), str(df.date.max().date())]
log["grid"] = dict(cells=int(len(cells)), cell_deg=C.CELL_DEG, block_hours=C.BLOCK_H)
say("data", rep, "cells", len(cells))

# ---- M3 hotspots (feature window only, for features) + full-period (for the map) -----------------
hs_feat, diag_feat = hs.run_hotspots(df[df.date <= C.FEATURE_END], cells)
hs_full, diag_full = hs.run_hotspots(df, cells)
log["hotspots_feature_window"] = diag_feat; log["hotspots_full_period"] = diag_full
static = ft.make_static(cells, df[df.date <= C.FEATURE_END], hs_feat)
hs_full.merge(cells, on="cell_id").to_csv(C.ART / "hotspots_full.csv", index=False)
say("hotspots", diag_feat["_compare"], diag_full["_compare"])

# ---- M4/M5 rows & time split ------------------------------------------------------------------
d = ctx.day_of
tr = ctx.rows(d(C.TRAIN_START), d(C.TRAIN_END))
va = ctx.rows(d(C.VAL_START), d(C.VAL_END))
te = ctx.rows(d(C.TEST_START), ndays - 1)
Xva = ft.build_features(ctx, static, *va[:3]); yva = va[3]
Xte = ft.build_features(ctx, static, *te[:3]); yte = te[3]
log["split"] = {k: dict(rows=int(len(v[3])), crashes=int(v[3].sum()), rate=float(v[3].mean()))
                for k, v in dict(train=tr, val=va, test=te).items()}
say("rows", log["split"])

def train_sample(neg_per_pos, seed):
    idx = md.sample_negatives(tr[3], neg_per_pos, np.random.default_rng(seed))
    return ft.build_features(ctx, static, tr[0][idx], tr[1][idx], tr[2][idx]), tr[3][idx]

# ---- M6: model zoo (one balanced sample), then balanced ensemble with tuning loop ---------------------
Xs, ys = train_sample(20, C.SEED)
runs, preds_val, preds_te = [], {}, {}
for name in ["LogReg (baseline)", "RandomForest", "XGBoost", "LightGBM", "MLP"]:
    pw = 1.0 if name in ("LogReg (baseline)", "RandomForest", "MLP") else float((ys == 0).sum() / (ys == 1).sum())
    m = md.make_model(name, pos_weight=pw).fit(Xs, ys)
    preds_val[name], preds_te[name] = md.proba(m, Xva), md.proba(m, Xte)
    runs.append(dict(model=name, val_pr_auc=float(average_precision_score(yva, preds_val[name]))))
    say(name, runs[-1])

# SMOTE ablation (imputed, because SMOTE cannot handle NaN)
imp = md.SimpleImputer(strategy="median").fit(Xs)
Xsm, ysm = SMOTE(random_state=C.SEED, k_neighbors=5).fit_resample(imp.transform(Xs), ys)
sm = md.make_model("LightGBM", pos_weight=1.0).fit(pd.DataFrame(Xsm, columns=ft.ALL_F), ysm)
preds_val["LightGBM + SMOTE (ablation)"] = md.proba(sm, pd.DataFrame(imp.transform(Xva), columns=ft.ALL_F))
preds_te["LightGBM + SMOTE (ablation)"] = md.proba(sm, pd.DataFrame(imp.transform(Xte), columns=ft.ALL_F))
runs.append(dict(model="LightGBM + SMOTE (ablation)", val_pr_auc=float(average_precision_score(yva, preds_val["LightGBM + SMOTE (ablation)"]))))

# tuning rounds: re-balance (neg:pos) and hyper-parameters; capped number of rounds
grid = [dict(neg=10, num_leaves=7, min_child_samples=100), dict(neg=10, num_leaves=15, min_child_samples=50),
        dict(neg=20, num_leaves=7, min_child_samples=100), dict(neg=5, num_leaves=7, min_child_samples=200)]
best = None
for r, cfg in enumerate(grid):
    Xb, yb = train_sample(cfg["neg"], C.SEED + 100)
    mb = md.make_model("LightGBM", pos_weight=cfg["neg"], num_leaves=cfg["num_leaves"],
                       min_child_samples=cfg["min_child_samples"]).fit(Xb, yb)
    ap = float(average_precision_score(yva, md.proba(mb, Xva)))
    runs.append(dict(tuning_round=r + 1, config=cfg, val_pr_auc=ap)); say("tune", runs[-1])
    if best is None or ap > best[0]: best = (ap, cfg)
cfg = best[1]
bags = []
for k in range(C.N_BAGS):
    Xb, yb = train_sample(cfg["neg"], C.SEED + 1000 + k)
    bags.append(md.make_model("LightGBM", pos_weight=cfg["neg"], num_leaves=cfg["num_leaves"],
                              min_child_samples=cfg["min_child_samples"], random_state=C.SEED + k).fit(Xb, yb))
bag_val = np.column_stack([md.proba(b, Xva) for b in bags]); bag_te = np.column_stack([md.proba(b, Xte) for b in bags])
NAME = "Balanced LightGBM ensemble (final)"
preds_val[NAME], preds_te[NAME] = bag_val.mean(1), bag_te.mean(1)
val_ap = float(average_precision_score(yva, preds_val[NAME])); val_lift = val_ap / yva.mean()
log["validation_acceptable"] = bool(val_lift >= 1.5)
log["tuning_runs"] = runs; log["chosen_config"] = cfg

# naive baselines
base_val = ft.build_features(ctx, static, *va[:3])["cnt_all"].values.astype(float)
for nm, pv, pt in [("Baseline: historical cell count", Xva["cnt_all"].values, Xte["cnt_all"].values),
                   ("Baseline: static hotspot rate (feature window)", Xva["feat_rate"].fillna(0).values, Xte["feat_rate"].fillna(0).values),
                   ("Baseline: random", rng.random(len(yva)), rng.random(len(yte)))]:
    preds_val[nm], preds_te[nm] = pv.astype(float), pt.astype(float)

# calibration (isotonic on validation), metrics on test
cal = md.Calibrator().fit(preds_val[NAME], yva)
table, cals = {}, {}
for nm in preds_te:
    thr = md.best_f1_threshold(yva, preds_val[nm])
    c = md.Calibrator().fit(preds_val[nm], yva); cals[nm] = c
    table[nm] = md.metrics(yte, preds_te[nm], c(preds_te[nm]), thr)
table[NAME]["pr_auc_ci95"] = md.bootstrap_ap(yte, preds_te[NAME])
table["Baseline: random"]["pr_auc_ci95"] = md.bootstrap_ap(yte, preds_te["Baseline: random"])
log["test_metrics"] = table
say("test", {k: round(v["pr_auc_lift"], 2) for k, v in table.items()})

# hotspot persistence check: do feature-window hotspots hold more of the TEST crashes than their area share?
te_crash = df[df.date >= C.TEST_START]
pers = {}
for lab, col in [("gi_star", "gi_hot"), ("dbscan", "db_hot")]:
    hot = set(static.index[static[col] == 1]); share = te_crash.cell_id.isin(hot).mean()
    pers[lab] = dict(hotspot_cells=len(hot), area_share=len(hot) / len(static), test_crash_share=float(share))
log["hotspot_persistence"] = pers

# ---- cross-city (leave-one-city-out) ----------------------------------------------------------------
loco = {}
Xall_tr = ft.build_features(ctx, static, *tr[:3]); city_tr = cells.city.values[tr[0]]
city_te = cells.city.values[te[0]]
for city in sorted(cells.city.unique()):
    m_tr = city_tr != city
    idx = np.flatnonzero(m_tr); yy = tr[3][idx]
    sel = idx[md.sample_negatives(yy, 10, np.random.default_rng(1))]
    m = md.make_model("LightGBM", pos_weight=10, num_leaves=7, min_child_samples=100).fit(Xall_tr.iloc[sel][ft.TRANSFERABLE_F], tr[3][sel])
    mask = city_te == city
    s = md.proba(m, Xte[mask][ft.TRANSFERABLE_F]); y = yte[mask]
    loco[city] = dict(pr_auc=float(average_precision_score(y, s)), prevalence=float(y.mean()),
                      lift=float(average_precision_score(y, s) / y.mean()), roc_auc=float(roc_auc_score(y, s)))
log["cross_city_loco"] = loco; say("loco", {k: round(v["lift"], 2) for k, v in loco.items()})

# ---- conditions (risk-score) + severity models on crash-level records ---------------------------------
COND_F = ["weather", "visibility", "traffic_density", "is_peak_hour", "hour", "dow", "road_type", "lanes", "traffic_signal", "temperature"]
CATS = {"weather": ["clear", "fog", "rain"], "visibility": ["high", "medium", "low"],
        "traffic_density": ["low", "medium", "high"], "road_type": ["highway", "rural", "urban"]}
def cond_X(d_):
    X = d_[COND_F].copy() if "dow" in d_ else d_.assign(dow=d_.date.dt.dayofweek)[COND_F].copy()
    for c, cats in CATS.items(): X[c] = pd.Categorical(X[c], categories=cats)
    return X
trm, vam, tem = df.date <= C.TRAIN_END, (df.date >= C.VAL_START) & (df.date <= C.VAL_END), df.date >= C.TEST_START
Xc = cond_X(df)
risk_m = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15, verbose=-1, random_state=C.SEED).fit(Xc[trm], df.risk_score[trm])
sev_m = LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=7, min_child_samples=100, verbose=-1, random_state=C.SEED).fit(Xc[trm], df.severe[trm])
log["conditions_model"] = dict(
    risk_r2_test=float(r2_score(df.risk_score[tem], risk_m.predict(Xc[tem]))),
    risk_mae_test=float(mean_absolute_error(df.risk_score[tem], risk_m.predict(Xc[tem]))),
    severe_auc_val=float(roc_auc_score(df.severe[vam], sev_m.predict_proba(Xc[vam])[:, 1])),
    severe_auc_test=float(roc_auc_score(df.severe[tem], sev_m.predict_proba(Xc[tem])[:, 1])),
    severe_rate=float(df.severe.mean()))
say("conditions", log["conditions_model"])

# ---- risk engine thresholds from the validation score distribution --------------------------------------
cal_final = cal
p_val = cal_final(preds_val[NAME]); p_ref = float(yva.mean())
# Relative risk is measured against the average cell of the SAME city and time slot. Without this,
# alerts would just rank cities by crashes-per-cell (a grid artefact), not hotspots.
grp = pd.Series(p_val).groupby([cells.city.values[va[0]], va[1], va[2]]).transform("mean").values
rel = p_val / np.maximum(grp, 1e-12)
scen = pd.DataFrame([(w, v, t) for w, v in [("clear", "high"), ("fog", "low"), ("rain", "medium")] for t in ["low", "medium", "high"]],
                    columns=["weather", "visibility", "traffic_density"])
scen["risk"] = risk_m.predict(cond_X(scen.assign(is_peak_hour=0, hour=12, dow=2, road_type="urban", lanes=3, traffic_signal=1, temperature=28)))
cond_grid = scen.risk.values
samp = rng.choice(len(rel), 400000)
scores = rel[samp] * rng.choice(cond_grid, len(samp))
thresholds = {k: float(np.quantile(scores, q)) for k, q in C.ALERT_QUANTILES.items()}
log["alert_thresholds"] = thresholds

# ---- explainability sample + artifacts -----------------------------------------------------------------
import shap
expl = shap.TreeExplainer(bags[0])
sv = np.abs(expl.shap_values(Xva.iloc[rng.choice(len(Xva), 5000)])); sv = sv[1] if isinstance(sv, list) else sv
log["shap_global_crash_model"] = dict(sorted(zip(ft.ALL_F, map(float, sv.mean(0))), key=lambda t: -t[1]))
risk_sv = shap.TreeExplainer(risk_m).shap_values(Xc.iloc[rng.choice(len(Xc), 3000)])
log["shap_global_conditions_model"] = dict(sorted(zip(COND_F, map(float, np.abs(risk_sv).mean(0))), key=lambda t: -t[1]))

city_temp = df[trm].groupby("city").temperature.median().to_dict()
joblib.dump(dict(bags=bags, calibrator=cal_final, p_ref=p_ref, thresholds=thresholds, risk_model=risk_m,
                 severity_model=sev_m, cond_features=COND_F, cats=CATS, city_temp=city_temp,
                 severity_auc_val=log["conditions_model"]["severe_auc_val"], val_lift=float(val_lift)), C.ART / "models.joblib")
joblib.dump(dict(cells=cells, spec=spec, static=static, ctx=ctx), C.ART / "context.joblib")
df.to_csv(C.ART / "crashes_clean.csv.gz", index=False)
json.dump(log, open(C.ART / "metrics.json", "w"), indent=2, default=float)

# ---- plots ------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for nm in [NAME, "LightGBM", "Baseline: historical cell count"]:
    p_, r_, _ = precision_recall_curve(yte, preds_te[nm]); ax[0].plot(r_, p_, label=nm)
ax[0].axhline(yte.mean(), color="k", ls=":", label="prevalence"); ax[0].set_xlabel("recall"); ax[0].set_ylabel("precision"); ax[0].set_yscale("log"); ax[0].legend(fontsize=7); ax[0].set_title("Test PR curve")
pc = cal_final(preds_te[NAME]); fy, fx = calibration_curve(yte, pc, n_bins=8, strategy="quantile")
ax[1].plot(fx, fy, "o-"); mx = max(fx.max(), fy.max()) * 1.1; ax[1].plot([0, mx], [0, mx], "k:"); ax[1].set_xlabel("predicted P(crash)"); ax[1].set_ylabel("observed"); ax[1].set_title("Calibration (test)")
plt.tight_layout(); plt.savefig(C.ART / "eval_plots.png", dpi=130); plt.close()
say("done")
