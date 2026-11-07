"""Extra honesty check: does the model beat chance WITHIN a city? (appends to metrics.json)"""
import json, numpy as np, pandas as pd, joblib
from sklearn.metrics import average_precision_score
import config as C, features as ft
m = joblib.load(C.ART / "models.joblib"); c = joblib.load(C.ART / "context.joblib")
ctx, st, cells = c["ctx"], c["static"], c["cells"]
te = ctx.rows(ctx.day_of(C.TEST_START), ctx.ndays - 1)
X = ft.build_features(ctx, st, *te[:3]); y = te[3]
s = np.mean([b.predict_proba(X)[:, 1] for b in m["bags"]], axis=0)
city = cells.city.values[te[0]]
res = {}
for cname in sorted(cells.city.unique()):
    k = city == cname; yy, ss = y[k], s[k]
    ap = average_precision_score(yy, ss)
    # permutation null: shuffle cell identity of the scores within the city (keeps score distribution)
    rng = np.random.default_rng(0); null = [average_precision_score(yy, rng.permutation(ss)) for _ in range(30)]
    res[cname] = dict(lift=float(ap / yy.mean()), null_lift_mean=float(np.mean(null) / yy.mean()),
                      null_lift_p95=float(np.percentile(null, 95) / yy.mean()))
# city-only baseline
rate = pd.Series(y).groupby(city).transform("mean")  # oracle city rate (upper bound for a city-only model)
tr_rate = c["cells"].groupby("city").size()
d = json.load(open(C.ART / "metrics.json"))
d["within_city_lift"] = res
d["city_only_oracle_lift"] = float(average_precision_score(y, rate.values) / y.mean())
json.dump(d, open(C.ART / "metrics.json", "w"), indent=2)
print(json.dumps(res, indent=1)); print("city-only oracle lift", d["city_only_oracle_lift"])
