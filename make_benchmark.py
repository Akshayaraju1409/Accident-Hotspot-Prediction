"""Planted-hotspot benchmark.

indian_roads_dataset.csv has no spatial structure (uniform locations), so a hotspot detector has nothing to find.
To prove the pipeline CAN find and predict hotspots when they exist, this script keeps every real row as the
background and adds extra crashes around KNOWN locations (Gaussian clusters, persistent over the whole period,
skewed to peak/night hours and rain/fog). The ground truth is written to data/benchmark_truth.json.

    python make_benchmark.py          # -> data/benchmark_planted.csv + data/benchmark_truth.json

This is a method-validation dataset, NOT evidence about real Indian roads.
"""
import json
import numpy as np
import pandas as pd
import config as C

N_PER_CITY = 4                 # planted hotspots per city
SIGMA_DEG = 0.006              # ~0.6 km spread, so each hotspot sits in roughly 1-4 grid cells
EXTRA_RANGE = (60, 200)        # extra crashes per hotspot over the whole period (background is ~15 per cell)
HOUR_W = np.where(np.isin(np.arange(24), [7, 8, 9, 17, 18, 19, 20, 21, 22]), 3.0, 1.0)


def build(src=C.DATA / "indian_roads_dataset.csv", seed=C.SEED):
    rng = np.random.default_rng(seed)
    df = pd.read_csv(src)
    df["is_planted"] = 0
    d0, d1 = pd.to_datetime(df.date).min(), pd.to_datetime(df.date).max()
    ndays = (d1 - d0).days + 1
    extra, truth, nxt = [], [], int(df.accident_id.max()) + 1
    for city, g in df.groupby("city"):
        lat0, lat1, lon0, lon1 = g.latitude.min(), g.latitude.max(), g.longitude.min(), g.longitude.max()
        # keep hotspot centres away from the box edge so clusters are not clipped
        for k in range(N_PER_CITY):
            clat = rng.uniform(lat0 + 0.04, lat1 - 0.04); clon = rng.uniform(lon0 + 0.04, lon1 - 0.04)
            n = int(rng.integers(*EXTRA_RANGE))
            w = np.where(g.weather.isin(["rain", "fog"]), 2.0, 1.0)
            r = g.sample(n, replace=True, weights=w, random_state=int(rng.integers(1e9))).copy()
            r["latitude"] = np.clip(clat + rng.normal(0, SIGMA_DEG, n), lat0, lat1)
            r["longitude"] = np.clip(clon + rng.normal(0, SIGMA_DEG, n), lon0, lon1)
            day = d0 + pd.to_timedelta(rng.integers(0, ndays, n), unit="D")
            hour = rng.choice(24, n, p=HOUR_W / HOUR_W.sum())
            r["date"] = day.strftime("%Y-%m-%d"); r["hour"] = hour; r["time"] = [f"{h}:00" for h in hour]
            r["day_of_week"] = day.day_name(); r["is_weekend"] = (day.dayofweek >= 5).astype(int)
            r["is_peak_hour"] = np.isin(hour, [7, 8, 9, 17, 18, 19, 20, 21]).astype(int)
            r["accident_id"] = np.arange(nxt, nxt + n); nxt += n
            r["is_planted"] = 1
            extra.append(r)
            truth.append(dict(city=city, lat=float(clat), lon=float(clon), extra_crashes=n))
    out = pd.concat([df] + extra, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    return out, truth


if __name__ == "__main__":
    out, truth = build()
    out.to_csv(C.DATA / "benchmark_planted.csv", index=False)
    json.dump(truth, open(C.DATA / "benchmark_truth.json", "w"), indent=1)
    print(f"{len(out)} rows ({int(out.is_planted.sum())} planted) -> data/benchmark_planted.csv; {len(truth)} hotspots in truth file")
