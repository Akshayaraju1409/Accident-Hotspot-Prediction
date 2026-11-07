"""M8-M10: PredictionService, RiskEngine and Explainer (prediction-time services)."""
import json
import joblib, numpy as np, pandas as pd
import config as C, features as ft
from data_prep import Context  # noqa: F401  (needed to unpickle the context)

LABELS = {
    "blk": "time of day (6-h block)", "dow": "day of week", "month": "month", "is_weekend": "weekend",
    "is_peak_block": "peak-hour block", "cnt_7d": "crashes in this cell, last 7 days",
    "cnt_30d": "crashes in this cell, last 30 days", "cnt_90d": "crashes in this cell, last 90 days",
    "cnt_all": "crashes in this cell, all history", "cnt_blk_90d": "crashes in this cell at this time of day, last 90 days",
    "gi_z": "Gi* hotspot z-score of the cell", "gi_hot": "cell is a Gi* hotspot", "db_hot": "cell is in a DBSCAN cluster",
    "db_share": "share of cell crashes inside DBSCAN clusters", "feat_rate": "historical crash rate of the cell",
    "urban_share": "share of urban-road crashes", "highway_share": "share of highway crashes",
    "mean_lanes": "average lanes", "signal_share": "share of crashes at signalised spots",
    "lat_c": "cell latitude", "lon_c": "cell longitude", "city_code": "city (overall crash density)",
    "weather": "weather", "visibility": "visibility", "traffic_density": "traffic density",
    "is_peak_hour": "peak hour", "hour": "hour of day", "road_type": "road type", "lanes": "lanes",
    "traffic_signal": "traffic signal", "temperature": "temperature"}
VIS = {"clear": "high", "fog": "low", "rain": "medium"}


class WeatherError(RuntimeError):
    pass


def fetch_open_meteo(city, start, horizon, session=None):
    """Hourly forecast -> {hour_ts: 'clear'|'rain'|'fog'}. Raises WeatherError on any failure."""
    import requests
    lat, lon = C.CITY_CENTRES[city]
    try:
        r = (session or requests).get(C.OPEN_METEO_URL, timeout=10, params=dict(
            latitude=lat, longitude=lon, hourly="precipitation,visibility,weather_code",
            forecast_days=3, timezone="auto"))
        r.raise_for_status(); h = r.json()["hourly"]
    except Exception as e:  # network, HTTP or schema problems
        raise WeatherError(f"Could not fetch forecast weather ({e}).") from e
    out = {}
    for t, pr, vis, code in zip(h["time"], h["precipitation"], h["visibility"], h["weather_code"]):
        if (pr or 0) >= 0.2: w = "rain"
        elif (vis is not None and vis < 1000) or code in (45, 48): w = "fog"
        else: w = "clear"
        out[pd.Timestamp(t)] = w
    want = [start.floor("h") + pd.Timedelta(hours=i) for i in range(1, horizon + 1)]
    if not all(t in out for t in want):
        raise WeatherError("Forecast does not cover the requested hours.")
    return {t: out[t] for t in want}


class RiskEngine:
    def __init__(self, thresholds, p_ref):
        self.thr, self.p_ref = thresholds, p_ref

    def score(self, rel, cond_risk):
        """rel = P(crash) relative to the average cell of the same city and hour."""
        return np.asarray(rel) * np.asarray(cond_risk)

    def level(self, score, thr=None):
        t = thr or self.thr
        return np.where(score >= t["HIGH"], "High", np.where(score >= t["MEDIUM"], "Medium", "Low"))


class Service:
    def __init__(self, art=C.ART):
        m = joblib.load(art / "models.joblib"); c = joblib.load(art / "context.joblib")
        self.m, self.cells, self.spec, self.static, self.ctx = m, c["cells"], c["spec"], c["static"], c["ctx"]
        self.engine = RiskEngine(m["thresholds"], m["p_ref"])
        self.cities = sorted(self.spec)
        self.metrics = json.load(open(art / "metrics.json"))
        self.hotspots = pd.read_csv(art / "hotspots_full.csv")
        self.crashes = pd.read_csv(art / "crashes_clean.csv.gz", parse_dates=["date", "ts"])
        self._shap = None

    # --- helpers -----------------------------------------------------------------------
    def _cond_frame(self, cells, ts, weather, density, temp):
        st = self.static.loc[cells.cell_id]
        road = np.where(st.urban_share.fillna(1) > 0.5, "urban", np.where(st.highway_share.fillna(0) > 0.5, "highway", "rural"))
        X = pd.DataFrame({"weather": weather, "visibility": VIS[weather], "traffic_density": density,
                          "is_peak_hour": int(ts.hour in (8, 9, 10, 17, 18, 19)), "hour": ts.hour, "dow": ts.dayofweek,
                          "road_type": road, "lanes": st.mean_lanes.fillna(3).round().clip(1, 6).values,
                          "traffic_signal": (st.signal_share.fillna(0.5) > 0.5).astype(int).values, "temperature": temp})
        for c, cats in self.m["cats"].items(): X[c] = pd.Categorical(X[c], categories=cats)
        return X[self.m["cond_features"]]

    def forecast(self, city, start, weather, density="medium", horizon=6, temperature=None):
        """weather: 'clear'|'rain'|'fog' (one scenario) or dict {hour_ts: weather} from the live API."""
        if city not in self.spec: raise ValueError(f"Unknown city '{city}'.")
        if not 1 <= horizon <= 6: raise ValueError("Horizon must be 1-6 hours.")
        if density not in ("low", "medium", "high"): raise ValueError("Traffic density must be low, medium or high.")
        start = pd.Timestamp(start).floor("h")
        cells = self.cells[self.cells.city == city].reset_index(drop=True)
        temp = temperature if temperature is not None else self.m["city_temp"][city]
        rows = []
        for i in range(1, horizon + 1):
            ts = start + pd.Timedelta(hours=i)
            w = weather[ts] if isinstance(weather, dict) else weather
            day = (ts.normalize() - self.ctx.start).days; blk = ts.hour // C.BLOCK_H
            X = ft.build_features(self.ctx, self.static, cells.cell_id.values, np.full(len(cells), day), np.full(len(cells), blk))
            raw = np.column_stack([b.predict_proba(X)[:, 1] for b in self.m["bags"]])
            mu, sd = raw.mean(1), raw.std(1)
            cal = self.m["calibrator"]
            cond = self.m["risk_model"].predict(self._cond_frame(cells, ts, w, density, temp))
            Xc = self._cond_frame(cells, ts, w, density, temp)
            psev = self.m["severity_model"].predict_proba(Xc)[:, 1]
            p = cal(mu); rel = p / max(p.mean(), 1e-12); score = self.engine.score(rel, cond); cv = sd / np.maximum(mu, 1e-9)
            rows.append(pd.DataFrame({
                "cell_id": cells.cell_id, "hour_ts": ts, "weather": w, "hazardous": w in C.HAZARDOUS_WEATHER,
                "p_crash_block": p, "rel_risk": rel, "p_lo": cal(np.maximum(mu - 1.96 * sd, 0)), "p_hi": cal(mu + 1.96 * sd),
                "cond_risk": cond, "score": score, "alert": self.engine.level(score),
                "confidence": np.where(cv < 0.25, "High", np.where(cv < 0.5, "Medium", "Low")),
                "p_severe": psev, "thr_medium": self.engine.thr["MEDIUM"], "thr_high": self.engine.thr["HIGH"]}))
        out = pd.concat(rows, ignore_index=True).merge(
            cells[["cell_id", "lat_min", "lon_min", "dlat", "dlon", "lat_c", "lon_c"]], on="cell_id")
        return out

    def explain(self, city, ts, cell_id, weather, density="medium", top=4):
        """Top reasons (SHAP) for one cell-hour: crash-propensity model + conditions model."""
        import shap
        ts = pd.Timestamp(ts); day = (ts.normalize() - self.ctx.start).days
        X = ft.build_features(self.ctx, self.static, np.array([cell_id]), np.array([day]), np.array([ts.hour // C.BLOCK_H]))
        if self._shap is None:
            self._shap = (shap.TreeExplainer(self.m["bags"][0]), shap.TreeExplainer(self.m["risk_model"]))
        sv = self._shap[0].shap_values(X); sv = sv[1] if isinstance(sv, list) else sv
        cells = self.cells[self.cells.cell_id == cell_id]
        Xc = self._cond_frame(cells, ts, weather, density, self.m["city_temp"][cells.city.iloc[0]])
        sc = self._shap[1].shap_values(Xc)
        rep = [("crash propensity", f, X[f].iloc[0], v) for f, v in zip(ft.ALL_F, sv[0])]
        rep += [("conditions risk", f, Xc[f].iloc[0], v) for f, v in zip(self.m["cond_features"], sc[0])]
        df = pd.DataFrame(rep, columns=["model", "feature", "value", "shap"])
        df["reason"] = df.feature.map(LABELS)
        # rank by relative importance inside each model so both contribute
        df["rel"] = df.groupby("model").shap.transform(lambda s: s / (s.abs().max() + 1e-12))
        return df.reindex(df.rel.abs().sort_values(ascending=False).index).head(top).reset_index(drop=True)

    def actual_crashes(self, city, start, horizon):
        t0 = pd.Timestamp(start).floor("h") + pd.Timedelta(hours=1); t1 = t0 + pd.Timedelta(hours=horizon)
        c = self.crashes
        return c[(c.city == city) & (c.ts >= t0) & (c.ts < t1)]
