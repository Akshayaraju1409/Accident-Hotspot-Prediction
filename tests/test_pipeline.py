import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, pytest
import config as C, data_prep as dp, features as ft, hotspots as hs
from predict import RiskEngine, fetch_open_meteo, WeatherError, Service

@pytest.fixture(scope="module")
def world():
    df, _ = dp.load_clean(); cells, spec = dp.build_grid(df)
    df["cell_id"] = dp.assign_cells(df.latitude, df.longitude, df.city, spec)
    ctx = dp.Context(df, cells, df.date.min(), (df.date.max() - df.date.min()).days + 1)
    return df, cells, spec, ctx

def test_every_crash_in_its_cell(world):
    df, cells, spec, _ = world
    c = cells.set_index("cell_id").loc[df.cell_id]
    assert ((df.latitude.values >= c.lat_min.values - 1e-9) & (df.latitude.values <= (c.lat_min + c.dlat).values + 1e-9)).all()
    assert (c.city.values == df.city.values).all()

def test_counts_conserved(world):
    df, _, _, ctx = world
    assert ctx.D.sum() == len(df)

def test_features_are_causal(world):
    df, cells, _, ctx = world
    static = ft.make_static(cells, df[df.date <= C.FEATURE_END], hs.run_hotspots(df[df.date <= C.FEATURE_END], cells)[0])
    d = 700; cell = np.arange(ctx.ncell); day = np.full(ctx.ncell, d); blk = np.zeros(ctx.ncell, int)
    base = ft.build_features(ctx, static, cell, day, blk)
    # inject crashes ON and AFTER day d: features for day d must not change
    ctx2 = dp.Context(df, cells, ctx.start, ctx.ndays); ctx2.D[:, d:, :] += 1
    ctx2.cum[:, 1:] = np.cumsum(ctx2.D.sum(2), axis=1); ctx2.cumb[:, 1:, :] = np.cumsum(ctx2.D, axis=1)
    mod = ft.build_features(ctx2, static, cell, day, blk)
    pd.testing.assert_frame_equal(base, mod)

def test_bh_fdr():
    p = np.array([0.001, 0.01, 0.04, 0.5, 0.9]); q = hs._bh(p)
    assert (q >= p - 1e-12).all() and (np.diff(q[np.argsort(p)]) >= -1e-12).all()

def test_risk_engine_levels():
    e = RiskEngine({"MEDIUM": 1.0, "HIGH": 2.0}, 0.01)
    assert list(e.level(np.array([0.5, 1.5, 2.5]))) == ["Low", "Medium", "High"]

def test_weather_failure_gives_no_forecast():
    class Bad:
        def get(self, *a, **k): raise OSError("offline")
    with pytest.raises(WeatherError):
        fetch_open_meteo("Delhi", pd.Timestamp("2025-01-01 10:00"), 6, session=Bad())

def test_weather_mapping():
    class R:
        def raise_for_status(self): pass
        def json(self):
            t = pd.date_range("2025-01-01 11:00", periods=6, freq="h").strftime("%Y-%m-%dT%H:%M")
            return {"hourly": {"time": list(t), "precipitation": [0, 1, 0, 0, 0, 0], "visibility": [20000, 20000, 500, 20000, 20000, 20000],
                               "weather_code": [0] * 6}}
    class S_:
        def get(self, *a, **k): return R()
    out = fetch_open_meteo("Delhi", pd.Timestamp("2025-01-01 10:20"), 6, session=S_())
    assert list(out.values())[:3] == ["clear", "rain", "fog"]

def test_service_validation_and_alerts():
    s = Service()
    with pytest.raises(ValueError): s.forecast("Atlantis", "2024-12-15 12:00", "rain")
    with pytest.raises(ValueError): s.forecast("Delhi", "2024-12-15 12:00", "rain", horizon=9)
    r = s.forecast("Delhi", "2024-12-15 12:00", "fog", "high", 3)
    assert set(r.alert) <= {"Low", "Medium", "High"} and r.hour_ts.nunique() == 3
    # hazardous + dense traffic must never be lower-risk than clear + sparse for the same cells
    a = s.forecast("Delhi", "2024-12-15 12:00", "clear", "low", 1); b = s.forecast("Delhi", "2024-12-15 12:00", "fog", "high", 1)
    assert b.score.mean() > a.score.mean()
    ex = s.explain("Delhi", r.hour_ts.iloc[0], int(r.cell_id.iloc[0]), "fog", "high")
    assert len(ex) == 4
