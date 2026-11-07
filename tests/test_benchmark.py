"""Method validation: the detectors must find planted hotspots and stay quiet on the spatially random original data."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pytest
import config as C, data_prep as dp, hotspots as hs, make_benchmark as mb, validate_benchmark as vb


@pytest.fixture(scope="module")
def planted():
    out, truth = mb.build()
    path = C.DATA / "_test_benchmark.csv"; out.to_csv(path, index=False)
    try:
        df, cells = vb._prep(path)
    finally:
        path.unlink(missing_ok=True)
    tab = hs.run_hotspots(df, cells)[0]
    return df, cells, tab


def test_gi_star_uses_a_real_z_score(planted):
    # regression: esda's .Zs understated z ~3x and made Gi* find nothing even for obvious clusters
    _, _, tab = planted
    assert tab.gi_z.max() > 3.5 and tab.gi_hot.sum() >= 20


def test_dbscan_recovers_every_planted_hotspot(planted):
    df, cells, tab = planted
    assert vb.hotspot_level(df, cells, tab)["recall_dbscan"] >= 0.9


def test_past_hotspots_predict_future_crashes(planted):
    df, cells, _ = planted
    assert vb.capture_rate(df, cells) > 0.18          # random = 0.10, oracle ceiling ~0.25


def test_no_hotspots_in_spatially_random_data():
    df, cells = vb._prep(C.DATA / "indian_roads_dataset.csv")
    tab = hs.run_hotspots(df, cells)[0]
    assert tab.gi_hot.mean() < 0.01
    assert abs(vb.capture_rate(df, cells) - 0.10) < 0.03
