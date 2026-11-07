"""Does the pipeline find and predict hotspots when they really exist (and stay quiet when they don't)?

Uses the planted-hotspot benchmark (make_benchmark.py) where ground truth is known, and the original
indian_roads_dataset.csv as the 'no hotspots exist' control.

    python make_benchmark.py && python validate_benchmark.py
Writes artifacts_benchmark/validation.json and prints a summary table.
"""
import json
import numpy as np
import pandas as pd
import config as C, data_prep as dp, hotspots as hs

TOP_FRAC = 0.10          # "patrol the top 10% of cells"
MIN_PLANTED = 20         # a cell counts as truly hot if >= this many planted crashes landed in it


def _prep(path):
    df, _ = dp.load_clean(path)
    cells, spec = dp.build_grid(df)
    df["cell_id"] = dp.assign_cells(df.latitude, df.longitude, df.city, spec)
    return df, cells


def capture_rate(df, cells, frac=TOP_FRAC):
    """Rank cells by crashes up to FEATURE_END (past only); what share of LATER crashes fall in the top cells?
    Random ranking would give `frac`. Computed per city then pooled so city size cannot inflate it."""
    past = df[df.date <= C.FEATURE_END]; future = df[df.date >= C.TEST_START]
    hit = tot = 0
    for city, cc in cells.groupby("city"):
        pc = past[past.city == city].groupby("cell_id").size().reindex(cc.cell_id).fillna(0)
        fc = future[future.city == city].groupby("cell_id").size().reindex(cc.cell_id).fillna(0)
        k = max(1, int(round(frac * len(cc))))
        top = pc.sort_values(ascending=False, kind="stable").index[:k]
        hit += fc[top].sum(); tot += fc.sum()
    return float(hit / tot)


def neighbours(cell_ids, cells):
    """Cell ids within one cell (8-neighbourhood) of the given cells, incl. themselves."""
    c = cells.set_index("cell_id")
    out = set()
    for cid in cell_ids:
        r = c.loc[cid]
        near = c[(c.city == r.city) & ((c.lat_c - r.lat_c).abs() <= 1.01 * r.dlat) & ((c.lon_c - r.lon_c).abs() <= 1.01 * r.dlon)]
        out.update(near.index)
    return out


def hotspot_level(df, cells, tab):
    """Per planted hotspot centre: was ANY detected cell within one cell of it? (fair for a spatial smoother)"""
    truth = json.load(open(C.DATA / "benchmark_truth.json"))
    _, spec = dp.build_grid(df)
    cid = dp.assign_cells(pd.Series([t["lat"] for t in truth]), pd.Series([t["lon"] for t in truth]),
                          pd.Series([t["city"] for t in truth]), spec)
    rows = []
    for t, c in zip(truth, cid):
        near = neighbours([c], cells)
        rows.append(dict(extra=t["extra_crashes"],
                         gi=bool(tab.loc[tab.cell_id.isin(near), "gi_hot"].any()),
                         db=bool(tab.loc[tab.cell_id.isin(near), "db_hot"].any())))
    r = pd.DataFrame(rows); r["strong"] = r.extra >= 100
    return dict(n_hotspots=len(r),
                recall_gi_star=float(r.gi.mean()), recall_dbscan=float(r.db.mean()),
                recall_gi_star_strong=float(r[r.strong].gi.mean()), recall_gi_star_weak=float(r[~r.strong].gi.mean()),
                recall_dbscan_strong=float(r[r.strong].db.mean()), recall_dbscan_weak=float(r[~r.strong].db.mean()))


def oracle_capture(df, cells, frac=TOP_FRAC):
    """Ceiling: rank cells by where crashes were really planted (knows the truth)."""
    future = df[df.date >= C.TEST_START]; hit = tot = 0
    for city, cc in cells.groupby("city"):
        pl = df[(df.city == city) & (df.is_planted == 1)].groupby("cell_id").size().reindex(cc.cell_id).fillna(0)
        fc = future[future.city == city].groupby("cell_id").size().reindex(cc.cell_id).fillna(0)
        top = pl.sort_values(ascending=False, kind="stable").index[:max(1, int(round(frac * len(cc))))]
        hit += fc[top].sum(); tot += fc.sum()
    return float(hit / tot)


def detection(df, cells):
    out = hs.run_hotspots(df, cells)
    tab = out[0] if isinstance(out, tuple) else out
    det_gi = set(tab[tab.gi_hot == 1].cell_id); det_db = set(tab[tab.db_hot == 1].cell_id)
    truth = df[df.is_planted == 1].groupby("cell_id").size()
    true_cells = set(truth[truth >= MIN_PLANTED].index)
    near_true = neighbours(true_cells, cells)
    res = {}
    for name, det in [("gi_star", det_gi), ("dbscan", det_db), ("either", det_gi | det_db)]:
        res[name] = dict(
            detected_cells=len(det),
            true_hot_cells=len(true_cells),
            recall=len(det & true_cells) / max(1, len(true_cells)),
            precision_exact=len(det & true_cells) / max(1, len(det)),
            precision_within_1_cell=len(det & near_true) / max(1, len(det)))
    res["per_hotspot"] = hotspot_level(df, cells, tab)
    return res


if __name__ == "__main__":
    report = {}
    bdf, bcells = _prep(C.DATA / "benchmark_planted.csv")
    report["benchmark_detection"] = detection(bdf, bcells)
    report["benchmark_capture_top10pct_cells"] = capture_rate(bdf, bcells)
    report["benchmark_oracle_capture_ceiling"] = oracle_capture(bdf, bcells)
    odf, ocells = _prep(C.DATA / "indian_roads_dataset.csv")
    otab = hs.run_hotspots(odf, ocells); otab = otab[0] if isinstance(otab, tuple) else otab
    report["original_gi_hot_cells"] = int(otab.gi_hot.sum()); report["original_dbscan_hot_cells"] = int(otab.db_hot.sum())
    report["original_cells"] = int(len(otab))
    report["original_capture_top10pct_cells"] = capture_rate(odf, ocells)
    report["random_baseline_capture"] = TOP_FRAC
    (C.ROOT / "artifacts_benchmark").mkdir(exist_ok=True)
    json.dump(report, open(C.ROOT / "artifacts_benchmark" / "validation.json", "w"), indent=1)
    print(json.dumps(report, indent=1))
