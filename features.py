"""M4: cell-block features. Rolling history is strictly causal (days < day of the row);
static hotspot/road features come only from the FEATURE window (before the training rows)."""
import numpy as np
import pandas as pd
import config as C

TIME_F = ["blk", "dow", "month", "is_weekend", "is_peak_block"]
HIST_F = ["cnt_7d", "cnt_30d", "cnt_90d", "cnt_all", "cnt_blk_90d"]
STATIC_F = ["gi_z", "gi_hot", "db_hot", "db_share", "feat_rate", "urban_share", "highway_share",
            "mean_lanes", "signal_share"]
POS_F = ["lat_c", "lon_c", "city_code"]
ALL_F = TIME_F + HIST_F + STATIC_F + POS_F
TRANSFERABLE_F = TIME_F + HIST_F + ["gi_z", "gi_hot", "db_hot", "db_share", "feat_rate",
                                     "urban_share", "highway_share", "mean_lanes", "signal_share"]


def make_static(cells, feat_crashes, hs):
    """Static per-cell table from the feature window only."""
    g = feat_crashes.groupby("cell_id")
    road = pd.DataFrame({
        "urban_share": g.road_type.apply(lambda s: (s == "urban").mean()),
        "highway_share": g.road_type.apply(lambda s: (s == "highway").mean()),
        "mean_lanes": g.lanes.mean(), "signal_share": g.traffic_signal.mean()})
    st = cells.set_index("cell_id").join(hs.set_index("cell_id")[["gi_z", "gi_hot", "db_hot", "db_share", "count"]])
    st = st.join(road)
    nd = max((pd.Timestamp(C.FEATURE_END) - pd.Timestamp("2022-01-01")).days + 1, 1)
    st["feat_rate"] = st["count"] / nd
    return st.drop(columns=["count"]).sort_index()


def build_features(ctx, static, cell, day, blk):
    n = ctx.ndays
    dcl = np.clip(day, 0, n)
    def win(w):
        return ctx.cum[cell, dcl] - ctx.cum[cell, np.clip(dcl - w, 0, None)]
    dates = ctx.start + pd.to_timedelta(day, unit="D")
    f = pd.DataFrame({
        "blk": blk.astype(np.int8), "dow": dates.dayofweek.values.astype(np.int8),
        "month": dates.month.values.astype(np.int8),
        "is_weekend": (dates.dayofweek.values >= 5).astype(np.int8),
        "is_peak_block": np.isin(blk, [1, 3]).astype(np.int8),   # 06-12 and 18-24 blocks contain the peak hours
        "cnt_7d": win(7), "cnt_30d": win(30), "cnt_90d": win(90), "cnt_all": ctx.cum[cell, dcl],
        "cnt_blk_90d": ctx.cumb[cell, dcl, blk] - ctx.cumb[cell, np.clip(dcl - 90, 0, None), blk]})
    s = static.loc[cell, STATIC_F + POS_F].reset_index(drop=True)
    f = pd.concat([f.reset_index(drop=True), s], axis=1)
    return f[ALL_F].astype(np.float32)
