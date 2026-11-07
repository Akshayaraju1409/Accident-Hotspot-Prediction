"""M3: Getis-Ord Gi* (with FDR control) and DBSCAN; compared; plus Moran's I diagnostic."""
import numpy as np
import pandas as pd
from scipy.stats import norm, poisson
from sklearn.cluster import DBSCAN
from libpysal.weights import KNN, fill_diagonal
from esda import G_Local, Moran
import config as C


def _bh(p, q=C.FDR_Q):
    p = np.asarray(p, float); n = len(p)
    order = np.argsort(p); ranked = p[order] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n); out[order] = np.minimum(adj, 1)
    return out


def _xy_km(lat, lon):
    return np.c_[np.asarray(lon) * np.cos(np.radians(np.asarray(lat))) * 111.32, np.asarray(lat) * 110.57]


def run_hotspots(crashes, cells):
    """crashes: DataFrame with city, latitude, longitude, cell_id. Returns per-cell table + diagnostics."""
    res = []; diag = {}
    cnt = crashes.groupby("cell_id").size()
    for city, cc in cells.groupby("city"):
        counts = cc.cell_id.map(cnt).fillna(0).values.astype(float)
        xy = _xy_km(cc.lat_c, cc.lon_c)
        w = KNN.from_array(xy, k=C.GI_K); w.transform = "r"
        wstar = fill_diagonal(KNN.from_array(xy, k=C.GI_K), 1.0); wstar.transform = "r"   # self included -> Gi*
        g = G_Local(counts, wstar, star=None, permutations=C.GI_PERMS, seed=C.SEED)
        z = np.nan_to_num(g.z_sim)  # permutation z-score; esda's .Zs is not the Gi* z and understates it ~3x
        p_hi = norm.sf(z)                       # one-sided: unusually HIGH counts
        q_hi = _bh(p_hi); q_lo = _bh(norm.cdf(z))
        np.random.seed(C.SEED); mi = Moran(counts, w, permutations=C.GI_PERMS)
        diag[city] = dict(moran_I=float(mi.I), moran_p=float(mi.p_sim), cells=int(len(cc)),
                          crashes=int(counts.sum()))
        # DBSCAN on raw crash locations (haversine)
        cr = crashes[crashes.city == city]
        lab = np.full(len(cr), -1)
        # min_samples from a complete-spatial-randomness null: smallest count that a Poisson
        # neighbourhood of radius eps exceeds with probability < 0.001 (never below DB_MIN)
        area = ((cc.lat_c.max() - cc.lat_c.min() + cc.dlat.iloc[0]) * 110.57) * \
               ((cc.lon_c.max() - cc.lon_c.min() + cc.dlon.iloc[0]) * 111.32 * np.cos(np.radians(cc.lat_c.mean())))
        lam = len(cr) / area * np.pi * C.DB_EPS_KM ** 2
        min_pts = int(max(C.DB_MIN, poisson.ppf(0.999, lam) + 1))
        diag[city]["dbscan_min_samples"] = min_pts
        if len(cr) >= min_pts:
            lab = DBSCAN(eps=C.DB_EPS_KM / 6371.0, min_samples=min_pts, metric="haversine").fit_predict(
                np.radians(cr[["latitude", "longitude"]].values))
        clustered = pd.Series((lab >= 0).astype(int), index=cr.index).groupby(cr.cell_id).sum()
        tot = pd.Series(1, index=cr.index).groupby(cr.cell_id).sum()
        share = (clustered / tot).reindex(cc.cell_id).fillna(0).values
        res.append(pd.DataFrame({
            "cell_id": cc.cell_id.values, "count": counts, "gi_z": z, "gi_p": p_hi, "gi_q": q_hi,
            "gi_hot": ((q_hi < C.FDR_Q) & (z > 0)).astype(int),
            "gi_cold": ((q_lo < C.FDR_Q) & (z < 0)).astype(int),
            "db_share": share, "db_hot": (share > 0.5).astype(int),
            "db_clusters": int(len(set(lab)) - (1 if -1 in lab else 0))}))
    out = pd.concat(res, ignore_index=True)
    a, b = out.gi_hot.astype(bool), out.db_hot.astype(bool)
    diag["_compare"] = dict(gi_hot_cells=int(a.sum()), dbscan_hot_cells=int(b.sum()),
                            overlap=int((a & b).sum()), union=int((a | b).sum()),
                            jaccard=float((a & b).sum() / max((a | b).sum(), 1)))
    return out, diag
