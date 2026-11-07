"""M1/M2: load, clean, grid, and cell x time-block count tensors."""
import numpy as np
import pandas as pd
import config as C


def load_clean(path=C.RAW_CSV):
    df = pd.read_csv(path)
    rep = {"rows_raw": int(len(df))}
    df = df.drop_duplicates("accident_id")
    rep["duplicate_ids_removed"] = rep["rows_raw"] - len(df)
    ok = df.latitude.between(*C.INDIA_LAT) & df.longitude.between(*C.INDIA_LON)
    rep["invalid_coords_removed"] = int((~ok).sum())
    df = df[ok].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    rep["invalid_dates_removed"] = int(df.date.isna().sum())
    df = df[df.date.notna()].copy()
    h = df.time.str.split(":").str[0].astype(int)
    rep["hour_time_mismatch"] = int((h != df.hour).sum())
    df["festival"] = df["festival"].fillna("None")
    df["ts"] = df.date + pd.to_timedelta(df.hour, unit="h")
    df["severe"] = df.accident_severity.isin(C.SEVERE_LEVELS).astype(int)
    rep["rows_clean"] = int(len(df))
    return df.reset_index(drop=True), rep


def build_grid(df):
    """Per-city grid that tiles the bounding box of that city's crashes exactly
    (equal-size cells, no partial edge cells, which would fake spatial autocorrelation)."""
    spec, frames, nxt = {}, [], 0
    for city, g in df.groupby("city"):
        lat0, lat1 = float(g.latitude.min()), float(g.latitude.max())
        lon0, lon1 = float(g.longitude.min()), float(g.longitude.max())
        ny = max(1, int(round((lat1 - lat0) / C.CELL_DEG)))
        nx = max(1, int(round((lon1 - lon0) / C.CELL_DEG)))
        dlat, dlon = (lat1 - lat0) / ny, (lon1 - lon0) / nx
        iy, ix = np.divmod(np.arange(nx * ny), nx)
        frames.append(pd.DataFrame({
            "cell_id": nxt + np.arange(nx * ny), "city": city, "ix": ix, "iy": iy,
            "lat_min": lat0 + iy * dlat, "lon_min": lon0 + ix * dlon, "dlat": dlat, "dlon": dlon}))
        spec[city] = dict(lat0=lat0, lon0=lon0, lat1=lat1, lon1=lon1, nx=nx, ny=ny,
                          dlat=dlat, dlon=dlon, first_id=nxt)
        nxt += nx * ny
    cells = pd.concat(frames, ignore_index=True)
    cells["lat_c"] = cells.lat_min + cells.dlat / 2
    cells["lon_c"] = cells.lon_min + cells.dlon / 2
    cells["city_code"] = cells.city.astype("category").cat.codes
    return cells, spec


def assign_cells(lat, lon, city, spec):
    out = np.full(len(lat), -1, dtype=int)
    lat, lon, city = np.asarray(lat), np.asarray(lon), np.asarray(city)
    for c, s in spec.items():
        m = city == c
        ix = np.clip(np.floor((lon[m] - s["lon0"]) / s["dlon"]).astype(int), 0, s["nx"] - 1)
        iy = np.clip(np.floor((lat[m] - s["lat0"]) / s["dlat"]).astype(int), 0, s["ny"] - 1)
        out[m] = s["first_id"] + iy * s["nx"] + ix
    return out


class Context:
    """Holds count tensors. D[cell, day, block]; cum[cell, d] = crashes on days < d."""

    def __init__(self, df, cells, start, ndays):
        self.start, self.ndays, self.ncell = pd.Timestamp(start), int(ndays), len(cells)
        day = (df.date - self.start).dt.days.values
        blk = (df.hour.values // C.BLOCK_H)
        n_blk = 24 // C.BLOCK_H
        self.D = np.zeros((self.ncell, self.ndays, n_blk), dtype=np.int16)
        np.add.at(self.D, (df.cell_id.values, day, blk), 1)
        self.n_blk = n_blk
        self.cum = np.zeros((self.ncell, self.ndays + 1), dtype=np.int32)
        self.cum[:, 1:] = np.cumsum(self.D.sum(2), axis=1)
        self.cumb = np.zeros((self.ncell, self.ndays + 1, n_blk), dtype=np.int32)
        self.cumb[:, 1:, :] = np.cumsum(self.D, axis=1)

    def day_of(self, date):
        return int((pd.Timestamp(date) - self.start).days)

    def rows(self, d0, d1):
        """All (cell, day, block) triples for days d0..d1 inclusive."""
        d1 = min(d1, self.ndays - 1)
        cell, day, blk = np.meshgrid(np.arange(self.ncell), np.arange(d0, d1 + 1),
                                     np.arange(self.n_blk), indexing="ij")
        cell, day, blk = cell.ravel(), day.ravel(), blk.ravel()
        y = (self.D[cell, day, blk] > 0).astype(np.int8)
        return cell, day, blk, y
