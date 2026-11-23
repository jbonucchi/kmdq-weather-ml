#!/usr/bin/env python3
"""Fetch HRRR f00 analysis values at the target grid point from the hrrrzarr archive (s3://hrrrzarr,
the same HRRR wrfsfc output as s3://noaa-hrrr-bdp-pds, re-chunked 150x150 so one point = one ~50 KB chunk/field).
Hours missing from zarr fall back to byte-range GRIB2 from noaa-hrrr-bdp-pds via .idx.
Writes one parquet per UTC day in data/hrrr_days/ (resumable: complete days are skipped).
Usage: fetch_hrrr.py --start 2023-10-01 --end 2026-10-20 [--workers 96]"""
import os, sys, argparse, datetime as dt, threading, time, tempfile
import numpy as np, pandas as pd, requests, numcodecs
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
BLOSC = numcodecs.Blosc(); _tl = threading.local()
def sess():
    if not hasattr(_tl, "s"):
        s = requests.Session(); s.mount("https://", requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=4, max_retries=3)); _tl.s = s
    return _tl.s
CY, CX, OY, OX = IY // CHUNK, IX // CHUNK, IY % CHUNK, IX % CHUNK

def zarr_value(day, hour, lev, var):
    u = ZARR.format(d=day, h=hour, lev=lev, var=var, cy=CY, cx=CX)
    for a in range(4):
        try:
            r = sess().get(u, timeout=30)
            if r.status_code in (403, 404): return "MISSING"
            r.raise_for_status()
            b = BLOSC.decode(r.content)   # dtype varies by archive era/field: f2 (pre-2024), f4, f8
            dtp = {2: "<f2", 4: "<f4", 8: "<f8"}[len(b) // (CHUNK * CHUNK)]
            if dtp == "<f2" and var in ("PRATE", "MASSDEN"): return "MISSING"   # tiny values lost in float16 -> take from GRIB
            v = float(np.frombuffer(b, dtype=dtp).reshape(CHUNK, CHUNK)[OY, OX])
            return np.nan if v in (-9999.0, -10000.0) or not np.isfinite(v) else v
        except Exception:
            time.sleep(1 + 2 * a)
    return "ERROR"

def grib_point(url, fields=FIELDS):
    """Byte-range fetch of the requested fields from one wrfsfcfXX GRIB2 file; returns {col: value} or None."""
    import pygrib
    r = sess().get(url + ".idx", timeout=30)
    if r.status_code != 200: return None
    lines = [l.split(":") for l in r.text.strip().splitlines()]
    want = {f"{v}:{GRIB_LEVEL[lev]}": col for (lev, v), col in fields.items()}
    ranges, got = [], set()
    for i, p in enumerate(lines):
        k = f"{p[3]}:{p[4]}"
        if k in want and want[k] not in got and "max" not in p[5] and "min" not in p[5] and "acc" not in p[5] and "ave" not in p[5]:
            end = int(lines[i + 1][1]) - 1 if i + 1 < len(lines) else ""
            ranges.append((int(p[1]), end, want[k])); got.add(want[k])
    out = {c: np.nan for c in fields.values()}
    def one(rg):
        s, e, col = rg
        for a in range(4):
            try:
                g = sess().get(url, headers={"Range": f"bytes={s}-{e}"}, timeout=60); g.raise_for_status(); break
            except Exception:
                if a == 3: raise
                time.sleep(2 + 3 * a)
        with tempfile.NamedTemporaryFile(suffix=".grib2") as t:
            t.write(g.content); t.flush()
            gr = pygrib.open(t.name)
            try: v = gr[1].values[IY, IX]
            finally: gr.close()
            return col, (np.nan if np.ma.is_masked(v) else float(v))
    for col, v in _grib_pool().map(one, ranges): out[col] = v
    return out

_GP = None
def _grib_pool():   # one shared pool (threads + their sessions are reused, avoiding a per-call leak)
    global _GP
    if _GP is None: _GP = ThreadPoolExecutor(16)
    return _GP

def fetch_day(day, pool, now):
    d0 = dt.datetime.strptime(day, "%Y%m%d")
    hours = [h for h in range(24) if d0 + dt.timedelta(hours=h) <= now - dt.timedelta(hours=2)]
    tasks = [(h, lev, var, col) for h in hours for (lev, var), col in FIELDS.items()]
    vals = list(pool.map(lambda t: zarr_value(day, t[0], t[1], t[2]), tasks))
    rows = {h: {"valid_utc": d0 + dt.timedelta(hours=h)} for h in hours}
    nbad = {h: {} for h in hours}
    for (h, lev, var, col), v in zip(tasks, vals):
        if isinstance(v, str): nbad[h][(lev, var)] = col; v = np.nan
        rows[h][col] = v
    for h, miss in nbad.items():   # fallback: GRIB2 from noaa-hrrr-bdp-pds (whole hour, or just the absent fields)
        if not miss: continue
        try: g = grib_point(GRIB.format(d=day, h=h), fields=FIELDS if len(miss) > 10 else miss)
        except Exception as e: print(f"  {day} {h:02d}z GRIB fallback failed: {e}", flush=True); g = None
        if g is not None:
            rows[h].update({k: v for k, v in g.items() if pd.notna(v) or k in miss.values()})
            rows[h]["src"] = "grib" if len(miss) > 10 else "zarr+grib"
        else: rows[h]["src"] = "missing"   # retried on the next run (day not complete)
    df = pd.DataFrame(rows.values())
    df["src"] = df["src"].fillna("zarr") if "src" in df else "zarr"
    return df

def run(start, end, workers=96, verbose=True):
    os.makedirs(HRRR_DIR, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    days = pd.date_range(start, end, freq="D").strftime("%Y%m%d")
    with ThreadPoolExecutor(workers) as pool:
        for i, day in enumerate(days[::-1]):   # newest first
            f = os.path.join(HRRR_DIR, f"{day}.parquet")
            if os.path.exists(f):
                old = pd.read_parquet(f)
                if len(old) == 24 and not (old["src"] == "missing").any(): continue
            t = time.time(); df = fetch_day(day, pool, now)
            if len(df): df.to_parquet(f, index=False)
            if verbose: print(f"{day}: {len(df)} h, src={df['src'].value_counts().to_dict() if len(df) else {}}, {time.time()-t:.1f}s", flush=True)

def load_all():
    fs = sorted(os.path.join(HRRR_DIR, f) for f in os.listdir(HRRR_DIR) if f.endswith(".parquet"))
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    return df[df["src"] != "missing"].sort_values("valid_utc").reset_index(drop=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--start", default="2023-10-01"); ap.add_argument("--end", default=dt.date.today().isoformat())
    ap.add_argument("--workers", type=int, default=96); a = ap.parse_args(); run(a.start, a.end, a.workers)

# ---------------- forecast (f01..f48) access ----------------
ZARR_FCST = "https://hrrrzarr.s3.amazonaws.com/sfc/{d}/{d}_{h:02d}z_fcst.zarr/{lev}/{var}/{lev}/{var}/0.{cy}.{cx}"
GRIB_F = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{d}/conus/hrrr.t{h:02d}z.wrfsfcf{f:02d}.grib2"

def fcst_zarr(run, workers=32):
    """All forecast hours of one run from hrrrzarr *_fcst.zarr (chunk = 48 or 18 lead times x 150 x 150).
    Returns DataFrame with run_utc, fhr, valid_utc + raw fields, or None if not (fully) available."""
    d, h = run.strftime("%Y%m%d"), run.hour
    def one(item):
        (lev, var), col = item
        u = ZARR_FCST.format(d=d, h=h, lev=lev, var=var, cy=CY, cx=CX)
        for a in range(4):
            try:
                r = sess().get(u, timeout=60)
                if r.status_code in (403, 404): return col, None
                r.raise_for_status(); b = BLOSC.decode(r.content)
                n = len(b) // (CHUNK * CHUNK * 4); dtp = "<f4"
                if len(b) % (CHUNK * CHUNK * 4) or n not in (18, 48): n = len(b) // (CHUNK * CHUNK * 8); dtp = "<f8"
                a3 = np.frombuffer(b, dtype=dtp).reshape(n, CHUNK, CHUNK)[:, OY, OX].astype(float)
                a3[a3 == -9999.0] = np.nan; return col, a3
            except Exception:
                time.sleep(1 + 2 * a)
        return col, None
    with ThreadPoolExecutor(workers) as ex: res = dict(ex.map(one, FIELDS.items()))
    good = {k: v for k, v in res.items() if v is not None}
    if len(good) < len(FIELDS) - 5: return None
    n = max(len(v) for v in good.values())
    df = pd.DataFrame({k: (good[k] if k in good else np.full(n, np.nan)) for k in FIELDS.values()})
    df.insert(0, "fhr", np.arange(1, n + 1)); df.insert(0, "run_utc", run)
    df.insert(2, "valid_utc", [run + dt.timedelta(hours=int(f)) for f in df.fhr]); df["src"] = "zarr_fcst"
    return df

def fcst_grib(run, fhrs=range(1, 49), workers=6):
    """Same as fcst_zarr but from the GRIB2 files on noaa-hrrr-bdp-pds (slower; used as fallback)."""
    d, h = run.strftime("%Y%m%d"), run.hour
    def one(f):
        o = grib_point(GRIB_F.format(d=d, h=h, f=f))
        return None if o is None else {"run_utc": run, "fhr": f, "valid_utc": run + dt.timedelta(hours=f), **o, "src": "grib"}
    with ThreadPoolExecutor(workers) as ex: rows = [r for r in ex.map(one, fhrs) if r]
    return pd.DataFrame(rows) if rows else None
