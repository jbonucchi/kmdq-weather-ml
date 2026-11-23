"""Extra sky-cover features (2026-11-14). Input: raw HRRR frame (config.FIELDS columns, SI units) sorted by valid_utc,
same rows as features.build_features. Time-window features use neighbouring hours of the same continuous sequence
(an analysis series in training, one run's f00..f48 in prediction)."""
import numpy as np, pandas as pd
from config import TARGET_LAT, TARGET_LON

def _rh(t, td):
    a, b = 17.625, 243.04; tc, tdc = t - 273.15, td - 273.15
    return 100 * np.exp(a * tdc / (b + tdc) - a * tc / (b + tc))

def _cosz(t):
    t = pd.to_datetime(t); doy = t.dt.dayofyear.values; hr = (t.dt.hour + t.dt.minute / 60).values
    g = 2 * np.pi / 365 * (doy - 1 + (hr - 12) / 24)
    decl = 0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    ha = np.radians((hr * 60 + eqt + 4 * TARGET_LON) / 4 - 180)
    la = np.radians(TARGET_LAT)
    return np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(ha)

def sky_features(r, groups=("A",)):
    r = r.reset_index(drop=True); e = {}
    agl = lambda z: (z.where((z > -1) & (z < 30000)) - r.hsfc).clip(lower=0)
    cb, ce, ct = agl(r.cbase), agl(r.ceil), agl(r.ctop)
    if "A" in groups:
        for lv in ("925", "850", "700", "500"): e[f"rh{lv}"] = _rh(r[f"t{lv}"], r[f"td{lv}"])
        e["rh_low_max"] = np.maximum(e["rh925"], e["rh850"]); e["rh_mid_max"] = np.maximum(e["rh700"], e["rh500"])
        e["rh_lowmid_max"] = np.maximum(e["rh_low_max"], e["rh700"])
        # cloud below 12,000 ft (ASOS/AWOS ceilometer limit = 3658 m): HRRR low cloud layer, plus total cloud when its base is below 12k ft
        e["cld_below12k"] = np.where(cb < 3658, np.maximum(r.lcdc, r.tcdc), r.lcdc)
        e["cbase_lt12k"] = (cb < 3658).astype(float); e["ceil_lt12k"] = (ce < 3658).astype(float)
        e["ceil_lt3k_ft"] = (ce < 914).astype(float)
        e["tcdc_minus_hcdc"] = r.tcdc - r.hcdc
        e["cloud_depth"] = (ct - cb)
        e["log_tcolw"] = np.log10(r.tcolw.clip(lower=0) + 1e-4); e["log_tcoli"] = np.log10(r.tcoli.clip(lower=0) + 1e-4)
        cz = _cosz(r.valid_utc); e["cosz"] = cz
        e["sw_clear_ratio"] = np.where(cz > 0.15, r.dswrf / (1098 * cz * np.exp(-0.057 / np.maximum(cz, 0.15))), np.nan)
        e["lw_minus_bb"] = r.dlwrf - 5.67e-8 * r.t2 ** 4
        e["lcdc_x_rhlow"] = r.lcdc * e["rh_low_max"] / 100
    if "W" in groups:   # centered +/-1 h and +/-2 h windows within a continuous sequence
        t = pd.to_datetime(r.valid_utc); seg = ((t - t.shift(1)) != pd.Timedelta(hours=1)).cumsum()
        base = pd.DataFrame({"tcdc": r.tcdc, "lcdc": r.lcdc, "blcdc": r.blcdc,
                             "cb12": (cb < 3658).astype(float), "rhlow": np.maximum(_rh(r.t925, r.td925), _rh(r.t850, r.td850))})
        for w in (3, 5):
            g = base.groupby(seg.values)
            mx = g.rolling(w, center=True, min_periods=1).max().reset_index(level=0, drop=True).sort_index()
            mn = g.rolling(w, center=True, min_periods=1).mean().reset_index(level=0, drop=True).sort_index()
            for c in base:
                e[f"w{w}max_{c}"] = mx[c].values; e[f"w{w}mean_{c}"] = mn[c].values
            e[f"w{w}min_lcdc"] = g.lcdc.rolling(w, center=True, min_periods=1).min().reset_index(level=0, drop=True).sort_index().values
        # 6 h trailing+leading cloud fraction (cloudy hours share)
        e["w7mean_tcdc"] = base.groupby(seg.values).tcdc.rolling(7, center=True, min_periods=1).mean().reset_index(level=0, drop=True).sort_index().values
    if "N" in groups:   # neighbourhood stats (columns added by the neighbourhood fetch; NaN if absent)
        for c in ("tcdc", "lcdc"):
            for k in ("n3mean", "n3max", "n7mean", "n7max", "n7min", "n15mean", "n15gt50"):
                col = f"{c}_{k}"; e[col] = r[col].values if col in r else np.nan
            e[f"{c}_n7mean_minus_pt"] = e[f"{c}_n7mean"] - r[c].values
    return pd.DataFrame(e)
