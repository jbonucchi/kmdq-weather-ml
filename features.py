"""Feature engineering shared by training, prediction and verification. Input: raw HRRR point values
(columns named as in config.FIELDS, SI units) with a valid_utc column, sorted in time; output: model feature frame."""
import numpy as np, pandas as pd

def _spd(u, v): return np.hypot(u, v)
def _dir_sc(u, v):
    d = np.arctan2(-u, -v); return np.sin(d), np.cos(d)

def _agl(z, hs):
    z = z.where((z > -1) & (z < 30000)); return (z - hs).clip(lower=0)

def build_features(raw):
    r = raw.sort_values("valid_utc").reset_index(drop=True)
    f = {}
    K = 273.15
    f["t2_c"] = r.t2 - K; f["td2_c"] = r.td2 - K; f["dpd2"] = r.t2 - r.td2; f["rh2"] = r.rh2; f["q2"] = r.q2 * 1000
    f["tskin_minus_t2"] = r.tskin - r.t2
    for lv in ("1000", "925", "850", "700", "500"):
        f[f"t{lv}_c"] = r[f"t{lv}"] - K; f[f"dpd{lv}"] = r[f"t{lv}"] - r[f"td{lv}"]
    f["lapse_850_500"] = r.t850 - r.t500; f["lapse_sfc_850"] = r.t2 - r.t850; f["inv_925"] = r.t925 - r.t2
    for nm, u, v in (("10m", "u10", "v10"), ("80m", "u80", "v80"), ("925", "u925", "v925"), ("850", "u850", "v850"),
                     ("700", "u700", "v700"), ("500", "u500", "v500")):
        f[f"wspd_{nm}"] = _spd(r[u], r[v]); f[f"wdir_sin_{nm}"], f[f"wdir_cos_{nm}"] = _dir_sc(r[u], r[v])
    f["gust"] = r.gust; f["gust_minus_wind"] = r.gust - f["wspd_10m"]
    f["shear01"] = _spd(r.ush1, r.vsh1); f["shear06"] = _spd(r.ush6, r.vsh6)
    f["mslp_hpa"] = r.mslp / 100; f["psfc_hpa"] = r.psfc / 100
    for c in ("z850", "z700", "z500"): f[c] = r[c]
    f["prate_mmhr"] = r.prate * 3600; f["log_prate"] = np.log10(f["prate_mmhr"] + 1e-3)
    for c in ("crain", "csnow", "cfrzr", "cicep", "cpofp"): f[c] = r[c]
    f["vis_km"] = r.vis / 1000; f["log_vis"] = np.log10(r.vis.clip(lower=1))
    f["snod"] = r.snod
    for c in ("tcdc", "lcdc", "mcdc", "hcdc", "blcdc"): f[c] = r[c]
    f["cbase_agl"] = _agl(r.cbase, r.hsfc); f["ceil_agl"] = _agl(r.ceil, r.hsfc); f["ctop_agl"] = _agl(r.ctop, r.hsfc)
    f["has_ceil"] = f["ceil_agl"].notna().astype(int); f["log_ceil"] = np.log10(f["ceil_agl"].fillna(20000) + 10)
    f["zlcl_agl"] = r.zlcl; f["z0c_agl"] = _agl(r.z0c, r.hsfc); f["zfrz_agl"] = _agl(r.zfrz, r.hsfc)
    for c in ("refc", "refd1km", "refd4km", "refd263k"): f[c] = r[c].clip(lower=-20)
    for c in ("vil", "tcolw", "tcoli", "rhpw", "pwat", "pbl", "dswrf", "dlwrf", "shtfl", "lhtfl", "w700"): f[c] = r[c]
    f["log_smoke"] = np.log10(r.smoke.clip(lower=0) * 1e9 + 0.1)
    for c in ("sbt113", "sbt114", "sbt123", "sbt124"): f[c] = r[c]
    f["sbt_ir_minus_t2"] = r.sbt124 - r.t2; f["sbt_wv_minus_ir"] = r.sbt123 - r.sbt124
    for c in ("sbcape", "sbcin", "mucape", "mucin", "mlcape", "mlcin", "cape03", "li", "bli", "srh1", "srh3"): f[c] = r[c]
    # time features (local solar-ish timing captured by UTC hour; day-of-year seasonality)
    t = pd.to_datetime(r.valid_utc)
    hr = t.dt.hour + t.dt.minute / 60; doy = t.dt.dayofyear
    f["hour_sin"] = np.sin(2 * np.pi * hr / 24); f["hour_cos"] = np.cos(2 * np.pi * hr / 24)
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25); f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    # tendencies from the previous 1 and 3 hours (only where the previous rows are truly 1/3 h earlier)
    for lag in (1, 3):
        ok = (t - t.shift(lag)) == pd.Timedelta(hours=lag)
        for c in ("t2_c", "td2_c", "mslp_hpa", "tcdc", "lcdc", "refc", "log_vis", "rh2"):
            f[f"d{lag}h_{c}"] = (f[c] - f[c].shift(lag)).where(ok)
    f = pd.DataFrame(f); f.insert(0, "valid_utc", t.values)
    return f

def feature_cols(f): return [c for c in f.columns if c != "valid_utc"]
