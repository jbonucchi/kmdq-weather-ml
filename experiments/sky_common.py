"""Shared setup for the 2026-11-14 sky-cover (clear vs overcast) experiments. Same split as the fog/rain work:
train < 2024-10-21 03Z, validation year -> 2025-10-21 03Z, test year -> 2026-10-21 03Z. Validation = model selection, test scored once."""
import os, sys, json, pickle, datetime as dt, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, "/workspace/hrrr_ml"); sys.path.insert(0, "/workspace/hrrr_ml/experiments")
import train as T, model_lib as M, features as X, lead_eval as LE
from labels import CLASSES
C = {c: i for i, c in enumerate(CLASSES)}
CL, PC, MC, OV, FOG, LR, RA, HR, TS = (C[k] for k in ("Clear", "Partly cloudy", "Mostly cloudy", "Overcast", "Fog/mist/haze", "Light rain/drizzle", "Rain", "Heavy rain", "Thunderstorm"))
SKY = [CL, PC, MC, OV]; PRECIP = [LR, RA, HR]
EXP = "/workspace/hrrr_ml/experiments"
va0, te0, te1 = pd.Timestamp("2024-10-21 03:00"), pd.Timestamp("2025-10-21 03:00"), pd.Timestamp("2026-10-21 03:00")
ds = pd.read_parquet("/workspace/hrrr_ml/data/dataset_KMDQ.parquet").sort_values("valid_utc").reset_index(drop=True)
ds["valid_utc"] = pd.to_datetime(ds.valid_utc)
lab = ds.dropna(subset=["label"]).set_index("valid_utc").label
lab = lab[lab.isin(CLASSES)]
an = ds.set_index("valid_utc")

def add_extra(f, r, groups=("A",)):
    """Extra features computed from the raw HRRR frame r (aligned row-wise with f, sorted by valid_utc, possibly several
    runs concatenated -> 'seg' column delimits continuous sequences for the time-window features)."""
    from features_sky import sky_features
    e = sky_features(r, groups)
    return pd.concat([f.reset_index(drop=True), e.reset_index(drop=True)], axis=1)

_NA = {}
def neigh_anl():
    if "a" not in _NA:
        import fetch_neigh as FN; _NA["a"] = FN.load_anl().drop_duplicates("valid_utc").set_index("valid_utc")
    return _NA["a"]
def with_neigh(r):
    na = neigh_anl(); r = r.copy(); vt = pd.to_datetime(r.valid_utc)
    for c in na.columns: r[c] = na[c].reindex(vt).values
    return r

def f00_frame(groups=None):
    d = with_neigh(ds) if groups and "N" in groups else ds
    f = X.build_features(d)
    if groups: f = add_extra(f, d, groups)
    f["label"] = ds.label.values
    f = f[f.label.isin(CLASSES)].reset_index(drop=True); f["y"] = f.label.map(C).astype(int)
    return f

def fc_frames(lo, hi, groups=None, tag=""):
    cache = f"{EXP}/cache_sky_{tag}{'_'.join(groups) if groups else 'base'}_{lo:%Y%m%d}_{hi:%Y%m%d}.pkl"
    if os.path.exists(cache): return pickle.load(open(cache, "rb"))
    Fs = []
    for x in sorted(os.listdir(LE.FD)):
        run = dt.datetime.strptime(x[:10], "%Y%m%d%H")
        if not (lo <= pd.Timestamp(run) and pd.Timestamp(run) + pd.Timedelta(hours=48) <= hi): continue
        fc = pd.read_parquet(os.path.join(LE.FD, x))
        prev = an.loc[[run - dt.timedelta(hours=k) for k in (2, 1, 0) if run - dt.timedelta(hours=k) in an.index]].reset_index()
        if groups and "N" in groups:
            prev = with_neigh(prev); nf = pd.read_parquet(f"/workspace/hrrr_ml/data/fcst_12z_neigh/{x[:10]}.parquet")
            fc = fc.merge(nf.drop(columns="valid_utc"), on="fhr", how="left")
        prev["fhr"] = 0; seq = pd.concat([prev, fc], ignore_index=True); seq["valid_utc"] = pd.to_datetime(seq.valid_utc)
        seq = seq.sort_values("valid_utc").reset_index(drop=True)
        g = X.build_features(seq)
        if groups: g = add_extra(g, seq, groups)
        g["fhr"] = seq.fhr.values; g["run"] = run
        g = g[(g.fhr >= 1) & g.valid_utc.isin(lab.index)]
        Fs.append(g)
    G = pd.concat(Fs, ignore_index=True); y = G.valid_utc.map(lab).map(C).values.astype(int)
    pickle.dump((G, y), open(cache, "wb")); return G, y

def fit(Xtr, ytr, Xva=None, yva=None, rounds=None, mult=None, seed=42, params=None, sky_mult=None):
    p = dict(T.PARAMS, seed=seed, **(params or {}))
    m_ = dict(T.CLASS_WEIGHT_MULT if mult is None else mult)
    if sky_mult: m_.update(sky_mult)
    w = T.weights(ytr, 0.25, m_)
    d = lgb.Dataset(Xtr, ytr, weight=w)
    if Xva is None: return lgb.train(p, d, num_boost_round=rounds)
    dv = lgb.Dataset(Xva, yva, weight=T.weights(yva, 0.25, m_), reference=d)
    return lgb.train(p, d, 3000, valid_sets=[dv], callbacks=[lgb.early_stopping(100, verbose=False)])

def summ(y, p):
    m = M.metrics(y, p); pc = m["per_class"]
    wy = np.isin(y, PRECIP); wp = np.isin(p, PRECIP); tp = (wy & wp).sum()
    o = dict(n=int(len(y)), acc=m["accuracy"], macro_f1=m["macro_f1"])
    for nm, k in (("clear", CL), ("pc", PC), ("mc", MC), ("ovc", OV), ("fog", FOG), ("lr", LR), ("ra", RA), ("ts", TS)):
        o[f"{nm}_f1"] = pc[CLASSES[k]]["f1"]; o[f"{nm}_r"] = pc[CLASSES[k]]["recall"]; o[f"{nm}_p"] = pc[CLASSES[k]]["precision"]
    o["sky_macro_f1"] = float(np.mean([pc[CLASSES[k]]["f1"] for k in SKY]))
    o["obsOVC_predCLR"] = int(((y == OV) & (p == CL)).sum()); o["obsCLR_predOVC"] = int(((y == CL) & (p == OV)).sum())
    o["obsBKNOVC_predCLR"] = int((np.isin(y, [MC, OV]) & (p == CL)).sum()); o["obsCLR_predBKNOVC"] = int(((y == CL) & np.isin(p, [MC, OV])).sum())
    o["anyprecip_f1"] = float(2 * tp / max(wy.sum() + wp.sum(), 1)); o["anyprecip_r"] = float(tp / max(wy.sum(), 1))
    o["fog_fa"] = int(((p == FOG) & (y != FOG)).sum())
    return o

def show(s, keys=("acc", "macro_f1", "sky_macro_f1", "clear_f1", "pc_f1", "mc_f1", "ovc_f1", "obsOVC_predCLR", "obsCLR_predOVC", "fog_f1", "anyprecip_f1")):
    return " ".join(f"{k}={s[k]:.3f}" if isinstance(s[k], float) else f"{k}={s[k]}" for k in keys)
