"""Shared setup for the 2026-11-03 precipitation-weighting experiments (same split/methodology as the fog work)."""
import os, sys, json, pickle, datetime as dt, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, "/workspace/hrrr_ml")
import train as T, model_lib as M, features as X, lead_eval as LE
from labels import CLASSES
C = {c: i for i, c in enumerate(CLASSES)}
FOG, LR, RA, HR, TS, CL, OV = (C[k] for k in ("Fog/mist/haze", "Light rain/drizzle", "Rain", "Heavy rain", "Thunderstorm", "Clear", "Overcast"))
PRECIP = [LR, RA, HR]; WET = [LR, RA, HR, TS]
f, ds = M.load_xy("KMDQ"); feats = X.feature_cols(f.drop(columns=["label", "y", "raw_idx"]))
Xa, ya = f[feats].astype(float), f.y.values; t = f.valid_utc
va0, te0, te1 = pd.Timestamp("2024-10-21 03:00"), pd.Timestamp("2025-10-21 03:00"), pd.Timestamp("2026-10-21 03:00")
tr = (t < va0).values; va = ((t >= va0) & (t < te0)).values; trva = (t < te0).values; te = ((t >= te0) & (t <= te1)).values
an = ds.copy(); an["valid_utc"] = pd.to_datetime(an.valid_utc); lab = an.dropna(subset=["label"]).set_index("valid_utc").label; an = an.set_index("valid_utc")

def W(y, mult):
    """class-weight power 0.25 (current) x per-class multipliers dict {class_idx: mult}; fog x2 is included by caller."""
    w = T.weights(y, .25, {}).copy()
    for k, m in mult.items(): w[y == k] *= m
    return w
def fitw(mask, mult, rounds=None, valmask=None):
    d = lgb.Dataset(Xa[mask], ya[mask], weight=W(ya[mask], mult))
    if valmask is None: return lgb.train(T.PARAMS, d, num_boost_round=rounds)
    dv = lgb.Dataset(Xa[valmask], ya[valmask], weight=W(ya[valmask], mult), reference=d)
    return lgb.train(T.PARAMS, d, 3000, valid_sets=[dv], callbacks=[lgb.early_stopping(100, verbose=False)])
def boost(P, sc):
    Q = np.asarray(P, float).copy()
    for k, s in sc.items(): Q[:, k] *= s
    return Q / Q.sum(1, keepdims=True)

def fc_feats(run, fc):
    prev = an.loc[[run - dt.timedelta(hours=k) for k in (2, 1, 0) if run - dt.timedelta(hours=k) in an.index]].reset_index()
    prev["fhr"] = 0; seq = pd.concat([prev, fc], ignore_index=True); seq["valid_utc"] = pd.to_datetime(seq.valid_utc)
    g = X.build_features(seq); g["fhr"] = seq.fhr.values; g = g[(g.fhr >= 1) & g.valid_utc.isin(lab.index)].copy(); g["run"] = run
    return g
def G_for(lo, hi):
    cache = f"/workspace/hrrr_ml/experiments/cache_G_{lo:%Y%m%d}_{hi:%Y%m%d}.pkl"
    if os.path.exists(cache): return pickle.load(open(cache, "rb"))
    Fs = []
    for x in sorted(os.listdir(LE.FD)):
        run = dt.datetime.strptime(x[:10], "%Y%m%d%H")
        if lo <= pd.Timestamp(run) and pd.Timestamp(run) + pd.Timedelta(hours=48) <= hi: Fs.append(fc_feats(run, pd.read_parquet(os.path.join(LE.FD, x))))
    G = pd.concat(Fs, ignore_index=True); y = G.valid_utc.map(lab).map(CLASSES.index).values
    pickle.dump((G, y), open(cache, "wb")); return G, y

def summ(y, p):
    m = M.metrics(y, p); pc = m["per_class"]
    wy = np.isin(y, PRECIP); wp = np.isin(p, PRECIP); tp = (wy & wp).sum()
    out = dict(n=len(y), acc=m["accuracy"], macro_f1=m["macro_f1"])
    for nm, k in (("fog", FOG), ("lr", LR), ("ra", RA), ("hr", HR), ("ts", TS)):
        out[f"{nm}_n"] = pc[CLASSES[k]]["support"]; out[f"{nm}_r"] = pc[CLASSES[k]]["recall"]; out[f"{nm}_p"] = pc[CLASSES[k]]["precision"]; out[f"{nm}_f1"] = pc[CLASSES[k]]["f1"]
    out["precip_mean_f1"] = float(np.mean([pc[CLASSES[k]]["f1"] for k in PRECIP]))
    # "any precipitation" detection: obs in {drizzle, rain, heavy rain} vs forecast in that set
    out["anyprecip_n"] = int(wy.sum()); out["anyprecip_r"] = float(tp / max(wy.sum(), 1)); out["anyprecip_p"] = float(tp / max(wp.sum(), 1))
    out["anyprecip_f1"] = float(2 * tp / max(wy.sum() + wp.sum(), 1))
    out["clear_p"] = pc["Clear"]["precision"]; out["overcast_p"] = pc["Overcast"]["precision"]; out["overcast_r"] = pc["Overcast"]["recall"]
    out["precip_called_overcast"] = int((wy & (p == OV)).sum())
    return out
