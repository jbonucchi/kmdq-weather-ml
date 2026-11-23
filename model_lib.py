"""Shared model utilities: dataset assembly, baselines, metrics, load/predict."""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from labels import CLASSES
import features as X
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support

def load_xy(stn):
    ds = pd.read_parquet(os.path.join(DATA, f"dataset_{stn}.parquet")).sort_values("valid_utc").reset_index(drop=True)
    f = X.build_features(ds)                       # tendencies computed on the continuous HRRR series
    f["label"] = ds["label"].values; f["raw_idx"] = ds.index
    f = f[f.label.isin(CLASSES)].reset_index(drop=True)
    f["y"] = f.label.map({c: i for i, c in enumerate(CLASSES)}).astype(int)
    f["fhr"] = 0                                    # analysis rows (forecast-hour feature, 2026-11-14)
    return f, ds

FCST_DIR = os.path.join(DATA, "fcst_12z")
def load_fcst_xy(ds, lo=None, hi=None):
    """Labeled 12z forecast hours (f01-f48) from the archive data/fcst_12z/<run>.parquet (written by lead_eval.py and,
    since 2026-11-14, by predict_latest.py), as model features with fhr. Only runs whose whole 48 h window is inside
    [lo, hi] (valid times) are used, so a run never straddles a split boundary. ds = merged dataset (labels + f00 analyses)."""
    import datetime as dt
    ds = ds.copy(); ds["valid_utc"] = pd.to_datetime(ds.valid_utc)
    lab = ds.dropna(subset=["label"]); lab = lab[lab.label.isin(CLASSES)].set_index("valid_utc").label
    an = ds.set_index("valid_utc"); out = []
    if not os.path.isdir(FCST_DIR): return None
    for x in sorted(os.listdir(FCST_DIR)):
        if not x.endswith(".parquet"): continue
        run = pd.Timestamp(dt.datetime.strptime(x[:10], "%Y%m%d%H"))
        if (lo is not None and run < lo) or (hi is not None and run + pd.Timedelta(hours=48) > hi): continue
        fc = pd.read_parquet(os.path.join(FCST_DIR, x))
        prev = an.loc[[run - pd.Timedelta(hours=k) for k in (2, 1, 0) if run - pd.Timedelta(hours=k) in an.index]].reset_index()
        prev["fhr"] = 0; seq = pd.concat([prev, fc], ignore_index=True); seq["valid_utc"] = pd.to_datetime(seq.valid_utc)
        g = X.build_features(seq); g["fhr"] = seq.fhr.values; g = g[(g.fhr >= 1) & g.valid_utc.isin(lab.index)].copy()
        g["run"] = run; out.append(g)
    if not out: return None
    G = pd.concat(out, ignore_index=True); G["label"] = G.valid_utc.map(lab).values
    G["y"] = G.label.map({c: i for i, c in enumerate(CLASSES)}).astype(int)
    return G

def rule_baseline(raw):
    """Rule-based weather type from /workspace/hrrr_point.py (wtype), mapped to the ML classes."""
    out = []
    for _, v in raw.iterrows():
        pr = (v.prate if pd.notna(v.prate) else 0) * 141.732   # in/hr
        flags = {k: (v[k] or 0) > 0.5 for k in ("crain", "csnow", "cfrzr", "cicep")}
        refc = v.refc if pd.notna(v.refc) else -10; cape = v.sbcape if pd.notna(v.sbcape) else 0
        if pr > 0.001 or any(flags.values()):
            if flags["cfrzr"] or flags["cicep"] or flags["csnow"]: out.append("Snow/ice"); continue
            if refc >= 40 and cape >= 100: out.append("Thunderstorm"); continue
            if pr >= 0.3: out.append("Heavy rain"); continue
            out.append("Light rain/drizzle" if pr < 0.01 and not flags["crain"] else "Rain"); continue
        vis_mi = v.vis / 1609.34 if pd.notna(v.vis) else 10
        if vis_mi < 3: out.append("Fog/mist/haze"); continue
        c = v.tcdc if pd.notna(v.tcdc) else 0
        out.append("Clear" if c < 37.5 else "Partly cloudy" if c < 62.5 else "Mostly cloudy" if c < 87.5 else "Overcast")
    return np.array([CLASSES.index(o) for o in out])

def metrics(y, p, name=""):
    labs = list(range(len(CLASSES)))
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=labs, zero_division=0)
    present = sup > 0
    return {"name": name, "n": int(len(y)), "accuracy": float(accuracy_score(y, p)),
            "macro_f1": float(np.mean(f1[present])), "weighted_f1": float(f1_score(y, p, labels=labs, average="weighted", zero_division=0)),
            "per_class": {CLASSES[i]: {"precision": float(pr[i]), "recall": float(rc[i]), "f1": float(f1[i]), "support": int(sup[i])} for i in labs},
            "confusion": confusion_matrix(y, p, labels=labs).tolist()}

def load_model(stn=STATION, kind="prod"):
    import lightgbm as lgb
    m = lgb.Booster(model_file=os.path.join(MODELS, f"model_{stn}_{kind}.txt"))
    meta = json.load(open(os.path.join(MODELS, f"model_{stn}_{kind}.json")))
    return m, meta

def class_boost(P, scales):
    """Multiply class probabilities by {class name: factor} and renormalize (decision bias toward rare classes, tuned on validation)."""
    P = np.asarray(P, dtype=float).copy()
    if scales:
        for c, s in scales.items(): P[:, CLASSES.index(c)] *= s
        P /= P.sum(1, keepdims=True)
    return P

def fog_boost(P, scale):
    """Back-compat: fog-only boost (2026-10-30 model)."""
    return class_boost(P, {"Fog/mist/haze": scale} if scale and scale != 1 else {})

def predict_proba(model, meta, feats):
    P = model.predict(feats[meta["features"]].astype(float), num_iteration=meta.get("best_iteration"))
    P = np.asarray(P).reshape(len(feats), len(CLASSES))
    if "class_prob_scale" in meta: return class_boost(P, meta["class_prob_scale"])
    return fog_boost(P, meta.get("fog_prob_scale", 1.0))
