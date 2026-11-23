import os, sys, json, datetime as dt, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, "/workspace/hrrr_ml")
import train as T, model_lib as M, features as X, lead_eval as LE
from labels import CLASSES
FOG = CLASSES.index("Fog/mist/haze")
f, ds = M.load_xy("KMDQ"); feats = X.feature_cols(f.drop(columns=["label","y","raw_idx"]))
Xa, ya = f[feats].astype(float), f.y.values; t = f.valid_utc
va0, te0, te1 = pd.Timestamp("2024-10-21 03:00"), pd.Timestamp("2025-10-21 03:00"), pd.Timestamp("2026-10-21 03:00")
trva = (t < te0).values; te = ((t >= te0) & (t <= te1)).values
def W(y, fm): return T.weights(y, .25, {}) * np.where(y == FOG, fm, 1.0)
def fitw(mask, fm, rounds):
    return lgb.train(T.PARAMS, lgb.Dataset(Xa[mask], ya[mask], weight=W(ya[mask], fm)), num_boost_round=rounds)
CFG = {"current (fog x1)": (1, 163), "fog x2": (2, 142), "fog x3": (3, 135)}
def summ(y, p):
    m = M.metrics(y, p); pc = m["per_class"]
    return dict(n=len(y), acc=m["accuracy"], macro_f1=m["macro_f1"], fog_n=pc["Fog/mist/haze"]["support"], fog_recall=pc["Fog/mist/haze"]["recall"],
                fog_precision=pc["Fog/mist/haze"]["precision"], fog_f1=pc["Fog/mist/haze"]["f1"], clear_precision=pc["Clear"]["precision"],
                fog_called_clear=int(((y == FOG) & (p == 0)).sum()), fog_forecasts=int((p == FOG).sum()))
out = {"test_f00": {}, "test_12z_f01_f48": {}, "recent_runs": {}}
old_eval, old_meta = M.load_model("KMDQ", "eval")
out["test_f00"]["saved current eval model"] = summ(ya[te], M.predict_proba(old_eval, old_meta, f[te]).argmax(1))
models = {}
for k, (fm, r) in CFG.items():
    models[k] = fitw(trva, fm, r); out["test_f00"][k] = summ(ya[te], models[k].predict(Xa[te]).argmax(1)); print(k, out["test_f00"][k], flush=True)
# archived 12z forecasts in test year
an = ds.copy(); an["valid_utc"] = pd.to_datetime(an.valid_utc); lab = an.dropna(subset=["label"]).set_index("valid_utc").label; an = an.set_index("valid_utc")
def fc_feats(run, fc):
    prev = an.loc[[run - dt.timedelta(hours=k) for k in (2, 1, 0) if run - dt.timedelta(hours=k) in an.index]].reset_index()
    prev["fhr"] = 0; seq = pd.concat([prev, fc], ignore_index=True); seq["valid_utc"] = pd.to_datetime(seq.valid_utc)
    g = X.build_features(seq); g["fhr"] = seq.fhr.values; g = g[(g.fhr >= 1) & g.valid_utc.isin(lab.index)]
    return g
def lead_eval(runs, mdl):
    Fs = []
    for run in runs:
        p = os.path.join(LE.FD, f"{run:%Y%m%d%H}.parquet")
        fc = pd.read_parquet(p) if os.path.exists(p) else LE.get(run)
        if fc is None: continue
        g = fc_feats(run, fc); g["run"] = run; Fs.append(g)
    G = pd.concat(Fs, ignore_index=True); y = G.valid_utc.map(lab).map(CLASSES.index).values
    return {k: summ(y, m.predict(G[feats].astype(float)).argmax(1)) for k, m in mdl.items()}, G
runs = sorted(dt.datetime.strptime(x[:10], "%Y%m%d%H") for x in os.listdir(LE.FD))
test_runs = [r for r in runs if te0 <= pd.Timestamp(r) and pd.Timestamp(r) + pd.Timedelta(hours=48) <= te1]
out["test_12z_f01_f48"], _ = lead_eval(test_runs, models); out["test_12z_runs"] = len(test_runs)
print(out["test_12z_f01_f48"], flush=True)
# recent runs Oct 11-29 2026 with models trained only on data before 2026-10-11 12z (out of sample)
cut = (t < pd.Timestamp("2026-10-11 12:00")).values
rm = {k: fitw(cut, fm, r) for k, (fm, r) in CFG.items()}
recent = [dt.datetime(2026, 10, d, 12) for d in range(11, 30)]
out["recent_runs"], G = lead_eval(recent, rm)
y = G.valid_utc.map(lab).map(CLASSES.index).values
per = []
for run, idx in G.groupby("run").groups.items():
    row = {"run": f"{run:%Y-%m-%d %HZ}", "hours": len(idx), "obs_fog": int((y[G.index.get_indexer(idx)] == FOG).sum())}
    for k, m in rm.items():
        p = m.predict(G.loc[idx, feats].astype(float)).argmax(1); yy = y[G.index.get_indexer(idx)]
        row[f"{k} acc"] = round(float((p == yy).mean()), 3); row[f"{k} fog hits"] = int(((p == FOG) & (yy == FOG)).sum()); row[f"{k} fog false alarms"] = int(((p == FOG) & (yy != FOG)).sum())
    per.append(row)
out["recent_per_run"] = per
json.dump(out, open("experiments/fog_test_results.json", "w"), indent=1, default=str)
print(json.dumps(out["recent_runs"], indent=0)); print(pd.DataFrame(per).to_string())
