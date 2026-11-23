import os, sys, json, datetime as dt, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, "/workspace/hrrr_ml"); sys.argv = ["x"]
exec(open("experiments/fog_test.py").read().split("out = {")[0])   # shared setup/helpers
an = ds.copy(); an["valid_utc"] = pd.to_datetime(an.valid_utc); lab = an.dropna(subset=["label"]).set_index("valid_utc").label; an = an.set_index("valid_utc")
def fc_feats(run, fc):
    prev = an.loc[[run - dt.timedelta(hours=k) for k in (2, 1, 0) if run - dt.timedelta(hours=k) in an.index]].reset_index()
    prev["fhr"] = 0; seq = pd.concat([prev, fc], ignore_index=True); seq["valid_utc"] = pd.to_datetime(seq.valid_utc)
    g = X.build_features(seq); g["fhr"] = seq.fhr.values; return g[(g.fhr >= 1) & g.valid_utc.isin(lab.index)]
def G_for(lo, hi):
    Fs = []
    for x in sorted(os.listdir(LE.FD)):
        run = dt.datetime.strptime(x[:10], "%Y%m%d%H")
        if lo <= pd.Timestamp(run) and pd.Timestamp(run) + pd.Timedelta(hours=48) <= hi: Fs.append(fc_feats(run, pd.read_parquet(os.path.join(LE.FD, x))))
    G = pd.concat(Fs, ignore_index=True); return G, G.valid_utc.map(lab).map(CLASSES.index).values
GV, yv = G_for(va0, te0); GT, yt = G_for(te0, te1); print("val fcst hours", len(GV), "test fcst hours", len(GT))
tr = (t < va0).values; va = ((t >= va0) & (t < te0)).values
rows = []
for fm, it in ((1, 142), (1.5, 123), (2, 124), (3, 118), (4, 114)):
    m = fitw(tr, fm, it); Pf = m.predict(GV[feats].astype(float)); P0 = m.predict(Xa[va])
    for s in (1, 1.25, 1.5, 2, 2.5, 3):
        Q = Pf.copy(); Q[:, FOG] *= s; Q0 = P0.copy(); Q0[:, FOG] *= s
        a = summ(yv, Q.argmax(1)); b = summ(ya[va], Q0.argmax(1))
        rows.append(dict(fm=fm, s=s, fc_acc=a["acc"], fc_mf1=a["macro_f1"], fc_fog_r=a["fog_recall"], fc_fog_p=a["fog_precision"], fc_fog_f1=a["fog_f1"], fc_clr_p=a["clear_precision"],
                         f00_acc=b["acc"], f00_mf1=b["macro_f1"], f00_fog_r=b["fog_recall"], f00_fog_f1=b["fog_f1"]))
    print(fm, flush=True)
r = pd.DataFrame(rows).round(4); r.to_csv("experiments/fog_val_fcst_grid.csv", index=False); print(r.to_string())
import pickle; pickle.dump((GT, yt), open("/tmp/gt.pkl", "wb"))
