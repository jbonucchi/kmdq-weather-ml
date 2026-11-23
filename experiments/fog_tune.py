import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, "/workspace/hrrr_ml")
import train as T, model_lib as M, features as X
from labels import CLASSES
FOG = CLASSES.index("Fog/mist/haze"); CL = CLASSES.index("Clear")
f, ds = M.load_xy("KMDQ"); feats = X.feature_cols(f.drop(columns=["label","y","raw_idx"]))
Xa, ya = f[feats].astype(float), f.y.values
va0, te0 = pd.Timestamp("2024-10-21 03:00"), pd.Timestamp("2025-10-21 03:00"); te1 = pd.Timestamp("2026-10-21 03:00")
t = f.valid_utc; tr = (t < va0).values; va = ((t >= va0) & (t < te0)).values
def W(y, power, fm):
    w = T.weights(y, power, {}); w = w * np.where(y == FOG, fm, 1.0); return w
import lightgbm as lgb
rows = []
for fm in (1, 1.5, 2, 3, 4, 6):
    dtr = lgb.Dataset(Xa[tr], ya[tr], weight=W(ya[tr], .25, fm)); dva = lgb.Dataset(Xa[va], ya[va], weight=W(ya[va], .25, fm), reference=dtr)
    m = lgb.train(T.PARAMS, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    P = m.predict(Xa[va], num_iteration=m.best_iteration)
    for s in (1, 1.25, 1.5, 2, 2.5, 3):
        Q = P.copy(); Q[:, FOG] *= s; p = Q.argmax(1); mm = M.metrics(ya[va], p); pc = mm["per_class"]
        rows.append(dict(fm=fm, s=s, it=m.best_iteration, acc=mm["accuracy"], mf1=mm["macro_f1"], fog_r=pc["Fog/mist/haze"]["recall"],
                         fog_p=pc["Fog/mist/haze"]["precision"], fog_f1=pc["Fog/mist/haze"]["f1"], clr_p=pc["Clear"]["precision"]))
    print(fm, m.best_iteration, flush=True)
r = pd.DataFrame(rows).round(4); r.to_csv("experiments/fog_val_grid.csv", index=False); print(r.to_string())
