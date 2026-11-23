#!/usr/bin/env python3
"""Option D: add archived 12z forecast hours (f01-f48, with fhr as a feature; f00 rows get fhr=0) to the training set,
so the model learns how HRRR forecast cloud relates to observed sky at each lead.
Selection (validation-year forecasts): train = f00 < 2024-10-21 + test-year forecasts (the only other archived forecast year;
hyper-parameter choice only). Final check (test-year forecasts, scored once): train = f00 < 2025-10-21 + validation-year forecasts.
Usage: sky_fcst_train.py GROUPS (base | A,W | N ...)"""
import sys
exec(open("/workspace/hrrr_ml/experiments/sky_common.py").read())
cfg = sys.argv[1]; groups = None if cfg == "base" else tuple(cfg.split(","))
mode = sys.argv[2] if len(sys.argv) > 2 else "val"
f = f00_frame(groups); f["fhr"] = 0
GV, yv = fc_frames(va0, te0, groups); GT, yt = fc_frames(te0, te1, groups)
feats = [c for c in f.columns if c not in ("valid_utc", "label", "y")]
t = f.valid_utc; tr = (t < va0).values; va = ((t >= va0) & (t < te0)).values; trva = (t < te0).values; te = ((t >= te0) & (t <= te1)).values
Xf, yf = f[feats].astype(float), f.y.values
res = {}
def run(name, train_f00, Gtr, ytr_fc, Gev, yev, fwt, rounds=None, seed=42, fc_only=False):
    Xtr = pd.concat(([] if fc_only else [Xf[train_f00]]) + [Gtr[feats].astype(float)], ignore_index=True)
    ytr = np.concatenate(([] if fc_only else [yf[train_f00]]) + [ytr_fc]).astype(int)
    w = T.weights(ytr, 0.25)
    if not fc_only: w[train_f00.sum():] *= fwt
    p = dict(T.PARAMS, seed=seed); d = lgb.Dataset(Xtr, ytr, weight=w)
    if rounds is None:
        dv = lgb.Dataset(Gev[feats].astype(float), yev, weight=T.weights(yev, 0.25), reference=d)
        m = lgb.train(p, d, 3000, valid_sets=[dv], callbacks=[lgb.early_stopping(100, verbose=False)]); it = m.best_iteration
    else: m = lgb.train(p, d, rounds); it = rounds
    P = m.predict(Gev[feats].astype(float), num_iteration=it if rounds is None else None)
    s = summ(yev, M.class_boost(P, T.CLASS_PROB_SCALE).argmax(1)); s["day1_acc"] = float((M.class_boost(P, T.CLASS_PROB_SCALE).argmax(1) == yev)[Gev.fhr.values <= 24].mean())
    s["day2_acc"] = float((M.class_boost(P, T.CLASS_PROB_SCALE).argmax(1) == yev)[Gev.fhr.values > 24].mean())
    res[name] = {"iters": it, **s}; print(name, it, show(s), f"d1={s['day1_acc']:.3f} d2={s['day2_acc']:.3f}", flush=True)
    return m, it, P
if mode == "val":
    for fwt in (1.0, 2.0):
        for sd in (42, 7):
            run(f"{cfg}|f00+fc w{fwt}|s{sd}", tr, GT, yt, GV, yv, fwt, seed=sd)
    for sd in (42, 7): run(f"{cfg}|fc only|s{sd}", tr, GT, yt, GV, yv, 1.0, seed=sd, fc_only=True)
json.dump(res, open(f"{EXP}/sky_fcst_train_{cfg.replace(',', '')}_{mode}.json", "w"), indent=1)
