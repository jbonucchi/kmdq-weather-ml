#!/usr/bin/env python3
"""Held-out test year, scored once for the configuration chosen on validation (sky_final_val.py):
option D (f00 + archived 12z forecasts, forecast rows x2) with feature groups A,W,N and sky probability scales
Partly cloudy x1.25, Mostly cloudy x1.5, Overcast x1.15 (on top of the fog/precip boosts). Train = f00 < 2025-10-21 03Z +
validation-year forecasts; rounds = 85 (validation early-stop mean 74 x 1.15). Compared with the live eval model (f00-only)
and with D on base features (attribution). Writes sky_final_test.json."""
exec(open("/workspace/hrrr_ml/experiments/sky_common.py").read())
SKYSC = {"Partly cloudy": 1.25, "Mostly cloudy": 1.5, "Overcast": 1.15}; ROUNDS = 85; FWT = 2.0
out = {}
def score(name, P, y, G=None):
    p = P.argmax(1); s = summ(y, p)
    if G is not None:
        s["day1_acc"] = float((p == y)[G.fhr.values <= 24].mean()); s["day2_acc"] = float((p == y)[G.fhr.values > 24].mean())
    out[name] = s; print(name, show(s), flush=True); return s
# live eval model (f00-only, base features)
em, emeta = M.load_model("KMDQ", "eval")
f = f00_frame(None); t = f.valid_utc; te = ((t >= te0) & (t <= te1)).values
GT, yt = fc_frames(te0, te1, None)
score("current|f00", M.predict_proba(em, emeta, f[te]), f.y.values[te]); score("current|fc", M.predict_proba(em, emeta, GT), yt, GT)
for cfg in ("base", "A,W,N"):
    groups = None if cfg == "base" else tuple(cfg.split(","))
    f = f00_frame(groups); f["fhr"] = 0; t = f.valid_utc; trva = (t < te0).values; te = ((t >= te0) & (t <= te1)).values
    GV, yv = fc_frames(va0, te0, groups); GT, yt = fc_frames(te0, te1, groups)
    feats = [c for c in f.columns if c not in ("valid_utc", "label", "y")]
    Xtr = pd.concat([f.loc[trva, feats], GV[feats]], ignore_index=True).astype(float); ytr = np.concatenate([f.y.values[trva], yv])
    w = T.weights(ytr, 0.25); w[trva.sum():] *= FWT
    for sd in (42, 7):
        m = lgb.train(dict(T.PARAMS, seed=sd), lgb.Dataset(Xtr, ytr, weight=w), ROUNDS)
        P0 = m.predict(f.loc[te, feats].astype(float)); P1 = m.predict(GT[feats].astype(float))
        for sn, sc in (("noskyscale", T.CLASS_PROB_SCALE), ("skyscale", dict(T.CLASS_PROB_SCALE, **SKYSC))):
            score(f"D-{cfg}|{sn}|s{sd}|f00", M.class_boost(P0, sc), f.y.values[te]); score(f"D-{cfg}|{sn}|s{sd}|fc", M.class_boost(P1, sc), yt, GT)
        if cfg == "A,W,N" and sd == 42:
            imp = pd.Series(m.feature_importance("gain"), index=feats).sort_values(ascending=False); out["top_gain_AWN"] = (imp / imp.sum()).head(25).round(4).to_dict()
            pc = M.class_boost(P1, dict(T.CLASS_PROB_SCALE, **SKYSC)).argmax(1)
            out["cm_fc_AWN_skyscale"] = pd.crosstab(pd.Categorical(np.array(CLASSES)[yt], CLASSES), pd.Categorical(np.array(CLASSES)[pc], CLASSES), dropna=False).loc[CLASSES[:6], CLASSES[:6]].to_dict()
json.dump(out, open(f"{EXP}/sky_final_test.json", "w"), indent=1)
