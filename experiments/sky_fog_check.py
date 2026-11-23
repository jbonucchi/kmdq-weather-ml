#!/usr/bin/env python3
"""The forecast-trained model raises test-year forecast fog false alarms (308 -> 451). Check a smaller fog prediction boost
(x1.5 -> x1.25 / x1.0) on validation-year forecasts (selection), then report the test year for the new eval model."""
exec(open("/workspace/hrrr_ml/experiments/sky_common.py").read())
f = f00_frame(None); f["fhr"] = 0; feats = [c for c in f.columns if c not in ("valid_utc", "label", "y")]
GV, yv = fc_frames(va0, te0, None); GT, yt = fc_frames(te0, te1, None); tr = (f.valid_utc < va0).values
Xtr = pd.concat([f.loc[tr, feats], GT[feats]], ignore_index=True).astype(float); ytr = np.concatenate([f.y.values[tr], yt]); w = T.weights(ytr, 0.25); w[tr.sum():] *= 2
m = lgb.train(T.PARAMS, lgb.Dataset(Xtr, ytr, weight=w), 73); Pv = m.predict(GV[feats].astype(float))
em, emeta = M.load_model("KMDQ", "eval"); Pt = em.predict(GT[emeta["features"]].astype(float))
# current-model val reference (f00-only model trained on train years)
m0 = fit(f.loc[tr, [c for c in feats if c != "fhr"]].astype(float), f.y.values[tr], rounds=132); P0 = m0.predict(GV[[c for c in feats if c != "fhr"]].astype(float))
s = summ(yv, M.class_boost(P0, {k: v for k, v in T.CLASS_PROB_SCALE.items() if k not in ("Partly cloudy", "Mostly cloudy", "Overcast")}).argmax(1)); print("VAL current-config", show(s), "fog_fa", s["fog_fa"], f"fog_r={s['fog_r']:.3f}")
for fs in (1.5, 1.25, 1.0):
    sc = dict(T.CLASS_PROB_SCALE, **{"Fog/mist/haze": fs})
    s = summ(yv, M.class_boost(Pv, sc).argmax(1)); print("VAL new fog x", fs, show(s), "fog_fa", s["fog_fa"], f"fog_r={s['fog_r']:.3f} fog_p={s['fog_p']:.3f}")
    s = summ(yt, M.class_boost(Pt, sc).argmax(1)); print("TEST new fog x", fs, show(s), "fog_fa", s["fog_fa"], f"fog_r={s['fog_r']:.3f} fog_p={s['fog_p']:.3f} precip_r={s['anyprecip_r']:.3f}")
