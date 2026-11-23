#!/usr/bin/env python3
"""Option C: prediction-time probability scaling among sky classes (Partly/Mostly cloudy/Overcast vs Clear), on top of the
existing fog/precip boosts. Tuned on validation-year 12z forecasts with the current (base) features. Writes sky_scale_val.csv."""
import itertools
exec(open("/workspace/hrrr_ml/experiments/sky_common.py").read())
f = f00_frame(None); feats = [c for c in f.columns if c not in ("valid_utc", "label", "y")]
Xa, ya, t = f[feats].astype(float), f.y.values, f.valid_utc; tr = (t < va0).values; va = ((t >= va0) & (t < te0)).values
GV, yv = fc_frames(va0, te0, None)
rows = []
for sd in (42, 7):
    m = fit(Xa[tr], ya[tr], Xa[va], ya[va], seed=sd)
    Pv = m.predict(GV[feats].astype(float), num_iteration=m.best_iteration); P0 = m.predict(Xa[va], num_iteration=m.best_iteration)
    for spc, smc, sov in itertools.product((0.8, 1.0, 1.25, 1.5), (0.8, 1.0, 1.25, 1.5, 2.0), (0.7, 0.85, 1.0, 1.15, 1.3)):
        sc = dict(T.CLASS_PROB_SCALE, **{"Partly cloudy": spc, "Mostly cloudy": smc, "Overcast": sov})
        s = summ(yv, M.class_boost(Pv, sc).argmax(1)); s0 = summ(ya[va], M.class_boost(P0, sc).argmax(1))
        rows.append({"seed": sd, "pc": spc, "mc": smc, "ov": sov, **{f"fc_{k}": v for k, v in s.items()}, **{f"f00_{k}": v for k, v in s0.items()}})
d = pd.DataFrame(rows); d.to_csv(f"{EXP}/sky_scale_val.csv", index=False)
g = d.groupby(["pc", "mc", "ov"]).mean(numeric_only=True).reset_index()
b = g[(g.pc == 1) & (g.mc == 1) & (g.ov == 1)].iloc[0]
ok = g[(g.fc_acc >= b.fc_acc - 0.005) & (g.f00_acc >= b.f00_acc - 0.01)].sort_values("fc_sky_macro_f1", ascending=False)
cols = ["pc", "mc", "ov", "fc_acc", "fc_sky_macro_f1", "fc_clear_f1", "fc_mc_f1", "fc_ovc_f1", "fc_obsOVC_predCLR", "fc_obsCLR_predOVC", "f00_acc", "f00_sky_macro_f1"]
print("baseline\n", b[cols].to_string()); print(ok[cols].head(12).to_string())
