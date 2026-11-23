#!/usr/bin/env python3
"""Validation-year selection for option D (train on f00 + archived 12z forecasts, forecast rows x2 weight) with feature
groups base / A,W / A,W,N, plus a sky-class probability-scale grid on top. 3 seeds. Writes sky_final_val.json/.csv."""
import itertools
exec(open("/workspace/hrrr_ml/experiments/sky_common.py").read())
FWT = 2.0; SEEDS = (42, 7, 123); res = {}; rows = []
for cfg in ("base", "A,W", "A,W,N"):
    groups = None if cfg == "base" else tuple(cfg.split(","))
    f = f00_frame(groups); f["fhr"] = 0
    GV, yv = fc_frames(va0, te0, groups); GT, yt = fc_frames(te0, te1, groups)
    feats = [c for c in f.columns if c not in ("valid_utc", "label", "y")]
    t = f.valid_utc; tr = (t < va0).values
    Xtr = pd.concat([f.loc[tr, feats], GT[feats]], ignore_index=True).astype(float); ytr = np.concatenate([f.y.values[tr], yt])
    w = T.weights(ytr, 0.25); w[tr.sum():] *= FWT
    for sd in SEEDS:
        d = lgb.Dataset(Xtr, ytr, weight=w); dv = lgb.Dataset(GV[feats].astype(float), yv, weight=T.weights(yv, 0.25), reference=d)
        m = lgb.train(dict(T.PARAMS, seed=sd), d, 3000, valid_sets=[dv], callbacks=[lgb.early_stopping(100, verbose=False)])
        P = m.predict(GV[feats].astype(float), num_iteration=m.best_iteration)
        for spc, smc, sov in itertools.product((1.0, 1.25), (1.0, 1.25, 1.5), (0.85, 1.0, 1.15)):
            sc = dict(T.CLASS_PROB_SCALE, **{"Partly cloudy": spc, "Mostly cloudy": smc, "Overcast": sov})
            p = M.class_boost(P, sc).argmax(1); s = summ(yv, p)
            rows.append({"cfg": cfg, "seed": sd, "iters": m.best_iteration, "pc": spc, "mc": smc, "ov": sov, **s})
        print(cfg, sd, m.best_iteration, flush=True)
D = pd.DataFrame(rows); D.to_csv(f"{EXP}/sky_final_val.csv", index=False)
g = D.groupby(["cfg", "pc", "mc", "ov"]).mean(numeric_only=True).reset_index()
cols = ["cfg", "pc", "mc", "ov", "iters", "acc", "macro_f1", "sky_macro_f1", "clear_f1", "pc_f1", "mc_f1", "ovc_f1", "obsOVC_predCLR", "obsCLR_predOVC", "fog_f1", "anyprecip_f1"]
pd.set_option("display.width", 250)
print(g[(g.pc == 1) & (g.mc == 1) & (g.ov == 1)][cols].to_string())
print(g.sort_values("sky_macro_f1", ascending=False)[cols].head(15).to_string())
