#!/usr/bin/env python3
"""Feature-group comparison (validation-year selection). Usage: sky_feat_test.py GROUPS... (e.g. base A W A,W)"""
import sys
exec(open("/workspace/hrrr_ml/experiments/sky_common.py").read())
res = {}
seeds = [42, 7, 123]
for cfg in sys.argv[1:]:
    groups = None if cfg == "base" else tuple(cfg.split(","))
    f = f00_frame(groups); feats = [c for c in f.columns if c not in ("valid_utc", "label", "y")]
    Xa, ya, t = f[feats].astype(float), f.y.values, f.valid_utc
    tr = (t < va0).values; va = ((t >= va0) & (t < te0)).values
    GV, yv = fc_frames(va0, te0, groups)
    for sd in seeds:
        m = fit(Xa[tr], ya[tr], Xa[va], ya[va], seed=sd)
        sc = T.CLASS_PROB_SCALE
        s0 = summ(ya[va], M.class_boost(m.predict(Xa[va], num_iteration=m.best_iteration), sc).argmax(1))
        s1 = summ(yv, M.class_boost(m.predict(GV[feats].astype(float), num_iteration=m.best_iteration), sc).argmax(1))
        res[f"{cfg}|{sd}"] = {"iters": m.best_iteration, "val_f00": s0, "val_fc": s1, "nfeat": len(feats)}
        print(cfg, sd, m.best_iteration, "\n  f00:", show(s0), "\n  fc :", show(s1), flush=True)
    for k in ("val_f00", "val_fc"):
        mean = {kk: float(np.mean([res[f"{cfg}|{sd}"][k][kk] for sd in seeds])) for kk in res[f"{cfg}|{seeds[0]}"][k]}
        res[f"{cfg}|mean_{k}"] = mean; print(cfg, "MEAN", k, show(mean), flush=True)
out = f"{EXP}/sky_feat_val.json"
old = json.load(open(out)) if os.path.exists(out) else {}
old.update(res); json.dump(old, open(out, "w"), indent=1)
