"""Seed-averaged validation grid (3 LightGBM seeds) for the precipitation emphasis; reduces single-run noise in rain_tune.py."""
exec(open("/workspace/hrrr_ml/experiments/rain_common.py").read())
import itertools
GV, yv = G_for(va0, te0)
CFGS = {"base": {FOG: 2}, "p1.5": {FOG: 2, LR: 1.5, RA: 1.5, HR: 1.5}, "p2": {FOG: 2, LR: 2, RA: 2, HR: 2}, "p3": {FOG: 2, LR: 3, RA: 3, HR: 3},
        "lr1.5_ra2": {FOG: 2, LR: 1.5, RA: 2, HR: 2}, "lr2_ra3": {FOG: 2, LR: 2, RA: 3, HR: 3}, "p2+ts2": {FOG: 2, LR: 2, RA: 2, HR: 2, TS: 2}}
rows = []
for seed in (42, 7, 2026):
    T.PARAMS["seed"] = seed
    for name, mult in CFGS.items():
        m = fitw(tr, mult, valmask=va); it = m.best_iteration
        Pf = m.predict(GV[feats].astype(float), num_iteration=it); P0 = m.predict(Xa[va], num_iteration=it)
        for sL, sR, sT in itertools.product((1, 1.25, 1.5, 2), (1, 1.5, 2, 3, 4, 5, 6), (1, 1.5)):
            sc = {FOG: 2, LR: sL, RA: sR, HR: sR, TS: sT}
            a = summ(yv, boost(Pf, sc).argmax(1)); b = summ(ya[va], boost(P0, sc).argmax(1))
            rows.append(dict(seed=seed, cfg=name, it=it, sL=sL, sR=sR, sT=sT, **{"fc_" + k: v for k, v in a.items()}, **{"f00_" + k: v for k, v in b.items()}))
        print(seed, name, it, flush=True)
pd.DataFrame(rows).to_csv("experiments/rain_val_grid_seeds.csv", index=False)
