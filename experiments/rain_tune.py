"""Tune precipitation emphasis on the VALIDATION year (2024-10-21 -> 2025-10-21), fog x2 weight + x2 boost kept.
Models trained on train years only (early stopping on val); scored on val f00 and archived 12z f01-f48 val forecasts."""
exec(open("/workspace/hrrr_ml/experiments/rain_common.py").read())
import itertools, time
GV, yv = G_for(va0, te0)
CFGS = {"base": {FOG: 2}, "p1.5": {FOG: 2, LR: 1.5, RA: 1.5, HR: 1.5}, "p2": {FOG: 2, LR: 2, RA: 2, HR: 2}, "p3": {FOG: 2, LR: 3, RA: 3, HR: 3},
        "p4": {FOG: 2, LR: 4, RA: 4, HR: 4}, "p2+ts2": {FOG: 2, LR: 2, RA: 2, HR: 2, TS: 2}, "lr2_ra3": {FOG: 2, LR: 2, RA: 3, HR: 3}}
rows = []
for name, mult in CFGS.items():
    t0 = time.time(); m = fitw(tr, mult, valmask=va); it = m.best_iteration
    Pf = m.predict(GV[feats].astype(float), num_iteration=it); P0 = m.predict(Xa[va], num_iteration=it)
    for sL, sR, sT in itertools.product((1, 1.25, 1.5, 2, 3), (1, 1.5, 2, 3, 4), (1, 1.5)):
        sc = {FOG: 2, LR: sL, RA: sR, HR: sR, TS: sT}
        a = summ(yv, boost(Pf, sc).argmax(1)); b = summ(ya[va], boost(P0, sc).argmax(1))
        rows.append(dict(cfg=name, it=it, sL=sL, sR=sR, sT=sT, **{"fc_" + k: v for k, v in a.items()}, **{"f00_" + k: v for k, v in b.items()}))
    print(name, it, round(time.time() - t0), flush=True)
r = pd.DataFrame(rows); r.to_csv("experiments/rain_val_grid.csv", index=False)
