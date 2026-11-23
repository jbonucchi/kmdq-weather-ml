"""Check: class-weight power 0 vs 0.25 with the chosen precip/fog emphasis, on validation (train.py's acc+macroF1 search picked 0.0)."""
exec(open("/workspace/hrrr_ml/experiments/rain_common.py").read())
GV, yv = G_for(va0, te0)
mult = {FOG: 2, LR: 1.5, RA: 2, HR: 2}; sc = {FOG: 2, LR: 1.5, RA: 3, HR: 3}
for pw in (0.0, 0.25):
    w = lambda y: T.weights(y, pw, {"Fog/mist/haze": 2, "Light rain/drizzle": 1.5, "Rain": 2, "Heavy rain": 2})
    d = lgb.Dataset(Xa[tr], ya[tr], weight=w(ya[tr])); dv = lgb.Dataset(Xa[va], ya[va], weight=w(ya[va]), reference=d)
    m = lgb.train(T.PARAMS, d, 3000, valid_sets=[dv], callbacks=[lgb.early_stopping(100, verbose=False)])
    for nm, scc in (("boost", sc), ("noboost", {})):
        a = summ(yv, boost(m.predict(GV[feats].astype(float), num_iteration=m.best_iteration), scc).argmax(1))
        b = summ(ya[va], boost(m.predict(Xa[va], num_iteration=m.best_iteration), scc).argmax(1))
        print(pw, nm, m.best_iteration, {k: round(a[k], 3) for k in ("acc", "macro_f1", "anyprecip_r", "anyprecip_p", "anyprecip_f1", "precip_mean_f1", "ra_r", "fog_r")},
              "f00", {k: round(b[k], 3) for k in ("acc", "macro_f1", "anyprecip_f1", "fog_r")}, flush=True)
