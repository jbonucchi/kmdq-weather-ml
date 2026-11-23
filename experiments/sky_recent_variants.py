#!/usr/bin/env python3
"""Recent-period (12z runs 2026-10-20..11-14) attribution: D with base features, with/without sky scales; D-A,W,N without scales.
Same cutoff as sky_recent.py. Appends to sky_recent.json."""
exec(open("/workspace/hrrr_ml/experiments/sky_common.py").read())
CUT = pd.Timestamp("2026-10-20 12:00"); NOW = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("h")
SK = dict(T.CLASS_PROB_SCALE, **{"Partly cloudy": 1.25, "Mostly cloudy": 1.5, "Overcast": 1.15})
out = json.load(open(f"{EXP}/sky_recent.json")); hrs = pd.read_csv(f"{EXP}/sky_recent_hours.csv")
for cfg in ("base", "A,W,N"):
    g = None if cfg == "base" else tuple(cfg.split(","))
    f = f00_frame(g); f["fhr"] = 0; feats = [c for c in f.columns if c not in ("valid_utc", "label", "y")]; tr = (f.valid_utc < CUT).values
    GA, _ = fc_frames(va0, CUT, g); GA = GA[GA.valid_utc < CUT]; yA = GA.valid_utc.map(lab).map(C).values.astype(int)
    G2, y2 = fc_frames(CUT, NOW + pd.Timedelta(hours=48), g, tag="recent")
    Xtr = pd.concat([f.loc[tr, feats], GA[feats]], ignore_index=True).astype(float); ytr = np.concatenate([f.y.values[tr], yA]); w = T.weights(ytr, 0.25); w[tr.sum():] *= 2.0
    P = lgb.train(T.PARAMS, lgb.Dataset(Xtr, ytr, weight=w), 85).predict(G2[feats].astype(float))
    for sn, sc in (("nosky", T.CLASS_PROB_SCALE), ("sky", SK)):
        if cfg == "A,W,N" and sn == "sky": continue
        p = M.class_boost(P, sc).argmax(1); k = f"D-{cfg}-{sn}"; out[k] = summ(y2, p); hrs[k] = np.array(CLASSES)[p]
        s = hrs[pd.to_datetime(hrs.valid_utc) >= "2026-11-06"]
        out[k]["acc_since_nov6"] = float((s[k] == s.obs).mean()); print(k, show(out[k]), "since Nov6", round(out[k]["acc_since_nov6"], 3), flush=True)
json.dump(out, open(f"{EXP}/sky_recent.json", "w"), indent=1); hrs.to_csv(f"{EXP}/sky_recent_hours.csv", index=False)
