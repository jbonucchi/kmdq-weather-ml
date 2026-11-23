#!/usr/bin/env python3
"""Recent out-of-sample check (12z runs 2026-10-20 .. 2026-11-14): prod-style models trained only on data valid before
2026-10-20 12Z. Current configuration (f00-only, base features, 151 rounds) vs new (f00 + forecasts x2, A,W,N features,
sky scales, 85 rounds). Writes sky_recent.json and sky_recent_hours.csv."""
exec(open("/workspace/hrrr_ml/experiments/sky_common.py").read())
CUT = pd.Timestamp("2026-10-20 12:00"); NOW = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("h")
SKYSC = dict(T.CLASS_PROB_SCALE, **{"Partly cloudy": 1.25, "Mostly cloudy": 1.5, "Overcast": 1.15})
out = {}; hrs = {}
# current configuration
f = f00_frame(None); feats = [c for c in f.columns if c not in ("valid_utc", "label", "y")]; tr = (f.valid_utc < CUT).values
G, y = fc_frames(CUT, NOW + pd.Timedelta(hours=48), None, tag="recent")   # runs whose 48 h window ends <= now+48h; labels only where observed
m = lgb.train(T.PARAMS, lgb.Dataset(f.loc[tr, feats].astype(float), f.y.values[tr], weight=T.weights(f.y.values[tr], 0.25)), 151)
Pc = M.class_boost(m.predict(G[feats].astype(float)), T.CLASS_PROB_SCALE); out["current"] = summ(y, Pc.argmax(1)); print("current", show(out["current"]))
hrs = G[["run", "fhr", "valid_utc"]].copy(); hrs["obs"] = np.array(CLASSES)[y]; hrs["current"] = np.array(CLASSES)[Pc.argmax(1)]
# new configuration
g = ("A", "W", "N"); f = f00_frame(g); f["fhr"] = 0; feats = [c for c in f.columns if c not in ("valid_utc", "label", "y")]; tr = (f.valid_utc < CUT).values
GA, yA = fc_frames(va0, CUT, g); GA = GA[GA.valid_utc < CUT]; yA = yA[: len(yA)] if False else GA.valid_utc.map(lab).map(C).values.astype(int)
G2, y2 = fc_frames(CUT, NOW + pd.Timedelta(hours=48), g, tag="recent")
Xtr = pd.concat([f.loc[tr, feats], GA[feats]], ignore_index=True).astype(float); ytr = np.concatenate([f.y.values[tr], yA]); w = T.weights(ytr, 0.25); w[tr.sum():] *= 2.0
m2 = lgb.train(T.PARAMS, lgb.Dataset(Xtr, ytr, weight=w), 85)
Pn = M.class_boost(m2.predict(G2[feats].astype(float)), SKYSC); out["new"] = summ(y2, Pn.argmax(1)); print("new    ", show(out["new"]))
assert (G2.valid_utc.values == G.valid_utc.values).all()
hrs["new"] = np.array(CLASSES)[Pn.argmax(1)]; hrs.to_csv(f"{EXP}/sky_recent_hours.csv", index=False)
for since in ("2026-11-06",):
    s = hrs[hrs.valid_utc >= since]
    out[f"since_{since}"] = {k: float((s[k] == s.obs).mean()) for k in ("current", "new")}; out[f"since_{since}"]["n"] = len(s)
    for k in ("current", "new"):
        out[f"since_{since}"][f"{k}_obsOVCBKN_predCLR"] = int((s.obs.isin(["Overcast", "Mostly cloudy"]) & (s[k] == "Clear")).sum())
        out[f"since_{since}"][f"{k}_obsCLR_predOVCBKN"] = int((s.obs.eq("Clear") & s[k].isin(["Overcast", "Mostly cloudy"])).sum())
    print(since, out[f"since_{since}"])
json.dump(out, open(f"{EXP}/sky_recent.json", "w"), indent=1)
