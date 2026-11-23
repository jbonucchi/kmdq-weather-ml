"""Score the chosen precipitation emphasis ONCE on the held-out test year (f00 + archived 12z f01-f48), vs the current
fog-only model; then rescore recent runs (2026-10-20 .. 11-02 12z) with models trained only on data before 2026-10-20 12z."""
exec(open("/workspace/hrrr_ml/experiments/rain_common.py").read())
T.PARAMS["seed"] = 42
GT, yt = G_for(te0, te1)
CUR = dict(mult={FOG: 2}, rounds=142, sc={FOG: 2})
NEW = dict(mult={FOG: 2, LR: 1.5, RA: 2, HR: 2}, rounds=int(113 * 1.15), sc={FOG: 2, LR: 1.5, RA: 3, HR: 3})
res = {}
for name, c in (("current (fog only)", CUR), ("new (fog + precip)", NEW)):
    m = fitw(trva, c["mult"], c["rounds"]); P = boost(m.predict(GT[feats].astype(float)), c["sc"]); p = P.argmax(1)
    res[name] = {"test_f00": summ(ya[te], boost(m.predict(Xa[te]), c["sc"]).argmax(1)), "test_12z_f01_f48": summ(yt, p),
                 "day1_acc": float((p == yt)[GT.fhr.values <= 24].mean()), "day2_acc": float((p == yt)[GT.fhr.values > 24].mean())}
    print(name, json.dumps(res[name], default=float), flush=True)
json.dump(res, open("/workspace/hrrr_ml/experiments/rain_final_test.json", "w"), indent=1, default=float)
# recent runs, out of sample
cut = (t < pd.Timestamp("2026-10-20 12:00")).values
runs = [dt.datetime(2026, 10, 20, 12) + dt.timedelta(days=k) for k in range(14)]
G = pd.concat([fc_feats(r, LE.get(r)) for r in runs], ignore_index=True); y = G.valid_utc.map(lab).map(CLASSES.index).values
out = {}
for name, c in (("current", CUR), ("new", NEW)):
    m = fitw(cut, c["mult"], c["rounds"]); P = boost(m.predict(G[feats].astype(float)), c["sc"]); p = P.argmax(1)
    out[name] = summ(y, p); G[name] = [CLASSES[i] for i in p]
    for k, cl in ((LR, "lr"), (RA, "ra"), (HR, "hr"), (OV, "ov")): G[f"{name}_{cl}pct"] = (P[:, k] * 100).round(0)
G["obs"] = [CLASSES[i] for i in y]
G[["run", "fhr", "valid_utc", "obs", "current", "new"] + [c for c in G if c.endswith("pct")]].to_csv("/workspace/hrrr_ml/experiments/rain_recent_hours.csv", index=False)
json.dump(out, open("/workspace/hrrr_ml/experiments/rain_recent.json", "w"), indent=1, default=float)
print(json.dumps(out, default=float))
