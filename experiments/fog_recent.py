exec(open("experiments/fog_test.py").read().split("out = {")[0])
an = ds.copy(); an["valid_utc"] = pd.to_datetime(an.valid_utc); lab = an.dropna(subset=["label"]).set_index("valid_utc").label; an = an.set_index("valid_utc")
def boost(P, s): Q = P.copy(); Q[:, FOG] *= s; return Q / Q.sum(1, keepdims=True)
Fs = []
for d in range(11, 30):
    run = dt.datetime(2026, 10, d, 12); fc = LE.get(run)
    prev = an.loc[[run - dt.timedelta(hours=k) for k in (2, 1, 0)]].reset_index(); prev["fhr"] = 0
    seq = pd.concat([prev, fc], ignore_index=True); seq["valid_utc"] = pd.to_datetime(seq.valid_utc)
    g = X.build_features(seq); g["fhr"] = seq.fhr.values; g = g[(g.fhr >= 1) & g.valid_utc.isin(lab.index)]; g["run"] = run; Fs.append(g)
G = pd.concat(Fs, ignore_index=True); y = G.valid_utc.map(lab).map(CLASSES.index).values
v = pd.read_parquet("logs/verification.parquet"); v = v[v.run_utc <= "2026-10-29 12Z"]
yl = v.obs.map(CLASSES.index).values; pl = v.predicted_type.map(CLASSES.index).values
print("AS LOGGED (deployed prod model):", summ(yl, pl), "rule acc", (v.rule_type == v.obs).mean())
cut = (t < pd.Timestamp("2026-10-11 12:00")).values
rows = []
for name, fm, r, s in (("current", 1, 163, 1), ("new", 2, 142, 2)):
    m = fitw(cut, fm, r); P = boost(m.predict(G[feats].astype(float)), s); p = P.argmax(1)
    print(name, summ(y, p)); G[name] = [CLASSES[i] for i in p]; G[name + "_fogpct"] = (P[:, FOG] * 100).round(0)
G["obs"] = [CLASSES[i] for i in y]
fogs = G[(G.obs == "Fog/mist/haze") | (G.current == "Fog/mist/haze") | (G.new == "Fog/mist/haze")]
print(fogs[["run", "fhr", "valid_utc", "obs", "current", "current_fogpct", "new", "new_fogpct"]].to_string())
for run, g in G.groupby("run"):
    print(f"{run:%m-%d}", len(g), "fog obs", (g.obs == "Fog/mist/haze").sum(), "acc cur %.3f new %.3f" % ((g.current == g.obs).mean(), (g.new == g.obs).mean()),
          "fog hits cur", ((g.current == g.obs) & (g.obs == "Fog/mist/haze")).sum(), "new", ((g.new == g.obs) & (g.obs == "Fog/mist/haze")).sum(),
          "fog FA cur", ((g.current == "Fog/mist/haze") & (g.obs != "Fog/mist/haze")).sum(), "new", ((g.new == "Fog/mist/haze") & (g.obs != "Fog/mist/haze")).sum())
