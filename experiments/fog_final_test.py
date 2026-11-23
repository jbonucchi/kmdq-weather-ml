import pickle
exec(open("experiments/fog_test.py").read().split("out = {")[0])
GT, yt = pickle.load(open("/tmp/gt.pkl", "rb"))
def boost(P, s): Q = P.copy(); Q[:, FOG] *= s; return Q / Q.sum(1, keepdims=True)
res = {}
for name, fm, r, s in (("current", 1, 163, 1), ("new fog x2 + boost x2", 2, 142, 2), ("weight x2 only", 2, 142, 1)):
    m = fitw(trva, fm, r)
    res[name] = {"test_f00": summ(ya[te], boost(m.predict(Xa[te]), s).argmax(1)), "test_12z_f01_f48": summ(yt, boost(m.predict(GT[feats].astype(float)), s).argmax(1))}
    P = boost(m.predict(GT[feats].astype(float)), s); res[name]["day1"] = summ(yt[GT.fhr.values <= 24], P[GT.fhr.values <= 24].argmax(1))["acc"]; res[name]["day2"] = summ(yt[GT.fhr.values > 24], P[GT.fhr.values > 24].argmax(1))["acc"]
    print(name, json.dumps(res[name], default=float), flush=True)
json.dump(res, open("experiments/fog_final_test.json", "w"), indent=1, default=float)
