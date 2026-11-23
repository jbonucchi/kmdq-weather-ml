#!/usr/bin/env python3
"""2026-11-14 sky-class diagnosis: confusion among Clear/Partly/Mostly/Overcast for the live eval model
(test-year f00 and archived 12z f01-f48) and recent operational forecasts. Writes sky_diag_20261114.json."""
import os, sys, json, numpy as np, pandas as pd
exec(open("/workspace/hrrr_ml/experiments/rain_common.py").read())
SKY = ["Clear", "Partly cloudy", "Mostly cloudy", "Overcast"]
def sky_cm(obs, pred):
    o = pd.Categorical(obs, CLASSES); p = pd.Categorical(pred, CLASSES)
    return pd.crosstab(o, p, dropna=False).loc[SKY + ["Fog/mist/haze", "Light rain/drizzle"], SKY + ["Fog/mist/haze", "Light rain/drizzle"]]
out = {}
em, emeta = M.load_model("KMDQ", "eval")
P = M.predict_proba(em, emeta, pd.DataFrame(Xa[te], columns=feats)); p = P.argmax(1)
o = np.array(CLASSES)[ya[te]]; pr = np.array(CLASSES)[p]
print("== f00 test year"); cm = sky_cm(o, pr); print(cm); out["f00_cm"] = cm.to_dict()
GT, yt = G_for(te0, te1); P2 = M.predict_proba(em, emeta, GT[feats]); p2 = P2.argmax(1)
o2 = np.array(CLASSES)[yt]; pr2 = np.array(CLASSES)[p2]
print("== 12z f01-48 test year"); cm2 = sky_cm(o2, pr2); print(cm2); out["fc_cm"] = cm2.to_dict()
for nm, oo, pp in (("f00", o, pr), ("fc", o2, pr2)):
    m = M.metrics(np.array([CLASSES.index(x) for x in oo]), np.array([CLASSES.index(x) for x in pp]))
    out[nm + "_sky_f1"] = {c: round(m["per_class"][c]["f1"], 3) for c in SKY}; print(nm, out[nm + "_sky_f1"], "acc", round(m["accuracy"], 3))
# HRRR state on clear<->overcast confusion (f00 test)
raw = ds.loc[f.raw_idx[te]].reset_index(drop=True); F0 = f[te].reset_index(drop=True)
def desc(mask, name):
    r = raw[mask]; g = F0[mask]
    d = {"n": int(mask.sum()), "tcdc_med": float(r.tcdc.median()), "lcdc_med": float(r.lcdc.median()), "mcdc_med": float(r.mcdc.median()), "hcdc_med": float(r.hcdc.median()),
         "frac_hrrr_ceil_lt_3658m": float((g.ceil_agl < 3658).mean()), "frac_lcdc_gt50": float((r.lcdc > 50).mean()), "frac_tcdc_lt20": float((r.tcdc < 20).mean())}
    print(name, d); return d
cases = {"obsOVC_predOVC": (o == "Overcast") & (pr == "Overcast"), "obsOVC_predCLR": (o == "Overcast") & (pr == "Clear"),
         "obsCLR_predCLR": (o == "Clear") & (pr == "Clear"), "obsCLR_predOVC": (o == "Clear") & (pr == "Overcast"),
         "obsBKN_predCLR": (o == "Mostly cloudy") & (pr == "Clear")}
out["hrrr_by_case_f00"] = {k: desc(v, k) for k, v in cases.items()}
# obs ceiling-height of OVC/BKN hours called clear (low stratus HRRR missed vs higher deck)
lb = pd.read_parquet("/workspace/hrrr_ml/data/labels_KMDQ.parquet").set_index("valid_utc")
t_te = F0.valid_utc
for k in ("obsOVC_predCLR", "obsBKN_predCLR", "obsOVC_predOVC"):
    hh = lb.reindex(t_te[cases[k]].values).obs_skyl1
    out[f"obs_base_ft_{k}"] = hh.describe().round(0).to_dict(); print(k, "obs lowest base ft", hh.describe().round(0).to_dict())
# Clear obs: how often HRRR had high cloud (above 12k) -> label CLR by ceilometer limit
c = cases["obsCLR_predCLR"] | cases["obsCLR_predOVC"]
print("obs Clear hrs w/ HRRR hcdc>50 & lcdc<20:", float(((raw.hcdc > 50) & (raw.lcdc < 20))[o == "Clear"].mean()))
# recent operational
v = pd.read_parquet("/workspace/hrrr_ml/logs/verification.parquet"); v = v[v.obs.notna()]
v["valid_dt"] = pd.to_datetime(v.valid_dt)
for nm, lo in (("ops_since_oct20", "2026-10-20"), ("ops_since_nov6", "2026-11-06")):
    s = v[v.valid_dt >= lo]; cm3 = sky_cm(s.obs.values, s.predicted_type.values); print("==", nm, len(s)); print(cm3); out[nm + "_cm"] = cm3.to_dict()
json.dump(out, open("/workspace/hrrr_ml/experiments/sky_diag_20261114.json", "w"), indent=1, default=str)
