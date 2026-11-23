#!/usr/bin/env python3
"""Compare a modest fog-weight reduction vs the live fog x2 + precip model (2026-11-11).

Does not overwrite models/ or reports/. Writes experiments/fog_reduce_20261111.json
and experiments/fog_reduce_nov9_10_hours.csv.
"""
import os, sys, json, datetime as dt, numpy as np, pandas as pd
from zoneinfo import ZoneInfo
sys.path.insert(0, "/workspace/hrrr_ml")
exec(open("/workspace/hrrr_ml/experiments/rain_common.py").read())
import lightgbm as lgb
CT = ZoneInfo("America/Chicago")
OUT = "/workspace/hrrr_ml/experiments/fog_reduce_20261111.json"
HOURS_OUT = "/workspace/hrrr_ml/experiments/fog_reduce_nov9_10_hours.csv"

PRECIP_MULT = {"Fog/mist/haze": 2.0, "Light rain/drizzle": 1.5, "Rain": 2.0, "Heavy rain": 2.0}
PRECIP_SCALE = {"Fog/mist/haze": 2.0, "Light rain/drizzle": 1.5, "Rain": 3.0, "Heavy rain": 3.0}

def with_fog(base, fog):
    d = dict(base); d["Fog/mist/haze"] = fog; return d

def train_eval(fog_w, seed_it=None):
    """Train an eval model (train+val) with early-stopped rounds unless seed_it is given."""
    mult = with_fog(PRECIP_MULT, fog_w)
    if seed_it is None:
        m = T.fit(Xa[tr], ya[tr], Xa[va], ya[va], power=0.25, mult=mult)
        it = int(m.best_iteration)
        print(f"  early-stop iters for fog_w={fog_w}: {it}", flush=True)
    else:
        it = seed_it
    rounds = int(it * 1.15)
    m = T.fit(Xa[trva], ya[trva], power=0.25, rounds=rounds, mult=mult)
    return m, it, rounds, mult

def train_prod_cut(fog_w, rounds, cut):
    """Prod-style model on labeled hours with valid_utc <= cut."""
    mult = with_fog(PRECIP_MULT, fog_w)
    return T.fit(Xa[cut], ya[cut], power=0.25, rounds=rounds, mult=mult)

def score_xy(model, X, y, scale):
    P = M.class_boost(model.predict(X), scale)
    p = P.argmax(1)
    s = rc.summ(y, p) if False else None
    # rain_common summ is in the exec'd namespace
    s = summ(y, p)
    s["fog_fa"] = int(((p == FOG) & (y != FOG)).sum())
    s["fog_miss"] = int(((p != FOG) & (y == FOG)).sum())
    s["fog_hit"] = int(((p == FOG) & (y == FOG)).sum())
    return s, P, p

def pack(s):
    keys = ["n", "acc", "macro_f1", "fog_n", "fog_r", "fog_p", "fog_f1", "fog_hit", "fog_fa", "fog_miss",
            "anyprecip_r", "anyprecip_p", "anyprecip_f1", "lr_r", "lr_p", "ra_r", "ra_p", "clear_p", "overcast_p"]
    return {k: s[k] for k in keys if k in s}

# -------------------------------------------------------------------------
# Test-year features (cached 12z forecasts, same as rain_final_test)
# -------------------------------------------------------------------------
print("loading test-year 12z cache...", flush=True)
GT, yt = G_for(te0, te1)
print(f"  test 12z hours {len(yt)}  f00 test {te.sum()}", flush=True)

print("scoring live eval model (fog train x2, boost x2)...", flush=True)
eval_m, eval_meta = M.load_model("KMDQ", "eval")
# predict_proba already applies class_prob_scale from meta
P0 = M.predict_proba(eval_m, eval_meta, pd.DataFrame(Xa[te], columns=feats))
p0 = P0.argmax(1)
s_f00_cur = summ(ya[te], p0)
s_f00_cur["fog_fa"] = int(((p0 == FOG) & (ya[te] != FOG)).sum())
s_f00_cur["fog_miss"] = int(((p0 != FOG) & (ya[te] == FOG)).sum())
s_f00_cur["fog_hit"] = int(((p0 == FOG) & (ya[te] == FOG)).sum())
P0f = M.predict_proba(eval_m, eval_meta, GT[feats])
p0f = P0f.argmax(1)
s_fc_cur = summ(yt, p0f)
s_fc_cur["fog_fa"] = int(((p0f == FOG) & (yt != FOG)).sum())
s_fc_cur["fog_miss"] = int(((p0f != FOG) & (yt == FOG)).sum())
s_fc_cur["fog_hit"] = int(((p0f == FOG) & (yt == FOG)).sum())
s_fc_cur["day1_acc"] = float((p0f == yt)[GT.fhr.values <= 24].mean())
s_fc_cur["day2_acc"] = float((p0f == yt)[GT.fhr.values > 24].mean())
print("  current f00", pack(s_f00_cur), flush=True)
print("  current 12z", pack(s_fc_cur), "day1", s_fc_cur["day1_acc"], "day2", s_fc_cur["day2_acc"], flush=True)

# Primary candidate plus a couple of nearby modest boosts on the same trees.
# Train weight 1.5 is the approved "a little" reduction; boosts 1.25/1.5/1.75 let us
# pick a nearby inference scale if one is clearly better.
print("training eval model with fog training weight 1.5...", flush=True)
m15, it15, rounds15, _ = train_eval(1.5)
print(f"  trained iters={it15} rounds={rounds15}", flush=True)

candidates = {
    "current (fog x2 / boost x2)": {"f00": s_f00_cur, "fcst": s_fc_cur, "fog_w": 2.0, "fog_s": 2.0,
                                    "iters": eval_meta.get("best_iteration"), "source": "live eval model"},
}

raw_te = m15.predict(Xa[te])
raw_fc = m15.predict(GT[feats].astype(float))
for fog_s in (1.25, 1.5, 1.75):
    name = f"new (fog train x1.5 / boost x{fog_s:g})"
    scale = with_fog(PRECIP_SCALE, fog_s)
    s_f00, _, p = score_xy(type("M", (), {"predict": lambda self, X: raw_te})(), Xa[te], ya[te], scale)
    # score_xy calls model.predict — easier to just boost the raw arrays
    P = M.class_boost(raw_te, scale); p = P.argmax(1); s_f00 = summ(ya[te], p)
    s_f00["fog_fa"] = int(((p == FOG) & (ya[te] != FOG)).sum())
    s_f00["fog_miss"] = int(((p != FOG) & (ya[te] == FOG)).sum())
    s_f00["fog_hit"] = int(((p == FOG) & (ya[te] == FOG)).sum())
    P = M.class_boost(raw_fc, scale); p = P.argmax(1); s_fc = summ(yt, p)
    s_fc["fog_fa"] = int(((p == FOG) & (yt != FOG)).sum())
    s_fc["fog_miss"] = int(((p != FOG) & (yt == FOG)).sum())
    s_fc["fog_hit"] = int(((p == FOG) & (yt == FOG)).sum())
    s_fc["day1_acc"] = float((p == yt)[GT.fhr.values <= 24].mean())
    s_fc["day2_acc"] = float((p == yt)[GT.fhr.values > 24].mean())
    candidates[name] = {"f00": s_f00, "fcst": s_fc, "fog_w": 1.5, "fog_s": fog_s, "iters": it15, "rounds": rounds15}
    print(" ", name, "f00", pack(s_f00), flush=True)
    print("   12z", pack(s_fc), "day1", s_fc["day1_acc"], "day2", s_fc["day2_acc"], flush=True)

# -------------------------------------------------------------------------
# Recent 12z forecasts, including the Nov 9-10 early-morning false alarms.
# Compare live prod vs a new-weight model trained on the same cutoff as live prod.
# -------------------------------------------------------------------------
print("fetching recent 12z forecasts (Nov 4-11 if missing)...", flush=True)
recent_runs = [dt.datetime(2026, 11, d, 12) for d in range(1, 12)]
fcs = {}
for run in recent_runs:
    fc = LE.get(run)
    if fc is None or not len(fc):
        print(f"  missing {run:%Y%m%d%H}", flush=True)
        continue
    fcs[run] = fc
    print(f"  got {run:%Y%m%d%H} {len(fc)} h", flush=True)

rows = []
for run, fc in fcs.items():
    try:
        g = fc_feats(run, fc)
    except Exception as e:
        print(f"  fc_feats failed for {run}: {e}", flush=True)
        continue
    if len(g):
        rows.append(g)
Grec = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
yrec = Grec.valid_utc.map(lab).map(CLASSES.index).values if len(Grec) else np.array([], int)
print(f"recent scored hours: {len(Grec)}", flush=True)

prod_m, prod_meta = M.load_model("KMDQ", "prod")
cut = (t <= pd.Timestamp(prod_meta["trained_on"][1])).values
print(f"training new prod-style models on {cut.sum()} hours through {prod_meta['trained_on'][1]}...", flush=True)
# Use the candidate's eval rounds (1.15 * early-stopped iters). Live prod used 129*1.15 from the x2 model.
m15_prod = train_prod_cut(1.5, rounds15, cut)

def predict_pack(model, meta_or_scale, G, y, is_meta=False):
    if not len(G):
        return {"n": 0}, None, None
    if is_meta:
        P = M.predict_proba(model, meta_or_scale, G)
    else:
        P = M.class_boost(model.predict(G[feats].astype(float)), meta_or_scale)
    p = P.argmax(1)
    s = summ(y, p)
    s["fog_fa"] = int(((p == FOG) & (y != FOG)).sum())
    s["fog_miss"] = int(((p != FOG) & (y == FOG)).sum())
    s["fog_hit"] = int(((p == FOG) & (y == FOG)).sum())
    s["day1_acc"] = float((p == y)[G.fhr.values <= 24].mean()) if len(G) else None
    s["day2_acc"] = float((p == y)[G.fhr.values > 24].mean()) if len(G) else None
    return s, P, p

recent = {}
s, Pcur, pcur = predict_pack(prod_m, prod_meta, Grec, yrec, is_meta=True)
recent["current prod (fog x2 / boost x2)"] = s
print("  current prod recent", pack(s), "day1", s.get("day1_acc"), "day2", s.get("day2_acc"), flush=True)

raw_rec = m15_prod.predict(Grec[feats].astype(float)) if len(Grec) else None
for fog_s in (1.25, 1.5, 1.75):
    name = f"new prod (fog train x1.5 / boost x{fog_s:g})"
    scale = with_fog(PRECIP_SCALE, fog_s)
    if raw_rec is None:
        recent[name] = {"n": 0}
        continue
    P = M.class_boost(raw_rec, scale); p = P.argmax(1)
    s = summ(yrec, p)
    s["fog_fa"] = int(((p == FOG) & (yrec != FOG)).sum())
    s["fog_miss"] = int(((p != FOG) & (yrec == FOG)).sum())
    s["fog_hit"] = int(((p == FOG) & (yrec == FOG)).sum())
    s["day1_acc"] = float((p == yrec)[Grec.fhr.values <= 24].mean())
    s["day2_acc"] = float((p == yrec)[Grec.fhr.values > 24].mean())
    recent[name] = s
    print(" ", name, pack(s), "day1", s["day1_acc"], "day2", s["day2_acc"], flush=True)

# Also: inference-only change on the live prod (no retraining) as a diagnostic.
if len(Grec):
    # invert not needed; re-predict raw (before boost) from the booster and apply new scale
    raw_live = prod_m.predict(Grec[feats].astype(float), num_iteration=prod_meta.get("best_iteration"))
    P = M.class_boost(raw_live, with_fog(PRECIP_SCALE, 1.5)); p = P.argmax(1)
    s = summ(yrec, p)
    s["fog_fa"] = int(((p == FOG) & (yrec != FOG)).sum())
    s["fog_miss"] = int(((p != FOG) & (yrec == FOG)).sum())
    s["fog_hit"] = int(((p == FOG) & (yrec == FOG)).sum())
    recent["live prod, boost only 2->1.5 (no retrain)"] = s
    print("  boost-only 1.5 recent", pack(s), flush=True)

# Nov 9-10 early-morning hours (all 12z runs that cover them)
focus = []
if len(Grec):
    Grec = Grec.copy()
    Grec["obs"] = [CLASSES[i] for i in yrec]
    Grec["valid_cst"] = pd.to_datetime(Grec.valid_utc).dt.tz_localize("UTC").dt.tz_convert(CT)
    P_live = M.predict_proba(prod_m, prod_meta, Grec)
    Grec["cur"] = [CLASSES[i] for i in P_live.argmax(1)]
    Grec["cur_fog"] = (P_live[:, FOG] * 100).round(1)
    Grec["cur_clear"] = (P_live[:, C["Clear"]] * 100).round(1)
    for fog_s in (1.25, 1.5, 1.75):
        P = M.class_boost(raw_rec, with_fog(PRECIP_SCALE, fog_s))
        Grec[f"new{fog_s:g}"] = [CLASSES[i] for i in P.argmax(1)]
        Grec[f"new{fog_s:g}_fog"] = (P[:, FOG] * 100).round(1)
        Grec[f"new{fog_s:g}_clear"] = (P[:, C["Clear"]] * 100).round(1)
    # inference-only
    P = M.class_boost(raw_live, with_fog(PRECIP_SCALE, 1.5))
    Grec["boost15"] = [CLASSES[i] for i in P.argmax(1)]
    Grec["boost15_fog"] = (P[:, FOG] * 100).round(1)
    mask = (
        (Grec.valid_cst.dt.strftime("%Y-%m-%d").isin(["2026-11-09", "2026-11-10"]))
        & (Grec.valid_cst.dt.hour.between(5, 9))
    )
    cols = ["run", "fhr", "valid_cst", "obs", "cur", "cur_fog", "cur_clear",
            "boost15", "boost15_fog", "new1.25", "new1.25_fog", "new1.25_clear",
            "new1.5", "new1.5_fog", "new1.5_clear", "new1.75", "new1.75_fog", "new1.75_clear"]
    tab = Grec.loc[mask, cols].sort_values(["valid_cst", "run"])
    tab.to_csv(HOURS_OUT, index=False)
    print("\n=== Nov 9-10 5-9 AM CST ===", flush=True)
    print(tab.to_string(index=False), flush=True)
    # spotlight the known false alarms
    fa_mask = (Grec.cur == "Fog/mist/haze") & (Grec.obs != "Fog/mist/haze") & mask
    print("\nknown-style FAs (current pred fog, obs not fog, 5-9 AM):", int(fa_mask.sum()), flush=True)
    focus = tab.to_dict(orient="records")

# recent fog FA hours among Nov 8-11 for context
if len(Grec):
    late = Grec[Grec.valid_cst >= "2026-11-08"]
    print("\n=== Nov 8+ fog-related (current vs new 1.5/1.5) ===", flush=True)
    rel = late[(late.obs == "Fog/mist/haze") | (late.cur == "Fog/mist/haze") | (late["new1.5"] == "Fog/mist/haze")]
    print(rel[["run", "fhr", "valid_cst", "obs", "cur", "cur_fog", "new1.5", "new1.5_fog"]].to_string(index=False), flush=True)

def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items() if k != "confusion"}
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, list):
        return [clean(x) for x in o]
    if isinstance(o, pd.Timestamp):
        return str(o)
    return o

out = {
    "note": "Modest fog-weight reduction, 2026-11-11. Other class weights unchanged. Test year split fixed at 2025-10-21..2026-10-21.",
    "live_eval_iters": eval_meta.get("best_iteration"),
    "new_eval_iters": it15,
    "new_eval_rounds": rounds15,
    "test_year": {k: clean(v) for k, v in candidates.items()},
    "recent_12z_Nov1_11": {k: clean(v) for k, v in recent.items()},
    "nov9_10_morning_hours": clean(focus),
}
# drop bulky confusion matrices already stripped
json.dump(out, open(OUT, "w"), indent=2, default=float)
print("\nWrote", OUT, flush=True)
print("DONE", flush=True)
