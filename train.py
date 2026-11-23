#!/usr/bin/env python3
"""Train LightGBM weather-type classifier from HRRR f00 point features -> station weather type.
Time-based split: test = last 12 months, validation = the 12 months before, train = everything earlier
(falls back to 60/20/20 chronological when < 3 years of data). Writes models/ and reports/.
Usage: train.py [--station KMDQ] [--prod-only]"""
import os, sys, json, argparse, time, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from labels import CLASSES
import features as X, model_lib as M
REP = os.path.join(ROOT, "reports")
PARAMS = dict(objective="multiclass", num_class=len(CLASSES), learning_rate=0.03, num_leaves=63, min_data_in_leaf=40,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=6, seed=42)

# Class emphasis, tuned on the validation year using archived 12z f01-f48 forecasts and scored once on the test year.
#  * Fog (2026-10-30, after repeated fog->Clear misses): originally Fog/mist/haze hours x2 training weight, fog probability x2
#    at prediction time (experiments/fog_val_fcst_grid.csv; pre-fog model in models/backup_20261030/).
#  * Precipitation (2026-11-03, after the Oct 31-Nov 2 runs called rain Overcast): Light rain/drizzle x1.5 and Rain/Heavy rain
#    x2 training weight; at prediction time drizzle probability x1.5 and Rain/Heavy rain x3 (experiments/rain_val_grid_seeds.csv,
#    experiments/rain_final_test.json; fog-only model in models/backup_20261103/). Thunderstorm left at x1 (no validation gain).
#  * Fog trim (2026-11-11, after early-morning fog false alarms on 9-10 Nov when KMDQ was clear): fog training weight and
#    prediction-time boost reduced from x2 to x1.5 (experiments/fog_reduce_20261111.py; previous model in models/backup_20261111/).
#    Other class weights unchanged. This recovered some f00 accuracy and fog precision, cut forecast fog false alarms, and
#    left 12z forecast accuracy flat; fog recall is a little lower, as intended.
#  * Sky cover (2026-11-14, after clear <-> overcast/mostly cloudy mix-ups): the model is now also trained on archived HRRR 12z
#    forecast hours (f01-f48 from data/fcst_12z/, ~750 runs since Oct 2024, weight x2 = FCST_TRAIN_WEIGHT) with the forecast hour
#    (fhr) as a feature (analysis rows fhr=0), so it learns how HRRR *forecast* cloud relates to what KMDQ reports, and the
#    sky-class probabilities are scaled Partly cloudy x1.25, Mostly cloudy x1.5, Overcast x1.15 (in CLASS_PROB_SCALE).
#    The forecast-trained model calls fog more often, so the fog prediction-time boost is now x1.0 (training weight stays x1.5);
#    this matched the previous model's validation-year fog false-alarm count (experiments/sky_fog_check.py).
#    Chosen on validation-year forecasts (experiments/sky_final_val.py), test year scored once (experiments/sky_final_test.py).
#    Previous model in models/backup_20261114/.
# Settings are stored in the model json (class_weight_mult / class_prob_scale) and applied by model_lib.predict_proba, so
# retrain.sh (train.py --prod-only) keeps them. Set the dicts to {} for the original (pre-2026-10-30) model.
# Class-weight power fixed at 0.25: the fog and precip emphasis above was tuned at 0.25. (train.py's old acc+macro-F1 search on f00
# picks 0.0 with these settings, which on validation forecasts cuts fog recall 23%->16% and rain recall 33%->27%; see
# experiments/rain_power_check.py.) Use --search-power to rerun that search.
CLASS_WEIGHT_POWER = 0.25
CLASS_WEIGHT_MULT = {"Fog/mist/haze": 1.5, "Light rain/drizzle": 1.5, "Rain": 2.0, "Heavy rain": 2.0}
CLASS_PROB_SCALE = {"Fog/mist/haze": 1.0, "Light rain/drizzle": 1.5, "Rain": 3.0, "Heavy rain": 3.0,
                    "Partly cloudy": 1.25, "Mostly cloudy": 1.5, "Overcast": 1.15}
FCST_TRAIN_WEIGHT = 2.0     # weight of archived 12z forecast rows relative to f00 rows (0 = f00-only, the pre-2026-11-14 model)
FOG_WEIGHT = CLASS_WEIGHT_MULT.get("Fog/mist/haze", 1.0); FOG_PROB_SCALE = CLASS_PROB_SCALE.get("Fog/mist/haze", 1.0)   # back-compat names
FOG = CLASSES.index("Fog/mist/haze")

def weights(y, power, mult=None):
    """Inverse-frequency class weights ** power, times per-class multipliers {class name: x} (default CLASS_WEIGHT_MULT)."""
    mult = CLASS_WEIGHT_MULT if mult is None else mult
    cnt = np.bincount(y, minlength=len(CLASSES)).astype(float); w = (cnt.sum() / np.maximum(cnt, 1)) ** power
    w = w / (w * cnt).sum() * cnt.sum()
    for c, m in mult.items(): w[CLASSES.index(c)] *= m
    return w[y]

def fit(Xtr, ytr, Xva=None, yva=None, power=0.5, rounds=3000, params=PARAMS, mult=None, row_mult=None):
    w = weights(ytr, power, mult)
    if row_mult is not None: w = w * row_mult
    dtr = lgb.Dataset(Xtr, ytr, weight=w)
    if Xva is None: return lgb.train(params, dtr, num_boost_round=rounds)
    dva = lgb.Dataset(Xva, yva, weight=weights(yva, power, mult), reference=dtr)
    return lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])

def settings(meta=None):
    """Class emphasis settings saved in the model json (falls back to the old fog-only keys, then to the module defaults)."""
    if meta is None: return dict(CLASS_WEIGHT_MULT), dict(CLASS_PROB_SCALE)
    if "class_weight_mult" in meta: return meta["class_weight_mult"], meta.get("class_prob_scale", {})
    if "fog_weight" in meta or "fog_prob_scale" in meta:
        return {"Fog/mist/haze": meta.get("fog_weight", 1.0)}, {"Fog/mist/haze": meta.get("fog_prob_scale", 1.0)}
    return {}, {}

def stack(fa, ya, G, mask_f, mask_g, fw):
    """f00 rows (mask_f) + forecast rows (mask_g) -> X, y, per-row weight multiplier."""
    if G is None or fw <= 0 or mask_g is None or not mask_g.any(): return fa[mask_f], ya[mask_f], None
    Xs = pd.concat([fa[mask_f], G.loc[mask_g, fa.columns].astype(float)], ignore_index=True)
    ys = np.concatenate([ya[mask_f], G.y.values[mask_g]])
    rm = np.r_[np.ones(mask_f.sum()), np.full(mask_g.sum(), fw)]
    return Xs, ys, rm

def split(f, test_end=None):
    t = f.valid_utc
    if test_end is not None:                       # fixed test-year end (reproduce an earlier split); later hours are unused for eval
        keep = (t <= test_end).values; t = t.where(keep)
    span = (t.max() - t.min()).days
    if span >= 3 * 365:
        te0 = t.max() - pd.Timedelta(days=365); va0 = te0 - pd.Timedelta(days=365)
    else:
        te0 = t.min() + pd.Timedelta(days=span * 0.8); va0 = t.min() + pd.Timedelta(days=span * 0.6)
    return (t < va0).values, ((t >= va0) & (t < te0)).values, (t >= te0).values, va0, te0

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--station", default=STATION); ap.add_argument("--prod-only", action="store_true")
    ap.add_argument("--search-power", action="store_true", help="re-search the class-weight power on validation f00 (default: fixed CLASS_WEIGHT_POWER)")
    ap.add_argument("--test-end", help="fix the end of the test year, e.g. '2026-10-21 03:00' (default: latest data)")
    a = ap.parse_args(); stn = a.station; os.makedirs(MODELS, exist_ok=True); os.makedirs(REP, exist_ok=True)
    f, ds = M.load_xy(stn); feats = X.feature_cols(f.drop(columns=["label", "y", "raw_idx"]))
    Xa, ya = f[feats].astype(float), f.y.values
    if a.prod_only:
        meta = json.load(open(os.path.join(MODELS, f"model_{stn}_eval.json")))
        mult, scale = settings(meta); fw = meta.get("fcst_train_weight", 0.0)
        G = M.load_fcst_xy(ds) if fw > 0 else None
        Xs, ys, rm = stack(Xa, ya, G, np.ones(len(ya), bool), None if G is None else np.ones(len(G), bool), fw)
        m = fit(Xs, ys, power=meta["class_weight_power"], rounds=int(meta["best_iteration"]), mult=mult, row_mult=rm)
        meta["n_fcst_train"] = 0 if G is None else int(len(G))
        meta["class_weight_mult"] = mult; meta["class_prob_scale"] = scale
        meta["fog_weight"] = mult.get("Fog/mist/haze", 1.0); meta["fog_prob_scale"] = scale.get("Fog/mist/haze", 1.0)
        m.save_model(os.path.join(MODELS, f"model_{stn}_prod.txt"))
        json.dump({**meta, "kind": "prod", "trained_on": [str(f.valid_utc.min()), str(f.valid_utc.max())], "n_train": int(len(f))},
                  open(os.path.join(MODELS, f"model_{stn}_prod.json"), "w"), indent=1)
        print("prod model retrained on", len(f), "analysis hours +", meta["n_fcst_train"], "forecast hours"); return
    tr, va, te, va0, te0 = split(f, pd.Timestamp(a.test_end) if a.test_end else None)
    te1 = f.valid_utc[te].max() + pd.Timedelta(hours=1)
    fw = FCST_TRAIN_WEIGHT; G = M.load_fcst_xy(ds) if fw > 0 else None
    if G is not None:   # whole runs only: validation / test windows; "before" = runs ending before the test year
        rend = G.run + pd.Timedelta(hours=48)
        g_va = ((G.run >= va0) & (rend <= te0)).values; g_te = ((G.run >= te0) & (rend <= te1)).values; g_pre = (rend <= te0).values
        g_pretr = (rend <= va0).values
    R = {"station": stn, "period": [str(f.valid_utc.min()), str(f.valid_utc.max())], "n_labeled": int(len(f)),
         "split": {"train": [str(f.valid_utc[tr].min()), str(f.valid_utc[tr].max()), int(tr.sum())],
                   "val": [str(f.valid_utc[va].min()), str(f.valid_utc[va].max()), int(va.sum())],
                   "test": [str(f.valid_utc[te].min()), str(f.valid_utc[te].max()), int(te.sum())]},
         "class_counts": {s: f.label[m].value_counts().reindex(CLASSES, fill_value=0).astype(int).to_dict() for s, m in (("train", tr), ("val", va), ("test", te))}}
    # baselines
    maj = np.bincount(ya[tr]).argmax()
    R["baseline_majority_test"] = M.metrics(ya[te], np.full(te.sum(), maj), "majority")
    rule = M.rule_baseline(ds.loc[f.raw_idx[te]])
    R["baseline_rule_test"] = M.metrics(ya[te], rule, "rule-based (hrrr_point.py)")
    # tune class-weight power on validation
    best = None; R["val_search"] = []
    for pw in ((0.0, 0.25, 0.5, 0.75) if a.search_power else (CLASS_WEIGHT_POWER,)):
        t0 = time.time()
        if G is not None and g_va.any():
            # rounds chosen by early stopping on validation-year forecasts. Training = f00 train years + archived forecasts outside the
            # validation window (before it if any, else the test-window runs: these only set the number of rounds; the scored eval model never sees them)
            g_sel = g_pretr if g_pretr.sum() > 5000 else g_te
            Xs, ys, rm = stack(Xa, ya, G, tr, g_sel, fw)
            m = fit(Xs, ys, G.loc[g_va, feats].astype(float), G.y.values[g_va], power=pw, row_mult=rm)
            p = M.class_boost(m.predict(G.loc[g_va, feats].astype(float), num_iteration=m.best_iteration), CLASS_PROB_SCALE).argmax(1); mm = M.metrics(G.y.values[g_va], p)
        else:
            m = fit(Xa[tr], ya[tr], Xa[va], ya[va], power=pw)
            p = M.class_boost(m.predict(Xa[va], num_iteration=m.best_iteration), CLASS_PROB_SCALE).argmax(1); mm = M.metrics(ya[va], p)
        R["val_search"].append({"power": pw, "iters": m.best_iteration, "acc": mm["accuracy"], "macro_f1": mm["macro_f1"], "sec": round(time.time() - t0)})
        print(R["val_search"][-1], flush=True)
        score = mm["macro_f1"] + mm["accuracy"]        # balance overall accuracy and rare-class skill
        if best is None or score > best[0]: best = (score, pw, m.best_iteration)
    _, pw, it = best
    # eval model: train+val, fixed iterations, scored once on the held-out test year
    trva = tr | va
    Xs, ys, rm = stack(Xa, ya, G, trva, None if G is None else g_pre, fw)
    m = fit(Xs, ys, power=pw, rounds=int(it * 1.15), row_mult=rm)
    P = M.class_boost(m.predict(Xa[te]), CLASS_PROB_SCALE); p = P.argmax(1)
    if G is not None and g_te.any():
        Pf = M.class_boost(m.predict(G.loc[g_te, feats].astype(float)), CLASS_PROB_SCALE).argmax(1)
        R["model_test_fcst12z"] = M.metrics(G.y.values[g_te], Pf, "LightGBM on test-year 12z f01-f48 forecasts")
    R["model_test"] = M.metrics(ya[te], p, f"LightGBM (class-weight power {pw}, class weight x{CLASS_WEIGHT_MULT}, prob x{CLASS_PROB_SCALE})"); R["class_weight_power"] = pw
    R["class_weight_mult"] = CLASS_WEIGHT_MULT; R["class_prob_scale"] = CLASS_PROB_SCALE; R["fog_weight"] = FOG_WEIGHT; R["fog_prob_scale"] = FOG_PROB_SCALE
    imp = pd.Series(m.feature_importance("gain"), index=feats).sort_values(ascending=False)
    R["top_features_gain"] = (imp / imp.sum()).head(25).round(4).to_dict()
    # monthly accuracy on test (seasonality of skill)
    tt = f.valid_utc[te].dt.to_period("M").astype(str).values
    R["test_monthly_acc"] = pd.Series(p == ya[te]).groupby(tt).mean().round(3).to_dict()
    m.save_model(os.path.join(MODELS, f"model_{stn}_eval.txt"))
    meta = {"station": stn, "classes": CLASSES, "features": feats, "class_weight_power": pw, "best_iteration": None,
            "class_weight_mult": CLASS_WEIGHT_MULT, "class_prob_scale": CLASS_PROB_SCALE, "fog_weight": FOG_WEIGHT, "fog_prob_scale": FOG_PROB_SCALE,
            "kind": "eval", "trained_on": [str(f.valid_utc[trva].min()), str(f.valid_utc[trva].max())], "n_train": int(trva.sum()),
            "test_accuracy": R["model_test"]["accuracy"], "test_macro_f1": R["model_test"]["macro_f1"],
            "fcst_train_weight": fw if G is not None else 0.0, "n_fcst_train": int(g_pre.sum()) if G is not None else 0}
    if "model_test_fcst12z" in R: meta["test_fcst_accuracy"] = R["model_test_fcst12z"]["accuracy"]; meta["test_fcst_macro_f1"] = R["model_test_fcst12z"]["macro_f1"]
    meta["best_iteration"] = int(it * 1.15)
    json.dump(meta, open(os.path.join(MODELS, f"model_{stn}_eval.json"), "w"), indent=1)
    pd.DataFrame({"valid_utc": f.valid_utc[te].values, "obs": f.label[te].values, "pred": [CLASSES[i] for i in p],
                  "rule": [CLASSES[i] for i in rule], **{f"p_{c}": P[:, i].round(4) for i, c in enumerate(CLASSES)}}
                 ).to_csv(os.path.join(REP, f"test_predictions_{stn}.csv"), index=False)
    # variant: + previous-hour observed weather type (nowcasting only; not usable beyond ~1 h lead)
    prev = f.set_index("valid_utc").y.reindex(f.valid_utc - pd.Timedelta(hours=1)).values
    Xp = Xa.copy(); Xp["prev_obs"] = prev
    mp = lgb.train({**PARAMS}, lgb.Dataset(Xp[trva], ya[trva], weight=weights(ya[trva], pw), categorical_feature=["prev_obs"]), num_boost_round=int(it * 1.15))   # f00-only nowcast variant
    R["variant_prev_obs_test"] = M.metrics(ya[te], M.class_boost(mp.predict(Xp[te]), CLASS_PROB_SCALE).argmax(1), "LightGBM + previous-hour obs")
    json.dump(R, open(os.path.join(REP, f"metrics_{stn}.json"), "w"), indent=1)
    # production model on all data
    Xs, ys, rm = stack(Xa, ya, G, np.ones(len(ya), bool), None if G is None else np.ones(len(G), bool), fw)
    mprod = fit(Xs, ys, power=pw, rounds=int(it * 1.15), row_mult=rm); mprod.save_model(os.path.join(MODELS, f"model_{stn}_prod.txt"))
    json.dump({**meta, "kind": "prod", "trained_on": [str(f.valid_utc.min()), str(f.valid_utc.max())], "n_train": int(len(f)), "n_fcst_train": 0 if G is None else int(len(G))},
              open(os.path.join(MODELS, f"model_{stn}_prod.json"), "w"), indent=1)
    for k in ("baseline_majority_test", "baseline_rule_test", "model_test", "variant_prev_obs_test", "model_test_fcst12z"):
        if k not in R: continue
        print(f"{R[k]['name']:40s} acc={R[k]['accuracy']:.3f} macroF1={R[k]['macro_f1']:.3f}")
if __name__ == "__main__": main()
