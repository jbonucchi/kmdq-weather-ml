#!/usr/bin/env python3
"""Verify logged forecasts (logs/forecast_log.parquet from predict_latest.py) against observed station weather types.
Refreshes recent METAR labels, scores every forecast hour that is now observed, and writes
logs/verification.parquet (one row per verified forecast hour), logs/verification_summary.{txt,json}.
Summary: most recent fully verified run (accuracy, macro-F1, per-class hits/misses), running 7-/30-day accuracy
for day-1 (f01-f24) and day-2 (f25-f48) leads, the rule-based HRRR baseline on the same hours, and the held-out test baseline.
Usage: verify.py [--station KMDQ] [--no-refresh]"""
import os, sys, json, argparse, datetime as dt, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from labels import CLASSES
import update_dataset as U, model_lib as M
LOG = os.path.join(ROOT, "logs")

def score(g):
    y = g.obs.map(CLASSES.index).values; p = g.predicted_type.map(CLASSES.index).values
    m = M.metrics(y, p); r = g.rule_type.map(CLASSES.index).values if "rule_type" in g else None
    return m, (float((r == y).mean()) if r is not None and len(y) else None)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--station", default=STATION); ap.add_argument("--no-refresh", action="store_true"); a = ap.parse_args()
    stn = a.station; fl = os.path.join(LOG, "forecast_log.parquet")
    if not os.path.exists(fl): print("no forecasts logged yet"); return
    if not a.no_refresh:
        today = dt.datetime.now(dt.timezone.utc).date(); U.update_labels(stn, today - dt.timedelta(days=3), today)
    lab = pd.read_parquet(U.labels_path(stn))[["valid_utc", "label", "metar"]]; lab["valid_utc"] = pd.to_datetime(lab.valid_utc)
    fc = pd.read_parquet(fl); fc = fc[fc.station == stn].copy(); fc["valid_dt"] = pd.to_datetime(fc.valid_utc.str.replace("Z", ""))
    v = fc.merge(lab.rename(columns={"valid_utc": "valid_dt", "label": "obs"}), on="valid_dt", how="inner")
    v["hit"] = v.predicted_type == v.obs; v["rule_hit"] = v.rule_type == v.obs
    v["lead"] = np.where(v.fhr <= 24, "day-1 (f01-f24)", "day-2 (f25-f48)")
    v.to_parquet(os.path.join(LOG, "verification.parquet"), index=False)
    if not len(v): print("nothing verifiable yet"); return
    lines, S = [], {"generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"), "station": stn}
    # most recent run with all its forecast hours already observed (else latest partially verified)
    nfc = fc.groupby("run_utc").size(); nv = v.groupby("run_utc").size()
    full = [r for r in nv.index if nv[r] >= nfc[r] - 2]
    run = max(full) if full else max(nv.index); g = v[v.run_utc == run]
    m, rb = score(g)
    S["latest_run"] = {"run_utc": run, "hours_verified": len(g), "accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "rule_baseline_accuracy": rb}
    lines.append(f"Run {run} vs {stn} obs: {len(g)} h verified, accuracy {m['accuracy']:.0%}, macro-F1 {m['macro_f1']:.2f} (rule-based HRRR: {rb:.0%})")
    for lead, gg in g.groupby("lead"):
        lines.append(f"   {lead}: {gg.hit.mean():.0%} of {len(gg)} h")
    pc = []
    for c in CLASSES:
        ob = (g.obs == c).sum(); pr = (g.predicted_type == c).sum(); hit = ((g.obs == c) & (g.predicted_type == c)).sum()
        if ob or pr: pc.append(f"{c}: observed {ob}, forecast {pr}, hits {hit}, misses {ob - hit}, false alarms {pr - hit}")
    S["latest_run"]["per_class"] = pc; lines += ["   " + x for x in pc]
    miss = g[~g.hit]
    if len(miss): lines.append("   Biggest confusions: " + ", ".join(f"{o} forecast as {p} ({n}h)" for (o, p), n in miss.groupby(["obs", "predicted_type"]).size().sort_values(ascending=False).head(3).items()))
    last = max(nv.index)
    if last != run:
        gl = v[v.run_utc == last]; ml, rl = score(gl)
        S["latest_partial_run"] = {"run_utc": last, "hours_verified": len(gl), "accuracy": ml["accuracy"], "macro_f1": ml["macro_f1"], "rule_baseline_accuracy": rl}
        lines.insert(0, f"Most recent run {last} (first {len(gl)} h observed so far): accuracy {ml['accuracy']:.0%}, macro-F1 {ml['macro_f1']:.2f} (rule-based HRRR: {rl:.0%}); "
                        + ", ".join(f"{o}->{p} {n}h" for (o, p), n in gl[~gl.hit].groupby(["obs", "predicted_type"]).size().sort_values(ascending=False).head(3).items()))
    now = pd.Timestamp.now("UTC").tz_localize(None); S["running"] = {}
    for days in (7, 30):
        w = v[v.valid_dt >= now - pd.Timedelta(days=days)]
        for lead, gg in w.groupby("lead"):
            if not len(gg): continue
            mm, rbb = score(gg); S["running"][f"{days}d {lead}"] = {"hours": len(gg), "accuracy": mm["accuracy"], "macro_f1": mm["macro_f1"], "rule_accuracy": rbb}
            lines.append(f"Last {days} days, {lead}: accuracy {mm['accuracy']:.0%} over {len(gg)} h, macro-F1 {mm['macro_f1']:.2f} (rule-based {rbb:.0%})")
    try:
        meta = json.load(open(os.path.join(MODELS, f"model_{stn}_eval.json")))
        S["test_baseline"] = {"accuracy": meta["test_accuracy"], "macro_f1": meta["test_macro_f1"]}
        lines.append(f"Reference: held-out test year on HRRR analyses (f00): accuracy {meta['test_accuracy']:.0%}, macro-F1 {meta['test_macro_f1']:.2f}")
        le = os.path.join(ROOT, "reports", f"lead_time_{stn}.json")
        if os.path.exists(le):
            L = json.load(open(le)); lines.append("Reference: test-year hindcasts by lead: " + ", ".join(f"{k} {x['accuracy']:.0%}" for k, x in L["by_lead_bucket"].items()))
    except FileNotFoundError: pass
    S["text"] = lines
    open(os.path.join(LOG, "verification_summary.txt"), "w").write("\n".join(lines) + "\n")
    json.dump(S, open(os.path.join(LOG, "verification_summary.json"), "w"), indent=1, default=str)
    print("\n".join(lines))
if __name__ == "__main__": main()
