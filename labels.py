"""METAR -> weather-type class labels, and station download from the Iowa Environmental Mesonet ASOS archive."""
import os, sys, io, re, datetime as dt, numpy as np, pandas as pd, requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *

CLASSES = ["Clear", "Partly cloudy", "Mostly cloudy", "Overcast", "Fog/mist/haze",
           "Light rain/drizzle", "Rain", "Heavy rain", "Thunderstorm", "Snow/ice"]
IEM = ("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station={s}&data=all&year1={a.year}&month1={a.month}&day1={a.day}"
       "&year2={b.year}&month2={b.month}&day2={b.day}&tz=Etc/UTC&format=onlycomma&latlon=no&elev=no&missing=empty&trace=T&direct=no&report_type=3")

def download_asos(station, start, end):
    """Routine (report_type=3) METARs between dates (inclusive start, exclusive end+1)."""
    a = pd.Timestamp(start); b = pd.Timestamp(end) + pd.Timedelta(days=1)
    r = requests.get(IEM.format(s=station[1:] if station.startswith("K") else station, a=a, b=b), timeout=300); r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text), low_memory=False)

SKY_RANK = {"CLR": 0, "SKC": 0, "NSC": 0, "FEW": 1, "SCT": 2, "BKN": 3, "OVC": 4, "VV": 4, "VV ": 4}
def classify(wx, vis, skies):
    """Priority: Thunderstorm > Snow/ice > Heavy rain > Rain > Light rain/drizzle > Fog/mist/haze > sky cover.
    wx = IEM wxcodes string, vis = statute miles, skies = list of skyc1..4."""
    w = wx if isinstance(wx, str) else ""
    toks = w.split()
    if "TS" in w: return "Thunderstorm"                      # TS, VCTS, +TSRA ... (thunder heard / lightning detected)
    if re.search(r"SN|SG|PL|IP|GS|GR|FZRA|FZDZ", w): return "Snow/ice"   # snow, sleet, freezing rain/drizzle, hail
    liq = [t for t in toks if re.search(r"RA|DZ|UP", t) and not t.startswith("VC")]
    if liq:
        if any(t.startswith("+") and "RA" in t for t in liq): return "Heavy rain"
        if any(not t.startswith("-") and "RA" in t for t in liq): return "Rain"
        return "Light rain/drizzle"                             # -RA, DZ, -DZ, +DZ, UP
    if re.search(r"FG|BR|HZ|FU", w) or (pd.notna(vis) and vis < 3): return "Fog/mist/haze"
    cover = max([SKY_RANK.get(str(s).strip(), -1) for s in skies if isinstance(s, str)] or [-1])
    if cover < 0: return None
    return ["Clear", "Clear", "Partly cloudy", "Mostly cloudy", "Overcast"][cover]

def hourly_labels(raw):
    """For each top-of-hour, take the routine METAR nearest to it within 30 min."""
    d = raw.copy(); d["valid"] = pd.to_datetime(d["valid"])
    d["label"] = [classify(w, v, s) for w, v, s in zip(d["wxcodes"], d["vsby"], d[["skyc1", "skyc2", "skyc3", "skyc4"]].values.tolist())]
    d = d.dropna(subset=["label"]).sort_values("valid")
    hrs = pd.DataFrame({"valid_utc": pd.date_range(d.valid.min().floor("h"), d.valid.max().ceil("h"), freq="h")})
    m = pd.merge_asof(hrs, d.rename(columns={"valid": "obs_time"}), left_on="valid_utc", right_on="obs_time",
                      direction="nearest", tolerance=pd.Timedelta("30min"))
    m = m.dropna(subset=["label"])
    keep = ["valid_utc", "obs_time", "label", "tmpf", "dwpf", "vsby", "skyc1", "skyl1", "wxcodes", "metar"]
    return m[keep].rename(columns={"tmpf": "obs_tmpf", "dwpf": "obs_dwpf", "vsby": "obs_vsby", "skyc1": "obs_skyc1", "skyl1": "obs_skyl1"})
