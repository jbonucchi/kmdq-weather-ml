"""Shared configuration for the HRRR -> station weather-type ML project."""
import json, os
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data"); HRRR_DIR = os.path.join(DATA, "hrrr_days"); MODELS = os.path.join(ROOT, "models")
TARGET_LAT, TARGET_LON = 34.8108024597168, -86.72720336914062
_g = json.load(open(os.path.join(ROOT, "gridpoint.json")))
IY, IX = _g["iy"], _g["ix"]          # HRRR grid indices (y, x) of nearest point
CHUNK = 150
STATION = os.environ.get("HRRR_ML_STATION", "KMDQ")
ZARR = "https://hrrrzarr.s3.amazonaws.com/sfc/{d}/{d}_{h:02d}z_anl.zarr/{lev}/{var}/{lev}/{var}/{cy}.{cx}"
GRIB = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{d}/conus/hrrr.t{h:02d}z.wrfsfcf00.grib2"
# LTNG omitted: its chunks are absent from hrrrzarr, so it cannot be trained on.
# (zarr level, zarr var) -> column name. Same idea as /workspace/hrrr_point.py, plus extras useful for wx type.
FIELDS = {
 ("2m_above_ground","TMP"):"t2", ("2m_above_ground","DPT"):"td2", ("2m_above_ground","RH"):"rh2", ("2m_above_ground","SPFH"):"q2",
 ("10m_above_ground","UGRD"):"u10", ("10m_above_ground","VGRD"):"v10", ("80m_above_ground","UGRD"):"u80", ("80m_above_ground","VGRD"):"v80",
 ("surface","GUST"):"gust", ("mean_sea_level","MSLMA"):"mslp", ("surface","PRES"):"psfc", ("surface","TMP"):"tskin",
 ("surface","PRATE"):"prate", ("surface","CRAIN"):"crain", ("surface","CSNOW"):"csnow", ("surface","CFRZR"):"cfrzr",
 ("surface","CICEP"):"cicep", ("surface","CPOFP"):"cpofp", ("surface","VIS"):"vis", ("surface","SNOD"):"snod",
 ("surface","DSWRF"):"dswrf", ("surface","DLWRF"):"dlwrf", ("surface","SHTFL"):"shtfl", ("surface","LHTFL"):"lhtfl",
 ("surface","HPBL"):"pbl", ("surface","HGT"):"hsfc",
 ("entire_atmosphere","TCDC"):"tcdc", ("low_cloud_layer","LCDC"):"lcdc", ("middle_cloud_layer","MCDC"):"mcdc",
 ("high_cloud_layer","HCDC"):"hcdc", ("boundary_layer_cloud_layer","TCDC"):"blcdc",
 ("cloud_base","HGT"):"cbase", ("cloud_ceiling","HGT"):"ceil", ("cloud_top","HGT"):"ctop",
 ("entire_atmosphere","REFC"):"refc", ("1000m_above_ground","REFD"):"refd1km", ("4000m_above_ground","REFD"):"refd4km",
 ("263K_level","REFD"):"refd263k", ("entire_atmosphere","VIL"):"vil",
 ("entire_atmosphere","TCOLW"):"tcolw", ("entire_atmosphere","TCOLI"):"tcoli", ("entire_atmosphere","RHPW"):"rhpw",
 ("entire_atmosphere_single_layer","PWAT"):"pwat", ("8m_above_ground","MASSDEN"):"smoke",
 ("top_of_atmosphere","SBT123"):"sbt123", ("top_of_atmosphere","SBT124"):"sbt124",
 ("top_of_atmosphere","SBT113"):"sbt113", ("top_of_atmosphere","SBT114"):"sbt114",
 ("surface","CAPE"):"sbcape", ("surface","CIN"):"sbcin", ("255_0mb_above_ground","CAPE"):"mucape", ("255_0mb_above_ground","CIN"):"mucin",
 ("90_0mb_above_ground","CAPE"):"mlcape", ("90_0mb_above_ground","CIN"):"mlcin", ("0_3000m_above_ground","CAPE"):"cape03",
 ("500_1000mb","LFTX"):"li", ("180_0mb_above_ground","4LFTX"):"bli",
 ("0_1000m_above_ground","VUCSH"):"ush1", ("0_1000m_above_ground","VVCSH"):"vsh1",
 ("0_6000m_above_ground","VUCSH"):"ush6", ("0_6000m_above_ground","VVCSH"):"vsh6",
 ("1000_0m_above_ground","HLCY"):"srh1", ("3000_0m_above_ground","HLCY"):"srh3",
 ("0C_isotherm","HGT"):"z0c", ("highest_tropospheric_freezing_level","HGT"):"zfrz",
 ("level_of_adiabatic_condensation_from_sfc","HGT"):"zlcl",
 ("1000mb","TMP"):"t1000", ("1000mb","DPT"):"td1000",
 ("925mb","TMP"):"t925", ("925mb","DPT"):"td925", ("925mb","UGRD"):"u925", ("925mb","VGRD"):"v925",
 ("850mb","TMP"):"t850", ("850mb","DPT"):"td850", ("850mb","UGRD"):"u850", ("850mb","VGRD"):"v850", ("850mb","HGT"):"z850",
 ("700mb","TMP"):"t700", ("700mb","DPT"):"td700", ("700mb","UGRD"):"u700", ("700mb","VGRD"):"v700", ("700mb","HGT"):"z700",
 ("700mb","DZDT"):"w700",
 ("500mb","TMP"):"t500", ("500mb","DPT"):"td500", ("500mb","UGRD"):"u500", ("500mb","VGRD"):"v500", ("500mb","HGT"):"z500",
}
# Matching GRIB2 idx keys (var:level) for the fallback / predict path (wrfsfcf00).
GRIB_LEVEL = {"2m_above_ground":"2 m above ground","10m_above_ground":"10 m above ground","80m_above_ground":"80 m above ground",
 "surface":"surface","mean_sea_level":"mean sea level","entire_atmosphere":"entire atmosphere","low_cloud_layer":"low cloud layer",
 "middle_cloud_layer":"middle cloud layer","high_cloud_layer":"high cloud layer","boundary_layer_cloud_layer":"boundary layer cloud layer",
 "cloud_base":"cloud base","cloud_ceiling":"cloud ceiling","cloud_top":"cloud top","1000m_above_ground":"1000 m above ground",
 "4000m_above_ground":"4000 m above ground","263K_level":"263 K level","entire_atmosphere_single_layer":"entire atmosphere (considered as a single layer)",
 "8m_above_ground":"8 m above ground","top_of_atmosphere":"top of atmosphere","255_0mb_above_ground":"255-0 mb above ground",
 "90_0mb_above_ground":"90-0 mb above ground","0_3000m_above_ground":"0-3000 m above ground","500_1000mb":"500-1000 mb",
 "180_0mb_above_ground":"180-0 mb above ground","0_1000m_above_ground":"0-1000 m above ground","0_6000m_above_ground":"0-6000 m above ground",
 "1000_0m_above_ground":"1000-0 m above ground","3000_0m_above_ground":"3000-0 m above ground","0C_isotherm":"0C isotherm",
 "highest_tropospheric_freezing_level":"highest tropospheric freezing level","level_of_adiabatic_condensation_from_sfc":"level of adiabatic condensation from sfc",
 "1000mb":"1000 mb","925mb":"925 mb","850mb":"850 mb","700mb":"700 mb","500mb":"500 mb"}
