import requests, tempfile, numpy as np, pygrib, math, json
url="https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20261019/conus/hrrr.t12z.wrfsfcf00.grib2"
idx=requests.get(url+".idx").text.splitlines()
for i,l in enumerate(idx):
    p=l.split(":")
    if p[3]=="HGT" and p[4]=="surface":
        s=int(p[1]); e=int(idx[i+1].split(":")[1])-1; break
d=requests.get(url,headers={"Range":f"bytes={s}-{e}"}).content
t=tempfile.NamedTemporaryFile(suffix=".grib2"); t.write(d); t.flush()
m=pygrib.open(t.name)[1]; la,lo=m.latlons(); lo=np.where(lo>180,lo-360,lo); h=m.values
def hav(a,b,c,d):
    a,b,c,d=map(np.radians,(a,b,c,d)); return 2*6371.0088*np.arcsin(np.sqrt(np.sin((c-a)/2)**2+np.cos(a)*np.cos(c)*np.sin((d-b)/2)**2))
LAT,LON=34.8108024597168,-86.72720336914062
dist=hav(LAT,LON,la,lo); iy,ix=np.unravel_index(dist.argmin(),dist.shape)
print("grid",iy,ix,la[iy,ix],lo[iy,ix],dist[iy,ix],"km elev",h[iy,ix])
st={"KMDQ":(34.8614,-86.5572),"KHSV":(34.6372,-86.7751),"KHUA":(34.6787,-86.6848)}
for k,(a,b) in st.items(): print(k, round(float(hav(LAT,LON,a,b)),2),"km")
json.dump({"iy":int(iy),"ix":int(ix),"lat":float(la[iy,ix]),"lon":float(lo[iy,ix]),"dist_km":float(dist[iy,ix]),"elev_m":float(h[iy,ix])},open("gridpoint.json","w"))

