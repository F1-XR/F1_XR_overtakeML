"""실제 시나리오 1건 드라이런 — 특정 (드라이버, 시각)에서 18피처가 제대로 뽑히나 확인.
measure_step4의 피처 로직 재사용(=features.py와 동일 계산). 부스터도 실제 순회."""
import json, math, bisect
from datetime import datetime
import numpy as np
exec(open("measure_step4.py").read().split("def main()")[0])  # 헬퍼 재사용

names, trees = parse_booster(MODEL)
iv = idx_driver(load("intervals"), ["interval"])
pos = idx_driver(load("position"), ["position"])
weather = sorted([(ts(w["date"]), w) for w in load("weather")])
stints_by = {}
for s in load("stints"): stints_by.setdefault(s["driver_number"], []).append(s)
laps_rows = load("laps")
laps_by = {}
for r in laps_rows:
    if r.get("date_start"):
        laps_by.setdefault(r["driver_number"], []).append((ts(r["date_start"]), r.get("lap_number"), r.get("duration_sector_1"), r.get("duration_sector_2")))
for k in laps_by: laps_by[k].sort()
car = idx_driver(load("car_data"), ["speed","drs"])
loc = idx_driver(load("location"), ["x","y"])
ref = build_reference(loc, laps_rows)

# 가장 접전인 순간 하나 고르기(gap 최소, 실제 추월로 이어진 것 우선)
best=None
for dn,series in iv.items():
    for t,gap in series:
        if gap and 0<gap<=0.6:
            best=(dn,t,gap); break
    if best: break
dn,t,gap=best
def car_at(dn,t,w=3.0):
    s=car.get(dn); 
    if not s: return None,None
    i=bisect.bisect_right([x[0] for x in s],t)-1
    return (s[i][1],s[i][2]) if i>=0 and t-s[i][0]<=w else (None,None)
def loc_at(dn,t,w=3.0):
    s=loc.get(dn)
    if not s: return None
    i=bisect.bisect_right([x[0] for x in s],t)-1
    return (s[i][1],s[i][2]) if i>=0 and t-s[i][0]<=w and s[i][1] is not None else None

f={"season":2025.0,"gap_ahead":float(gap)}
series=iv[dn]; times=[s[0] for s in series]
j=bisect.bisect_right(times,t-3.0)-1
if j>=0 and series[j][1] is not None: f["gap_trend"]=float(gap)-float(series[j][1])
p=before(pos.get(dn,[]),t)
if p and p[1] is not None: f["position"]=float(p[1])
lp=before(laps_by.get(dn,[]),t)
if lp and lp[1]:
    cl=lp[1]; f["is_lap1"]=1.0 if cl==1 else 0.0
    st=[s for s in stints_by.get(dn,[]) if s.get("lap_start") and s["lap_start"]<=cl]
    cur=max(st,key=lambda s:s["lap_start"]) if st else None
    if cur: f["tyre_age"]=float((cl-cur["lap_start"])+(cur.get("tyre_age_at_start") or 0))
    el,s1,s2=t-lp[0],lp[2],lp[3]
    if s1: f["sector"]=1.0 if el<s1 else (2.0 if s2 and el<s1+s2 else 3.0)
wi=bisect.bisect_right([w[0] for w in weather],t)-1
if wi>=0:
    w=weather[wi][1]
    for fld in ("air_temperature","track_temperature","humidity","rainfall"):
        if w.get(fld) is not None: f[fld]=float(w[fld])
sp,drs=car_at(dn,t)
if sp is not None:
    f["speed"]=float(sp); f["drs_active"]=1.0 if drs in (10,12,14) else 0.0
    spp,_=car_at(dn,t-3.0)
    if spp is not None: f["speed_delta"]=float(sp)-float(spp)
xy=loc_at(dn,t)
if xy and ref:
    pr=project(ref,xy[0],xy[1]); f["track_progress"]=pr
    f["track_progress_sin"]=math.sin(2*math.pi*pr); f["track_progress_cos"]=math.cos(2*math.pi*pr)
    f["segment"]=float(min(int(pr*30),29))

print(f"시나리오: {int(dn)}번, gap={gap:.2f}s, 시각={datetime.fromtimestamp(t)}")
print(f"\n계산된 피처 {len(f)}/26:")
for n in names:
    v=f.get(n)
    print(f"  {n:24} {'-1.0(결측)' if v is None else round(v,3)}")
vec=[float(f.get(n,-1.0)) for n in names]
raw=raw_score(trees,vec)
print(f"\n부스터 raw score={raw:.3f} (양수↑=추월 가능성↑ · 순위비교용, 절대확률은 실서버 lightgbm에서)")
