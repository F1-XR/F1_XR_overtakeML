"""4단계 검증 — step3(car_data+sector)에 track_progress(위치 기하)까지 추가하면
축소 모델이 gap baseline을 넘는지 측정. lightgbm/sklearn 없이 부스터 순회 + numpy 최근접.

트랙 기준선: 깨끗한 한 바퀴 location으로 1회 구성(경기당 1번) → 각 시점 (x,y) 최근접 투영.
"""
import json, math, bisect
from pathlib import Path
from datetime import datetime
import numpy as np

RAW = Path("data/raw/2025_spa_race")
MODEL = Path("data/models/races_initial_event_type_final_label_overtake.txt")
CAL = Path("data/models/races_initial_event_type_final_label_overtake_calibration.json")
HORIZON, BATTLE_GAP, STEP, SEGMENTS = 30.0, 2.0, 3, 30

def ts(s): return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
def load(n): return json.load(open(RAW / f"{n}.json"))

def parse_booster(path):
    txt = path.read_text(); trees = []
    for block in txt.split("Tree=")[1:]:
        d = {}
        for line in block.splitlines():
            if "=" in line:
                k, v = line.split("=", 1); d[k.strip()] = v.strip()
        A = lambda k, f: [f(x) for x in d[k].split()] if d.get(k) else []
        trees.append({"sf": A("split_feature", int), "th": A("threshold", float),
                      "lc": A("left_child", int), "rc": A("right_child", int), "lv": A("leaf_value", float)})
    names = [l for l in txt.splitlines() if l.startswith("feature_names=")][0].split("=", 1)[1].split()
    return names, trees

def tout(t, x):
    n = 0
    while True:
        c = t["lc"][n] if x[t["sf"][n]] <= t["th"][n] else t["rc"][n]
        if c < 0: return t["lv"][~c]
        n = c
def raw_score(trees, x): return sum(tout(t, x) for t in trees)

def idx_driver(rows, keys):
    d = {}
    for r in rows:
        dn = r.get("driver_number")
        if dn is None or not r.get("date"): continue
        d.setdefault(dn, []).append((ts(r["date"]), *[r.get(k) for k in keys]))
    for k in d: d[k].sort()
    return d
def before(series, t):
    i = bisect.bisect_right([s[0] for s in series], t) - 1
    return series[i] if i >= 0 else None

# ── 트랙 기준선 ──
def build_reference(loc_by_driver, laps_rows):
    # 깨끗한 랩(lap>1, pit-out 아님) 하나의 location 구간으로 기준선 구성
    laps_by = {}
    for r in laps_rows:
        if r.get("date_start") and r.get("lap_duration") and r.get("lap_number", 0) > 1 and not r.get("is_pit_out_lap"):
            st = ts(r["date_start"]); laps_by.setdefault(r["driver_number"], []).append((st, st + float(r["lap_duration"])))
    for dn, laps in laps_by.items():
        series = loc_by_driver.get(dn)
        if not series: continue
        times = [s[0] for s in series]
        for st, en in sorted(laps):
            lo = bisect.bisect_left(times, st); hi = bisect.bisect_right(times, en)
            pts = [(series[i][1], series[i][2]) for i in range(lo, hi)]
            pts = [(x, y) for x, y in pts if x is not None and y is not None]
            if len(pts) < 200: continue
            xy = np.array(pts, dtype=float)
            if len(xy) > 1200: xy = xy[np.linspace(0, len(xy)-1, 1200).astype(int)]
            step = np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-6]
            xy = xy[step]
            if len(xy) < 50: continue
            dist = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
            if dist[-1] <= 0: continue
            print(f"[ref] driver={dn} points={len(xy)}")
            return {"xy": xy, "progress": dist/dist[-1]}
    return None

def project(ref, x, y):
    d2 = ((ref["xy"] - np.array([x, y]))**2).sum(1)
    return float(ref["progress"][int(d2.argmin())])

def roc_auc(sc, y):
    pr = sorted(zip(sc, y)); rk = [0.0]*len(pr); i = 0
    while i < len(pr):
        j = i
        while j+1 < len(pr) and pr[j+1][0] == pr[i][0]: j += 1
        for k in range(i, j+1): rk[k] = (i+j)/2.0+1
        i = j+1
    pos = sum(y); neg = len(y)-pos
    if not pos or not neg: return None
    return (sum(r for r, (_, l) in zip(rk, pr) if l == 1) - pos*(pos+1)/2)/(pos*neg)
def pr_auc(sc, y):
    o = sorted(range(len(sc)), key=lambda i: -sc[i]); tp=fp=0; pos=sum(y); ap=p=0.0
    if not pos: return None
    for i in o:
        tp += y[i]; fp += (1-y[i]); rec = tp/pos; ap += (tp/(tp+fp))*(rec-p); p = rec
    return ap

def main():
    names, trees = parse_booster(MODEL)
    cal = json.load(open(CAL)); rt, dv = cal["raw_thresholds"], cal["display_values"]; fill = -1.0

    iv = idx_driver(load("intervals"), ["interval"])
    pos = idx_driver(load("position"), ["position"])
    weather = sorted([(ts(w["date"]), w) for w in load("weather")])
    stints_by = {}
    for s in load("stints"): stints_by.setdefault(s["driver_number"], []).append(s)
    laps_rows = load("laps")
    laps_by = {}
    for r in laps_rows:
        if r.get("date_start"):
            laps_by.setdefault(r["driver_number"], []).append(
                (ts(r["date_start"]), r.get("lap_number"), r.get("duration_sector_1"), r.get("duration_sector_2")))
    for k in laps_by: laps_by[k].sort()

    print("car_data 로딩…"); car = idx_driver(load("car_data"), ["speed", "drs"])
    print("location 로딩…"); loc = idx_driver(load("location"), ["x", "y"])
    ref = build_reference(loc, laps_rows)
    if ref is None: print("기준선 실패"); return

    def car_at(dn, t, w=3.0):
        s = car.get(dn)
        if not s: return None, None
        i = bisect.bisect_right([x[0] for x in s], t)-1
        if i < 0 or t-s[i][0] > w: return None, None
        return s[i][1], s[i][2]
    def loc_at(dn, t, w=3.0):
        s = loc.get(dn)
        if not s: return None
        i = bisect.bisect_right([x[0] for x in s], t)-1
        if i < 0 or t-s[i][0] > w or s[i][1] is None: return None
        return (s[i][1], s[i][2])

    X, y, gaponly = [], [], []
    for dn, series in iv.items():
        times = [s[0] for s in series]
        for k in range(0, len(series), STEP):
            t, gap = series[k]
            if gap is None or not (0 < gap <= BATTLE_GAP): continue
            f = {"season": 2025.0, "gap_ahead": float(gap)}
            j = bisect.bisect_right(times, t-3.0)-1
            if j >= 0 and series[j][1] is not None: f["gap_trend"] = float(gap)-float(series[j][1])
            p = before(pos.get(dn, []), t)
            if p and p[1] is not None: f["position"] = float(p[1])
            lp = before(laps_by.get(dn, []), t)
            if lp and lp[1] is not None:
                cl = lp[1]; f["is_lap1"] = 1.0 if cl == 1 else 0.0
                cur = None
                for s in stints_by.get(dn, []):
                    ls, le = s.get("lap_start"), s.get("lap_end")
                    if ls and ls <= cl and (le is None or cl <= le): cur = s
                if cur is None:
                    st = [s for s in stints_by.get(dn, []) if s.get("lap_start") and s["lap_start"] <= cl]
                    cur = max(st, key=lambda s: s["lap_start"]) if st else None
                if cur: f["tyre_age"] = float((cl-cur["lap_start"])+(cur.get("tyre_age_at_start") or 0))
                el, s1, s2 = t-lp[0], lp[2], lp[3]
                if s1 is not None:
                    f["sector"] = 1.0 if el < s1 else (2.0 if (s2 and el < s1+s2) else 3.0)
            wi = bisect.bisect_right([w[0] for w in weather], t)-1
            if wi >= 0:
                w = weather[wi][1]
                for fld in ("air_temperature", "track_temperature", "humidity", "rainfall"):
                    if w.get(fld) is not None: f[fld] = float(w[fld])
            sp, drs = car_at(dn, t)
            if sp is not None:
                f["speed"] = float(sp); f["drs_active"] = 1.0 if drs in (10,12,14) else 0.0
                spp, _ = car_at(dn, t-3.0)
                if spp is not None: f["speed_delta"] = float(sp)-float(spp)
            # ── track_progress ──
            xy = loc_at(dn, t)
            if xy is not None:
                pr = project(ref, xy[0], xy[1])
                f["track_progress"] = pr
                f["track_progress_sin"] = math.sin(2*math.pi*pr)
                f["track_progress_cos"] = math.cos(2*math.pi*pr)
                f["segment"] = float(min(int(pr*SEGMENTS), SEGMENTS-1))
            x = [float(f.get(n, fill)) for n in names]
            pn = p[1] if p else None
            if pn is None: continue
            imp = 0
            for pt, pv in pos.get(dn, []):
                if t < pt <= t+HORIZON and pv is not None and pv < pn: imp = 1; break
            X.append(x); y.append(imp); gaponly.append(-float(gap))

    raws = [raw_score(trees, xx) for xx in X]
    base = sum(y)/len(y)
    fm = lambda v: f"{v:9.3f}" if v is not None else f"{'n/a':>9}"
    print(f"\n샘플 {len(y)}개 · 양성 {sum(y)} ({100*base:.1f}%)")
    print(f"{'':26}{'ROC-AUC':>9}{'PR-AUC':>9}")
    print(f"{'축소+car+sector+track':26}{fm(roc_auc(raws,y))}{fm(pr_auc(raws,y))}")
    print(f"{'gap만 (baseline)':26}{fm(roc_auc(gaponly,y))}{fm(pr_auc(gaponly,y))}")
    print("(이전: 축소만 0.745/0.084 · +car+sector 0.780/0.188)")

if __name__ == "__main__":
    main()
