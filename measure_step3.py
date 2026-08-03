"""3단계 검증 — car_data(speed·drs·speed_delta) + sector 를 '실제 값'으로 채우면
축소 모델이 gap baseline을 넘는지 측정. (track_progress 기하는 아직 -1.0 유지)

measure_reduced.py 와 동일 셋업 + car_data/sector 추가. lightgbm 없이 부스터 직접 순회.
"""
import json, math, bisect
from pathlib import Path
from datetime import datetime

RAW = Path("data/raw/2025_spa_race")
MODEL = Path("data/models/races_initial_event_type_final_label_overtake.txt")
CAL = Path("data/models/races_initial_event_type_final_label_overtake_calibration.json")
HORIZON, BATTLE_GAP, STEP = 30.0, 2.0, 3

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
                      "lc": A("left_child", int), "rc": A("right_child", int),
                      "lv": A("leaf_value", float)})
    names = [l for l in txt.splitlines() if l.startswith("feature_names=")][0].split("=", 1)[1].split()
    return names, trees

def tout(t, x):
    n = 0
    while True:
        c = t["lc"][n] if x[t["sf"][n]] <= t["th"][n] else t["rc"][n]
        if c < 0: return t["lv"][~c]
        n = c
def raw_score(trees, x): return sum(tout(t, x) for t in trees)
def sigmoid(z): return 1/(1+math.exp(-z)) if z > -700 else 0.0

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

def roc_auc(sc, y):
    pr = sorted(zip(sc, y)); rk = [0.0]*len(pr); i = 0
    while i < len(pr):
        j = i
        while j+1 < len(pr) and pr[j+1][0] == pr[i][0]: j += 1
        for k in range(i, j+1): rk[k] = (i+j)/2.0+1
        i = j+1
    pos = sum(y); neg = len(y)-pos
    if not pos or not neg: return None
    sp = sum(r for r, (_, l) in zip(rk, pr) if l == 1)
    return (sp - pos*(pos+1)/2)/(pos*neg)
def pr_auc(sc, y):
    o = sorted(range(len(sc)), key=lambda i: -sc[i]); tp = fp = 0; pos = sum(y); ap = pr = 0.0
    if not pos: return None
    for i in o:
        if y[i]: tp += 1
        else: fp += 1
        rec = tp/pos; ap += (tp/(tp+fp))*(rec-pr); pr = rec
    return ap

def main():
    names, trees = parse_booster(MODEL)
    cal = json.load(open(CAL)); rt, dv = cal["raw_thresholds"], cal["display_values"]
    fill = -1.0

    iv = idx_driver(load("intervals"), ["interval"])
    pos = idx_driver(load("position"), ["position"])
    weather = sorted([(ts(w["date"]), w) for w in load("weather")])
    stints_by = {}
    for s in load("stints"): stints_by.setdefault(s["driver_number"], []).append(s)
    laps_by = {}
    for r in load("laps"):
        if r.get("date_start"):
            laps_by.setdefault(r["driver_number"], []).append(
                (ts(r["date_start"]), r.get("lap_number"),
                 r.get("duration_sector_1"), r.get("duration_sector_2")))
    for k in laps_by: laps_by[k].sort()

    # car_data (대용량) → 드라이버별 (t, speed, drs) 인덱스만 남기고 원본 버림
    print("car_data 로딩 중(대용량)…")
    car = idx_driver(load("car_data"), ["speed", "drs"])
    print("car_data 인덱스 완료:", sum(len(v) for v in car.values()), "포인트")

    def car_at(dn, t, within=3.0):
        s = car.get(dn)
        if not s: return None, None
        i = bisect.bisect_right([x[0] for x in s], t) - 1
        if i < 0: return None, None
        rec = s[i]
        if t - rec[0] > within: return None, None
        return rec[1], rec[2]   # speed, drs

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
                cur_lap = lp[1]; f["is_lap1"] = 1.0 if cur_lap == 1 else 0.0
                cur = None
                for s in stints_by.get(dn, []):
                    ls, le = s.get("lap_start"), s.get("lap_end")
                    if ls and ls <= cur_lap and (le is None or cur_lap <= le): cur = s
                if cur is None:
                    st = [s for s in stints_by.get(dn, []) if s.get("lap_start") and s["lap_start"] <= cur_lap]
                    cur = max(st, key=lambda s: s["lap_start"]) if st else None
                if cur: f["tyre_age"] = float((cur_lap-cur["lap_start"])+(cur.get("tyre_age_at_start") or 0))
                # sector: 랩 시작 이후 경과시간 vs 섹터 시간
                lap_start, s1, s2 = lp[0], lp[2], lp[3]
                el = t - lap_start
                if s1 is not None:
                    if el < s1: f["sector"] = 1.0
                    elif s2 is not None and el < s1+s2: f["sector"] = 2.0
                    else: f["sector"] = 3.0
            wi = bisect.bisect_right([w[0] for w in weather], t)-1
            if wi >= 0:
                w = weather[wi][1]
                for fld in ("air_temperature", "track_temperature", "humidity", "rainfall"):
                    if w.get(fld) is not None: f[fld] = float(w[fld])
            # ── car_data: speed, drs_active, speed_delta ──
            sp, drs = car_at(dn, t)
            if sp is not None:
                f["speed"] = float(sp)
                f["drs_active"] = 1.0 if drs in (10, 12, 14) else 0.0
                sp_prev, _ = car_at(dn, t-3.0)
                if sp_prev is not None: f["speed_delta"] = float(sp)-float(sp_prev)
            x = [float(f.get(n, fill)) for n in names]
            pn = p[1] if p else None
            if pn is None: continue
            imp = 0
            for pt, pv in pos.get(dn, []):
                if t < pt <= t+HORIZON and pv is not None and pv < pn: imp = 1; break
            X.append(x); y.append(imp); gaponly.append(-float(gap))

    import numpy as np
    raws = [raw_score(trees, x) for x in X]
    base = sum(y)/len(y)
    print(f"\n샘플 {len(y)}개 · 양성 {sum(y)} ({100*base:.1f}%)")
    print(f"{'':22}{'ROC-AUC':>9}{'PR-AUC':>9}")
    fm = lambda v: f"{v:9.3f}" if v is not None else f"{'n/a':>9}"
    print(f"{'축소+car_data+sector':22}{fm(roc_auc(raws,y))}{fm(pr_auc(raws,y))}")
    print(f"{'gap만 (baseline)':22}{fm(roc_auc(gaponly,y))}{fm(pr_auc(gaponly,y))}")
    print(f"{'(이전 축소만: 0.745 / 0.084)':22}")

if __name__ == "__main__":
    main()
