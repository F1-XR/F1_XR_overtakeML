"""축소 피처(튜토리얼 배관) 성능 측정 — LightGBM lib 없이 부스터 트리를 직접 순회.

목적: F1_XR_AI가 계산 가능한 ~10개 피처만 진짜 값으로 주고 나머지 16개를 -1.0으로 채웠을 때,
      추월 예측 모델이 실제 경기(Spa 2025)에서 얼마나 순위를 매기는지(ROC-AUC/PR-AUC) 측정.

주의:
  - ROC-AUC/PR-AUC는 '순위' 지표라 부스터의 average-init 상수와 무관(순위 불변) → lib 없이도 정확.
  - 라벨은 position.json 기반 '30초 내 순위 상승' 프록시(모델의 엄격한 on-track 라벨과 다름).
    따라서 절대 수치는 공식 held-out(0.853)과 직접 비교 금지, '축소 재료로 신호가 남아있나'의 지표로 본다.
"""
import json, math, bisect
from pathlib import Path

RAW = Path("data/raw/2025_spa_race")
MODEL = Path("data/models/races_initial_event_type_final_label_overtake.txt")
CAL = Path("data/models/races_initial_event_type_final_label_overtake_calibration.json")
HORIZON = 30.0      # 라벨: 앞으로 30초
BATTLE_GAP = 2.0    # 배틀 상황(앞차 gap ≤ 2s)만
STEP = 3            # 배틀 레코드 subsample 간격

def load(name): return json.load(open(RAW / f"{name}.json"))
def ts(s):      # ISO → epoch seconds
    from datetime import datetime
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()

# ── 부스터 파싱 ───────────────────────────────────────────
def parse_booster(path):
    txt = path.read_text()
    feat_names = None
    trees = []
    for block in txt.split("Tree=")[1:]:
        d = {}
        for line in block.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                d[k.strip()] = v.strip()
        def arr(k, f): return [f(x) for x in d[k].split()] if d.get(k) else []
        trees.append({
            "split_feature": arr("split_feature", int),
            "threshold":     arr("threshold", float),
            "left_child":    arr("left_child", int),
            "right_child":   arr("right_child", int),
            "leaf_value":    arr("leaf_value", float),
        })
    m = [l for l in txt.splitlines() if l.startswith("feature_names=")][0]
    feat_names = m.split("=", 1)[1].split()
    return feat_names, trees

def tree_out(t, x):
    node = 0
    while True:
        f = t["split_feature"][node]
        go_left = x[f] <= t["threshold"][node]
        child = t["left_child"][node] if go_left else t["right_child"][node]
        if child < 0:
            return t["leaf_value"][~child]   # ~child = -child-1 = leaf idx
        node = child

def raw_score(trees, x):
    return sum(tree_out(t, x) for t in trees)

# ── 시계열 인덱스 ─────────────────────────────────────────
def by_driver_sorted(rows, val_keys):
    idx = {}
    for r in rows:
        d = r.get("driver_number")
        if d is None or not r.get("date"): continue
        idx.setdefault(d, []).append((ts(r["date"]), *[r.get(k) for k in val_keys]))
    for d in idx: idx[d].sort()
    return idx

def latest_before(series, t):   # series: list of (time, ...)
    i = bisect.bisect_right([s[0] for s in series], t) - 1
    return series[i] if i >= 0 else None

def main():
    feat_names, trees = parse_booster(MODEL)
    cal = json.load(open(CAL)); rt = cal["raw_thresholds"]; dv = cal["display_values"]
    print(f"부스터: {len(trees)} trees · 피처 {len(feat_names)}개")

    intervals = load("intervals")   # gap_ahead
    position  = load("position")
    weather   = sorted([(ts(w["date"]), w) for w in weather_rows(load("weather"))])
    stints    = load("stints")
    laps      = load("laps")

    iv_idx = by_driver_sorted(intervals, ["interval"])
    pos_idx = by_driver_sorted(position, ["position"])
    laps_idx = {}
    for r in laps:
        d = r.get("driver_number")
        if d is None or not r.get("date_start"): continue
        laps_idx.setdefault(d, []).append((ts(r["date_start"]), r.get("lap_number")))
    for d in laps_idx: laps_idx[d].sort()
    stints_by = {}
    for s in stints: stints_by.setdefault(s["driver_number"], []).append(s)

    # season 상수
    SEASON = 2025.0
    fill = -1.0
    order = feat_names

    X, y, gap_only = [], [], []
    for d, series in iv_idx.items():
        for k in range(0, len(series), STEP):
            t, gap = series[k]
            if gap is None or not (0 < gap <= BATTLE_GAP):
                continue
            # ── 피처 ──
            feats = {"season": SEASON, "gap_ahead": float(gap)}
            # gap_trend: 3초 전 gap
            j = bisect.bisect_right([s[0] for s in series], t - 3.0) - 1
            if j >= 0 and series[j][1] is not None:
                feats["gap_trend"] = float(gap) - float(series[j][1])
            # position
            p = latest_before(pos_idx.get(d, []), t)
            if p and p[1] is not None: feats["position"] = float(p[1])
            # current lap → is_lap1, tyre_age
            lp = latest_before(laps_idx.get(d, []), t)
            cur_lap = lp[1] if lp else None
            if cur_lap is not None:
                feats["is_lap1"] = 1.0 if cur_lap == 1 else 0.0
                cur = None
                for s in stints_by.get(d, []):
                    ls, le = s.get("lap_start"), s.get("lap_end")
                    if ls is not None and ls <= cur_lap and (le is None or cur_lap <= le): cur = s
                if cur is None:
                    st = [s for s in stints_by.get(d, []) if s.get("lap_start") and s["lap_start"] <= cur_lap]
                    cur = max(st, key=lambda s: s["lap_start"]) if st else None
                if cur:
                    feats["tyre_age"] = float((cur_lap - cur["lap_start"]) + (cur.get("tyre_age_at_start") or 0))
            # weather
            wi = bisect.bisect_right([w[0] for w in weather], t) - 1
            if wi >= 0:
                w = weather[wi][1]
                for f in ("air_temperature", "track_temperature", "humidity", "rainfall"):
                    if w.get(f) is not None: feats[f] = float(w[f])
            # ── 벡터(축소: 나머지 -1.0) ──
            x = [float(feats.get(n, fill)) for n in order]
            # ── 라벨: 30초 내 순위 상승(프록시) ──
            pos_now = p[1] if p else None
            if pos_now is None: continue
            improved = 0
            for pt, pv in pos_idx.get(d, []):
                if t < pt <= t + HORIZON and pv is not None and pv < pos_now:
                    improved = 1; break
            X.append(x); y.append(improved); gap_only.append(-float(gap))
    return evaluate(trees, rt, dv, order, X, y, gap_only)

def weather_rows(w): return w

def sigmoid(z): return 1/(1+math.exp(-z)) if z > -700 else 0.0

def roc_auc(scores, labels):
    pairs = sorted(zip(scores, labels))
    # rank-sum (Mann-Whitney U); ties averaged
    ranks = [0.0]*len(pairs); i = 0
    while i < len(pairs):
        j = i
        while j+1 < len(pairs) and pairs[j+1][0] == pairs[i][0]: j += 1
        avg = (i + j)/2.0 + 1
        for k in range(i, j+1): ranks[k] = avg
        i = j+1
    pos = sum(labels); neg = len(labels) - pos
    if pos == 0 or neg == 0: return None
    sum_pos = sum(r for r, (_, l) in zip(ranks, pairs) if l == 1)
    return (sum_pos - pos*(pos+1)/2) / (pos*neg)

def pr_auc(scores, labels):  # average precision
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    tp = fp = 0; pos = sum(labels); ap = 0.0; prev_r = 0.0
    if pos == 0: return None
    for i in order:
        if labels[i] == 1: tp += 1
        else: fp += 1
        prec = tp/(tp+fp); rec = tp/pos
        ap += prec*(rec-prev_r); prev_r = rec
    return ap

def evaluate(trees, rt, dv, order, X, y, gap_only):
    import numpy as np
    raws = [raw_score(trees, x) for x in X]
    disp = [float(np.interp(sigmoid(r), rt, dv)) for r in raws]  # 순위엔 영향 없음
    base = sum(y)/len(y) if y else 0
    print(f"\n샘플 {len(y)}개 · 양성(30초내 순위상승) {sum(y)} ({100*base:.1f}%)")
    print(f"{'':16}{'ROC-AUC':>9}{'PR-AUC':>9}")
    print(f"{'축소 모델':16}{fmt(roc_auc(raws,y))}{fmt(pr_auc(raws,y))}")
    print(f"{'gap만(baseline)':16}{fmt(roc_auc(gap_only,y))}{fmt(pr_auc(gap_only,y))}")
    print(f"{'무작위 기준':16}{0.5:9.3f}{base:9.3f}")
    return

def fmt(v): return f"{v:9.3f}" if v is not None else f"{'n/a':>9}"

if __name__ == "__main__":
    main()
