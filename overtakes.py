"""실제 순위 상승(추월 추정) 순간을 시각과 함께 목록으로 출력한다.
사전조건: python pipeline.py 를 한 번 돌려 data/raw/position.json 이 있어야 함.
실행: python overtakes.py
주의: position 기준 '순위 상승'이라 피트/리타이어로 인한 상승도 섞일 수 있음(대략 확인용).
"""
import json
import pandas as pd
import config as C


def main():
    pos = pd.DataFrame(json.loads((C.RAW_DIR / "position.json").read_text()))
    pos["date"] = pd.to_datetime(pos["date"], utc=True).dt.tz_localize(None)
    pos = pos.dropna(subset=["date"]).sort_values(["driver_number", "date"])
    t0 = pos["date"].min()

    events = []
    for dn, g in pos.groupby("driver_number"):
        prev = None
        for _, r in g.iterrows():
            if prev is not None and r["position"] < prev:   # 순위 숫자 감소 = 상승
                sec = (r["date"] - t0).total_seconds()
                events.append((r["date"], sec, int(dn), int(prev), int(r["position"])))
            prev = r["position"]
    events.sort()

    print(f"경기 시작 기준. 총 순위상승(추월 추정) {len(events)}건\n")
    print(f"{'시각':>10}  {'경기시간':>8}   차량   순위변화")
    print("-" * 44)
    for dt, sec, dn, frm, to in events:
        mmss = f"{int(sec // 60)}:{int(sec % 60):02d}"
        print(f"  {dt.strftime('%H:%M:%S')}   {mmss:>6}   #{dn:<3}  P{frm} -> P{to}")


if __name__ == "__main__":
    main()
