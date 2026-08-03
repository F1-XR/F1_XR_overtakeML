"""예측을 통합 패널(이벤트 UI)로 재생한다.
사전조건: python pipeline.py 실행 -> data/processed/samples.parquet
실행: python visualize.py         (창)
      python visualize.py --gif   (event_ui.gif 저장)
각 순간 '추월 확률이 가장 높은 차'의 통합 패널을 보여준다.
추월/타이어/간격/예상랩타임 = 실제 값. 언더컷 = 규칙 기반 추정(여러 경기 모으면 학습 교체).
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, FancyBboxPatch
from lightgbm import LGBMClassifier

import config as C

FEATS = ["gap_ahead", "gap_trend", "position", "tyre_age", "tyre_delta", "drs_range"]
COMP_KR = {"SOFT": "소프트", "MEDIUM": "미디엄", "HARD": "하드",
           "INTERMEDIATE": "인터", "WET": "웻", "UNKNOWN": "미상"}
COMP_COLOR = {"SOFT": "#E24B4A", "MEDIUM": "#EF9F27", "HARD": "#B4B2A9",
              "INTERMEDIATE": "#3B6D11", "WET": "#185FA5", "UNKNOWN": "#888780"}
HOT = "#D85A30"; GRAY = "#B4B2A9"; MUT = "#8A8A85"; INK = "#2C2C2A"


def load_predictions():
    df = pd.read_parquet(C.PROC_DIR / "samples.parquet")
    X = df[FEATS].astype(float).fillna(-1)
    y = df["label"].astype(int)
    if y.nunique() < 2:
        print("positive 부족 - HORIZON 늘리거나 경기 추가."); sys.exit(1)
    m = LGBMClassifier(n_estimators=300, learning_rate=0.05,
                       class_weight="balanced", verbose=-1)
    m.fit(X, y)
    df["prob"] = m.predict_proba(X)[:, 1]
    p = df["prob"]
    print(f"[prob] min={p.min():.2f} mean={p.mean():.2f} max={p.max():.2f} "
          f"| >0.5: {(p>0.5).sum()} rows")
    return df


def _wear(age):
    if not np.isfinite(age):
        return "미상", 0.3
    lvl = "높음" if age >= 25 else ("보통" if age >= 15 else "낮음")
    return lvl, min(age / 35.0, 1.0)


def _bar(ax, x, y, w, frac, color):
    ax.add_patch(Rectangle((x, y - 0.11), w, 0.22, fill=False, ec="#D3D1C7", lw=1))
    ax.add_patch(Rectangle((x, y - 0.11), w * max(0, min(frac, 1)), 0.22, color=color))


def animate(df, save_gif=False):
    times = np.array(sorted(df["t"].unique()))[::5]
    fig, ax = plt.subplots(figsize=(5.4, 5.4))

    def draw(t):
        ax.clear(); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
        rows = df[df["t"] == t]
        if len(rows) == 0:
            return
        r = rows.sort_values("prob", ascending=False).iloc[0]
        comp = str(r.get("compound", "UNKNOWN")).upper()
        age = float(r.get("tyre_age", np.nan))
        prob = float(r["prob"]); gap = float(r["gap_ahead"])
        drs = int(r.get("drs_range", 0)) == 1
        wlvl, wfrac = _wear(age)

        ax.add_patch(FancyBboxPatch((0.3, 0.4), 9.4, 9.2, boxstyle="round,pad=0.02,rounding_size=0.25",
                                    fill=True, fc="#FBFBFA", ec="#E3E1DA", lw=1))
        ax.text(5, 9.75, pd.Timestamp(t).strftime("%H:%M:%S"), ha="center", fontsize=9, color=MUT)
        # 헤더
        ax.add_patch(Rectangle((0.9, 8.55), 0.9, 0.9, color=COMP_COLOR.get(comp, GRAY)))
        ax.text(1.35, 9.0, str(int(r["driver"])), ha="center", va="center", fontsize=13,
                color="white", weight="bold")
        ax.text(2.1, 9.0, f"#{int(r['driver'])} · P{int(r['position'])}", va="center",
                fontsize=15, color=INK, weight="bold")

        y = 7.9
        ax.text(0.9, y, "타이어", fontsize=11, color=MUT)
        ax.text(3.0, y, f"{COMP_KR.get(comp,'미상')} {int(age) if np.isfinite(age) else '-'}랩 · 마모 {wlvl}",
                fontsize=11, color=INK)
        _bar(ax, 7.0, y, 2.3, wfrac, COMP_COLOR.get(comp, GRAY))

        ax.plot([0.9, 9.3], [7.35, 7.35], color="#E3E1DA", lw=1)

        y = 6.7
        ax.text(0.9, y, "안 들어가면", fontsize=11, color=MUT)
        ax.text(3.0, y, "추월 위험", fontsize=11, color=INK)
        _bar(ax, 5.6, y, 2.6, prob, HOT if prob > 0.5 else GRAY)
        ax.text(8.4, y, f"{prob*100:.0f}%", va="center", fontsize=11,
                color="#993C1D" if prob > 0.5 else MUT, weight="bold")

        y = 5.7
        uc = float(r.get("undercut_prob", np.nan))
        ax.text(0.9, y, "지금 들어가면", fontsize=11, color=MUT)
        ax.text(3.0, y, "언더컷 성공", fontsize=11, color=INK)
        if np.isfinite(uc):
            _bar(ax, 5.6, y, 2.6, uc, "#185FA5")
            ax.text(8.4, y, f"{uc*100:.0f}%", va="center", fontsize=11, color="#0C447C", weight="bold")
        else:
            ax.text(5.6, y, "(데이터 부족)", fontsize=10, color="#B9B7B0", style="italic")

        y = 4.8
        ax.text(0.9, y, "앞차 간격", fontsize=11, color=MUT)
        ax.text(3.0, y, f"{gap:.1f}s" + ("  · DRS" if drs else ""), fontsize=11, color=INK)

        y = 4.0
        dl = float(r.get("laptime_delta", np.nan))
        ax.text(0.9, y, "예상 랩타임", fontsize=11, color=MUT)
        if np.isfinite(dl):
            ax.text(3.0, y, f"2랩 뒤 {dl:+.1f}s", fontsize=11, color=INK)
        else:
            ax.text(3.0, y, "(데이터 부족)", fontsize=10, color="#B9B7B0", style="italic")

        if prob > 0.5 and wlvl == "높음":
            ax.add_patch(FancyBboxPatch((0.9, 2.7), 8.4, 0.8, boxstyle="round,pad=0.02,rounding_size=0.15",
                                        fc="#FAECE7", ec="none"))
            ax.text(1.3, 3.1, "→  타이어 마모로 추월 위험", fontsize=11.5, color="#993C1D", weight="bold")

    anim = FuncAnimation(fig, draw, frames=times, interval=200)
    if save_gif:
        anim.save("event_ui.gif", writer="pillow", fps=5); print("event_ui.gif 저장")
    else:
        plt.show()


if __name__ == "__main__":
    animate(load_predictions(), save_gif="--gif" in sys.argv)
