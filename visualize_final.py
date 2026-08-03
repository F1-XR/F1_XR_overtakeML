"""Animate the 5-circuit final model bundle in a Matplotlib Figure UI.

Run:
    python visualize_final.py
    python visualize_final.py --season 2025 --circuit Sakhir
    python visualize_final.py --season 2025 --circuit Monza --driver 16
    python visualize_final.py --season 2025 --circuit Sakhir --static
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyBboxPatch, Rectangle


RUN = "races_initial_event_type_final"

TARGETS = {
    "label_overtake": "Overtake",
    "label_position_gain": "Gain",
    "label_position_loss": "Loss",
    "label_position_change": "Change",
}
TARGET_COLORS = {
    "label_overtake": "#1f6fba",
    "label_position_gain": "#16864b",
    "label_position_loss": "#c45136",
    "label_position_change": "#7651b4",
}
COMPOUND_COLORS = {
    "SOFT": "#e24b4a",
    "MEDIUM": "#efb227",
    "HARD": "#bbb9ae",
    "INTERMEDIATE": "#3b7d2a",
    "WET": "#185fa5",
    "UNKNOWN": "#8a8a85",
}
TEXT = "#2c2c2a"
MUTED = "#85827a"
LINE = "#dfddd4"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--circuit", default="Sakhir")
    parser.add_argument("--driver", type=int, help="Show one driver through time")
    parser.add_argument("--rank", type=int, default=0, help="0 means highest overtake candidate at each time")
    parser.add_argument("--step", type=int, default=5, help="Use every Nth second for animation")
    parser.add_argument("--max-frames", type=int, help="Limit animation frames by evenly sampling the timeline")
    parser.add_argument("--interval", type=int, default=220, help="Animation interval in ms")
    parser.add_argument("--static", action="store_true", help="Show one selected row instead of animating")
    parser.add_argument("--run", default=RUN)
    parser.add_argument("--sample-path", help="Override sample parquet path; useful for external race tests")
    parser.add_argument("--save", help="Save a static PNG when --static is used")
    parser.add_argument("--gif", help="Save animation as GIF")
    parser.add_argument("--no-show", action="store_true", help="Save without opening a Figure window")
    return parser.parse_args()


def paths(run, sample_path=None):
    root = Path(__file__).resolve().parent
    return {
        "models": root / "data" / "models",
        "samples": Path(sample_path) if sample_path else root / "data" / "processed" / f"{run}_samples.parquet",
    }


def load_bundle(run):
    p = paths(run)
    bundle = {}
    for target in TARGETS:
        meta_path = p["models"] / f"{run}_{target}.json"
        model_path = p["models"] / f"{run}_{target}.txt"
        calibration_path = p["models"] / f"{run}_{target}_calibration.json"
        meta = json.loads(meta_path.read_text())
        calibration = json.loads(calibration_path.read_text())
        bundle[target] = {
            "features": meta["features"],
            "model": lgb.Booster(model_file=str(model_path)),
            "calibration": calibration,
        }
    return bundle


def apply_calibration(raw, calibration):
    x = np.array(calibration.get("raw_thresholds", []), dtype=float)
    y = np.array(calibration.get("display_values", []), dtype=float)
    if len(x) == 0:
        return raw
    return np.interp(raw, x, y)


def load_predictions(args):
    p = paths(args.run, args.sample_path)
    df = pd.read_parquet(p["samples"])
    df = df[
        (df["season"] == args.season)
        & (df["circuit_short_name"].astype(str) == args.circuit)
    ].copy()
    if args.driver is not None:
        df = df[df["driver"] == args.driver].copy()
    if df.empty:
        raise SystemExit(
            f"No samples found: season={args.season}, circuit={args.circuit}, driver={args.driver}"
        )

    bundle = load_bundle(args.run)
    for target, item in bundle.items():
        x = df[item["features"]].astype(float).fillna(-1)
        raw = item["model"].predict(x)
        df[f"raw_{target}"] = raw
        df[f"display_{target}"] = apply_calibration(raw, item["calibration"])
    return df.sort_values("t").reset_index(drop=True)


def pick_row(rows, rank):
    ranked = rows.sort_values(
        ["display_label_overtake", "display_label_position_change"],
        ascending=False,
    )
    rank = max(0, min(rank, len(ranked) - 1))
    return ranked.iloc[rank]


def fmt_num(value, suffix="", digits=1, missing="-"):
    if value is None or pd.isna(value):
        return missing
    return f"{float(value):.{digits}f}{suffix}"


def draw_bar(ax, x, y, w, h, frac, color, label, value_text):
    frac = float(np.clip(frac, 0, 1))
    ax.text(x, y + h * 0.5, label, va="center", ha="left", fontsize=12, color=TEXT)
    bx = x + 2.35
    ax.add_patch(Rectangle((bx, y), w, h, fill=False, ec="#d2d0c7", lw=1.1))
    ax.add_patch(Rectangle((bx, y), w * frac, h, color=color, ec="none"))
    ax.text(
        bx + w + 0.25,
        y + h * 0.5,
        value_text,
        va="center",
        ha="left",
        fontsize=12,
        color=color,
        weight="bold",
    )


def draw_panel(ax, row, args, frame_idx=None, frame_count=None):
    ax.clear()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.add_patch(
        FancyBboxPatch(
            (0.45, 0.35),
            9.1,
            9.25,
            boxstyle="round,pad=0.02,rounding_size=0.18",
            fill=True,
            fc="#fbfbfa",
            ec="#e4e2db",
            lw=1.2,
        )
    )

    t = pd.Timestamp(row["t"]).strftime("%H:%M:%S") if "t" in row else "-"
    driver = int(row["driver"]) if not pd.isna(row.get("driver")) else -1
    pos = int(row["position"]) if not pd.isna(row.get("position")) else -1
    ahead = row.get("ahead_driver", np.nan)
    ahead_txt = f"#{int(ahead)}" if not pd.isna(ahead) else "-"
    compound = str(row.get("compound", "UNKNOWN")).upper()
    compound_color = COMPOUND_COLORS.get(compound, COMPOUND_COLORS["UNKNOWN"])
    frame_txt = f" · frame {frame_idx + 1}/{frame_count}" if frame_count else ""

    ax.text(
        5,
        9.77,
        f"{args.season} {args.circuit} · {t}{frame_txt}",
        ha="center",
        fontsize=10,
        color=MUTED,
    )
    ax.add_patch(Rectangle((0.95, 8.75), 0.85, 0.65, color=compound_color, ec="none"))
    ax.text(1.38, 9.08, str(driver), ha="center", va="center", fontsize=13, color="white", weight="bold")
    ax.text(2.05, 9.08, f"#{driver} · P{pos}", ha="left", va="center", fontsize=18, color=TEXT, weight="bold")
    ax.text(7.25, 9.08, f"Ahead {ahead_txt}", ha="left", va="center", fontsize=12, color=MUTED)

    ax.plot([0.95, 9.05], [8.45, 8.45], color=LINE, lw=1)

    y = 7.82
    for target, label in TARGETS.items():
        prob = float(row[f"display_{target}"])
        draw_bar(ax, 0.95, y, 3.35, 0.2, prob, TARGET_COLORS[target], label, f"{prob * 100:.1f}%")
        y -= 0.62

    ax.plot([0.95, 9.05], [5.12, 5.12], color=LINE, lw=1)

    facts = [
        ("Gap ahead", f"{fmt_num(row.get('gap_ahead'), 's')} · {'DRS' if int(row.get('drs_range', 0) or 0) == 1 else 'No DRS'}"),
        ("Gap trend", fmt_num(row.get("gap_trend"), "s", digits=2)),
        ("Tyre", f"{compound} {fmt_num(row.get('tyre_age'), ' laps', digits=0)}"),
        ("Tyre delta", fmt_num(row.get("tyre_age_delta"), " laps", digits=0)),
        ("Speed", fmt_num(row.get("speed"), " km/h", digits=0)),
        ("Speed delta", fmt_num(row.get("speed_delta"), " km/h", digits=1)),
        (
            "Track",
            f"progress {fmt_num(row.get('track_progress'), digits=3)} · "
            f"sector {fmt_num(row.get('sector'), digits=0)} · "
            f"segment {fmt_num(row.get('segment'), digits=0)}",
        ),
    ]
    y = 4.62
    for label, value in facts:
        ax.text(0.95, y, label, fontsize=11.5, color=MUTED, ha="left", va="center")
        ax.text(3.05, y, value, fontsize=11.5, color=TEXT, ha="left", va="center")
        y -= 0.47

    ax.plot([0.95, 9.05], [1.45, 1.45], color=LINE, lw=1)
    actual = [
        f"actual overtake={int(row.get('label_overtake', 0))}",
        f"gain={int(row.get('label_position_gain', 0))}",
        f"loss={int(row.get('label_position_loss', 0))}",
        f"change={int(row.get('label_position_change', 0))}",
    ]
    ax.text(0.95, 1.08, " · ".join(actual), fontsize=10.5, color=MUTED, ha="left", va="center")
    ax.text(
        0.95,
        0.72,
        "Display probabilities are calibrated. Default animation picks the top overtake candidate.",
        fontsize=9.2,
        color="#9a978f",
        ha="left",
        va="center",
    )


def frame_rows(df, args):
    times = np.array(sorted(df["t"].unique()))
    step = max(1, int(args.step))
    times = times[::step]
    if args.max_frames and len(times) > args.max_frames:
        indices = np.linspace(0, len(times) - 1, args.max_frames).astype(int)
        times = times[indices]
    rows = []
    for t in times:
        group = df[df["t"] == t]
        if not group.empty:
            rows.append(pick_row(group, args.rank))
    return rows


def show_static(df, args):
    row = pick_row(df, args.rank)
    print(f"rows={len(df)} selected driver=#{int(row['driver'])} P{int(row['position'])}")
    for target, label in TARGETS.items():
        print(f"{label}: raw={row[f'raw_{target}']:.4f} display={row[f'display_{target}']:.4f}")

    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(7.0, 6.6))
    draw_panel(ax, row, args)
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=160)
        print(f"saved: {args.save}")
    if not args.no_show:
        plt.show()


def show_animation(df, args):
    rows = frame_rows(df, args)
    if not rows:
        raise SystemExit("No animation frames found.")

    first = rows[0]
    print(
        f"frames={len(rows)} rows={len(df)} first driver=#{int(first['driver'])} "
        f"P{int(first['position'])}"
    )
    print("Close the Figure window to end.")

    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(7.0, 6.6))

    def update(i):
        draw_panel(ax, rows[i], args, i, len(rows))
        return []

    anim = FuncAnimation(fig, update, frames=len(rows), interval=args.interval, repeat=True)
    fig.tight_layout()
    if args.gif:
        anim.save(args.gif, writer="pillow", fps=max(1, int(1000 / args.interval)))
        print(f"saved: {args.gif}")
    if not args.no_show:
        plt.show()


def main():
    args = parse_args()
    df = load_predictions(args)
    if args.static:
        show_static(df, args)
    else:
        show_animation(df, args)


if __name__ == "__main__":
    main()
