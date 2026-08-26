"""Compare the deployed generic overtake model with a Suzuka-specialized model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import train_races


def _episodes(frame: pd.DataFrame, mask: np.ndarray, max_gap_s: float = 2.0) -> list[np.ndarray]:
    selected = frame.loc[mask, ["driver", "t"]].copy()
    selected["_row"] = np.flatnonzero(mask)
    selected["t"] = pd.to_datetime(selected["t"], utc=True)
    episodes = []
    for _, group in selected.sort_values(["driver", "t"]).groupby("driver"):
        split = group["t"].diff().dt.total_seconds().fillna(max_gap_s + 1) > max_gap_s
        for _, episode in group.groupby(split.cumsum()):
            episodes.append(episode["_row"].to_numpy())
    return episodes


def metrics(frame: pd.DataFrame, y: np.ndarray, probability: np.ndarray) -> dict:
    rows = []
    positive_episodes = _episodes(frame, y == 1)
    event_details = []
    for index in positive_episodes:
        episode = frame.iloc[index].sort_values("t")
        event_time = pd.to_datetime(episode["t"].iloc[-1], utc=True) + pd.Timedelta(seconds=1)
        driver = int(episode["driver"].iloc[0])
        position = (
            int(episode["position"].dropna().iloc[0])
            if "position" in episode and not episode["position"].dropna().empty
            else None
        )
        event_probability = probability[index]
        event_details.append({
            "driver": driver,
            "position_before": position,
            "label_window_start": str(pd.to_datetime(episode["t"].iloc[0], utc=True)),
            "estimated_event_time": str(event_time),
            "max_probability": float(np.max(event_probability)),
            "first_peak_time": str(
                pd.to_datetime(episode.iloc[int(np.argmax(event_probability))]["t"], utc=True)
            ),
        })
    for threshold in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        fired = probability >= threshold
        tp = int(np.sum(fired & (y == 1)))
        fp = int(np.sum(fired & (y == 0)))
        fn = int(np.sum(~fired & (y == 1)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        detected_events = sum(bool(fired[index].any()) for index in positive_episodes)
        false_alert_episodes = len(_episodes(frame, fired & (y == 0), max_gap_s=5.0))
        hamilton = [
            event for event in event_details
            if event["driver"] == 44 and event["position_before"] == 8
        ]
        rows.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "fires": int(fired.sum()),
            "event_recall": detected_events / len(positive_episodes) if positive_episodes else 0.0,
            "detected_events": detected_events,
            "actual_events": len(positive_episodes),
            "false_alert_episodes": false_alert_episodes,
            "hamilton_8_to_7_detected": bool(hamilton) and all(
                event["max_probability"] >= threshold for event in hamilton
            ),
        })
    return {
        "rows": int(len(y)),
        "positives": int(y.sum()),
        "pr_auc": float(average_precision_score(y, probability)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "thresholds": rows,
        "event_details": event_details,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--generic-model", required=True)
    parser.add_argument("--specialized-model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    frame = pd.read_parquet(args.samples)
    frame = frame[frame["season"] == 2025].dropna(subset=["label_overtake"]).copy()
    quality, _ = train_races.target_training_mask(frame, "label_overtake", [2025])
    frame = frame[quality]
    x = frame[train_races.FEATS].astype(float).fillna(train_races.MISSING_VALUE_FILL)
    y = frame["label_overtake"].astype(int).to_numpy()

    report = {}
    for name, path in {
        "generic_15_race": args.generic_model,
        "suzuka_2024_h10": args.specialized_model,
    }.items():
        probability = lgb.Booster(model_file=path).predict(x)
        report[name] = metrics(frame, y, probability)

    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
