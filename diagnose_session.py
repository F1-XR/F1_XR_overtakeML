"""Rebuild event_type labels for one cached/live session without retraining models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests

import config as C
import pipeline


RESOURCES = ("drivers", "position", "intervals", "laps", "stints", "pit", "race_control", "weather")


def fetch_json(url: str) -> list[dict]:
    response = requests.get(url, timeout=90)
    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=int, required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    tag = f"diagnostic_{args.session}"
    C.SEASON = args.year
    C.RAW_DIR = C.DATA_DIR / "raw" / tag
    C.PROC_DIR = C.DATA_DIR / "processed" / tag
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    C.PROC_DIR.mkdir(parents=True, exist_ok=True)

    sessions = fetch_json(f"{args.server}/f1/sessions?year={args.year}")
    session = next(s for s in sessions if int(s["session_key"]) == args.session)
    meta = {
        "season": args.year,
        "session_key": args.session,
        "meeting_key": session.get("meeting_key"),
        "circuit_key": session.get("circuit_key"),
        "circuit_short_name": session.get("circuit_short_name"),
        "circuit_type": "Unknown",
        "session": session.get("session_name", "Race"),
        "country": session.get("country_name", ""),
    }
    (C.RAW_DIR / "session_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for resource in RESOURCES:
        path = C.RAW_DIR / f"{resource}.json"
        if not path.exists():
            rows = fetch_json(f"{args.server}/f1/{args.session}/{resource}")
            path.write_text(json.dumps(rows), encoding="utf-8")
            print(f"[collect] {resource}: {len(rows)}")

    frame = pipeline.build_dataset()
    positives = frame[frame["event_type"] == pipeline.EVENT_TYPE_BY_NAME["on_track_overtake"]]
    summary = {
        "session_key": args.session,
        "rows": int(len(frame)),
        "on_track_positive_rows": int(len(positives)),
        "positive_drivers": sorted(int(v) for v in positives["driver"].unique()),
        "event_type_counts": {
            str(k): int(v) for k, v in frame["change_reason"].value_counts().to_dict().items()
        },
    }
    out = C.PROC_DIR / "label_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
