"""Audit label balance, split leakage, event parsing, and feature reliance."""
import argparse
import json
import re
from pathlib import Path

import lightgbm as lgb
import pandas as pd

import pipeline
import train_races


def main():
    args = parse_args()
    sample_path = Path(args.sample_path) if args.sample_path else Path("data/processed") / f"{args.run_name}_samples.parquet"
    df = pd.read_parquet(sample_path)

    report = {
        "run_name": args.run_name,
        "sample_path": str(sample_path),
        "rows": int(len(df)),
        "targets": {
            target: target_report(df, target, args.train_years, args.test_year)
            for target in train_races.TARGETS
        },
        "event_type_distribution": event_type_distribution(df),
        "race_control_spot_checks": race_control_spot_checks(Path(args.raw_dir)),
        "feature_importance": {
            target: feature_importance(args.run_name, Path(args.model_dir), target)
            for target in train_races.TARGETS
        },
        "notes": [
            "class_weight=balanced is enabled in train_races.make_model().",
            "No retraining is needed for this audit/export change because labels and features were not changed.",
            "Retrain all affected targets if feature order, feature values, labels, or quality filters change.",
        ],
    }

    out_path = Path(args.out) if args.out else Path(args.model_dir) / f"{args.run_name}_risk_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_summary(report, out_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="races_initial_event_type_final")
    parser.add_argument("--sample-path")
    parser.add_argument("--model-dir", default="data/models")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--train-years", nargs="+", type=int, default=[2023, 2024])
    parser.add_argument("--test-year", type=int, default=2025)
    parser.add_argument("--out")
    return parser.parse_args()


def target_report(df, target, train_years, test_year):
    d = df.dropna(subset=[target]).copy()
    years = sorted(int(v) for v in d["season"].dropna().unique())
    final_mask, final_filter = train_races.target_training_mask(d, target, years)

    eval_quality, eval_filter = train_races.target_training_mask(d, target, train_years)
    train_mask = d["season"].isin(train_years) & eval_quality
    test_mask = (d["season"] == test_year) & eval_quality
    split = train_races.eval_split_report(d, target, train_years, test_year, train_mask, test_mask)

    return {
        "final_training_filter": final_filter,
        "eval_training_filter": eval_filter,
        "eval_split_report": split,
        "positive_absolute_risk": positive_absolute_risk(final_filter["train_balance_after"]),
    }


def positive_absolute_risk(balance):
    positives = int(balance["positive"])
    if positives < 500:
        return "high"
    if positives < 2000:
        return "medium"
    return "low"


def event_type_distribution(df):
    if "change_reason" not in df.columns:
        return []
    grouped = (
        df.groupby(["season", "circuit_short_name", "change_reason"])
        .size()
        .reset_index(name="rows")
        .sort_values(["season", "circuit_short_name", "change_reason"])
    )
    return grouped.to_dict("records")


def race_control_spot_checks(raw_root):
    checks = []
    for race_dir in sorted(raw_root.glob("*_race")):
        race_control_path = race_dir / "race_control.json"
        meta_path = race_dir / "session_meta.json"
        if not race_control_path.exists():
            continue

        rows = json.loads(race_control_path.read_text(encoding="utf-8"))
        rc = pd.DataFrame(rows)
        if not rc.empty and "date" in rc.columns:
            rc["date"] = pipeline._dt(rc["date"])
        windows = pipeline._race_control_windows(rc)
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        checks.append({
            "race_dir": race_dir.name,
            "season": meta.get("season"),
            "circuit_short_name": meta.get("circuit_short_name"),
            "race_control_rows": len(rows),
            "keyword_counts": keyword_counts(rows),
            "control_windows": [window_record(item) for item in windows],
        })
    return checks


def keyword_counts(rows):
    patterns = {
        "red_flag": r"\bRED\s+FLAG\b|\bSESSION\s+SUSPENDED\b",
        "safety_car": r"\bSAFETY\s+CAR\b",
        "virtual_safety_car": r"\bVIRTUAL\s+SAFETY\s+CAR\b",
        "penalty": r"\bPENALTY\b|\bDRIVE\s+THROUGH\b|\bSTOP\s*(?:-|AND\s+)?GO\b|\bTIME\s+PENALTY\b",
        "retirement": r"\bSTOPPED\b|\bRETIRED\b|\bCAR\s+STOPPED\b|\bPULLED\s+OFF\b",
    }
    counts = {name: 0 for name in patterns}
    for row in rows:
        text = " ".join(str(row.get(name, "")) for name in ("category", "flag", "message")).upper()
        for name, pattern in patterns.items():
            if re.search(pattern, text):
                counts[name] += 1
    return counts


def window_record(item):
    start, end, kind = item[0], item[1], item[2]
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    return {
        "kind": kind,
        "start": start_ts.isoformat(),
        "end": end_ts.isoformat(),
        "duration_seconds": float((end_ts - start_ts).total_seconds()),
    }


def feature_importance(run_name, model_dir, target):
    model_path = model_dir / f"{run_name}_{target}.txt"
    meta_path = model_dir / f"{run_name}_{target}.json"
    if not model_path.exists() or not meta_path.exists():
        return {"error": "model or metadata not found"}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    features = meta.get("features", train_races.FEATS)
    booster = lgb.Booster(model_file=str(model_path))
    gains = booster.feature_importance(importance_type="gain")
    rows = [
        {"feature": feature, "gain": float(gain), "rank": idx + 1}
        for idx, (feature, gain) in enumerate(
            sorted(zip(features, gains), key=lambda item: item[1], reverse=True)
        )
    ]
    watched = {
        name: next((row for row in rows if row["feature"] == name), None)
        for name in ["season", "circuit_key", "circuit_type_code", "track_progress", "segment", "sector"]
    }
    return {"top_10": rows[:10], "watched_features": watched}


def print_summary(report, out_path):
    print(f"wrote audit: {out_path}")
    for target, item in report["targets"].items():
        balance = item["final_training_filter"]["train_balance_after"]
        split = item["eval_split_report"]
        print(
            f"{target}: final positives={balance['positive']}/{balance['rows']} "
            f"rate={balance['positive_rate']:.4f} risk={item['positive_absolute_risk']} "
            f"eval_overlap={split['overlap_session_keys']}"
        )
    for target, item in report["feature_importance"].items():
        top = item.get("top_10", [])
        if top:
            watched = item.get("watched_features", {})
            circuit_rank = watched.get("circuit_key", {}).get("rank") if watched.get("circuit_key") else None
            season_rank = watched.get("season", {}).get("rank") if watched.get("season") else None
            print(f"{target}: top_feature={top[0]['feature']} circuit_key_rank={circuit_rank} season_rank={season_rank}")


if __name__ == "__main__":
    main()
