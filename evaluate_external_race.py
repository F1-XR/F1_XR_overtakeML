"""Evaluate a trained model bundle on races that were not in training."""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

import train_races


def main():
    args = parse_args()
    years = args.years or train_races.YEARS
    races = train_races.select_races(args.races)
    run_name = args.run_name or build_external_run_name(races, years)

    samples = load_or_build_samples(args, races, years, run_name)
    predictions, report = evaluate(samples, args.model_run, years)

    out_dir = Path("data/processed")
    pred_path = out_dir / f"{run_name}_predictions.parquet"
    report_path = Path("data/models") / f"{run_name}_vs_{args.model_run}_report.json"
    predictions.to_parquet(pred_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n===== external evaluation: {run_name} using {args.model_run} =====")
    print(f"samples={len(samples)} predictions={pred_path}")
    print(f"report={report_path}")
    print_generalization(report)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-run", default="races_initial_event_type_final")
    parser.add_argument("--races", nargs="+", default=["spa"])
    parser.add_argument("--years", nargs="+", type=int, default=train_races.YEARS)
    parser.add_argument("--run-name")
    parser.add_argument("--sample-path")
    parser.add_argument("--skip-build", action="store_true")
    return parser.parse_args()


def build_external_run_name(races, years):
    race_part = "_".join(race.slug for race in races)
    return f"{race_part}_external_test_{min(years)}_{max(years)}"


def load_or_build_samples(args, races, years, run_name):
    if args.sample_path:
        return pd.read_parquet(args.sample_path)

    out_path = Path("data/processed") / f"{run_name}_samples.parquet"
    if args.skip_build and out_path.exists():
        return pd.read_parquet(out_path)

    samples = []
    for race in races:
        for year in years:
            meta = train_races.collect_race(year, race)
            df = train_races.build_race(year, race, meta)
            samples.append(df)
    if not samples:
        raise RuntimeError("no external samples were built")

    merged = pd.concat(samples, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path)
    train_races.print_summary(merged, out_path)
    return merged


def evaluate(df, model_run, years):
    model_dir = Path("data/models")
    train_sample_path = Path("data/processed") / f"{model_run}_samples.parquet"
    training_samples = pd.read_parquet(train_sample_path) if train_sample_path.exists() else pd.DataFrame()

    predictions = df.copy()
    report = {
        "model_run": model_run,
        "external_rows": int(len(df)),
        "external_years": [int(v) for v in years],
        "external_circuits": sorted(str(v) for v in df["circuit_short_name"].dropna().unique()),
        "leakage_check": leakage_check(training_samples, df),
        "targets": {},
    }

    for target in train_races.TARGETS:
        meta = read_json(model_dir / f"{model_run}_{target}.json")
        calibration = read_json(model_dir / f"{model_run}_{target}_calibration.json")
        features = meta["features"]
        x = predictions[features].astype(float).fillna(meta.get("missing_value_fill", train_races.MISSING_VALUE_FILL))
        model = lgb.Booster(model_file=str(model_dir / f"{model_run}_{target}.txt"))
        raw = model.predict(x)
        display = apply_calibration(raw, calibration)
        predictions[f"raw_{target}"] = raw
        predictions[f"display_{target}"] = display

        quality_mask, quality_filter = train_races.target_training_mask(predictions, target, years)
        report["targets"][target] = {
            "model_oof_raw_metrics": meta.get("raw_metrics", {}),
            "model_oof_display_metrics": meta.get("display_metrics", {}),
            "external_all_rows": metric_block(predictions, target, np.ones(len(predictions), dtype=bool)),
            "external_quality_filtered": metric_block(predictions, target, quality_mask.to_numpy(dtype=bool)),
            "quality_filter": quality_filter,
            "thresholds": threshold_table(predictions, target, quality_mask.to_numpy(dtype=bool)),
        }

    return predictions, report


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def apply_calibration(raw, calibration):
    x = np.array(calibration.get("raw_thresholds", []), dtype=float)
    y = np.array(calibration.get("display_values", []), dtype=float)
    if len(x) == 0:
        return raw
    return np.interp(raw, x, y)


def metric_block(df, target, mask):
    values = df.loc[mask].copy()
    y = values[target].astype(int)
    raw = values[f"raw_{target}"].astype(float)
    display = values[f"display_{target}"].astype(float)
    block = balance_block(y)
    if len(values) == 0 or y.nunique() < 2:
        block.update({"raw_roc_auc": None, "raw_pr_auc": None, "raw_brier": None, "display_brier": None})
        return block
    block.update({
        "raw_roc_auc": float(roc_auc_score(y, raw)),
        "raw_pr_auc": float(average_precision_score(y, raw)),
        "raw_brier": float(brier_score_loss(y, raw)),
        "display_brier": float(brier_score_loss(y, display)),
        "mean_raw_probability": float(raw.mean()),
        "mean_display_probability": float(display.mean()),
    })
    return block


def balance_block(y):
    rows = int(len(y))
    positives = int(y.sum()) if rows else 0
    return {
        "rows": rows,
        "positive": positives,
        "negative": rows - positives,
        "positive_rate": float(positives / rows) if rows else 0.0,
    }


def threshold_table(df, target, mask):
    values = df.loc[mask].copy()
    if values.empty:
        return []
    y = values[target].astype(int).to_numpy()
    p = values[f"display_{target}"].astype(float).to_numpy()
    rows = []
    for threshold in [0.1, 0.2, 0.3, 0.5]:
        pred = p >= threshold
        tp = int(((y == 1) & pred).sum())
        fp = int(((y == 0) & pred).sum())
        fn = int(((y == 1) & ~pred).sum())
        rows.append({
            "display_threshold": threshold,
            "predicted_positive": int(pred.sum()),
            "true_positive": tp,
            "false_positive": fp,
            "precision": float(tp / (tp + fp)) if tp + fp else None,
            "recall": float(tp / (tp + fn)) if tp + fn else None,
        })
    return rows


def leakage_check(training, external):
    if training.empty:
        return {"training_samples_found": False}

    train_sessions = int_values(training, "session_key")
    external_sessions = int_values(external, "session_key")
    train_circuits = int_values(training, "circuit_key")
    external_circuits = int_values(external, "circuit_key")
    return {
        "training_samples_found": True,
        "training_session_keys": train_sessions,
        "external_session_keys": external_sessions,
        "overlap_session_keys": sorted(set(train_sessions).intersection(external_sessions)),
        "training_circuit_keys": train_circuits,
        "external_circuit_keys": external_circuits,
        "external_circuit_seen_in_training": sorted(set(train_circuits).intersection(external_circuits)),
    }


def int_values(df, col):
    if col not in df.columns:
        return []
    values = pd.to_numeric(df[col], errors="coerce").dropna().astype(int)
    return sorted(int(v) for v in values.unique())


def print_generalization(report):
    leak = report["leakage_check"]
    print(f"external circuits={report['external_circuits']}")
    print(f"session overlap={leak.get('overlap_session_keys')}")
    print(f"external circuit seen in training={leak.get('external_circuit_seen_in_training')}")
    for target, item in report["targets"].items():
        metrics = item["external_quality_filtered"]
        oof = item["model_oof_raw_metrics"]
        print(
            f"{target}: rows={metrics['rows']} pos={metrics['positive']} "
            f"rate={metrics['positive_rate']:.4f} "
            f"ROC-AUC={fmt(metrics['raw_roc_auc'])} PR-AUC={fmt(metrics['raw_pr_auc'])} "
            f"display Brier={fmt(metrics['display_brier'])} "
            f"OOF PR-AUC={fmt(oof.get('pr_auc'))}"
        )


def fmt(value):
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


if __name__ == "__main__":
    main()
