"""Export a fixed inference contract for Unity/server integration."""
import argparse
import json
from pathlib import Path

import pandas as pd

import train_races


TARGET_OUTPUTS = {
    "label_overtake": {
        "output_name": "overtake_probability",
        "meaning": "Probability that the selected driver overtakes the current ahead driver within 30 seconds.",
    },
    "label_position_gain": {
        "output_name": "position_gain_probability",
        "meaning": "Probability that the selected driver's position number improves within 30 seconds.",
    },
    "label_position_loss": {
        "output_name": "position_loss_probability",
        "meaning": "Probability that the selected driver's position number gets worse within 30 seconds.",
    },
    "label_position_change": {
        "output_name": "position_change_probability",
        "meaning": "Probability that either position gain or position loss happens within 30 seconds.",
    },
}


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    sample_path = Path(args.sample_path) if args.sample_path else Path("data/processed") / f"{args.run_name}_samples.parquet"
    out_path = Path(args.out) if args.out else model_dir / f"{args.run_name}_unity_contract.json"

    metas = {target: read_json(model_dir / f"{args.run_name}_{target}.json") for target in train_races.TARGETS}
    first_meta = next(iter(metas.values()))
    features = first_meta.get("features", train_races.FEATS)

    contract = {
        "run_name": args.run_name,
        "schema_version": first_meta.get("feature_schema_version", train_races.FEATURE_SCHEMA_VERSION),
        "model_format": "LightGBM text booster",
        "horizon_seconds": 30.0,
        "feature_count": len(features),
        "feature_order": features,
        "missing_value_fill": first_meta.get("missing_value_fill", train_races.MISSING_VALUE_FILL),
        "input_rule": {
            "dtype": "float32_or_float64",
            "order_must_match_feature_order": True,
            "unknown_or_missing_numeric_value": first_meta.get("missing_value_fill", train_races.MISSING_VALUE_FILL),
            "calibrated_display_probability": "linear interpolation over raw_thresholds/display_values",
        },
        "circuit_key_mapping": circuit_key_mapping(sample_path),
        "event_type_definition": first_meta.get("event_type_definition", event_type_definition()),
        "outputs": {
            target: output_contract(args.run_name, model_dir, target, metas[target])
            for target in train_races.TARGETS
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print(f"wrote contract: {out_path}")
    print(f"features={len(features)} targets={len(train_races.TARGETS)}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="races_initial_event_type_final")
    parser.add_argument("--model-dir", default="data/models")
    parser.add_argument("--sample-path")
    parser.add_argument("--out")
    return parser.parse_args()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def output_contract(run_name, model_dir, target, meta):
    calibration_path = model_dir / f"{run_name}_{target}_calibration.json"
    calibration = read_json(calibration_path)
    return {
        **TARGET_OUTPUTS[target],
        "target": target,
        "model_file": str(model_dir / f"{run_name}_{target}.txt"),
        "metadata_file": str(model_dir / f"{run_name}_{target}.json"),
        "calibration_file": str(calibration_path),
        "probability_for_ui": meta.get("probability_for_ui", "display_probability"),
        "raw_probability": "LightGBM predict_proba positive class",
        "calibration": {
            "method": calibration.get("method"),
            "raw_thresholds": calibration.get("raw_thresholds", []),
            "display_values": calibration.get("display_values", []),
        },
    }


def circuit_key_mapping(sample_path):
    if not sample_path.exists():
        return []
    df = pd.read_parquet(sample_path)
    cols = [
        "race_slug",
        "circuit_short_name",
        "circuit_key",
        "circuit_type",
        "circuit_type_code",
    ]
    cols = [col for col in cols if col in df.columns]
    if not cols:
        return []
    rows = df[cols].drop_duplicates().sort_values(cols).to_dict("records")
    return rows


def event_type_definition():
    return {
        "0": "no_change",
        "1": "on_track_overtake",
        "2": "pit_related_change",
        "3": "retirement_related_change",
        "4": "lapping_pass",
        "5": "penalty_related_change",
        "6": "restart_overtake",
        "7": "uncertain",
    }


if __name__ == "__main__":
    main()
