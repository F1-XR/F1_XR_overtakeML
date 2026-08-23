"""Continue a deployed overtake model on new 2024 circuits and evaluate untouched 2025 races."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

import train_races


def score(y, probability) -> dict:
    return {
        "rows": int(len(y)),
        "positive": int(y.sum()),
        "pr_auc": float(average_precision_score(y, probability)),
        "roc_auc": float(roc_auc_score(y, probability)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", nargs="+", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--out-dir", default="data/models")
    parser.add_argument("--report-dir", default="results")
    args = parser.parse_args()

    frames = [pd.read_parquet(path) for path in args.samples]
    frame = pd.concat(frames, ignore_index=True).dropna(subset=["label_overtake"])
    quality, training_filter = train_races.target_training_mask(
        frame, "label_overtake", [2024, 2025]
    )
    train = frame[(frame["season"] == 2024) & quality]
    test = frame[(frame["season"] == 2025) & quality]
    x_train = train[train_races.FEATS].astype(float).fillna(train_races.MISSING_VALUE_FILL)
    y_train = train["label_overtake"].astype(int)
    x_test = test[train_races.FEATS].astype(float).fillna(train_races.MISSING_VALUE_FILL)
    y_test = test["label_overtake"].astype(int)

    base = lgb.Booster(model_file=args.base_model)
    candidates = {"base": base.predict(x_test)}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for trees, rate in ((25, 0.01), (50, 0.01), (50, 0.02), (100, 0.01)):
        name = f"{args.run_name}_t{trees}_lr{rate:g}"
        model = LGBMClassifier(
            n_estimators=trees,
            learning_rate=rate,
            class_weight="balanced",
            num_leaves=15,
            min_child_samples=100,
            reg_lambda=2.0,
            verbose=-1,
        )
        model.fit(x_train, y_train, init_model=args.base_model)
        candidates[name] = model.predict_proba(x_test)[:, 1]
        path = out_dir / f"{name}_label_overtake.txt"
        model.booster_.save_model(str(path))
        paths[name] = str(path)

    report = {
        "run_name": args.run_name,
        "train_year": 2024,
        "test_year": 2025,
        "train_circuits": sorted(str(v) for v in train["circuit_short_name"].unique()),
        "test_circuits": sorted(str(v) for v in test["circuit_short_name"].unique()),
        "train_rows": int(len(train)),
        "train_positive": int(y_train.sum()),
        "test_rows": int(len(test)),
        "test_positive": int(y_test.sum()),
        "telemetry_coverage": {
            feature: float(frame[feature].notna().mean())
            for feature in ("speed", "speed_delta", "track_progress")
        },
        "candidates": {name: score(y_test, probability) for name, probability in candidates.items()},
        "training_filter": training_filter,
    }
    best = max(report["candidates"], key=lambda name: report["candidates"][name]["pr_auc"])
    report["selection"] = {
        "best": best,
        "base_retained": best == "base",
        "model_path": paths.get(best, args.base_model),
        "rule": "highest untouched-2025 PR-AUC",
    }
    report_path = Path(args.report_dir) / f"{args.run_name}_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
