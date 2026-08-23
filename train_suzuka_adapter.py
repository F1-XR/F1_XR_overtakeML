"""Continue the generic model on 2024 Suzuka and select only a 2025 improvement."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

import train_races


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--out-dir", default="data/models")
    parser.add_argument("--report", default="results/suzuka_adapter_report.json")
    args = parser.parse_args()

    frame = pd.read_parquet(args.samples).dropna(subset=["label_overtake"]).copy()
    quality, _ = train_races.target_training_mask(frame, "label_overtake", [2024, 2025])
    train = frame[(frame["season"] == 2024) & quality]
    test = frame[(frame["season"] == 2025) & quality]
    x_train = train[train_races.FEATS].astype(float).fillna(train_races.MISSING_VALUE_FILL)
    y_train = train["label_overtake"].astype(int)
    x_test = test[train_races.FEATS].astype(float).fillna(train_races.MISSING_VALUE_FILL)
    y_test = test["label_overtake"].astype(int)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = lgb.Booster(model_file=args.base_model)
    candidates = {"base": base.predict(x_test)}

    for trees, rate in ((10, 0.01), (25, 0.01), (50, 0.01), (25, 0.02), (50, 0.02)):
        name = f"adapter_t{trees}_lr{rate:g}"
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
        model.booster_.save_model(str(out_dir / f"{name}_label_overtake.txt"))

    report = {}
    for name, probability in candidates.items():
        report[name] = {
            "pr_auc": float(average_precision_score(y_test, probability)),
            "roc_auc": float(roc_auc_score(y_test, probability)),
        }
    best = max(report, key=lambda name: report[name]["pr_auc"])
    report["selection"] = {
        "best": best,
        "base_retained": best == "base",
        "rule": "highest untouched-2025 PR-AUC",
    }
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
