"""Train final Bahrain models using 2023, 2024, and 2025 race samples."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

import config as C
from train_bahrain_years import FEATS, TARGETS


TRAIN_YEARS = [2023, 2024, 2025]
RUN_NAME = "bahrain_train_2023_2025_final"
SAMPLES_PATH = C.DATA_DIR / "processed" / "bahrain_2023_2025_samples.parquet"


def main():
    if not SAMPLES_PATH.exists():
        raise FileNotFoundError(f"missing samples: {SAMPLES_PATH}")

    df = pd.read_parquet(SAMPLES_PATH)
    missing = [name for name in FEATS if name not in df.columns]
    if missing:
        raise RuntimeError(f"missing feature columns: {missing}")

    print(f"[samples] {SAMPLES_PATH}")
    print(f"rows={len(df)} seasons={sorted(int(v) for v in df['season'].unique())}")
    print(label_summary(df).to_string())

    for target in TARGETS:
        train_final_target(df, target)


def train_final_target(df, target):
    d = df[df["season"].isin(TRAIN_YEARS)].dropna(subset=[target]).copy()
    X = d[FEATS].astype(float).fillna(-1)
    y = d[target].astype(int)

    if y.nunique() < 2:
        print(f"\n===== target={target} skipped: positive sample shortage =====")
        return

    raw_oof, y_oof, folds = season_oof_predictions(X, y, d["season"])
    calibrator, calibration_info = fit_calibrator(raw_oof, y_oof, folds)
    display_oof = calibrator.transform(raw_oof)
    raw_metrics = score(y_oof, raw_oof)
    display_metrics = score(y_oof, display_oof)

    model = make_model()
    model.fit(X, y)
    save_model(model, target, d, raw_metrics, display_metrics, calibrator, calibration_info)

    print(f"\n===== final target={target} | train={TRAIN_YEARS} =====")
    print(f"OOF raw ROC-AUC : {raw_metrics['roc_auc']:.3f}")
    print(f"OOF raw PR-AUC  : {raw_metrics['pr_auc']:.3f}")
    print(f"OOF raw Brier   : {raw_metrics['brier']:.3f}")
    print(f"OOF display ROC-AUC : {display_metrics['roc_auc']:.3f}")
    print(f"OOF display PR-AUC  : {display_metrics['pr_auc']:.3f}")
    print(f"OOF display Brier   : {display_metrics['brier']:.3f}")
    print("feature importance:")
    for feature, importance in sorted(zip(FEATS, model.feature_importances_), key=lambda item: -item[1]):
        print(f"  {feature:20s} {importance}")


def season_oof_predictions(X, y, seasons):
    raw_parts = []
    y_parts = []
    folds = []
    train_years = set(TRAIN_YEARS)

    for valid_year in TRAIN_YEARS:
        valid_mask = seasons == valid_year
        fit_mask = seasons.isin(train_years - {valid_year})
        if y[fit_mask].nunique() < 2 or y[valid_mask].nunique() < 2:
            continue

        model = make_model()
        model.fit(X[fit_mask], y[fit_mask])
        valid_p = model.predict_proba(X[valid_mask])[:, 1]
        raw_parts.append(valid_p)
        y_parts.append(y[valid_mask].to_numpy())
        folds.append({
            "fit_seasons": sorted(int(v) for v in train_years - {valid_year}),
            "valid_season": int(valid_year),
            "rows": int(valid_mask.sum()),
        })

    if not raw_parts:
        raise RuntimeError("could not create season OOF predictions")

    return np.concatenate(raw_parts), np.concatenate(y_parts), folds


def fit_calibrator(raw, y, folds):
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(raw, y)
    return calibrator, {
        "method": "isotonic_season_oof",
        "rows": int(len(y)),
        "folds": folds,
    }


def make_model():
    return LGBMClassifier(n_estimators=400, learning_rate=0.05, class_weight="balanced", verbose=-1)


def score(y_true, probability):
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "brier": float(brier_score_loss(y_true, probability)),
    }


def label_summary(df):
    labels = [name for name in TARGETS if name in df.columns]
    summary = df.groupby("season")[labels].sum().astype(int)
    summary["rows"] = df.groupby("season").size()
    return summary


def save_model(model, target, df, raw_metrics, display_metrics, calibrator, calibration_info):
    model_dir = C.DATA_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{RUN_NAME}_{target}.txt"
    meta_path = model_dir / f"{RUN_NAME}_{target}.json"
    calibration_path = model_dir / f"{RUN_NAME}_{target}_calibration.json"

    model.booster_.save_model(str(model_path))
    calibration_payload = calibration_info.copy()
    calibration_payload.update({
        "raw_thresholds": [float(value) for value in calibrator.X_thresholds_],
        "display_values": [float(value) for value in calibrator.y_thresholds_],
    })
    calibration_path.write_text(json.dumps(calibration_payload, indent=2))

    meta = {
        "target": target,
        "features": FEATS,
        "train_seasons": TRAIN_YEARS,
        "validation": "season_oof",
        "rows": int(len(df)),
        "positive": int(df[target].sum()),
        "positive_rate": float(df[target].mean()),
        "probability_for_ui": "display_probability",
        "raw_oof_metrics": raw_metrics,
        "display_oof_metrics": display_metrics,
        "calibration": calibration_info,
        "calibration_path": str(calibration_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"saved final model: {model_path}")
    print(f"saved final calibration: {calibration_path}")


if __name__ == "__main__":
    main()
