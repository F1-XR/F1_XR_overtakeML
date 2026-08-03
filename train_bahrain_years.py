"""Collect and train Bahrain race samples for multiple OpenF1 seasons."""
import json

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.isotonic import IsotonicRegression

import config as C
import pipeline


YEARS = [2023, 2024, 2025]
TRAIN_YEARS = [2023, 2024]
TEST_YEAR = 2025
RUN_NAME = "bahrain_train_2023_2024_test_2025"
COUNTRY = "Bahrain"
SESSION = "Race"
TARGETS = [
    "label_overtake",
    "label_position_gain",
    "label_position_loss",
    "label_position_change",
]
FEATS = [
    "gap_ahead",
    "gap_trend",
    "position",
    "tyre_age",
    "tyre_age_delta",
    "position_delta",
    "same_lap",
    "drs_range",
    "speed",
    "speed_delta",
    "drs_active",
    "track_progress",
    "track_progress_sin",
    "track_progress_cos",
    "sector",
    "segment",
]
ENDPOINTS = [
    "drivers",
    "position",
    "intervals",
    "laps",
    "stints",
    "pit",
    "race_control",
    "car_data",
    "location",
]


def main():
    root_raw = C.DATA_DIR / "raw"
    root_proc = C.DATA_DIR / "processed"
    root_proc.mkdir(parents=True, exist_ok=True)

    samples = []
    for year in YEARS:
        session_key = collect_year(year, root_raw, root_proc)
        df = build_year(year, session_key, root_raw, root_proc)
        samples.append(df)

    merged = pd.concat(samples, ignore_index=True)
    merged_path = root_proc / "bahrain_2023_2025_samples.parquet"
    merged.to_parquet(merged_path)
    merged.to_parquet(root_proc / "samples.parquet")
    print_summary(merged, merged_path)

    for target in TARGETS:
        train_target(merged, target)


def collect_year(year, root_raw, root_proc):
    set_context(year, root_raw, root_proc)
    meta_path = C.RAW_DIR / "session_meta.json"
    if meta_path.exists() and all((C.RAW_DIR / f"{name}.json").exists() for name in ENDPOINTS):
        meta = json.loads(meta_path.read_text())
        print(f"[cache] {year} raw data exists. session_key={meta['session_key']}")
        return meta["session_key"]

    session_key = pipeline.get_session_key()
    pipeline.collect(session_key)
    meta_path.write_text(json.dumps({
        "season": year,
        "country": COUNTRY,
        "session": SESSION,
        "session_key": session_key,
    }))
    return session_key


def build_year(year, session_key, root_raw, root_proc):
    set_context(year, root_raw, root_proc)
    df = pipeline.build_dataset()
    df = pipeline.add_tire(df)
    df = pipeline.add_undercut(df)
    df["season"] = year
    df["country"] = COUNTRY
    df["session_name"] = SESSION
    df["session_key"] = session_key
    out = C.PROC_DIR / "samples.parquet"
    df.to_parquet(out)
    print(f"[year] {year} rows={len(df)} -> {out}")
    return df


def set_context(year, root_raw, root_proc):
    C.SEASON = year
    C.COUNTRY = COUNTRY
    C.SESSION = SESSION
    C.FETCH_CAR_DATA = True
    C.FETCH_LOCATION_DATA = True
    C.RAW_DIR = root_raw / f"{year}_bahrain_race"
    C.PROC_DIR = root_proc / f"{year}_bahrain_race"
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    C.PROC_DIR.mkdir(parents=True, exist_ok=True)


def print_summary(df, path):
    print(f"\n===== merged Bahrain samples -> {path} =====")
    print(f"rows={len(df)} seasons={sorted(df['season'].unique())}")
    labels = [c for c in TARGETS if c in df.columns]
    by_year = df.groupby("season")[labels].sum().astype(int)
    by_year["rows"] = df.groupby("season").size()
    print(by_year.to_string())


def train_target(df, target):
    d = df.dropna(subset=[target]).copy()
    X = d[FEATS].astype(float).fillna(-1)
    y = d[target].astype(int)
    train_mask = d["season"].isin(TRAIN_YEARS)
    test_mask = d["season"] == TEST_YEAR

    if y[train_mask].nunique() < 2 or y[test_mask].nunique() < 2:
        print(f"\n===== target={target} skipped: train/test positive sample shortage =====")
        print(label_counts(d, target).to_string())
        return

    calibrator, calibration_info = fit_calibrator(X, y, d["season"], target)
    model = make_model()
    model.fit(X[train_mask], y[train_mask])
    raw_p = model.predict_proba(X[test_mask])[:, 1]
    display_p = calibrator.transform(raw_p) if calibrator is not None else raw_p
    y_test = y[test_mask]
    raw_metrics = score(y_test, raw_p)
    display_metrics = score(y_test, display_p)

    print(f"\n===== target={target} | train={TRAIN_YEARS} test={TEST_YEAR} =====")
    print(label_counts(d, target).to_string())
    print(f"raw ROC-AUC : {raw_metrics['roc_auc']:.3f}")
    print(f"raw PR-AUC  : {raw_metrics['pr_auc']:.3f}")
    print(f"raw Brier   : {raw_metrics['brier']:.3f}")
    print(f"display ROC-AUC : {display_metrics['roc_auc']:.3f}")
    print(f"display PR-AUC  : {display_metrics['pr_auc']:.3f}")
    print(f"display Brier   : {display_metrics['brier']:.3f}")
    base = ((X[test_mask]["gap_ahead"] < 1.0) & (X[test_mask]["gap_trend"] < 0)).astype(int)
    print(f"baseline ROC-AUC: {roc_auc_score(y_test, base):.3f}")
    save_model(model, target, y_test, raw_p, display_p, d, calibrator, calibration_info)
    print("feature importance:")
    for f, imp in sorted(zip(FEATS, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {f:12s} {imp}")


def make_model():
    return LGBMClassifier(n_estimators=400, learning_rate=0.05, class_weight="balanced", verbose=-1)


def fit_calibrator(X, y, seasons, target):
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
        print(f"[calibration] {target}: skipped")
        return None, {"method": "none", "folds": []}

    raw = np.concatenate(raw_parts)
    labels = np.concatenate(y_parts)
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(raw, labels)
    print(f"[calibration] {target}: isotonic oof rows={len(labels)} folds={len(folds)}")
    return calibrator, {
        "method": "isotonic_oof_by_train_season",
        "rows": int(len(labels)),
        "folds": folds,
    }


def score(y_true, p):
    return {
        "roc_auc": float(roc_auc_score(y_true, p)),
        "pr_auc": float(average_precision_score(y_true, p)),
        "brier": float(brier_score_loss(y_true, p)),
    }


def label_counts(df, target):
    g = df.groupby("season")[target].agg(["count", "sum"])
    g = g.rename(columns={"count": "rows", "sum": "positive"})
    g["rate"] = np.where(g["rows"] > 0, g["positive"] / g["rows"], 0.0)
    return g


def save_model(model, target, y_test, raw_p, display_p, df, calibrator, calibration_info):
    model_dir = C.DATA_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{RUN_NAME}_{target}.txt"
    meta_path = model_dir / f"{RUN_NAME}_{target}.json"
    calibration_path = model_dir / f"{RUN_NAME}_{target}_calibration.json"
    model.booster_.save_model(str(model_path))
    raw_metrics = score(y_test, raw_p)
    display_metrics = score(y_test, display_p)
    calibration_payload = calibration_info.copy()
    if calibrator is not None:
        calibration_payload.update({
            "raw_thresholds": [float(v) for v in calibrator.X_thresholds_],
            "display_values": [float(v) for v in calibrator.y_thresholds_],
        })
    calibration_path.write_text(json.dumps(calibration_payload, indent=2))
    meta = {
        "target": target,
        "features": FEATS,
        "train_seasons": TRAIN_YEARS,
        "test_season": TEST_YEAR,
        "rows": int(len(df)),
        "train_rows": int((df["season"].isin(TRAIN_YEARS)).sum()),
        "test_rows": int((df["season"] == TEST_YEAR).sum()),
        "positive": int(df[target].sum()),
        "probability_for_ui": "display_probability",
        "raw_metrics": raw_metrics,
        "display_metrics": display_metrics,
        "roc_auc": raw_metrics["roc_auc"],
        "pr_auc": raw_metrics["pr_auc"],
        "brier": raw_metrics["brier"],
        "display_roc_auc": display_metrics["roc_auc"],
        "display_pr_auc": display_metrics["pr_auc"],
        "display_brier": display_metrics["brier"],
        "calibration": calibration_info,
        "calibration_path": str(calibration_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"saved model: {model_path}")
    print(f"saved calibration: {calibration_path}")


if __name__ == "__main__":
    main()
