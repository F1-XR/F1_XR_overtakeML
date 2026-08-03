"""Collect, process, and train models for multiple F1 race circuits."""
import argparse
import json
import re
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

import config as C
import pipeline


YEARS = [2023, 2024, 2025]
TARGETS = [
    "label_overtake",
    "label_position_gain",
    "label_position_loss",
    "label_position_change",
]
TRAIN_GROUP_COLS = ["season", "circuit_short_name"]
SPLIT_GROUP_COLS = ["season", "session_key", "circuit_short_name"]
FEATURE_SCHEMA_VERSION = "event_type_v1_26"
MISSING_VALUE_FILL = -1.0
EXCLUDE_RED_FLAG_TRAIN_RACES = False
ON_TRACK_OVERTAKE_TYPES = {0, 1}
STRICT_OVERTAKE_EXCLUDE_TYPES = {2, 3, 4, 5, 6, 7}
FEATS = [
    "season",
    "circuit_key",
    "circuit_type_code",
    "is_lap1",
    "restart_phase",
    "air_temperature",
    "track_temperature",
    "humidity",
    "rainfall",
    "weather_regime_code",
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
    "weather",
    "car_data",
    "location",
]


@dataclass(frozen=True)
class RaceSpec:
    name: str
    country_name: str
    circuit_short_name: str
    slug: str


RACES = {
    "bahrain": RaceSpec("Bahrain", "Bahrain", "Sakhir", "bahrain"),
    "monza": RaceSpec("Monza", "Italy", "Monza", "monza"),
    "monaco": RaceSpec("Monaco", "Monaco", "Monte Carlo", "monaco"),
    "spa": RaceSpec("Spa-Francorchamps", "Belgium", "Spa-Francorchamps", "spa"),
    "silverstone": RaceSpec("Silverstone", "United Kingdom", "Silverstone", "silverstone"),
    "singapore": RaceSpec("Singapore", "Singapore", "Singapore", "singapore"),
}
RACE_GROUPS = {
    "bahrain": ["bahrain"],
    "initial": ["bahrain", "monza", "monaco", "silverstone", "singapore"],
    "initial_spa": ["bahrain", "monza", "monaco", "silverstone", "singapore", "spa"],
}


def main():
    args = parse_args()
    years = args.years or YEARS
    races = select_races(args.races)
    run_name = args.run_name or build_run_name(races, years, args.mode)

    if args.dry_run:
        for race in races:
            for year in years:
                print(json.dumps(resolve_session(year, race), indent=2))
        return

    samples = []
    for race in races:
        for year in years:
            meta = collect_race(year, race)
            if args.collect_only:
                continue
            df = build_race(year, race, meta)
            samples.append(df)

    if args.collect_only:
        return
    if not samples:
        raise RuntimeError("no samples were built")

    merged = pd.concat(samples, ignore_index=True)
    out_path = C.DATA_DIR / "processed" / f"{run_name}_samples.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path)
    print_summary(merged, out_path)

    if args.mode == "eval":
        train_years = [year for year in years if year != max(years)]
        test_year = max(years)
        for target in TARGETS:
            train_eval_target(merged, target, train_years, test_year, run_name)
    else:
        for target in TARGETS:
            train_final_target(merged, target, years, run_name)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--races", nargs="+", default=["bahrain"], help="Race ids or groups: bahrain, initial, monza, ...")
    parser.add_argument("--years", nargs="+", type=int, default=YEARS)
    parser.add_argument("--mode", choices=["eval", "final"], default="eval")
    parser.add_argument("--run-name")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def select_races(names):
    selected = []
    for name in names:
        key = name.lower()
        keys = RACE_GROUPS.get(key, [key])
        for race_key in keys:
            if race_key not in RACES:
                raise ValueError(f"unknown race: {name}")
            race = RACES[race_key]
            if race not in selected:
                selected.append(race)
    return selected


def build_run_name(races, years, mode):
    race_part = "initial" if [r.slug for r in races] == RACE_GROUPS["initial"] else "_".join(r.slug for r in races)
    if mode == "eval":
        return f"races_{race_part}_train_{min(years)}_{max(years) - 1}_test_{max(years)}"
    return f"races_{race_part}_train_{min(years)}_{max(years)}_final"


def collect_race(year, race):
    set_context(year, race)
    meta_path = C.RAW_DIR / "session_meta.json"
    meta = resolve_session(year, race)
    meta_path.write_text(json.dumps(meta, indent=2))

    if all((C.RAW_DIR / f"{name}.json").exists() for name in ENDPOINTS):
        print(f"[cache] {year} {race.name} raw data exists. session_key={meta['session_key']}")
        return meta

    pipeline.collect(meta["session_key"])
    return meta


def build_race(year, race, meta):
    set_context(year, race)
    df = pipeline.build_dataset()
    df = pipeline.add_tire(df)
    df = pipeline.add_undercut(df)
    red_flag_count = count_red_flags()
    df["race_slug"] = race.slug
    df["red_flag_count"] = red_flag_count
    df["has_red_flag"] = int(red_flag_count > 0)
    df["circuit_key"] = pd.to_numeric(df["circuit_key"], errors="coerce")
    df["circuit_id"] = df["circuit_key"]
    df["circuit_type_code"] = pd.to_numeric(df["circuit_type_code"], errors="coerce")
    out = C.PROC_DIR / "samples.parquet"
    df.to_parquet(out)
    red_text = f" red_flags={red_flag_count}" if red_flag_count else ""
    print(f"[race] {year} {race.name} rows={len(df)} session_key={meta['session_key']}{red_text} -> {out}")
    return df


def set_context(year, race):
    C.SEASON = year
    C.COUNTRY = race.country_name
    C.SESSION = "Race"
    C.FETCH_CAR_DATA = True
    C.FETCH_LOCATION_DATA = True
    C.RAW_DIR = C.DATA_DIR / "raw" / f"{year}_{race.slug}_race"
    C.PROC_DIR = C.DATA_DIR / "processed" / f"{year}_{race.slug}_race"
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    C.PROC_DIR.mkdir(parents=True, exist_ok=True)


def resolve_session(year, race):
    sessions = get_json("/sessions", {
        "year": year,
        "country_name": race.country_name,
        "session_name": "Race",
    })
    if not sessions:
        raise RuntimeError(f"session not found: {year} {race.name}")

    candidates = [
        item for item in sessions
        if normalize(item.get("circuit_short_name")) == normalize(race.circuit_short_name)
    ]
    if not candidates:
        candidates = [
            item for item in sessions
            if normalize(race.circuit_short_name) in normalize(item.get("circuit_short_name"))
        ]
    if not candidates:
        known = sorted({item.get("circuit_short_name") for item in sessions})
        raise RuntimeError(f"circuit not found: {year} {race.name}. candidates={known}")

    session = sorted(candidates, key=lambda item: item["date_start"])[0]
    meeting = first_or_empty(get_json("/meetings", {"meeting_key": session["meeting_key"]}))
    return {
        "season": int(year),
        "country": race.country_name,
        "session": "Race",
        "session_key": int(session["session_key"]),
        "meeting_key": int(session["meeting_key"]),
        "circuit_key": int(session["circuit_key"]) if session.get("circuit_key") is not None else None,
        "circuit_short_name": session.get("circuit_short_name") or race.circuit_short_name,
        "circuit_type": meeting.get("circuit_type", "Unknown"),
        "country_code": session.get("country_code") or meeting.get("country_code"),
        "date_start": session.get("date_start"),
        "date_end": session.get("date_end"),
    }


def get_json(path, params):
    url = C.BASE_URL + path
    for attempt in range(4):
        response = requests.get(url, params=params, timeout=90)
        if response.status_code == 200:
            time.sleep(C.REQUEST_SLEEP)
            return response.json()
        if response.status_code == 429 or 500 <= response.status_code < 600:
            time.sleep(2 * (attempt + 1))
            continue
        response.raise_for_status()
    raise RuntimeError(f"request failed: {url} {params}")


def normalize(value):
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def first_or_empty(items):
    return items[0] if items else {}


def count_red_flags():
    path = C.RAW_DIR / "race_control.json"
    if not path.exists():
        return 0
    rows = json.loads(path.read_text())
    count = 0
    for row in rows:
        flag = str(row.get("flag", "")).strip().upper()
        message = str(row.get("message", "")).strip().upper()
        has_red_flag_message = bool(re.search(r"\bRED\s+FLAG\b", message))
        has_suspended_message = bool(re.search(r"\bSESSION\s+SUSPENDED\b", message))
        if flag in {"RED", "RED FLAG"} or has_red_flag_message or has_suspended_message:
            count += 1
    return count


def print_summary(df, path):
    print(f"\n===== merged race samples -> {path} =====")
    print(f"rows={len(df)} seasons={sorted(int(v) for v in df['season'].unique())}")
    print(df.groupby(["season", "circuit_short_name"])[TARGETS].sum().astype(int).to_string())
    if "change_reason" in df.columns:
        print("\nevent_type counts:")
        print(
            df.groupby(["season", "circuit_short_name", "change_reason"])
            .size()
            .unstack(fill_value=0)
            .to_string()
        )


def target_training_mask(df, target, train_years):
    quality_mask = pd.Series(True, index=df.index)
    year_mask = df["season"].isin(train_years)
    excluded = []

    def exclude(reason, mask):
        nonlocal quality_mask
        mask = quality_mask & mask.fillna(False)
        train_mask = year_mask & mask
        rows = int(train_mask.sum())
        positives = int(df.loc[train_mask, target].sum()) if rows else 0
        quality_mask.loc[mask] = False
        if rows <= 0:
            return
        positive_rate = float(positives / rows) if rows else 0.0
        excluded.append({
            "target": target,
            "rows": rows,
            "positive": positives,
            "positive_rate": positive_rate,
            "reason": reason,
        })

    if target == "label_overtake" and "event_type" in df.columns:
        event_type = pd.to_numeric(df["event_type"], errors="coerce")
        exclude(
            "strict_on_track_event_type_exclusion",
            ~event_type.isin(ON_TRACK_OVERTAKE_TYPES),
        )
        for col in ["control_window", "pit_window", "retirement_window", "penalty_window", "restart_phase"]:
            if col in df.columns:
                exclude(f"{col}_exclusion", pd.to_numeric(df[col], errors="coerce").fillna(0) > 0)
        if "same_lap" in df.columns:
            exclude("not_same_lap_exclusion", pd.to_numeric(df["same_lap"], errors="coerce") != 1)
    else:
        if "control_window" in df.columns:
            exclude("control_window_exclusion", pd.to_numeric(df["control_window"], errors="coerce").fillna(0) > 0)
        if "event_type" in df.columns:
            event_type = pd.to_numeric(df["event_type"], errors="coerce")
            exclude("uncertain_event_type_exclusion", event_type == 7)

    if EXCLUDE_RED_FLAG_TRAIN_RACES:
        train_rows = df[year_mask & quality_mask]
        for (season, circuit), group in train_rows.groupby(TRAIN_GROUP_COLS):
            red_flag_count = group_red_flag_count(group)
            if red_flag_count <= 0:
                continue
            rows = int(len(group))
            positives = int(group[target].sum())
            positive_rate = float(positives / rows) if rows else 0.0
            quality_mask.loc[group.index] = False
            excluded.append({
                "season": int(season),
                "circuit_short_name": str(circuit),
                "target": target,
                "reason": "red_flag_race",
                "red_flag_count": red_flag_count,
                "rows": rows,
                "positive": positives,
                "positive_rate": positive_rate,
            })

    before = target_balance(df, target, year_mask)
    after = target_balance(df, target, year_mask & quality_mask)
    return quality_mask, {
        "method": "target_event_quality_filter",
        "exclude_red_flag_train_races": EXCLUDE_RED_FLAG_TRAIN_RACES,
        "overtake_positive_event_type": 1,
        "overtake_allowed_event_types": sorted(ON_TRACK_OVERTAKE_TYPES),
        "overtake_excluded_event_types": sorted(STRICT_OVERTAKE_EXCLUDE_TYPES),
        "group_columns": TRAIN_GROUP_COLS,
        "train_rows_before": before["rows"],
        "train_rows_after": after["rows"],
        "train_positive_before": before["positive"],
        "train_positive_after": after["positive"],
        "train_positive_rate_before": before["positive_rate"],
        "train_positive_rate_after": after["positive_rate"],
        "train_balance_before": before,
        "train_balance_after": after,
        "excluded_rows": before["rows"] - after["rows"],
        "excluded_groups": excluded,
    }


def target_balance(df, target, mask):
    rows = int(mask.sum())
    positives = int(df.loc[mask, target].sum()) if rows else 0
    return {
        "rows": rows,
        "positive": positives,
        "negative": rows - positives,
        "positive_rate": float(positives / rows) if rows else 0.0,
    }


def group_red_flag_count(group):
    if "red_flag_count" in group.columns:
        values = pd.to_numeric(group["red_flag_count"], errors="coerce").fillna(0)
        return int(values.max()) if len(values) else 0
    if "has_red_flag" in group.columns:
        values = pd.to_numeric(group["has_red_flag"], errors="coerce").fillna(0)
        return int(values.max()) if len(values) else 0
    return 0


def print_training_filter(target, training_filter):
    before = training_filter.get("train_balance_before", {})
    after = training_filter.get("train_balance_after", {})
    if before and after:
        print(
            f"[train-balance] target={target} "
            f"rows={before['rows']}->{after['rows']} "
            f"positive={before['positive']}->{after['positive']} "
            f"rate={before['positive_rate']:.4f}->{after['positive_rate']:.4f}"
        )
    excluded = training_filter["excluded_groups"]
    if not excluded:
        return
    print(f"[train-filter] target={target} excluded training rows:")
    for item in excluded:
        detail = (
            f"  reason={item['reason']} "
            f"rows={item['rows']} positive={item['positive']}"
        )
        if "season" in item and "circuit_short_name" in item:
            detail += f" race={item['season']} {item['circuit_short_name']}"
        if "red_flag_count" in item:
            detail += f" red_flags={item['red_flag_count']}"
        print(detail)


def eval_split_report(df, target, train_years, test_year, train_mask, test_mask):
    train_sessions = unique_int_values(df.loc[train_mask, "session_key"]) if "session_key" in df.columns else []
    test_sessions = unique_int_values(df.loc[test_mask, "session_key"]) if "session_key" in df.columns else []
    overlap = sorted(set(train_sessions).intersection(test_sessions))
    if overlap:
        raise RuntimeError(f"train/test session leakage detected for {target}: session_key overlap={overlap}")

    train_balance = target_balance(df, target, train_mask)
    test_balance = target_balance(df, target, test_mask)
    return {
        "method": "season_holdout",
        "group_columns": SPLIT_GROUP_COLS,
        "train_years": [int(v) for v in train_years],
        "test_year": int(test_year),
        "train_session_keys": train_sessions,
        "test_session_keys": test_sessions,
        "overlap_session_keys": overlap,
        "train_balance": train_balance,
        "test_balance": test_balance,
    }


def unique_int_values(series):
    values = pd.to_numeric(series, errors="coerce").dropna().astype(int)
    return sorted(int(v) for v in values.unique())


def print_eval_split(report):
    train = report["train_balance"]
    test = report["test_balance"]
    print(
        "[split] "
        f"train_years={report['train_years']} test_year={report['test_year']} "
        f"overlap_session_keys={report['overlap_session_keys']}"
    )
    print(
        "[split-balance] "
        f"train positive={train['positive']}/{train['rows']} ({train['positive_rate']:.4f}) "
        f"test positive={test['positive']}/{test['rows']} ({test['positive_rate']:.4f})"
    )


def train_eval_target(df, target, train_years, test_year, run_name):
    d = df.dropna(subset=[target]).copy()
    X = d[FEATS].astype(float).fillna(MISSING_VALUE_FILL)
    y = d[target].astype(int)
    quality_mask, training_filter = target_training_mask(d, target, train_years)
    train_mask = d["season"].isin(train_years) & quality_mask
    test_mask = (d["season"] == test_year) & quality_mask
    split_report = eval_split_report(d, target, train_years, test_year, train_mask, test_mask)

    if y[train_mask].nunique() < 2 or y[test_mask].nunique() < 2:
        print(f"\n===== target={target} skipped: train/test positive sample shortage =====")
        return

    print_training_filter(target, training_filter)
    print_eval_split(split_report)
    calibrator, calibration_info = fit_year_oof_calibrator(X, y, d["season"], train_years, quality_mask)
    model = make_model()
    model.fit(X[train_mask], y[train_mask])
    raw_p = model.predict_proba(X[test_mask])[:, 1]
    display_p = calibrator.transform(raw_p) if calibrator is not None else raw_p
    y_test = y[test_mask]
    raw_metrics = score(y_test, raw_p)
    display_metrics = score(y_test, display_p)
    per_circuit = per_circuit_metrics(d[test_mask], y_test, raw_p, display_p, target)

    print(f"\n===== target={target} | train={train_years} test={test_year} =====")
    print(f"raw ROC-AUC : {raw_metrics['roc_auc']:.3f}")
    print(f"raw PR-AUC  : {raw_metrics['pr_auc']:.3f}")
    print(f"raw Brier   : {raw_metrics['brier']:.3f}")
    print(f"display Brier: {display_metrics['brier']:.3f}")
    print_per_circuit(per_circuit)
    save_model(
        model,
        target,
        run_name,
        train_years,
        test_year,
        raw_metrics,
        display_metrics,
        calibrator,
        calibration_info,
        per_circuit,
        training_filter,
        split_report,
    )


def train_final_target(df, target, years, run_name):
    d = df[df["season"].isin(years)].dropna(subset=[target]).copy()
    X = d[FEATS].astype(float).fillna(MISSING_VALUE_FILL)
    y = d[target].astype(int)
    quality_mask, training_filter = target_training_mask(d, target, years)
    if y[quality_mask].nunique() < 2:
        return

    print_training_filter(target, training_filter)
    raw_oof, y_oof, folds = year_oof_predictions(X, y, d["season"], years, quality_mask)
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(raw_oof, y_oof)
    raw_metrics = score(y_oof, raw_oof)
    display_metrics = score(y_oof, calibrator.transform(raw_oof))
    calibration_info = {"method": "isotonic_year_oof", "rows": int(len(y_oof)), "folds": folds}

    model = make_model()
    model.fit(X[quality_mask], y[quality_mask])
    print(f"\n===== final target={target} | train={years} =====")
    print(f"OOF raw ROC-AUC : {raw_metrics['roc_auc']:.3f}")
    print(f"OOF raw PR-AUC  : {raw_metrics['pr_auc']:.3f}")
    print(f"OOF display Brier: {display_metrics['brier']:.3f}")
    save_model(
        model,
        target,
        run_name,
        years,
        None,
        raw_metrics,
        display_metrics,
        calibrator,
        calibration_info,
        [],
        training_filter,
        None,
    )


def fit_year_oof_calibrator(X, y, seasons, train_years, quality_mask):
    raw_parts, y_parts, folds = [], [], []
    train_years_set = set(train_years)
    for valid_year in train_years:
        valid_mask = (seasons == valid_year) & quality_mask
        fit_mask = seasons.isin(train_years_set - {valid_year}) & quality_mask
        if y[fit_mask].nunique() < 2 or y[valid_mask].nunique() < 2:
            continue
        model = make_model()
        model.fit(X[fit_mask], y[fit_mask])
        raw_parts.append(model.predict_proba(X[valid_mask])[:, 1])
        y_parts.append(y[valid_mask].to_numpy())
        folds.append({
            "fit_years": sorted(int(v) for v in train_years_set - {valid_year}),
            "valid_year": int(valid_year),
            "rows": int(valid_mask.sum()),
        })
    if not raw_parts:
        return None, {"method": "none", "folds": []}

    raw = np.concatenate(raw_parts)
    labels = np.concatenate(y_parts)
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(raw, labels)
    return calibrator, {"method": "isotonic_year_oof", "rows": int(len(labels)), "folds": folds}


def year_oof_predictions(X, y, seasons, years, quality_mask):
    raw_parts, y_parts, folds = [], [], []
    years_set = set(years)
    for valid_year in years:
        valid_mask = (seasons == valid_year) & quality_mask
        fit_mask = seasons.isin(years_set - {valid_year}) & quality_mask
        if y[fit_mask].nunique() < 2 or y[valid_mask].nunique() < 2:
            continue
        model = make_model()
        model.fit(X[fit_mask], y[fit_mask])
        raw_parts.append(model.predict_proba(X[valid_mask])[:, 1])
        y_parts.append(y[valid_mask].to_numpy())
        folds.append({
            "fit_years": sorted(int(v) for v in years_set - {valid_year}),
            "valid_year": int(valid_year),
            "rows": int(valid_mask.sum()),
        })
    if not raw_parts:
        raise RuntimeError("could not create OOF predictions")
    return np.concatenate(raw_parts), np.concatenate(y_parts), folds


def make_model():
    return LGBMClassifier(n_estimators=400, learning_rate=0.05, class_weight="balanced", verbose=-1)


def score(y_true, probability):
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "brier": float(brier_score_loss(y_true, probability)),
    }


def per_circuit_metrics(test_df, y_true, raw_p, display_p, target):
    result = []
    values = pd.DataFrame({
        "circuit_short_name": test_df["circuit_short_name"].to_numpy(),
        "y": y_true.to_numpy() if hasattr(y_true, "to_numpy") else y_true,
        "raw_p": raw_p,
        "display_p": display_p,
    })
    for circuit, group in values.groupby("circuit_short_name"):
        if group["y"].nunique() < 2:
            continue
        result.append({
            "circuit_short_name": circuit,
            "rows": int(len(group)),
            "positive": int(group["y"].sum()),
            "positive_rate": float(group["y"].mean()),
            "raw_roc_auc": float(roc_auc_score(group["y"], group["raw_p"])),
            "raw_pr_auc": float(average_precision_score(group["y"], group["raw_p"])),
            "raw_brier": float(brier_score_loss(group["y"], group["raw_p"])),
            "display_brier": float(brier_score_loss(group["y"], group["display_p"])),
        })
    return result


def print_per_circuit(metrics):
    if not metrics:
        return
    print("per-circuit:")
    for row in metrics:
        print(
            f"  {row['circuit_short_name']:14s} "
            f"PR-AUC={row['raw_pr_auc']:.3f} ROC-AUC={row['raw_roc_auc']:.3f} "
            f"pos={row['positive']}/{row['rows']}"
        )


def save_model(
    model,
    target,
    run_name,
    train_years,
    test_year,
    raw_metrics,
    display_metrics,
    calibrator,
    calibration_info,
    per_circuit,
    training_filter,
    split_report=None,
):
    model_dir = C.DATA_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{run_name}_{target}.txt"
    meta_path = model_dir / f"{run_name}_{target}.json"
    calibration_path = model_dir / f"{run_name}_{target}_calibration.json"

    model.booster_.save_model(str(model_path))
    calibration_payload = calibration_info.copy()
    if calibrator is not None:
        calibration_payload.update({
            "raw_thresholds": [float(v) for v in calibrator.X_thresholds_],
            "display_values": [float(v) for v in calibrator.y_thresholds_],
        })
    calibration_path.write_text(json.dumps(calibration_payload, indent=2))
    meta = {
        "target": target,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "features": FEATS,
        "feature_count": len(FEATS),
        "missing_value_fill": MISSING_VALUE_FILL,
        "event_type_definition": {
            "0": "no_change",
            "1": "on_track_overtake",
            "2": "pit_related_change",
            "3": "retirement_related_change",
            "4": "lapping_pass",
            "5": "penalty_related_change",
            "6": "restart_overtake",
            "7": "uncertain",
        },
        "train_years": train_years,
        "test_year": test_year,
        "probability_for_ui": "display_probability",
        "raw_metrics": raw_metrics,
        "display_metrics": display_metrics,
        "calibration": calibration_info,
        "calibration_path": str(calibration_path),
        "per_circuit_metrics": per_circuit,
        "training_filter": training_filter,
        "split_report": split_report,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"saved model: {model_path}")
    print(f"saved calibration: {calibration_path}")


if __name__ == "__main__":
    main()
