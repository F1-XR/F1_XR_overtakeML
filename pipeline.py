"""2024 바레인 GP 한 경기로 추월 예측 파이프라인을 처음부터 끝까지 관통시킨다.
목적: 정확한 모델이 아니라 '수집 -> 라벨 -> 학습 -> 평가'가 실제로 이어지는지 확인.
실행: python pipeline.py
"""
import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests

import config as C


F1_STATIC_BASE = "https://livetiming.formula1.com/static/"
EVENT_TYPES = {
    0: "no_change",
    1: "on_track_overtake",
    2: "pit_related_change",
    3: "retirement_related_change",
    4: "lapping_pass",
    5: "penalty_related_change",
    6: "restart_overtake",
    7: "uncertain",
}
EVENT_TYPE_BY_NAME = {v: k for k, v in EVENT_TYPES.items()}


def _get(path, params):
    url = C.BASE_URL + path
    for attempt in range(4):
        r = requests.get(url, params=params, timeout=90)
        if r.status_code == 200:
            time.sleep(C.REQUEST_SLEEP)
            return r.json()
        if r.status_code == 429 or 500 <= r.status_code < 600:
            time.sleep(2 * (attempt + 1))
            continue
        r.raise_for_status()
    raise RuntimeError(f"request failed: {url} {params}")


def get_session_key():
    data = _get("/sessions", {"year": C.SEASON, "country_name": C.COUNTRY,
                              "session_name": C.SESSION})
    if not data:
        raise RuntimeError("세션을 못 찾음. config 의 SEASON/COUNTRY/SESSION 확인")
    s = data[0]
    print(f"[session] key={s['session_key']} {s['year']} {s['country_name']} {s['session_name']}")
    return s["session_key"]


def collect(session_key):
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in ["drivers", "position", "intervals", "laps", "stints", "pit", "race_control", "weather"]:
        out = C.RAW_DIR / f"{name}.json"
        if out.exists():
            print(f"[cache] {name}: {out}")
            continue
        try:
            data = _get(f"/{name}", {"session_key": session_key})
        except requests.HTTPError as e:
            if name not in {"pit", "race_control", "weather"} or e.response is None or e.response.status_code != 404:
                raise
            data = []
            print(f"[collect] {name}: unavailable for this session, using 0 rows")
        out.write_text(json.dumps(data))
        print(f"[collect] {name}: {len(data)} rows")
    if C.FETCH_CAR_DATA:
        _collect_driver_endpoint(session_key, "car_data")
    if getattr(C, "FETCH_LOCATION_DATA", False):
        _collect_driver_endpoint(session_key, "location")


def _collect_driver_endpoint(session_key, name):
    out = C.RAW_DIR / f"{name}.json"
    if out.exists():
        print(f"[cache] {name}: {out}")
        return

    drivers = json.loads((C.RAW_DIR / "drivers.json").read_text())
    rows_all = []
    for d in drivers:
        try:
            rows = _get(f"/{name}", {
                "session_key": session_key,
                "driver_number": d["driver_number"],
            })
        except requests.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else None
            if status_code not in {404, 422}:
                raise
            rows = []
            print(f"[collect] {name} #{d['driver_number']}: unavailable ({status_code}), using 0 rows")
        rows_all.extend(rows)
        print(f"[collect] {name} #{d['driver_number']}: {len(rows)}")
    out.write_text(json.dumps(rows_all))


def _load(name):
    return pd.DataFrame(json.loads((C.RAW_DIR / f"{name}.json").read_text()))


def _load_optional(name):
    path = C.RAW_DIR / f"{name}.json"
    if not path.exists():
        return pd.DataFrame()
    return _load(name)


def _race_start(laps):
    """laps 의 1랩 시작 시각 = 그린 라이트 근처. 경기 전(그리드/포메이션) 컷 기준."""
    if laps.empty or "date_start" not in laps.columns or "lap_number" not in laps.columns:
        return None
    l1 = laps[laps["lap_number"] == 1]
    if l1.empty:
        return None
    d = _dt(l1["date_start"]).dropna()
    return d.min() if len(d) else None


def _dt(series):
    try:
        parsed = pd.to_datetime(series, utc=True, errors="coerce", format="mixed")
    except TypeError:
        parsed = pd.to_datetime(series, utc=True, errors="coerce")
    return parsed.dt.tz_localize(None)


def _load_static_json(url):
    r = requests.get(url, timeout=90)
    if r.status_code != 200:
        return None
    return json.loads(r.content.decode("utf-8-sig"))


def _session_path_from_static(year, session_key):
    schedule = _load_static_json(urljoin(F1_STATIC_BASE, f"{year}/Index.json"))
    if schedule is None:
        return None, None

    for meeting in schedule.get("Meetings", []):
        for session in meeting.get("Sessions", []):
            if session.get("Key") == session_key:
                return session.get("Path"), meeting.get("Key")
    return None, None


def _stream_offset(raw):
    h, m, s = raw.split(":")
    return pd.Timedelta(hours=int(h), minutes=int(m), seconds=float(s))


def _split_stream_line(line):
    idx = line.find("{")
    if idx < 0:
        return None, None
    try:
        return _stream_offset(line[:idx]), json.loads(line[idx:])
    except Exception:
        return None, None


def _estimate_t0_from_timing_data(laps, lines):
    if laps.empty or "date_start" not in laps.columns:
        return None

    laps_ok = laps.dropna(subset=["date_start", "driver_number", "lap_number"]).copy()
    if laps_ok.empty:
        return None

    laps_ok["date_start"] = _dt(laps_ok["date_start"])
    lap_starts = {
        (int(row.driver_number), int(row.lap_number)): row.date_start
        for row in laps_ok.itertuples()
        if pd.notna(row.date_start)
    }
    candidates = []
    for line in lines:
        if "NumberOfLaps" not in line:
            continue
        offset, obj = _split_stream_line(line)
        if offset is None:
            continue
        for driver, data in obj.get("Lines", {}).items():
            if not isinstance(data, dict) or "NumberOfLaps" not in data:
                continue
            try:
                key = (int(driver), int(data["NumberOfLaps"]) + 1)
            except Exception:
                continue
            date_start = lap_starts.get(key)
            if date_start is not None:
                candidates.append(date_start - offset)

    if not candidates:
        return None
    return pd.Series(candidates).median()


def _pit_from_timing_data(laps):
    meta_path = C.RAW_DIR / "session_meta.json"
    if not meta_path.exists():
        print("[pit] empty /pit and no session_meta.json; cannot fetch TimingData fallback")
        return pd.DataFrame()

    cache_path = C.RAW_DIR / "pit_timing_data.json"
    if cache_path.exists():
        pit = pd.DataFrame(json.loads(cache_path.read_text()))
        print(f"[pit] loaded TimingData cache rows: {len(pit)}")
        return pit

    meta = json.loads(meta_path.read_text())
    year = int(meta["season"])
    session_key = int(meta["session_key"])
    session_path, meeting_key = _session_path_from_static(year, session_key)
    if not session_path:
        print(f"[pit] empty /pit and static session path not found: year={year} session_key={session_key}")
        return pd.DataFrame()

    r = requests.get(urljoin(F1_STATIC_BASE, session_path + "TimingData.jsonStream"), timeout=90)
    if r.status_code != 200:
        print(f"[pit] empty /pit and TimingData unavailable: status={r.status_code}")
        return pd.DataFrame()

    lines = [line for line in r.text.split("\r\n") if line]
    t0 = _estimate_t0_from_timing_data(laps, lines)
    if t0 is None:
        print("[pit] empty /pit and TimingData t0 alignment failed")
        return pd.DataFrame()

    states = {}
    events = []
    for line in lines:
        if "InPit" not in line and "NumberOfPitStops" not in line and "NumberOfLaps" not in line:
            continue
        offset, obj = _split_stream_line(line)
        if offset is None:
            continue
        date = t0 + offset
        for driver, data in obj.get("Lines", {}).items():
            if not isinstance(data, dict):
                continue
            try:
                driver = int(driver)
            except Exception:
                continue

            state = states.setdefault(driver, {
                "in_pit": None,
                "entry": None,
                "laps": 0,
                "pit_count": 0,
                "open_event": None,
            })
            if "NumberOfLaps" in data:
                try:
                    state["laps"] = int(data["NumberOfLaps"])
                except Exception:
                    pass

            if "InPit" in data:
                in_pit = bool(data["InPit"])
                if in_pit and not state["in_pit"]:
                    state["entry"] = date
                elif not in_pit and state["in_pit"]:
                    event = state["open_event"]
                    if event is not None and event["date_end"] is None:
                        event["date"] = date
                        event["date_end"] = date
                        event["lane_duration"] = (date - event["date_start"]).total_seconds()
                        event["pit_duration"] = event["lane_duration"]
                    state["entry"] = None
                    state["open_event"] = None
                state["in_pit"] = in_pit

            if "NumberOfPitStops" in data:
                try:
                    pit_count = int(data["NumberOfPitStops"])
                except Exception:
                    pit_count = state["pit_count"]
                if pit_count > state["pit_count"]:
                    start = state["entry"] if state["entry"] is not None else date
                    event = {
                        "date": None,
                        "date_start": start,
                        "date_end": None,
                        "driver_number": driver,
                        "lap_number": int(state["laps"]) + 1,
                        "meeting_key": meeting_key,
                        "session_key": session_key,
                        "lane_duration": np.nan,
                        "pit_duration": np.nan,
                        "stop_duration": np.nan,
                        "source": "timing_data",
                    }
                    events.append(event)
                    state["open_event"] = event
                state["pit_count"] = max(state["pit_count"], pit_count)

    pit = pd.DataFrame(events)
    if pit.empty:
        return pit
    pit["date"] = pit["date"].fillna(pit["date_start"])
    pit["date_end"] = pit["date_end"].fillna(pit["date"])
    cache_path.write_text(json.dumps(pit.to_dict("records"), default=str))
    print(f"[pit] rebuilt from TimingData rows: {len(pit)}")
    return pit


def _normalize_pit(pit):
    if pit.empty:
        return pit

    pit = pit.copy()
    if "date" in pit.columns:
        pit["date"] = _dt(pit["date"])
    if "date_end" in pit.columns:
        pit["date_end"] = _dt(pit["date_end"])
    elif "date" in pit.columns:
        pit["date_end"] = pit["date"]

    if "date_start" in pit.columns:
        pit["date_start"] = _dt(pit["date_start"])
    elif "date_end" in pit.columns:
        if "lane_duration" in pit.columns:
            duration = pd.to_numeric(pit["lane_duration"], errors="coerce").fillna(0)
            pit["date_start"] = pit["date_end"] - pd.to_timedelta(duration, unit="s")
        else:
            pit["date_start"] = pit["date_end"]
    return pit


def build_dataset():
    C.PROC_DIR.mkdir(parents=True, exist_ok=True)
    pos = _load("position")
    pos["date"] = _dt(pos["date"])
    itv = _load("intervals")
    itv["date"] = _dt(itv["date"])
    itv["interval"] = pd.to_numeric(itv["interval"], errors="coerce")
    car = _load_optional("car_data")
    if not car.empty:
        car["date"] = _dt(car["date"])
        car["speed"] = pd.to_numeric(car["speed"], errors="coerce")
        car["drs"] = pd.to_numeric(car["drs"], errors="coerce")
    loc = _load_optional("location")
    if not loc.empty:
        loc["date"] = _dt(loc["date"])
        for col in ["x", "y", "z"]:
            loc[col] = pd.to_numeric(loc[col], errors="coerce")
    laps = _load("laps")
    stints = _load("stints")
    pit = _load("pit")
    if not pit.empty:
        pit = _normalize_pit(pit)
    else:
        pit = _normalize_pit(_pit_from_timing_data(laps))
    race_control = _load_optional("race_control")
    if not race_control.empty and "date" in race_control.columns:
        race_control["date"] = _dt(race_control["date"])
    weather = _load_optional("weather")
    if not weather.empty and "date" in weather.columns:
        weather["date"] = _dt(weather["date"])
        for col in ["air_temperature", "track_temperature", "humidity", "rainfall"]:
            if col in weather.columns:
                weather[col] = pd.to_numeric(weather[col], errors="coerce")

    t0, t1 = pos["date"].min(), pos["date"].max()
    grid = pd.date_range(t0, t1, freq=f"{int(C.GRID_S)}s")
    grid_np = grid.to_numpy()
    drivers = sorted(pos["driver_number"].unique())

    laps_ok = laps.dropna(subset=["date_start"]).copy()
    if not laps_ok.empty:
        laps_ok["date_start"] = _dt(laps_ok["date_start"])
        for col in ["duration_sector_1", "duration_sector_2", "lap_duration"]:
            if col in laps_ok.columns:
                laps_ok[col] = pd.to_numeric(laps_ok[col], errors="coerce")
        laps_ok = _attach_lap_end(laps_ok)
    track_ref = _build_track_reference(loc, laps_ok)

    frames = []
    for dn in drivers:
        g = pd.DataFrame({"t": grid})
        g["driver"] = dn
        g["position"] = _resample(pos, dn, "position", grid)
        g["gap_ahead"] = _resample(itv, dn, "interval", grid)
        g["speed"] = _resample(car, dn, "speed", grid)
        drs = _resample(car, dn, "drs", grid)
        g["drs_active"] = np.isin(drs, [10, 12, 14]).astype(int)
        g["air_temperature"] = _resample_global(weather, "air_temperature", grid)
        g["track_temperature"] = _resample_global(weather, "track_temperature", grid)
        g["humidity"] = _resample_global(weather, "humidity", grid)
        rainfall = _resample_global(weather, "rainfall", grid)
        g["rainfall"] = rainfall
        g["weather_regime_code"] = _weather_regime_code(rainfall)
        g["x"] = _resample(loc, dn, "x", grid)
        g["y"] = _resample(loc, dn, "y", grid)
        g["z"] = _resample(loc, dn, "z", grid)
        # 시간 -> 랩 번호
        dl = laps_ok[laps_ok.driver_number == dn].sort_values("date_start")
        if not dl.empty:
            ds = dl["date_start"].to_numpy()
            g["lap"] = np.clip(np.searchsorted(ds, grid_np, side="right"), 1, None)
        else:
            g["lap"] = np.nan
        g["is_lap1"] = (g["lap"] == 1).astype(int)
        g = _attach_tire(g, stints[stints.driver_number == dn])
        g = _attach_sector(g, laps_ok[laps_ok.driver_number == dn])
        g = _attach_track_progress(g, track_ref)
        frames.append(g)

    df = pd.concat(frames, ignore_index=True)
    df = _attach_ahead(df)
    df = _features(df)
    df = _event_labels(df, pit, race_control)
    rs = _race_start(laps)                      # 경기 전(그리드/포메이션) 컷
    if rs is not None:
        before = len(df)
        df = df[df["t"] >= rs]
        print(f"[dataset] race start >= {rs} (경기 전 {before - len(df)} rows 제거)")
    df = df[df["gap_ahead"] <= C.BATTLE_GAP_S].reset_index(drop=True)  # 배틀 상황만
    df = _attach_session_meta(df)
    out = C.PROC_DIR / "samples.parquet"
    df.to_parquet(out)
    print(
        f"[dataset] samples={len(df)} "
        f"overtake={int(df['label_overtake'].sum())} "
        f"position_gain={int(df['label_position_gain'].sum())} "
        f"position_loss={int(df['label_position_loss'].sum())} "
        f"position_change={int(df['label_position_change'].sum())} -> {out}"
    )
    return df


def _attach_session_meta(df):
    meta_path = C.RAW_DIR / "session_meta.json"
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())

    circuit_key = meta.get("circuit_key")
    df["season"] = meta.get("season", getattr(C, "SEASON", np.nan))
    df["country"] = meta.get("country", getattr(C, "COUNTRY", ""))
    df["session_name"] = meta.get("session", getattr(C, "SESSION", ""))
    df["session_key"] = meta.get("session_key", np.nan)
    df["meeting_key"] = meta.get("meeting_key", np.nan)
    df["circuit_key"] = circuit_key if circuit_key is not None else np.nan
    df["circuit_id"] = df["circuit_key"]
    df["circuit_short_name"] = meta.get("circuit_short_name", "")
    df["circuit_type"] = meta.get("circuit_type", "Unknown")
    df["circuit_type_code"] = _circuit_type_code(df["circuit_type"].iloc[0])
    return df


def _circuit_type_code(value):
    value = str(value or "").strip().lower()
    mapping = {
        "permanent": 0,
        "temporary - street": 1,
        "temporary - road": 2,
    }
    return mapping.get(value, -1)


def _resample(src, dn, col, grid):
    if src.empty or col not in src.columns:
        return np.full(len(grid), np.nan)
    s = src[src.driver_number == dn][["date", col]].dropna(subset=["date"])
    if s.empty:
        return np.full(len(grid), np.nan)
    s = s.set_index("date")[col].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    idx = s.index.union(grid)
    return s.reindex(idx).ffill().reindex(grid).to_numpy()


def _resample_global(src, col, grid):
    if src.empty or col not in src.columns or "date" not in src.columns:
        return np.full(len(grid), np.nan)
    s = src[["date", col]].dropna(subset=["date"]).sort_values("date")
    if s.empty:
        return np.full(len(grid), np.nan)
    s = s.set_index("date")[col].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    idx = s.index.union(grid)
    return s.reindex(idx).ffill().reindex(grid).to_numpy()


def _weather_regime_code(rainfall):
    rain = np.nan_to_num(rainfall, nan=0.0) > 0
    code = np.where(rain, 2, 0)
    if len(code) <= 1:
        return code

    transition_steps = int(round(float(getattr(C, "WEATHER_TRANSITION_S", 300.0)) / float(C.GRID_S)))
    changes = np.flatnonzero(rain[1:] != rain[:-1]) + 1
    for idx in changes:
        end = min(len(code), idx + max(transition_steps, 1))
        code[idx:end] = 1
    return code


def _attach_lap_end(laps):
    laps = laps.sort_values(["driver_number", "date_start"]).copy()
    laps["date_end"] = laps.groupby("driver_number")["date_start"].shift(-1)
    if "lap_duration" in laps.columns:
        inferred = laps["date_start"] + pd.to_timedelta(laps["lap_duration"], unit="s")
        laps["date_end"] = laps["date_end"].fillna(inferred)
    return laps


def _build_track_reference(loc, laps):
    if loc.empty or laps.empty:
        return None

    loc_ok = loc.dropna(subset=["date", "driver_number", "x", "y"]).sort_values("date")
    if loc_ok.empty or "date_end" not in laps.columns:
        return None

    for dn in sorted(loc_ok["driver_number"].dropna().unique()):
        loc_driver = loc_ok[loc_ok.driver_number == dn]
        laps_driver = laps[(laps.driver_number == dn) & (laps["lap_number"] > 1)].copy()
        if "is_pit_out_lap" in laps_driver.columns:
            laps_driver = laps_driver[laps_driver["is_pit_out_lap"] != True]
        laps_driver = laps_driver.dropna(subset=["date_start", "date_end"]).sort_values("date_start")

        for row in laps_driver.itertuples():
            if row.date_end <= row.date_start:
                continue
            pts = loc_driver[(loc_driver["date"] >= row.date_start) & (loc_driver["date"] < row.date_end)]
            if len(pts) < 200:
                continue
            ref = _make_track_reference(pts)
            if ref is not None:
                print(f"[location] track reference driver={int(dn)} lap={int(row.lap_number)} points={len(ref['xy'])}")
                return ref
    print("[location] track reference unavailable")
    return None


def _make_track_reference(points):
    xy = points.sort_values("date")[["x", "y"]].to_numpy(dtype=float)
    xy = xy[np.isfinite(xy).all(axis=1)]
    if len(xy) < 200:
        return None

    if len(xy) > 1200:
        idx = np.linspace(0, len(xy) - 1, 1200).astype(int)
        xy = xy[idx]

    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    keep = np.r_[True, step > 1e-6]
    xy = xy[keep]
    if len(xy) < 50:
        return None

    dist = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    if dist[-1] <= 0:
        return None

    from sklearn.neighbors import KDTree

    return {
        "xy": xy,
        "progress": dist / dist[-1],
        "tree": KDTree(xy),
    }


def _attach_sector(g, laps):
    g["sector"] = np.nan
    if laps.empty or "date_end" not in laps.columns:
        return g

    times = g["t"].to_numpy()
    sector = np.full(len(g), np.nan)
    for row in laps.dropna(subset=["date_start", "date_end"]).itertuples():
        s1 = getattr(row, "duration_sector_1", np.nan)
        s2 = getattr(row, "duration_sector_2", np.nan)
        if pd.isna(s1) or pd.isna(s2):
            continue

        start = np.datetime64(row.date_start)
        s1_end = np.datetime64(row.date_start + pd.to_timedelta(float(s1), unit="s"))
        s2_end = np.datetime64(row.date_start + pd.to_timedelta(float(s1 + s2), unit="s"))
        end = np.datetime64(row.date_end)

        mask = (times >= start) & (times < end)
        sector[mask] = 3
        sector[(times >= start) & (times < s1_end)] = 1
        sector[(times >= s1_end) & (times < s2_end)] = 2

    g["sector"] = sector
    return g


def _attach_track_progress(g, reference):
    g["track_progress"] = np.nan
    g["track_progress_sin"] = np.nan
    g["track_progress_cos"] = np.nan
    g["segment"] = np.nan
    if reference is None:
        return g

    xy = g[["x", "y"]].to_numpy(dtype=float)
    valid = np.isfinite(xy).all(axis=1)
    if not valid.any():
        return g

    _, idx = reference["tree"].query(xy[valid], k=1)
    progress = reference["progress"][idx[:, 0]]
    g.loc[valid, "track_progress"] = progress
    g.loc[valid, "track_progress_sin"] = np.sin(2 * np.pi * progress)
    g.loc[valid, "track_progress_cos"] = np.cos(2 * np.pi * progress)

    n_segments = int(getattr(C, "TRACK_SEGMENTS", 30))
    segment = np.floor(progress * n_segments)
    segment = np.clip(segment, 0, n_segments - 1)
    g.loc[valid, "segment"] = segment
    return g


def _attach_tire(g, st):
    g["compound"] = "UNKNOWN"
    g["tyre_age"] = np.nan
    for _, s in st.iterrows():
        m = (g["lap"] >= s["lap_start"]) & (g["lap"] <= s["lap_end"])
        g.loc[m, "compound"] = s.get("compound", "UNKNOWN")
        g.loc[m, "tyre_age"] = s.get("tyre_age_at_start", 0) + (g.loc[m, "lap"] - s["lap_start"])
    return g


def _attach_ahead(df):
    ahead = df[["t", "position", "driver", "lap", "tyre_age", "speed"]].copy()
    ahead = ahead.rename(columns={
        "position": "ahead_position",
        "driver": "ahead_driver",
        "lap": "ahead_lap",
        "tyre_age": "ahead_tyre_age",
        "speed": "ahead_speed",
    })
    ahead["position"] = ahead["ahead_position"] + 1  # 내 순위 바로 앞차와 매칭
    return df.merge(ahead, on=["t", "position"], how="left")


def _features(df):
    df = df.sort_values(["driver", "t"]).reset_index(drop=True)
    df["gap_trend"] = df.groupby("driver")["gap_ahead"].diff(5)   # 음수면 붙는 중
    df["drs_range"] = (df["gap_ahead"] < 1.0).astype(int)
    df["speed_delta"] = df["speed"] - df["ahead_speed"]
    df["position_delta"] = df["position"] - df["ahead_position"]
    df["same_lap"] = np.where(
        df["lap"].notna() & df["ahead_lap"].notna(),
        (df["lap"] == df["ahead_lap"]).astype(int),
        np.nan,
    )
    if "track_progress" in df.columns and "sector" in df.columns:
        sector_from_progress = np.floor(df["track_progress"] * 3) + 1
        sector_from_progress = sector_from_progress.clip(1, 3)
        df["sector"] = df["sector"].fillna(sector_from_progress)
    df["tyre_age_delta"] = df["tyre_age"] - df["ahead_tyre_age"]  # +면 내 타이어가 더 낡음
    df["tyre_delta"] = df["tyre_age_delta"]
    return df


def _labels(df, pit):
    df = df.sort_values(["driver", "t"]).reset_index(drop=True)
    h = int(round(C.HORIZON_S / C.GRID_S))
    pit_by = {}
    if not pit.empty:
        for dn, gg in pit.groupby("driver_number"):
            pit_by[dn] = gg["date"].to_numpy()
    parts = []
    for dn, g in df.groupby("driver"):
        g = g.copy()
        pos = g["position"].to_numpy()
        tt = g["t"].to_numpy()
        n = len(pos)
        label = np.zeros(n)
        pit_win = np.zeros(n)
        pits = pit_by.get(dn, np.array([], dtype="datetime64[ns]"))
        for i in range(n):
            end = min(n, i + h + 1)
            fut = pos[i + 1:end]
            if fut.size and np.nanmin(fut) < pos[i]:   # 앞으로 순위 상승
                label[i] = 1
            if pits.size:
                w0, w1 = tt[i], tt[min(n - 1, end - 1)]
                if ((pits >= w0) & (pits <= w1)).any():  # 윈도우에 본인 피트
                    pit_win[i] = 1
        g["label"] = label
        g["pit_in_window"] = pit_win
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    return out[out["pit_in_window"] == 0].drop(columns=["pit_in_window"])


def _event_labels(df, pit, race_control):
    df = df.sort_values(["driver", "t"]).reset_index(drop=True)
    h = int(round(C.HORIZON_S / C.GRID_S))
    position_grid = df.pivot_table(index="t", columns="driver", values="position", aggfunc="last").sort_index()
    pit_by = _pit_windows_by_driver(pit)
    control_windows = _race_control_windows(race_control)
    restart_windows = _restart_windows(control_windows)
    retirement_by = _driver_event_windows(
        race_control,
        [
            r"\bSTOPPED\b",
            r"\bRETIRED\b",
            r"\bCAR\s+STOPPED\b",
            r"\bPULLED\s+OFF\b",
        ],
    )
    penalty_by = _driver_event_windows(
        race_control,
        [
            r"\bPENALTY\b",
            r"\bDRIVE\s+THROUGH\b",
            r"\bSTOP\s*(?:-|AND\s+)?GO\b",
            r"\bTIME\s+PENALTY\b",
        ],
    )

    parts = []
    for dn, g in df.groupby("driver"):
        g = g.copy()
        pos = g["position"].to_numpy(dtype=float)
        times = g["t"].to_numpy()
        ahead = g["ahead_driver"].to_numpy()
        same_lap = g["same_lap"].to_numpy(dtype=float)
        label_overtake = np.zeros(len(g), dtype=int)
        label_position_gain = np.zeros(len(g), dtype=int)
        label_position_loss = np.zeros(len(g), dtype=int)
        label_position_change = np.zeros(len(g), dtype=int)
        event_type = np.zeros(len(g), dtype=int)
        pit_window = np.zeros(len(g), dtype=int)
        control_window = np.zeros(len(g), dtype=int)
        retirement_window = np.zeros(len(g), dtype=int)
        penalty_window = np.zeros(len(g), dtype=int)
        restart_phase = np.zeros(len(g), dtype=int)

        for i in range(len(g)):
            end = min(len(g), i + h + 1)
            if end <= i + 1 or np.isnan(pos[i]):
                continue

            w0 = times[i]
            w1 = times[end - 1]
            own_pit = _driver_pits_in_window(pit_by, int(dn), w0, w1)
            control = _overlaps_control_window(control_windows, w0, w1)
            restart = _overlaps_window(restart_windows, w0, w1)
            ahead_driver = int(ahead[i]) if not pd.isna(ahead[i]) else None
            ahead_pit = ahead_driver is not None and _driver_pits_in_window(pit_by, ahead_driver, w0, w1)
            retirement = _driver_event_in_window(retirement_by, int(dn), w0, w1)
            penalty = _driver_event_in_window(penalty_by, int(dn), w0, w1)
            if ahead_driver is not None:
                retirement = retirement or _driver_event_in_window(retirement_by, ahead_driver, w0, w1)
                penalty = penalty or _driver_event_in_window(penalty_by, ahead_driver, w0, w1)

            pit_window[i] = int(own_pit or ahead_pit)
            control_window[i] = int(control)
            retirement_window[i] = int(retirement)
            penalty_window[i] = int(penalty)
            restart_phase[i] = int(restart)

            future_times = g["t"].iloc[i + 1:end]
            future_i = pos[i + 1:end]
            valid_i = ~np.isnan(future_i)
            if valid_i.any():
                if np.nanmin(future_i[valid_i]) < pos[i]:
                    label_position_gain[i] = 1
                if np.nanmax(future_i[valid_i]) > pos[i]:
                    label_position_loss[i] = 1
                if label_position_gain[i] or label_position_loss[i]:
                    label_position_change[i] = 1

            candidate_overtake = False
            if ahead_driver is not None and ahead_driver in position_grid.columns:
                future_j = position_grid[ahead_driver].reindex(future_times).to_numpy(dtype=float)
                valid_pair = ~np.isnan(future_i) & ~np.isnan(future_j)
                candidate_overtake = bool(valid_pair.any() and np.any(future_i[valid_pair] < future_j[valid_pair]))

            if candidate_overtake or label_position_change[i]:
                same_lap_pair = not np.isnan(same_lap[i]) and int(same_lap[i]) == 1
                if control:
                    event_type[i] = EVENT_TYPE_BY_NAME["uncertain"]
                elif own_pit or ahead_pit:
                    event_type[i] = EVENT_TYPE_BY_NAME["pit_related_change"]
                elif retirement:
                    event_type[i] = EVENT_TYPE_BY_NAME["retirement_related_change"]
                elif penalty:
                    event_type[i] = EVENT_TYPE_BY_NAME["penalty_related_change"]
                elif candidate_overtake and not same_lap_pair:
                    event_type[i] = EVENT_TYPE_BY_NAME["lapping_pass"]
                elif candidate_overtake and restart:
                    event_type[i] = EVENT_TYPE_BY_NAME["restart_overtake"]
                elif candidate_overtake and same_lap_pair:
                    event_type[i] = EVENT_TYPE_BY_NAME["on_track_overtake"]
                else:
                    event_type[i] = EVENT_TYPE_BY_NAME["uncertain"]
            label_overtake[i] = int(event_type[i] == EVENT_TYPE_BY_NAME["on_track_overtake"])

        g["label_overtake"] = label_overtake
        g["label_position_gain"] = label_position_gain
        g["label_position_loss"] = label_position_loss
        g["label_position_change"] = label_position_change
        g["label"] = label_overtake
        g["event_type"] = event_type
        g["change_reason"] = [EVENT_TYPES.get(int(v), "uncertain") for v in event_type]
        g["pit_window"] = pit_window
        g["control_window"] = control_window
        g["retirement_window"] = retirement_window
        g["penalty_window"] = penalty_window
        g["restart_phase"] = restart_phase
        parts.append(g)

    out = pd.concat(parts, ignore_index=True)
    counts = out["change_reason"].value_counts().to_dict()
    print(f"[labels] event_type counts: {counts}")
    print(
        "[labels] flagged windows: "
        f"pit={int(out['pit_window'].sum())} "
        f"control={int(out['control_window'].sum())} "
        f"retirement={int(out['retirement_window'].sum())} "
        f"penalty={int(out['penalty_window'].sum())} "
        f"restart={int(out['restart_phase'].sum())}"
    )
    return out


def _driver_pits_in_window(pit_by, driver, w0, w1):
    return any(start <= w1 and end >= w0 for start, end in pit_by.get(int(driver), []))


def _driver_event_in_window(event_by, driver, w0, w1):
    return any(start <= w1 and end >= w0 for start, end in event_by.get(int(driver), []))


def _pit_windows_by_driver(pit):
    if pit.empty or "driver_number" not in pit.columns:
        return {}

    pit = _normalize_pit(pit)
    if "date_start" not in pit.columns or "date_end" not in pit.columns:
        return {}

    pit_by = {}
    for dn, gg in pit.dropna(subset=["date_start", "date_end"]).groupby("driver_number"):
        windows = []
        for row in gg.itertuples():
            start = row.date_start
            end = row.date_end
            if end < start:
                start, end = end, start
            windows.append((np.datetime64(start), np.datetime64(end)))
        pit_by[int(dn)] = windows
    return pit_by


def _driver_event_windows(race_control, patterns):
    if race_control.empty or "date" not in race_control.columns or "driver_number" not in race_control.columns:
        return {}

    event_by = {}
    horizon = pd.to_timedelta(float(getattr(C, "HORIZON_S", 30.0)), unit="s")
    for row in race_control.dropna(subset=["date", "driver_number"]).itertuples():
        text = " ".join(str(getattr(row, name, "")) for name in ("category", "flag", "message")).upper()
        if not any(re.search(pattern, text) for pattern in patterns):
            continue
        try:
            driver = int(getattr(row, "driver_number"))
        except (TypeError, ValueError):
            continue
        start = np.datetime64(getattr(row, "date"))
        end = np.datetime64(getattr(row, "date") + horizon)
        event_by.setdefault(driver, []).append((start, end))
    return event_by


def _race_control_windows(race_control):
    if race_control.empty or "date" not in race_control.columns:
        return []

    windows = []
    sc_start = None
    vsc_start = None
    red_start = None
    rc = race_control.dropna(subset=["date"]).sort_values("date")
    if rc.empty:
        return windows

    last_date = rc["date"].max()
    for _, row in rc.iterrows():
        date = row["date"]
        category = str(row.get("category", "")).upper()
        flag = str(row.get("flag", "")).strip().upper()
        message = str(row.get("message", "")).strip().upper()
        text = f"{category} {flag} {message}"

        if _is_red_flag(flag, message):
            red_start = red_start or date
        elif red_start is not None and (
            "GREEN" in flag
            or "SESSION RESUMED" in text
            or "SESSION STARTED" in text
        ):
            windows.append((np.datetime64(red_start), np.datetime64(date), "red"))
            red_start = None

        if category != "SAFETYCAR" or "SAFETY CAR" not in text:
            continue

        is_virtual = "VIRTUAL" in text
        is_ending = any(token in text for token in ("ENDING", "ENDED", "IN THIS LAP", "WITHDRAWN"))
        if is_virtual:
            if is_ending and vsc_start is not None:
                windows.append((np.datetime64(vsc_start), np.datetime64(date), "vsc"))
                vsc_start = None
            elif not is_ending and vsc_start is None:
                vsc_start = date
        else:
            if is_ending and sc_start is not None:
                windows.append((np.datetime64(sc_start), np.datetime64(date), "sc"))
                sc_start = None
            elif not is_ending and sc_start is None:
                sc_start = date

    for start, kind in ((sc_start, "sc"), (vsc_start, "vsc"), (red_start, "red")):
        if start is not None:
            windows.append((np.datetime64(start), np.datetime64(last_date), kind))
    return windows


def _is_red_flag(flag, message):
    return (
        flag in {"RED", "RED FLAG"}
        or bool(re.search(r"\bRED\s+FLAG\b", message))
        or bool(re.search(r"\bSESSION\s+SUSPENDED\b", message))
    )


def _restart_windows(control_windows):
    duration = pd.to_timedelta(float(getattr(C, "RESTART_PHASE_S", getattr(C, "HORIZON_S", 30.0))), unit="s")
    windows = []
    for item in control_windows:
        start, end = item[0], item[1]
        if pd.isna(end):
            continue
        restart_start = pd.Timestamp(end)
        restart_end = restart_start + duration
        windows.append((np.datetime64(restart_start), np.datetime64(restart_end)))
    return windows


def _overlaps_window(windows, w0, w1):
    return any(start <= w1 and end >= w0 for start, end in windows)


def _overlaps_control_window(windows, w0, w1):
    return any(item[0] <= w1 and item[1] >= w0 for item in windows)



def add_tire(df):
    """laps 랩타임으로 타이어 마모 회귀. df 에 laptime_delta(2랩 뒤 증가초) 추가."""
    try:
        from lightgbm import LGBMRegressor
    except Exception:
        df["laptime_delta"] = np.nan
        return df
    laps = _load("laps")
    if laps.empty or "lap_duration" not in laps.columns:
        df["laptime_delta"] = np.nan
        return df
    laps = laps.copy()
    laps["lap_duration"] = pd.to_numeric(laps["lap_duration"], errors="coerce")
    laps["lap_number"] = pd.to_numeric(laps["lap_number"], errors="coerce")
    laps = laps.dropna(subset=["lap_duration", "lap_number", "driver_number"])
    laps = laps[laps["lap_number"] > 1]
    if "is_pit_out_lap" in laps.columns:
        laps = laps[laps["is_pit_out_lap"] != True]
    key = (df.groupby(["driver", "lap"])
             .agg(compound=("compound", "first"), tyre_age=("tyre_age", "first"))
             .reset_index())
    comp_map = {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTERMEDIATE": 3, "WET": 4}
    t = laps.merge(key, left_on=["driver_number", "lap_number"],
                   right_on=["driver", "lap"], how="inner").dropna(subset=["tyre_age"])
    med = t.groupby("driver_number")["lap_duration"].transform("median")
    t = t[t["lap_duration"] <= med * 1.12]              # 세이프티카/트래픽 이상치 제거
    t["comp"] = t["compound"].astype(str).str.upper().map(comp_map).fillna(-1)
    if len(t) < 40:
        print("[tire] 랩 데이터 부족 -> laptime_delta 생략")
        df["laptime_delta"] = np.nan
        return df
    Xc = ["tyre_age", "lap_number", "comp"]
    reg = LGBMRegressor(n_estimators=200, learning_rate=0.05, verbose=-1)
    reg.fit(t[Xc].astype(float), t["lap_duration"].astype(float))
    k = key.dropna(subset=["tyre_age"]).copy()
    k["comp"] = k["compound"].astype(str).str.upper().map(comp_map).fillna(-1)
    k["lap_number"] = k["lap"]
    now = reg.predict(k[Xc].astype(float))
    kf = k.copy(); kf["tyre_age"] += 2; kf["lap_number"] += 2
    fut = reg.predict(kf[Xc].astype(float))
    k["laptime_delta"] = fut - now
    df = df.merge(k[["driver", "lap", "laptime_delta"]], on=["driver", "lap"], how="left")
    print(f"[tire] laptime 회귀 rows={len(t)} | delta 중앙값={np.nanmedian(k['laptime_delta']):+.2f}s")
    return df


def add_undercut(df):
    """언더컷 성공 추정. 한 경기론 학습 샘플이 거의 없어 규칙 기반 추정(휴리스틱).
    여러 경기 모으면 pit+position 라벨로 학습 모델 교체 예정."""
    gap = df["gap_ahead"].fillna(3.0)
    my_age = df["tyre_age"].fillna(0.0)
    ahead_age = df["ahead_tyre_age"].fillna(0.0)
    # 내 타이어가 앞차보다 낡을수록(교체 이득) + gap 작을수록 + 앞차도 아직 신선할수록 유리
    score = 0.55 * (my_age - ahead_age) / 10.0 + 0.6 * (1.5 - gap) + 0.15 * (20 - ahead_age) / 20.0
    df["undercut_prob"] = 1.0 / (1.0 + np.exp(-score))
    return df


def train(df):
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

    target = getattr(C, "TARGET_LABEL", "label_overtake")
    if target not in df.columns:
        target = "label"
    feats = [
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
    d = df.dropna(subset=[target]).copy()
    X = d[feats].astype(float).fillna(-1)
    y = d[target].astype(int)
    if y.nunique() < 2:
        print("[train] positive 샘플 부족 -> HORIZON 늘리거나 경기 추가")
        return

    cut = int(len(d) * 0.7)                 # 단일 경기: 시간 기준 앞70/뒤30
    order = np.argsort(d["t"].to_numpy())
    tr, te = order[:cut], order[cut:]
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]

    model = LGBMClassifier(n_estimators=300, learning_rate=0.05, class_weight="balanced")
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]

    print("\n===== 평가 (관통용 · '동작 확인'이 목적, 정확도는 부차) =====")
    if yte.nunique() > 1:
        print(f"  ROC-AUC : {roc_auc_score(yte, p):.3f}")
        print(f"  PR-AUC  : {average_precision_score(yte, p):.3f}")
        print(f"  Brier   : {brier_score_loss(yte, p):.3f}")
        base = ((Xte['gap_ahead'] < 1.0) & (Xte['gap_trend'] < 0)).astype(int)
        print(f"  baseline ROC-AUC(규칙): {roc_auc_score(yte, base):.3f}")
    print("  피처 중요도:")
    for f, imp in sorted(zip(feats, model.feature_importances_), key=lambda x: -x[1]):
        print(f"    {f:12s} {imp}")


def main():
    print("=== F1 추월 예측 파이프라인 관통 (2024 Bahrain GP) ===")
    key = get_session_key()
    collect(key)
    df = build_dataset()
    df = add_tire(df)
    df = add_undercut(df)
    df.to_parquet(C.PROC_DIR / "samples.parquet")
    train(df)
    print("\n완료. data/raw, data/processed 확인.")


if __name__ == "__main__":
    main()
