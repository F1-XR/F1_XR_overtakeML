"""파이프라인 설정. 여기 값만 바꾸면 대상 경기/윈도우를 조정할 수 있다."""
from pathlib import Path

# --- 대상 세션 (기본: 2024 바레인 GP 레이스) ---
BASE_URL = "https://api.openf1.org/v1"
SEASON = 2024
COUNTRY = "Bahrain"
SESSION = "Race"

# --- 처리 파라미터 ---
GRID_S = 1.0          # 공통 시간격자 간격(초)
HORIZON_S = 30.0      # 라벨: 앞으로 몇 초 안의 순위 변동을 볼지
BATTLE_GAP_S = 2.0    # 배틀 상황 필터: 앞차 gap 이 이 값 이하인 샘플만 사용

# --- 수집 옵션 ---
FETCH_CAR_DATA = False   # drs/속도(car_data). 매우 큼. 코어가 돌아간 뒤 True 로.
FETCH_LOCATION_DATA = False
# Free historical access is also limited to 30 requests/minute. Keep a small
# margin so long driver-by-driver telemetry collections do not get throttled.
REQUEST_SLEEP = 2.1

# --- 경로 (프로젝트 폴더 기준 상대경로 -> 어디로 옮겨도 동작) ---
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
TARGET_LABEL = "label_overtake"
TRACK_SEGMENTS = 30
RESTART_PHASE_S = 60.0  # SC/VSC/red flag 해제 직후를 별도 상태로 보는 시간(초)
WEATHER_TRANSITION_S = 300.0  # rainfall 상태가 바뀐 뒤 transition으로 보는 시간(초)
