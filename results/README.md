# results/ — 큐레이션된 산출물

학습 데이터·모델(`data/`)은 용량이 커서 커밋하지 않는다(`.gitignore`).
대신 이 폴더에 **재현 없이도 결과를 확인할 수 있는 핵심 산출물**만 골라 담았다.
기준 모델 런: **`races_initial_event_type_final`** (5개 서킷 · 2023–2024 학습).

| 파일 | 내용 |
| --- | --- |
| `races_initial_event_type_final_unity_contract.json` | Unity/서버 연동 계약 — 피처 순서(26), 4개 출력, 보정(raw→표시확률) 규칙 |
| `external_eval_spa_2025_report.json` | **미학습 서킷(Spa) held-out 평가** — 누수검증 + 4타깃 지표(ROC/PR-AUC/Brier) |
| `races_initial_event_type_final_risk_audit.json` | 라벨 균형·분할 누수·이벤트 파싱·피처 의존 감사 |
| `spa_external_2025_panel.png` | 예측 확률 재생 패널(정지 이미지) |
| `spa_trained_2025_anim.gif` | 예측 확률 재생 애니메이션 |

> 모델 `.txt` 부스터와 원시 데이터가 필요하면 `python train_races.py --races initial` 로 재생성한다.
