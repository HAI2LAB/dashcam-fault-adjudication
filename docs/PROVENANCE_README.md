# TIV 논문 결과·코드 패키지 (2026-09-21, 최무선 정리)

## 원칙
- `01_code_original/`: 허지웅 박사님 원본 코드 (미수정). `e2e_model3.py`의 `E2E_CHAIN=hard` 스위치는 박사님 추가분.
- `02_code_ms/`: 최무선 추가·변형 코드. 파일명 `_ms` 접미사. 인자/환경변수 미지정 시 원본과 동일 동작.
- 체크포인트(`best_*`)는 용량 문제로 제외. 서버 `~/TS_code/result_*/best_*/`에 있음.

## 02_code_ms 설명
| 파일 | 역할 | 스위치 |
|---|---|---|
| e2e_model3_ms.py | 모델 (원본 사본) | `E2E_CHAIN=none` → cmp/ratio 헤드가 h만 입력 (no-chain) |
| e2e_train3_ms.py | 학습 (원본 사본, e2e_model3_ms import) | `--lambda_cmp`(기본1.0), `--select {f1_w10,w10}`(기본 f1_w10) |
| run_nochain_ms.sh | no-chain 3시드 순차 실행 + `agg` 비교표 | E2E_CHAIN=none |
| run_ratioonly_ms.sh | ratio-only baseline 3시드 | E2E_CHAIN=none --lambda_sit 0 --lambda_cmp 0 --select w10 |
| make_fig3_qualitative_ms_v4.py | Fig.3 정성예시 조립 (fig3_clips.txt + 프레임 + preds CSV) | --columns 2 --frames 4 |
| make_fig4_confusion_ms.py | Fig.4 혼동행렬 (preds CSV) | |
| make_impact_shots_ms.py / _fig3.py | 충돌 프레임 추출 | |
| discard_count_ms.py | 라벨 필터링 폐기 수 (8/7,574) | |
| zeroshot_recalc_ms.py | zero-shot 진단 재채점 (exact match / macro recall) | |

## 04_results — 모두 InternVL3-8B 32f, 첫 번째 test split (n=2,837)
| 폴더 | 실험 | 실행 |
|---|---|---|
| result_e2e3_bb1 | 논문 본 결과 18런 (seed 42) | 박사님 |
| result_e2e3_bb1_s44 / _s45 | soft 시드 재현 (4개 구성) | 박사님 |
| result_e2e3_bb1_hard / _hard_s44 / _hard_s45 | hard-chain (one-hot detach) 3시드 | 최무선 (박사님 스위치 사용) |
| result_e2e3_bb1_nochain_s42/44/45 | no-chain 3시드 | 최무선 |
| result_e2e3_bb1_ratioonly_s42/44/45 | ratio-only baseline 3시드 (진행 중, 9/23 완료 예정) | 최무선 |
각 폴더: `e2e_*_eval.json`(eval-mode 최종 지표), `preds_*_eval.csv`(클립별 예측), `e2e_*.json`(학습 중 지표).
`AGG_soft_hard_nochain.txt`: 3시드 평균±σ 비교표.

## 05_oracle_zeroshot_videomae
- oracle_bb1.json: oracle ladder (13-class / fine-grained lookup, train fit → test 평가)
- zeroshot_summary/: 프롬프팅 zero-shot 진단 요약 JSON (클립별 원본 3,281건은 서버 ~/TS_code/backup/agent1/result/ 참조, 1.7GB)
- vmae.log / result_vmae*: VideoMAE-B 외부 baseline

## 06_figures / 07_evidence
- Fig.3·Fig.4 출력, 논문 수치 근거·로그·검증 결과(코드북 일치율, 헤드 일치율, 데이터 분포 등)
