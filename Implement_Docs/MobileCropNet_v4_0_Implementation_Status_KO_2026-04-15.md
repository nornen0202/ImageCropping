# MobileCropNet v4.0 구현 상태 및 실험 계획

## 1. v4.0 설계 분석 요약

`MobileCropNet_v4_0.md`의 핵심은 v3의 정적 후보 재랭킹 구조를 유지 보수하는 것이 아니라, Teacher 내부 topology에 의존하지 않는 mobile cropper를 새로 정의하는 것이다. v4.0의 핵심 계약은 다음 네 가지다.

- learned proposal generator: 정적 micro-bank를 핵심 알고리즘에서 제거하고, AR 조건과 global image context를 사용해 crop proposal을 직접 생성한다.
- operator-safe candidate encoding: ROIAlign/grid_sample 없이 feature-grid mask pooling으로 inside/boundary/context 표현을 만든다.
- relation-aware set ranking: 후보를 독립 점수화하지 않고 후보 간 상대 geometry와 IoU 관계를 반영한다.
- baseline-aware policy: full/max-area baseline 대비 crop 여부와 개선량을 함께 예측해 현재 라벨의 baseline-relative decision semantics를 보존한다.

따라서 v4.0 구현은 기존 `src/mobilecropnet` v3 코드를 수정하는 방식이 아니라 `src/mobilecropnet_v4`로 분리했다. v3 산출물 및 기존 스크립트와 충돌하지 않게 하면서, v4 전용 학습/평가/시각화/추론 경로를 새로 구성했다.

## 2. 최신 GAIC 라벨 스키마 반영

대상 라벨 디렉터리:

`data/GAIC/All/artifacts/training_labels/gaic_personv6_server_v1_leftover_ignore_monotonic`

확인된 최신 batch JSONL 핵심 필드는 다음과 같다.

- `baseline`: baseline candidate id, normalized bbox, `score_policy`, `crop_utility_raw`
- `decision_target`: `decision_type`, `decision_id`, `delta_vs_base`, baseline/winner candidate id
- `matching_targets`: positive crop targets, `score_targets.crop_utility_prob`, macro/checklist/safety 정보
- `candidate_pool`: positive/soft-positive/hard-negative/unsafe/ignore/overflow 후보
- `ignored_candidates`, `overflow_candidates`: 기본 학습 후보에서 제외하고, 필요 시 별도 정책으로 포함해야 하는 후보
- `routing`: subject mode id 및 routing metadata

v4 adapter는 baseline을 항상 첫 후보로 넣고, matching target을 positive supervision과 proposal supervision으로 사용한다. `candidate_pool`은 ignore/overflow를 기본 제외하고 positive, soft-positive, hard-negative, high-score 후보 순으로 제한된 `candidate_k`에 채운다.

좌표계는 v4 문서의 keep-AR 입력 원칙에 맞춰 letterbox resize를 사용한다. 원본 normalized box는 학습 시 letterbox normalized box로 변환하고, 평가/시각화/추론 출력은 다시 원본 normalized 좌표로 되돌린다.

## 3. 구현된 코드

- `src/mobilecropnet_v4/data.py`
  - current batch JSONL dataset adapter
  - baseline/matching/candidate_pool 통합
  - letterbox image loading 및 box coordinate transform
  - positive proposal supervision 구성
  - `A_macro/S_macro/C_macro/T_macro` auxiliary target 반영

- `src/mobilecropnet_v4/model.py`
  - mobile shared encoder
  - AR-conditioned learned proposal head
  - inside/boundary/context mask pooling
  - RelationLite pairwise candidate refinement
  - utility/positive/risk/macro/route/decision/delta heads
  - listwise, pairwise, proposal, policy loss 통합

- `src/scripts/prepare_mobilecropnet_v4_splits.py`
  - COCO train/test key 기반 leakage-aware JSONL split

- `src/scripts/train_mobilecropnet_v4.py`
  - replay candidate + learned proposal auxiliary 학습
  - AMP, cosine schedule, early stop, checkpoint 저장

- `src/scripts/evaluate_mobilecropnet_v4.py`
  - candidate top-1 hit, recall@K, SRCC/PCC, NDCG@K, IoU, proposal recall 평가

- `src/scripts/visualize_mobilecropnet_v4_predictions.py`
  - 고해상도 PNG overlay 및 contact sheet 저장

- `src/scripts/compare_mobilecropnet_v4_methods.py`
  - v4, baseline candidate, Teacher score oracle, positive oracle, proposal-only 비교

- `src/scripts/infer_mobilecropnet_v4.py`
  - 단일 이미지 추론. 후보 미입력 시 learned proposal을 직접 ranking

- `src/scripts/build_mobilecropnet_v4_report.py`
  - 학습/평가/시각화/비교 결과를 Markdown 리포트로 묶음

- `tests/test_mobilecropnet_v4.py`
  - dataset/model/loss/train/eval/compare/viz/infer smoke test

## 4. 현재 검증 결과

로컬 검증:

- `pytest tests/test_mobilecropnet_v4.py -q`: 통과
- `py_compile` for v4 package/scripts: 통과
- 실제 GAIC split 생성: 통과
  - train-dev rows: 4380
  - val-dev rows: 477
  - test rows: 625
  - unassigned rows: 1338
- 실제 GAIC 소량 CPU smoke 학습: 통과
  - image load, latest JSONL schema, loss backward, checkpoint 저장 확인
  - macro auxiliary target이 실제 라벨에서 non-zero로 들어오는 것 확인
- 평가/비교/PNG 시각화/단일 이미지 추론 smoke: 통과

생성된 로컬 smoke 산출물:

- split: `artifacts/mobilecropnet_v4/gaic_splits`
- local smoke run: `artifacts/mobilecropnet_v4/local_smoke_real`
- macro-target 보정 smoke: `artifacts/mobilecropnet_v4/local_smoke_real_macro`

## 5. 원격 GPU 실행 상태

공유 스토리지 동기화:

- 코드: `/group-volume/users/jaden.ju/Sources/ImageCropping/src/mobilecropnet_v4`
- 스크립트: `/group-volume/users/jaden.ju/Sources/ImageCropping/src/scripts/*mobilecropnet_v4*.py`
- split: `/group-volume/users/jaden.ju/Sources/ImageCropping/artifacts/mobilecropnet_v4/gaic_splits`

원격 CPU 서버에서 v4 코드 py_compile 및 실제 JSONL 1-row dataset load 확인 완료.

MLP GPU smoke workload:

- run name: `mcn-v4-smoke-20260415-184105`
- run id: `1093515`
- status: Pending
- 원인: A100 pool에 다수의 running/pending workload가 있어 큐 대기 중

notebook형 GPU IP `10.8.103.16`은 로컬 및 N6 CPU 서버에서 SSH/8888 접속이 timeout이므로 현재 자동 실행 경로로 사용할 수 없다.

## 6. GPU 수 판단

현재 GAIC v4 학습은 단일 A100 1장으로 충분하다. 모델 규모가 작고 replay candidate `candidate_k=16~32`, proposal query `Q=16~32`, input 256~320 범위이므로 DDP가 필요한 메모리/시간 병목은 아니다.

2개 이상의 GPU가 필요한 경우는 단일 학습 run이 아니라 실험 병렬화 목적이다. 권장 방식은 multi-GPU 한 run이 아니라 single-GPU workload를 여러 개 생성해 Q/input/token/hyperparameter sweep을 병렬 수행하는 것이다. 이는 v4 모델의 on-device 목표와도 맞고, 실험 실패 격리에도 유리하다.

## 7. 다음 실험 계획

GPU smoke 통과 후 다음 single-GPU workload를 병렬 생성한다.

- `mcn-v4-q16-256-w075`: Balanced baseline. Q16, input 256, candidate_k 24, width 0.75, token 128
- `mcn-v4-q24-288-w075`: proposal recall 중심. Q24, input 288, candidate_k 24, width 0.75, token 128
- `mcn-v4-q16-256-w100`: ranker capacity 확인. Q16, input 256, candidate_k 24, width 1.0, token 128

각 run은 train-dev 학습, val-dev 평가, test JSONL 평가, PNG visualization, comparison report, Markdown report를 생성한다. 최종 선택 기준은 val-dev `candidate_top1_hit`, `ndcg_at_5`, `proposal_recall_at_5_iou_0_5`, `top1_iou_to_best_positive`, decision accuracy를 함께 본다.

## 8. 제품/논문 수준 추가 과제

- GAIC test set에서 v4, baseline, Teacher oracle, 기존 v3 best checkpoint, 공개 cropper prediction을 동일 metric schema로 비교
- public cropper prediction을 `compare_mobilecropnet_v4_methods.py` 외부 입력으로 연결
- latency/throughput 측정: CPU, CUDA, target NPU 후보 runtime
- ONNX/TFLite/QAT export: operator whitelist 확인 후 별도 phase로 진행
- failure clustering: target AR, subject mode, baseline-delta, high-risk negative bucket별 실패 유형 분석
- qualitative appendix: top success/failure, proposal failure, baseline-regret failure, decision mismatch를 PNG contact sheet로 정리
