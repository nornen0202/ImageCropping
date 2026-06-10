# 10K_local Subject-Mode + Stage10 결과 보고서 (상세)

- 데이터셋: `data/SSTK/10K_local`
- 실행 템플릿: `README_SSTK_Curation.md`의 **3.9 + 3.9.1 동시 실행 흐름**
- 기준 run_tag: `rerun1_public_e2e_260303`
- 비교 run_tag(이전 보고서): `rerun1_public_e2e_subject_mode`

## 0) 산출물 검증 결과

### 0.1 무결성 체크 (PASS)

- Filtered parquet rows: **500**
- Candidate jsonl rows: **500** (이미지 1행)
- Teacher score jsonl rows: **500** (이미지 1행, AR 5개 내장)
- VLM label jsonl rows: **2500** (**이미지×AR = 500×5**, task 1행)
- VLM meta jsonl rows: **500** (이미지 1행)

핵심 파일 존재/검증:

- Candidate overview: `artifacts/candidates/candidates_ar_rerun1_public_e2e_260303_overview.json` 존재
- Teacher overview/QA: `artifacts/teacher/overview/*rerun1_public_e2e_260303*`, `artifacts/teacher/qa/*rerun1_public_e2e_260303*` 존재
- VLM summary: `artifacts/vlm_teacher/summary/vlm_teacher_summary_rerun1_public_e2e_260303.json` 존재
- Visualizations overview:
  - precompute: `components_rerun1_public_e2e_260303/viz_overview.json` → selected=12, rendered=12, missing=0
  - teacher: `teacher_scorer_rerun1_public_e2e_260303/viz_overview.json` → tasks=60, rendered=60, missing=0
  - vlm: `vlm_teacher_rerun1_public_e2e_260303/viz_overview.json` → render_tasks=60, rendered=60, missing=0

### 0.2 실행 품질 체크

- Public proposal 주입률: **1.000** (`candidate_overview.proposal_injected_rate`)
- Teacher expensive(real) 적용: **2500/2500 AR-task**
- VLM backend: `qwen25_vl` + `Qwen/Qwen3-VL-4B-Instruct`, multi-gpu 8 shard
- VLM placeholder 설명문(`<1 sentence summary>`) 비율: **0%**
- VLM fallback backend(heuristic) 사용 task: **0**

판정: **실행/산출물 기준으로는 통과(PASS)**.

## 1) 파이프라인 한눈에 보기

1. Subject-mode enrich: `feats_c2c3c5_v2_strict_enriched.jsonl` → `..._routed.jsonl`
2. Candidate 생성(+public teacher proposal 주입)
3. Teacher scorer(cheap→expensive→Top-K)
4. Stage-10 VLM teacher(AR task 단위 라벨)
5. Component/Teacher/VLM 시각화 + analytics 산출

## 2) 단계별 입력/출력 스키마 (처음 보는 사용자용)

### 2.1 Candidate (`candidates_ar_rerun1_public_e2e_260303.jsonl`)

- 단위: **이미지 1행**
- 핵심 필드:
  - `routing.subject_mode`, `routing.policy_id`, `routing.router_rule_id`
  - `proposal_injected` (해당 이미지에 teacher seed 주입 여부)
  - `candidates_by_ar` (AR별 후보 리스트)
  - `iou_to_teacher_top1_by_ar` (주입 seed와 후보 top1 정렬 정도)

### 2.2 Teacher (`teacher_scores_ar_rerun1_public_e2e_260303.jsonl`)

- 단위: **이미지 1행**, 내부 `results_by_ar`에 AR 5개
- AR별 핵심 필드:
  - `decision.decision_type`: `keep_full|minimal_crop|crop`
  - `decision.delta_improve`: top1과 baseline의 개선량
  - `decision.tau_improve`: decision 경계 임계값
  - `selected_topk`: 최종 Top-K 후보(점수/체크리스트 포함)
  - `proposal_injection`: seed 기여 추적(AR별)

### 2.3 VLM Label (`crop_label_v1_rerun1_public_e2e_260303.jsonl`)

- 단위: **(이미지, AR) task 1행**
  - 예: `sample_id = "sstk_image_xxx|ar=1:1"`
- 핵심 필드:
  - `decision_type`, `delta_improve_vs_baseline`, `selected_topk`
  - `explanations.short/long`
  - `meta_norm_v1` (category/main_subject/intent 등)

참고: 이 포맷 때문에 `2500 rows = 500 이미지 × 5 AR`가 정상입니다.

## 3) 정량 결과 요약

### 3.1 Candidate

- 이미지 수: **500**
- 후보 수/이미지: mean **533.332**, p50 **530.0**, p90 **612.1**
- proposal_injected_rate: **1.000**
- 주 피사체 source 분포 상위:
  - `{'c2_routing_union': 229, 'c2det_person_c2_c3_routing_union': 106, 'c2det_person_c2_c3': 88, 'c2': 70}`

### 3.2 Teacher (AR-task=2500)

- decision 분포: `{'minimal_crop': 1517, 'crop': 983}`
- decision 비율: `{'minimal_crop': 0.6068, 'crop': 0.3932}`
- 최종 score 분포:
  - mean **-3.8283**, p50 **-1.4200**, p90 **1.0619**, neg_rate **0.7404**
- proposal 기여 KPI:
  - teacher_seed_top1_rate_all: **0.0872**
  - teacher_seed_selected_any_rate_all: **0.4560**
  - proposal_rescue_rate_proxy: **0.0872**
- guard KPI:
  - guard_no_person_for_portrait_rate: **0.0000**
  - guard_no_text_signal_rate: **0.0360**
  - guard_low_blank_ratio_copyspace_rate: **0.0000**

### 3.3 VLM (AR-task=2500)

- decision 분포(재계산): `{'keep_full': 1604, 'crop': 886, 'minimal_crop': 10}`
- selected_k 분포(재계산): `{5: 2294, 4: 27, 1: 86, 2: 44, 3: 49}`
- teacher-vlm decision 일치율: **0.2024** (506/2500)
- 설명문 fallback 템플릿 사용률(`"...기준으로 안정적인 후보..."`): **0.7900**
  - rank별 fallback 비율: `{1: 0.0004, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0}`
- VLM analytics summary:
  - tasks: **2500**, selected_rows: **11899**
  - why_tags 상위는 `ar_fits_well`, `avoid_face_cut`, `context_loss` 계열이 우세

## 4) 이전 리포트(`rerun1_public_e2e_subject_mode`) 대비 변화

|항목|이전 `rerun1_public_e2e_subject_mode`|신규 `rerun1_public_e2e_260303`|해석|
|---|---:|---:|---|
|Candidate 평균 수/이미지|402.128|533.332|proposal seed 주입으로 후보군 확대|
|Proposal injected rate|0.000|1.000|이번 run은 공개 proposal 주입 정상|
|Teacher crop rate|0.3976|0.3932|유사(큰 분포 변형 없음)|
|Teacher minimal_crop rate|0.6024|0.6068|유사|
|VLM crop count|765|886|crop 수 증가|
|VLM keep_full count|1733|1604|keep_full 편향 완화|
|VLM minimal_crop count|2|10|희소 클래스 소폭 증가|

## 5) 시각화 자산과 읽는 법

### 5.1 Candidate 집계 시각화

![hist_total_candidates.png](../../candidates/visualizations/candidates_ar_rerun1_public_e2e_260303_viz/hist_total_candidates.png)
![bar_avg_candidates_by_ar.png](../../candidates/visualizations/candidates_ar_rerun1_public_e2e_260303_viz/bar_avg_candidates_by_ar.png)
![stacked_source_mix_by_ar.png](../../candidates/visualizations/candidates_ar_rerun1_public_e2e_260303_viz/stacked_source_mix_by_ar.png)
![scatter_subject_centroids.png](../../candidates/visualizations/candidates_ar_rerun1_public_e2e_260303_viz/scatter_subject_centroids.png)
![sample_candidate_boxes.png](../../candidates/visualizations/candidates_ar_rerun1_public_e2e_260303_viz/sample_candidate_boxes.png)

`sample_candidate_boxes.png` 범례(코드 기준):
- **빨강**: `must_keep=True` 후보(주로 baseline 강제 포함)
- **파랑**: 일반 후보
- **초록**: `subject_prior` 박스(C2/C3/routing 기반 주 피사체 prior)

`stacked_source_mix_by_ar.png` source 그룹 설명:
- `baseline`: max-area center/slide 계열 기본 후보
- `grid`: 격자(anchor) 탐색 후보
- `phi`: rule-of-thirds / phi-grid 계열
- `jitter`: 위치/스케일 미세 교란 후보
- `teacher_seed`: GAIC/CACNet/CGS proposal 주입 계열
- `other`: object/copyspace 등 기타 템플릿

### 5.2 Precompute (12샘플)

- 경로: `../../precompute/visualizations/components_rerun1_public_e2e_260303`
- 요약: selected=12, rendered=12, missing=0

`selected=12, rendered=12, missing=0` 의미:
- selected: 샘플 ID 조건으로 선택된 시각화 대상 수
- rendered: 실제 이미지 렌더 성공 수
- missing: 원본 로드 실패 등으로 렌더 누락된 수

### 5.3 Teacher (12샘플 × 5AR)

- 경로: `../../teacher/visualizations/teacher_scorer_rerun1_public_e2e_260303`
- 요약: tasks_selected=60, rendered=60, missing=0

![teacher_selected_k_distribution.png](assets/analytics/teacher_topk/teacher_selected_k_distribution.png)
![teacher_topk_final_score_hist.png](assets/analytics/teacher_topk/teacher_topk_final_score_hist.png)
![teacher_topk_source_mix_by_rank.png](assets/analytics/teacher_topk/teacher_topk_source_mix_by_rank.png)

Teacher overlay 상단 텍스트 해석:
- `decision`: `keep_full|minimal_crop|crop`
- `delta`: top1과 baseline final score 차이
- `tau`: decision 경계 임계값
- `A_raw/A_norm/cos`: expensive 단계 신호(미적/정렬)
- `headroom/lookroom/horizon/context`: composition check pass/fail

### 5.4 VLM (12샘플 × 5AR)

- 경로: `../../vlm_teacher/visualizations/vlm_teacher_rerun1_public_e2e_260303`
- 요약: render_tasks=60, rendered=60, missing=0

![vlm_decision_type_distribution.png](assets/analytics/vlm_teacher/vlm_decision_type_distribution.png)
![vlm_selected_k_distribution.png](assets/analytics/vlm_teacher/vlm_selected_k_distribution.png)
![vlm_teacher_confidence_hist.png](assets/analytics/vlm_teacher/vlm_teacher_confidence_hist.png)
![vlm_why_tags_top.png](assets/analytics/vlm_teacher/vlm_why_tags_top.png)
![vlm_source_mix_by_rank.png](assets/analytics/vlm_teacher/vlm_source_mix_by_rank.png)

VLM overlay 해석:
- **주황 박스**: baseline 후보
- **색상 박스 #1~#5**: VLM selected_topk
- **회색 박스**: also_considered(선택 탈락 후보)
- 상단 텍스트: `decision`, `delta`, `k`, `conf`, `top1_why`

## 6) Super Category별 12샘플 추적

|super_cat|image_id|원본|Precompute Combined|Teacher(1:1)|VLM(1:1)|샘플 JSON|
|---|---|---|---|---|---|---|
|animals|bigstock_image_149147297|![](assets/samples/original/bigstock_image_149147297.jpg)|![](../../precompute/visualizations/components_rerun1_public_e2e_260303/combined_all/bigstock_image_149147297.jpg)|![](../../teacher/visualizations/teacher_scorer_rerun1_public_e2e_260303/by_ar/1x1/bigstock_image_149147297.jpg)|![](../../vlm_teacher/visualizations/vlm_teacher_rerun1_public_e2e_260303/by_ar/1x1/bigstock_image_149147297.jpg)|[json](assets/samples/json/bigstock_image_149147297.json)|
|architecture_exterior|bigstock_image_163220822|![](assets/samples/original/bigstock_image_163220822.jpg)|![](../../precompute/visualizations/components_rerun1_public_e2e_260303/combined_all/bigstock_image_163220822.jpg)|![](../../teacher/visualizations/teacher_scorer_rerun1_public_e2e_260303/by_ar/1x1/bigstock_image_163220822.jpg)|![](../../vlm_teacher/visualizations/vlm_teacher_rerun1_public_e2e_260303/by_ar/1x1/bigstock_image_163220822.jpg)|[json](assets/samples/json/bigstock_image_163220822.json)|
|documents_text|bigstock_image_129575693|![](assets/samples/original/bigstock_image_129575693.jpg)|![](../../precompute/visualizations/components_rerun1_public_e2e_260303/combined_all/bigstock_image_129575693.jpg)|![](../../teacher/visualizations/teacher_scorer_rerun1_public_e2e_260303/by_ar/1x1/bigstock_image_129575693.jpg)|![](../../vlm_teacher/visualizations/vlm_teacher_rerun1_public_e2e_260303/by_ar/1x1/bigstock_image_129575693.jpg)|[json](assets/samples/json/bigstock_image_129575693.json)|
|food|bigstock_image_165044747|![](assets/samples/original/bigstock_image_165044747.jpg)|![](../../precompute/visualizations/components_rerun1_public_e2e_260303/combined_all/bigstock_image_165044747.jpg)|![](../../teacher/visualizations/teacher_scorer_rerun1_public_e2e_260303/by_ar/1x1/bigstock_image_165044747.jpg)|![](../../vlm_teacher/visualizations/vlm_teacher_rerun1_public_e2e_260303/by_ar/1x1/bigstock_image_165044747.jpg)|[json](assets/samples/json/bigstock_image_165044747.json)|
|indoor_interior|bigstock_image_112983323|![](assets/samples/original/bigstock_image_112983323.jpg)|![](../../precompute/visualizations/components_rerun1_public_e2e_260303/combined_all/bigstock_image_112983323.jpg)|![](../../teacher/visualizations/teacher_scorer_rerun1_public_e2e_260303/by_ar/1x1/bigstock_image_112983323.jpg)|![](../../vlm_teacher/visualizations/vlm_teacher_rerun1_public_e2e_260303/by_ar/1x1/bigstock_image_112983323.jpg)|[json](assets/samples/json/bigstock_image_112983323.json)|
|landscape_nature|bigstock_image_218991439|![](assets/samples/original/bigstock_image_218991439.jpg)|![](../../precompute/visualizations/components_rerun1_public_e2e_260303/combined_all/bigstock_image_218991439.jpg)|![](../../teacher/visualizations/teacher_scorer_rerun1_public_e2e_260303/by_ar/1x1/bigstock_image_218991439.jpg)|![](../../vlm_teacher/visualizations/vlm_teacher_rerun1_public_e2e_260303/by_ar/1x1/bigstock_image_218991439.jpg)|[json](assets/samples/json/bigstock_image_218991439.json)|
|other_ambiguous|bigstock_image_207940879|![](assets/samples/original/bigstock_image_207940879.jpg)|![](../../precompute/visualizations/components_rerun1_public_e2e_260303/combined_all/bigstock_image_207940879.jpg)|![](../../teacher/visualizations/teacher_scorer_rerun1_public_e2e_260303/by_ar/1x1/bigstock_image_207940879.jpg)|![](../../vlm_teacher/visualizations/vlm_teacher_rerun1_public_e2e_260303/by_ar/1x1/bigstock_image_207940879.jpg)|[json](assets/samples/json/bigstock_image_207940879.json)|
|people_multi|bigstock_image_122199020|![](assets/samples/original/bigstock_image_122199020.jpg)|![](../../precompute/visualizations/components_rerun1_public_e2e_260303/combined_all/bigstock_image_122199020.jpg)|![](../../teacher/visualizations/teacher_scorer_rerun1_public_e2e_260303/by_ar/1x1/bigstock_image_122199020.jpg)|![](../../vlm_teacher/visualizations/vlm_teacher_rerun1_public_e2e_260303/by_ar/1x1/bigstock_image_122199020.jpg)|[json](assets/samples/json/bigstock_image_122199020.json)|
|people_single|sstk_image_1772011034|![](assets/samples/original/sstk_image_1772011034.jpg)|![](../../precompute/visualizations/components_rerun1_public_e2e_260303/combined_all/sstk_image_1772011034.jpg)|![](../../teacher/visualizations/teacher_scorer_rerun1_public_e2e_260303/by_ar/1x1/sstk_image_1772011034.jpg)|![](../../vlm_teacher/visualizations/vlm_teacher_rerun1_public_e2e_260303/by_ar/1x1/sstk_image_1772011034.jpg)|[json](assets/samples/json/sstk_image_1772011034.json)|
|product_object|bigstock_image_141430643|![](assets/samples/original/bigstock_image_141430643.jpg)|![](../../precompute/visualizations/components_rerun1_public_e2e_260303/combined_all/bigstock_image_141430643.jpg)|![](../../teacher/visualizations/teacher_scorer_rerun1_public_e2e_260303/by_ar/1x1/bigstock_image_141430643.jpg)|![](../../vlm_teacher/visualizations/vlm_teacher_rerun1_public_e2e_260303/by_ar/1x1/bigstock_image_141430643.jpg)|[json](assets/samples/json/bigstock_image_141430643.json)|
|sports|bigstock_image_134277740|![](assets/samples/original/bigstock_image_134277740.jpg)|![](../../precompute/visualizations/components_rerun1_public_e2e_260303/combined_all/bigstock_image_134277740.jpg)|![](../../teacher/visualizations/teacher_scorer_rerun1_public_e2e_260303/by_ar/1x1/bigstock_image_134277740.jpg)|![](../../vlm_teacher/visualizations/vlm_teacher_rerun1_public_e2e_260303/by_ar/1x1/bigstock_image_134277740.jpg)|[json](assets/samples/json/bigstock_image_134277740.json)|
|transportation|bigstock_image_109682993|![](assets/samples/original/bigstock_image_109682993.jpg)|![](../../precompute/visualizations/components_rerun1_public_e2e_260303/combined_all/bigstock_image_109682993.jpg)|![](../../teacher/visualizations/teacher_scorer_rerun1_public_e2e_260303/by_ar/1x1/bigstock_image_109682993.jpg)|![](../../vlm_teacher/visualizations/vlm_teacher_rerun1_public_e2e_260303/by_ar/1x1/bigstock_image_109682993.jpg)|[json](assets/samples/json/bigstock_image_109682993.json)|

## 7) 샘플 JSON에서 꼭 봐야 할 포인트

샘플 파일 예시: `assets/samples/json/bigstock_image_149147297.json`

1. `candidate.candidate_counts_by_ar`
- 후보 풀이 AR별로 얼마나 구성됐는지

2. `teacher.by_ar[*].decision_type`, `delta_improve`, `tau_improve`
- Stage-9에서 왜 crop/minimal을 택했는지

3. `vlm_teacher.by_ar[*].selected_topk[*].why_tags/why_text`
- Stage-10이 어떤 근거 태그로 설명했는지

4. `teacher vs vlm decision` 일치 여부
- 불일치는 오류가 아니라, 수치 최적화(teacher)와 언어/시각 판단(VLM)의 정책 차이를 의미

## 8) 운영 관점 해석과 리스크

### 8.1 이번 run의 강점

- proposal injection이 실제로 활성화됨(1.0)
- teacher real expensive + VLM multi-gpu 전량 완료
- 시각화 자산(components/teacher/vlm) 모두 렌더 성공

### 8.2 남아있는 리스크

- VLM 설명 fallback 템플릿 사용률이 여전히 높음(**79.00%**)
  - 특히 rank2~5는 거의 fallback 문구 중심
- teacher-vlm decision 불일치율이 높음(**79.76%**)
  - 학습 타깃으로 단일 사용 시 policy drift 가능

### 8.3 권장 후속 점검

1. Stage-10 학습 데이터 사용 시 `teacher-vlm 합의 샘플`과 `불일치 샘플`을 분리
2. fallback 비율을 QA gate로 관리(예: rank2~5 fallback < 0.9 목표)
3. `proposal_rescue_rate_proxy`를 run별 추적해 proposal 품질 스윕

## 9) 결론

`rerun1_public_e2e_260303` 실행 산출물은 **무결성/완성도 기준 통과**이며, 이전 `rerun1_public_e2e_subject_mode` 대비
- 공개 proposal 주입이 정상 반영되었고,
- VLM 결과의 `keep_full` 편향이 일부 완화되었으며,
- 보고/디버깅에 필요한 자산(샘플 JSON/시각화/요약 지표)이 모두 연결된 상태입니다.

실무적으로는 `teacher-vlm 불일치`와 `fallback 설명문 비율`을 다음 릴리즈의 핵심 개선 지표로 두고 관리하는 것을 권장합니다.
