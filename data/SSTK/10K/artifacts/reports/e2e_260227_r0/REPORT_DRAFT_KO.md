# 10K Cropping Pipeline 결과 보고서 초안 (국문)

- 데이터셋: `data/SSTK/10K`
- 기준 문서: `README_SSTK_Curation.md`, `ImplementPlan_Docs/SSTK_Cropping_DataFactory_QwenLabeler_Reorganized_KO_v1_9.md`
- 기준 실행 산출물: `run_tag=e2e_260227_r0`
- 작성 시각: 2026-02-27 17:45:27

## 1) 전체 결과 요약

- Filtered 이미지 수: **10,000장**
- Phase A super category 분포(12종): people_multi=835, architecture_exterior=834, landscape_nature=834, 나머지 대부분 833 내외
- Candidate 생성: 평균 **533.4583** / image, 주입률 **1.000**
- Public Teacher raw/proposal: **30,000 / 10,000**
- Teacher Scorer QA: `crop` 0.3736, `minimal_crop` 0.6264
- Teacher 점수(최종): mean=-3.9795, p90=1.0558, neg_rate=0.7337
- VLM Teacher(Stage 10): tasks=**49,998**, backend=`qwen25_vl` 100%, validator fallback note 비어있음 100%
- VLM label 분포: decision `crop=18680`, `minimal_crop=31318`, selected_k `{1:1856, 2:873, 3:675, 4:745, 5:45849}`
- 보고서용 보강 시각화: precompute(12장) + teacher(60 task) 생성 완료

## 2) 파이프라인 단계별 로직/역할

### 2.1 Phase A (Filter & Curate)
- 입력: SSTK tar 메타/태그/품질 점수
- 역할: 품질/카테고리 기준 curated pool 구성, 이후 단계 공통 입력 parquet 생성
- 출력: `filtered_sstk_100.parquet`

### 2.2 Phase B-1 (Perception Precompute: C1/C2/C3/C5)
- C2: object/segmentation, C3: person pose/face/headpose, C5: horizon/symmetry
- 역할: candidate/teacher scoring에 필요한 구조적 feature 전처리
- 출력: `artifacts/precompute/feats_c2c3c5_v2_strict_enriched.jsonl`

### 2.3 v1.9 8.2.0a Public Teacher Proposal 주입
- GAIC/CACNet/CGS 추론 결과를 free-form seed proposal로 변환
- 역할: grid/rule 기반 후보의 blind spot 보완
- 출력: `artifacts/public_teachers/raw/*.jsonl`, `artifacts/public_teachers/proposals/*.jsonl`

### 2.4 Candidate Generator
- 입력: filtered parquet + precompute(C2/C3) + public teacher proposal
- 역할: AR별 후보군 대량 생성, baseline/grid/phi/jitter/teacher seed 통합
- 출력: `artifacts/candidates/candidates_ar_e2e_260227_r0.jsonl` + overview/viz

### 2.5 Teacher Scorer (Cheap -> Expensive -> Top-K)
- 역할: 후보를 점수화하고 최종 crop decision(`crop|minimal_crop|keep_full`) 도출
- 현재 산출물 기준: `use_real_expensive=true` (QA의 `expensive_source_counts={real: 49998}`)
- 출력: `artifacts/teacher/scores/teacher_scores_ar_e2e_260227_r0.jsonl` + overview/qa/viz
- Top-K 해석(핵심):
  - `candidate.candidate_counts_by_ar`는 Teacher 입력 후보 수(`num_input_candidates`)이며, 최종 K가 아님
  - Teacher 내부 흐름: `num_input_candidates -> num_valid_candidates -> num_effective_candidates -> cheap_top_m(<=30) -> selected_topk(<=top_k)`
  - 설정값 `top_k=5`는 상한(upper bound)이며, 항상 5개가 보장되지는 않음
  - 실제 최종 K는 `selected_topk` 길이(`K_actual`)로 판단
- Top-K 전체 분포(10,000장 x AR 5종, 일부 AR 결손 포함):
  - task 수: 49,998, `K_actual` 분포: {'1': 1856, '2': 873, '3': 675, '4': 745, '5': 45849}
  - 평균 K=4.7572, p50=5.0, p90=5.0
  - K<5 비율: 0.0830 (4,149 / 49,998)
  - Top-K final score(min/mean/p50/p90/max): -78.772 / -6.069 / -2.898 / 1.012 / 1.178
  - Top-K source 상위: grid(90337), baseline_maxarea_slide(44369), teacher:jitter(42566), baseline_maxarea_center(39314), jitter(10619), object_template(3917)

### 2.6 Teacher Score 계산식 (코드 기준)
- 1) Cheap score
  - `cheap = λ_cov*cov - λ_cut*p_cut - λ_text*p_text + λ_comp*r_comp + λ_hr*r_headroom + λ_lr*r_lookroom + λ_sym*r_sym + λ_ctx*r_context + λ_cs*r_copyspace`
  - `r_comp = w_third*r_third + w_phi*r_phi + w_center*r_center + w_horizon*r_horizon`
  - 기본 가중치(`TeacherScorerConfig`): `lambda_cov=1.20, lambda_cut=1.80, lambda_comp=1.00, lambda_hr=0.65, lambda_lr=0.55, lambda_sym=0.25, lambda_ctx=0.45, lambda_cs=0.35`
  - 라우팅(shot_type/flags)에 따라 `effective_lambdas`로 재가중됨(예: copy-space면 `ctx/cs` 강화)
- 2) Expensive prior/proxy (real 모델 미사용 시)
  - `aesthetic_proxy = 0.45*r_comp + 0.20*sym + 0.20*cov + 0.15*context_fit - 0.30*min(1,p_cut)`
  - `ca_proxy = 0.65*cov + 0.35*context_fit`
- 3) Expensive + Final score
  - `expensive = w_a*A_norm + w_ca*cos + w_cov*cov - w_cut*p_cut - w_text*p_text + w_edge*r_edge`
  - `final = expensive + w_area*log(area_ratio)`
  - 기본 가중치: `w_a=1.0, w_ca=0.3, w_cov=0.5, w_cut=2.0, w_text=0.0, w_edge=0.5`
  - `A_norm = clamp((A_raw - aesthetic_score_min) / (aesthetic_score_max - aesthetic_score_min), 0, 1)` (`min=1, max=10`)
  - `w_area`는 라우팅 기반: copy-space/landscape `0.12`, product `0.06`, 기본 `0.10`
- 4) Keep-vs-Crop decision
  - `delta = best_final - baseline_final`
  - `delta < tau_improve`이면 baseline 유지 경로(`baseline_full`이면 `keep_full`, 아니면 `minimal_crop`)
  - `delta >= tau_improve`이면 `crop`
  - `tau_improve`는 라우팅 기반: copy-space/landscape `0.055`, group(또는 인물 2+) `0.045`, portrait `0.03`, product `0.015`, 기본 `0.035`
- 5) Top-K 선택
  - `cheap_top_m`(기본 30) 후보를 expensive 재평가 후 정렬
  - `select_topk_diverse(k=5, tau_div=0.75)`로 IoU 다양성 제약을 적용해 선택
  - 다양성 제약으로 모자라면 남은 상위 후보로 fill하여 최대 `top_k`까지 채움

### 2.7 VLM/MLLM Teacher Labeler (Stage 10)
- 입력:
  - `teacher_scores_ar` jsonl의 image-level row(`teacher_scorer.results_by_ar[AR]`)를 AR task로 펼쳐 사용
  - curated image(`data/SSTK/10K/images`)
- 내부 입력 태스크 스키마(`teacher_ab_input_v1`) 핵심 필드:
  - `sample_id`, `image(width,height)`, `target_ar`
  - `features.route_global`, `features.subject_prior`
  - `candidates`(cheap_top_m + selected_topk + hard_negatives + baseline/best dedupe)
  - `policy(topm, topk, keep_policy)`, `decision`(Teacher numeric), `baseline_candidate`, `numeric_topk`
- 출력 라벨 스키마(`crop_label_v1`) 핵심 필드:
  - `image_id`, `target_ar`, `decision_type`, `delta_improve_vs_baseline`
  - `selected_topk(rank, candidate_id, bbox_norm_xyxy, why_tags, why_text)`
  - `also_considered(candidate_id, reject_tags, reject_text)`
  - `composition_checks`, `teacher(backend, teacher_id, teacher_confidence)`
  - `validator(schema_ok, numeric_consistency_ok, notes)`, `timing`, `input_refs`, `meta_norm_v1`
- 출력:
  - `artifacts/vlm_teacher/labels/*.jsonl` (`crop_label_v1`)
  - `artifacts/vlm_teacher/meta/*.jsonl` (`meta_norm_v1`)
  - `artifacts/vlm_teacher/summary/*.json`
- 스키마 전수 점검(`crop_label_v1` 전체):
  - total_tasks=49998, explanations 존재=49998/49998, explanations.short/long 존재=49998/49998·49998/49998
  - selected_topk why_text 존재=237852/237852 (missing=0)
  - validator schema_ok/numeric_ok=True = 49998/49998 / 49998/49998, notes_empty=49998/49998
  - teacher/input_refs/meta_norm_v1 존재=49998/49998 / 49998/49998 / 49998/49998
- 샘플별 확인 경로:
  - 각 샘플 JSON(`assets/samples/json/<image_id>.json`)에 `vlm_teacher.input_summary.by_ar`(입력)와 `vlm_teacher.output_summary.by_ar`(출력)를 추가해 두었음
  - 1:1 기준 빠른 확인: `num_input_candidates`, `cheap_top_m_size`, `decision_type`, `selected_k`, `selected_candidate_ids`, `teacher_confidence`

## 3) 단계별 입출력 포맷 정의 (예시 샘플: `sstk_image_2190537205`)

- Phase A row 예시: [`assets/samples/json/sstk_image_2190537205.json`](assets/samples/json/sstk_image_2190537205.json) 의 `phaseA`
- Precompute row 예시: 같은 파일의 `precompute` + merged jsonl의 `c2_seg/c3_pose/c5_geom`
- Public Teacher proposal 예시: 같은 파일의 `public_teacher` (teacher별 `free_form` bbox/score)
- Candidate row 예시: 같은 파일의 `candidate` (`candidate_counts_by_ar`, `proposal_injected`)
- Teacher row 예시: 같은 파일의 `teacher.by_ar` (`decision_type`, `delta_improve`, `tau_improve`, composition check)
- VLM label row 예시: `artifacts/vlm_teacher/labels/crop_label_v1*.jsonl` (`schema_version=crop_label_v1`)
- 샘플 JSON Stage10 예시: `assets/samples/json/<image_id>.json`의 `vlm_teacher`
  - 입력 스키마 요약: `vlm_teacher.input_summary.by_ar[AR]` (`num_input_candidates`, `cheap_top_m_size`, `baseline_candidate_id`, `teacher_numeric_decision`) 
  - 출력 스키마 요약: `vlm_teacher.output_summary.by_ar[AR]` (`decision_type`, `selected_k`, `selected_candidate_ids`, `explanations`, `selected_topk_preview`, `teacher`, `validator`, `timing`) 
  - 메타 정규화: `vlm_teacher.meta_norm_v1` (`category`, `main_subject`, `intent`, `special_flags`) 

## 4) Visualization 키 항목 설명 (Teacher Overlay)

- `decision`: baseline 대비 최종 정책 (`crop`, `minimal_crop`, `keep_full`)
- `delta`: `best_score - baseline_score` 개선량
- `tau`: 개선량 임계치(카테고리/정책 기반 threshold)
- `headroom/lookroom/horizon/context`: composition rule별 pass/fail 체크
- `top1 final`: 최종 선택된 crop 후보 점수
- `A_raw`: 미학 점수 원시값(실모델 Aesthetic head 출력, 기본 스케일 1~10 근처)
- `A_norm`: `A_raw`를 `[0,1]`로 정규화한 값 (`(A_raw-1)/(10-1)`; 범위 클리핑)
- `cos`: crop 이미지 임베딩과 태그 텍스트 임베딩의 cosine similarity (`[-1,1]`)
- `src`: expensive score 소스 (`real`=실모델, `proxy`=대체값)

## 5) 단계별 시각화 산출물

### 5.1 Candidate 단계 (집계 시각화)
![hist_total_candidates.png](../../candidates/visualizations/candidates_ar_e2e_260227_r0_viz/hist_total_candidates.png)
![bar_avg_candidates_by_ar.png](../../candidates/visualizations/candidates_ar_e2e_260227_r0_viz/bar_avg_candidates_by_ar.png)
![stacked_source_mix_by_ar.png](../../candidates/visualizations/candidates_ar_e2e_260227_r0_viz/stacked_source_mix_by_ar.png)
![scatter_subject_centroids.png](../../candidates/visualizations/candidates_ar_e2e_260227_r0_viz/scatter_subject_centroids.png)
![sample_candidate_boxes.png](../../candidates/visualizations/candidates_ar_e2e_260227_r0_viz/sample_candidate_boxes.png)

- `sample_candidate_boxes.png` 해석: 검은 테두리=정규화 캔버스, 빨강=`must_keep`, 파랑=일반 후보, 초록=`subject_prior`
- `stacked_source_mix_by_ar.png` 해석: y축 ratio(해당 AR 내부 비율), source는 상위 항목 + other
- source 용어: `baseline`(안전 앵커), `grid`(전역 coverage), `phi`(`phi_thirds`), `jitter`(국소 탐색), `teacher seed`(`teacher:*`, `teacher:jitter`)

### 5.2 Precompute 단계 (보고서용 12샘플)
- 경로: `../../precompute/visualizations/components_e2e_260227_r0_report12`
- 요약: selected=12, rendered=12, missing=0
- `selected`: 시각화 대상으로 선택된 이미지 수
- `rendered`: 실제 로딩/오버레이 저장 완료 수
- `missing`: 선택되었지만 로딩 실패한 수

대표 샘플(`sstk_image_2190537205`)
![orig](../../precompute/visualizations/components_e2e_260227_r0_report12/original/sstk_image_2190537205.jpg)
![c2](../../precompute/visualizations/components_e2e_260227_r0_report12/c2_seg/sstk_image_2190537205.jpg)
![c3](../../precompute/visualizations/components_e2e_260227_r0_report12/c3_pose/sstk_image_2190537205.jpg)
![c5](../../precompute/visualizations/components_e2e_260227_r0_report12/c5_geom/sstk_image_2190537205.jpg)
![combined](../../precompute/visualizations/components_e2e_260227_r0_report12/combined_all/sstk_image_2190537205.jpg)

### 5.3 Teacher 단계 (보고서용 12샘플 x 5AR)
- 경로: `../../teacher/visualizations/teacher_scorer_e2e_260227_r0_report12`
- 요약: selected_tasks=60, rendered=60, missing=0

![teacher_1x1](../../teacher/visualizations/teacher_scorer_e2e_260227_r0_report12/by_ar/1x1/sstk_image_2190537205.jpg)

Top-K 분포 시각화(전체 task 기준):
![teacher_selected_k_distribution.png](assets/analytics/teacher_topk/teacher_selected_k_distribution.png)
![teacher_topk_final_score_hist.png](assets/analytics/teacher_topk/teacher_topk_final_score_hist.png)
![teacher_topk_source_mix_by_rank.png](assets/analytics/teacher_topk/teacher_topk_source_mix_by_rank.png)
- `teacher_selected_k_distribution`: `(image,AR)`별 최종 K 분포
- `teacher_topk_final_score_hist`: selected_topk row들의 final score 분포
- `teacher_topk_source_mix_by_rank`: rank(1~5)별 source 비율

### 5.4 VLM Teacher 단계 (Stage 10 검증 시각화)
- 경로: `../../vlm_teacher/visualizations/vlm_teacher_e2e_260227_r0`
- 요약: analytics_tasks=49,998, selected_rows=237,852, rendered=60, missing=0

![vlm_decision_type_distribution.png](assets/analytics/vlm_teacher/vlm_decision_type_distribution.png)
![vlm_selected_k_distribution.png](assets/analytics/vlm_teacher/vlm_selected_k_distribution.png)
![vlm_teacher_confidence_hist.png](assets/analytics/vlm_teacher/vlm_teacher_confidence_hist.png)
![vlm_why_tags_top.png](assets/analytics/vlm_teacher/vlm_why_tags_top.png)
![vlm_source_mix_by_rank.png](assets/analytics/vlm_teacher/vlm_source_mix_by_rank.png)

- 오버레이 색상 규칙:
  - 주황색: `baseline_used_candidate_id`
  - 컬러 박스(#1~#5): `selected_topk`
  - 회색 박스: `also_considered` (비선정 후보)
- 대표 샘플(`sstk_image_2190537205`, AR=1:1):
![vlm_1x1](../../vlm_teacher/visualizations/vlm_teacher_e2e_260227_r0/by_ar/1x1/sstk_image_2190537205.jpg)

## 6) Super Category별 예시 샘플 (원본 + 단계 결과 매칭)

|super_cat|image_id|원본|Precompute Combined|Teacher(AR 1:1)|샘플 JSON|
|---|---|---|---|---|---|
|animals|sstk_image_2257975789|![](assets/samples/original/sstk_image_2257975789.jpg)|![](../../precompute/visualizations/components_e2e_260227_r0_report12/combined_all/sstk_image_2257975789.jpg)|![](../../teacher/visualizations/teacher_scorer_e2e_260227_r0_report12/by_ar/1x1/sstk_image_2257975789.jpg)|[json](assets/samples/json/sstk_image_2257975789.json)|
|architecture_exterior|sstk_image_1165522843|![](assets/samples/original/sstk_image_1165522843.jpg)|![](../../precompute/visualizations/components_e2e_260227_r0_report12/combined_all/sstk_image_1165522843.jpg)|![](../../teacher/visualizations/teacher_scorer_e2e_260227_r0_report12/by_ar/1x1/sstk_image_1165522843.jpg)|[json](assets/samples/json/sstk_image_1165522843.json)|
|documents_text|sstk_image_2129282159|![](assets/samples/original/sstk_image_2129282159.jpg)|![](../../precompute/visualizations/components_e2e_260227_r0_report12/combined_all/sstk_image_2129282159.jpg)|![](../../teacher/visualizations/teacher_scorer_e2e_260227_r0_report12/by_ar/1x1/sstk_image_2129282159.jpg)|[json](assets/samples/json/sstk_image_2129282159.json)|
|food|sstk_image_2024012429|![](assets/samples/original/sstk_image_2024012429.jpg)|![](../../precompute/visualizations/components_e2e_260227_r0_report12/combined_all/sstk_image_2024012429.jpg)|![](../../teacher/visualizations/teacher_scorer_e2e_260227_r0_report12/by_ar/1x1/sstk_image_2024012429.jpg)|[json](assets/samples/json/sstk_image_2024012429.json)|
|indoor_interior|sstk_image_629195588|![](assets/samples/original/sstk_image_629195588.jpg)|![](../../precompute/visualizations/components_e2e_260227_r0_report12/combined_all/sstk_image_629195588.jpg)|![](../../teacher/visualizations/teacher_scorer_e2e_260227_r0_report12/by_ar/1x1/sstk_image_629195588.jpg)|[json](assets/samples/json/sstk_image_629195588.json)|
|landscape_nature|sstk_image_1104016514|![](assets/samples/original/sstk_image_1104016514.jpg)|![](../../precompute/visualizations/components_e2e_260227_r0_report12/combined_all/sstk_image_1104016514.jpg)|![](../../teacher/visualizations/teacher_scorer_e2e_260227_r0_report12/by_ar/1x1/sstk_image_1104016514.jpg)|[json](assets/samples/json/sstk_image_1104016514.json)|
|other_ambiguous|sstk_image_174059567|![](assets/samples/original/sstk_image_174059567.jpg)|![](../../precompute/visualizations/components_e2e_260227_r0_report12/combined_all/sstk_image_174059567.jpg)|![](../../teacher/visualizations/teacher_scorer_e2e_260227_r0_report12/by_ar/1x1/sstk_image_174059567.jpg)|[json](assets/samples/json/sstk_image_174059567.json)|
|people_multi|pond5_image_99604568|![](assets/samples/original/pond5_image_99604568.jpg)|![](../../precompute/visualizations/components_e2e_260227_r0_report12/combined_all/pond5_image_99604568.jpg)|![](../../teacher/visualizations/teacher_scorer_e2e_260227_r0_report12/by_ar/1x1/pond5_image_99604568.jpg)|[json](assets/samples/json/pond5_image_99604568.json)|
|people_single|sstk_image_2190537205|![](assets/samples/original/sstk_image_2190537205.jpg)|![](../../precompute/visualizations/components_e2e_260227_r0_report12/combined_all/sstk_image_2190537205.jpg)|![](../../teacher/visualizations/teacher_scorer_e2e_260227_r0_report12/by_ar/1x1/sstk_image_2190537205.jpg)|[json](assets/samples/json/sstk_image_2190537205.json)|
|product_object|sstk_image_1955755765|![](assets/samples/original/sstk_image_1955755765.jpg)|![](../../precompute/visualizations/components_e2e_260227_r0_report12/combined_all/sstk_image_1955755765.jpg)|![](../../teacher/visualizations/teacher_scorer_e2e_260227_r0_report12/by_ar/1x1/sstk_image_1955755765.jpg)|[json](assets/samples/json/sstk_image_1955755765.json)|
|sports|sstk_image_1178976352|![](assets/samples/original/sstk_image_1178976352.jpg)|![](../../precompute/visualizations/components_e2e_260227_r0_report12/combined_all/sstk_image_1178976352.jpg)|![](../../teacher/visualizations/teacher_scorer_e2e_260227_r0_report12/by_ar/1x1/sstk_image_1178976352.jpg)|[json](assets/samples/json/sstk_image_1178976352.json)|
|transportation|sstk_image_1977782153|![](assets/samples/original/sstk_image_1977782153.jpg)|![](../../precompute/visualizations/components_e2e_260227_r0_report12/combined_all/sstk_image_1977782153.jpg)|![](../../teacher/visualizations/teacher_scorer_e2e_260227_r0_report12/by_ar/1x1/sstk_image_1977782153.jpg)|[json](assets/samples/json/sstk_image_1977782153.json)|

### 6.1 샘플별 VLM Teacher 입력/출력(JSON) 확인표
- 아래 표는 각 샘플 JSON(`assets/samples/json/<image_id>.json`)의 `vlm_teacher` 블록에서 집계한 값입니다.
- `입력(1:1 input/top_m)`은 `num_input_candidates/cheap_top_m_size`, `출력(1:1 decision/K)`은 `decision_type/selected_k`를 의미합니다.
- 각 샘플 JSON의 `vlm_teacher.output_summary.by_ar[AR].explanations`에서 short/long 설명을 확인할 수 있습니다.

|super_cat|image_id|AR task 수|decision 분포|selected_k 분포|입력(1:1 input/top_m)|출력(1:1 decision/K)|1:1 top1 candidate_id|1:1 conf|샘플 JSON|
|---|---|---:|---|---|---|---|---|---:|---|
|people_multi|pond5_image_99604568|5|minimal_crop:5|1:2, 5:3|108/1|minimal_crop/1|1x1_g12x12_smaxc_i0001|0.5|[json](assets/samples/json/pond5_image_99604568.json)|
|landscape_nature|sstk_image_1104016514|5|crop:1, minimal_crop:4|5:5|114/30|minimal_crop/5|1x1_g12x12_smaxc_i0001|1.0|[json](assets/samples/json/sstk_image_1104016514.json)|
|architecture_exterior|sstk_image_1165522843|5|crop:4, minimal_crop:1|5:5|103/30|crop/5|1x1_g12x12_smy0_i0002|1.0|[json](assets/samples/json/sstk_image_1165522843.json)|
|sports|sstk_image_1178976352|5|crop:1, minimal_crop:4|3:1, 5:4|122/31|crop/5|1x1_g12x12_smy2_i0004|1.0|[json](assets/samples/json/sstk_image_1178976352.json)|
|other_ambiguous|sstk_image_174059567|5|crop:5|5:5|123/30|crop/5|1x1_g12x12_steach_cgs_jit0_10_i0004|0.717496|[json](assets/samples/json/sstk_image_174059567.json)|
|product_object|sstk_image_1955755765|5|crop:2, minimal_crop:3|4:2, 5:3|182/4|minimal_crop/4|1x1_g12x12_steach_gaic_jit0_22_i0004|0.7104909999999998|[json](assets/samples/json/sstk_image_1955755765.json)|
|transportation|sstk_image_1977782153|5|crop:2, minimal_crop:3|5:5|128/30|minimal_crop/5|1x1_g12x12_smaxc_i0001|1.0|[json](assets/samples/json/sstk_image_1977782153.json)|
|food|sstk_image_2024012429|5|minimal_crop:5|5:5|104/30|minimal_crop/5|1x1_g12x12_smaxc_i0001|0.536654|[json](assets/samples/json/sstk_image_2024012429.json)|
|documents_text|sstk_image_2129282159|5|minimal_crop:5|1:2, 5:3|128/1|minimal_crop/1|1x1_g12x12_smx2_i0003|0.5|[json](assets/samples/json/sstk_image_2129282159.json)|
|people_single|sstk_image_2190537205|5|minimal_crop:5|1:1, 5:4|136/1|minimal_crop/1|1x1_g12x12_steach_cacnet_jit0_13_i0004|0.5|[json](assets/samples/json/sstk_image_2190537205.json)|
|animals|sstk_image_2257975789|5|crop:1, minimal_crop:4|5:5|125/30|minimal_crop/5|1x1_g12x12_smaxc_i0001|0.5|[json](assets/samples/json/sstk_image_2257975789.json)|
|indoor_interior|sstk_image_629195588|5|minimal_crop:5|5:5|109/31|minimal_crop/5|1x1_g12x12_smaxc_i0001|0.593186|[json](assets/samples/json/sstk_image_629195588.json)|

## 7) 공개 Teacher(5.5) 결과 요약

- raw jsonl rows: **30,000**
- teacher별 rows: `{'gaic': 10000, 'cacnet': 10000, 'cgs': 10000}`
- proposal non-zero rows: `{'cacnet': 10000, 'cgs': 9990, 'gaic': 9990}`
- error rows(raw): `{}`

## 8) 산출물 경로 체크 (보고서 기준)

- candidates: `data/SSTK/10K/artifacts/candidates/candidates_ar_e2e_260227_r0.jsonl`
- teacher: `data/SSTK/10K/artifacts/teacher/scores/teacher_scores_ar_e2e_260227_r0.jsonl`
- precompute viz(report12): `data/SSTK/10K/artifacts/precompute/visualizations/components_e2e_260227_r0_report12`
- teacher viz(report12): `data/SSTK/10K/artifacts/teacher/visualizations/teacher_scorer_e2e_260227_r0_report12`
- topk analytics: `data/SSTK/10K/artifacts/reports/e2e_260227_r0/assets/analytics/teacher_topk`
- vlm labels/meta/summary: `data/SSTK/10K/artifacts/vlm_teacher/{labels,meta,summary}`
- vlm viz/report analytics: `data/SSTK/10K/artifacts/vlm_teacher/visualizations/vlm_teacher_e2e_260227_r0`, `data/SSTK/10K/artifacts/reports/e2e_260227_r0/assets/analytics/vlm_teacher`

## 9) 재현 명령어 (이번 보고서 보강분)

```bash
# precompute viz (12 super_cat 샘플)
python3 src/visualize_components.py \
  --parquet data/SSTK/10K/filtered_sstk_100.parquet \
  --merged_jsonl data/SSTK/10K/artifacts/precompute/feats_c2c3c5_v2_strict_enriched.jsonl \
  --tar_dir /sstk/20230916/sstk_100 \
  --image_dir data/SSTK/10K/images \
  --out_dir data/SSTK/10K/artifacts/precompute/visualizations/components_e2e_260227_r0_report12 \
  --image_ids_file data/SSTK/10K/artifacts/reports/e2e_260227_r0/sample_ids_supercat12.txt \
  --num_samples 12 --draw_combined 1

# teacher viz (12 샘플 x 5 AR)
python3 src/visualize_teacher_scores.py \
  --teacher_scores_jsonl data/SSTK/10K/artifacts/teacher/scores/teacher_scores_ar_e2e_260227_r0.jsonl \
  --parquet data/SSTK/10K/filtered_sstk_100.parquet \
  --tar_dir /sstk/20230916/sstk_100 \
  --image_dir data/SSTK/10K/images \
  --out_dir data/SSTK/10K/artifacts/teacher/visualizations/teacher_scorer_e2e_260227_r0_report12 \
  --image_ids_file data/SSTK/10K/artifacts/reports/e2e_260227_r0/sample_ids_supercat12.txt \
  --target_ar all --decision_filter all --num_samples 0

# VLM stage-10 viz (12 샘플 x 5 AR + 전체 analytics)
bash src/scripts/run_visualize_vlm_teacher.sh \
  data/SSTK/10K/artifacts/vlm_teacher/labels/crop_label_v1_e2e_260227_r0.jsonl \
  data/SSTK/10K/filtered_sstk_100.parquet \
  /sstk/20230916/sstk_100 \
  data/SSTK/10K/artifacts/vlm_teacher/visualizations/vlm_teacher_e2e_260227_r0 \
  --teacher_scores_jsonl data/SSTK/10K/artifacts/teacher/scores/teacher_scores_ar_e2e_260227_r0.jsonl \
  --image_dir data/SSTK/10K/images \
  --image_ids_file data/SSTK/10K/artifacts/reports/e2e_260227_r0/sample_ids_supercat12.txt \
  --num_samples 0 \
  --target_ar all \
  --analytics_use_full_labels 1
```

## 10) 검토 포인트(초안 단계)

- 현재 run_tag 기준 teacher 결과는 `use_real_expensive=true` 경로로 일관되게 생성됨
- Top-K 분포/score/source 분석 자산은 보고서 `assets/analytics/teacher_topk`에 정리됨
- Stage-10도 동일 run_tag 기준으로 검증 완료(`qwen25_vl`, fallback note 없음, rendered=60/60)
- 추가 요청 시 동일 포맷으로 EN 버전 보고서 및 PPT 요약본 변환 가능