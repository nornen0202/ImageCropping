# SSTK Subject-Mode Routing + C2 Top-N 패치 구현 보고서 (KO)

## 1) 목적
- `SSTK_Dataset_Comparison_RnD_and_SubjectMode_PatchDesign_KO.md`의 보완안(섹션 3/4/5/6/7/8/10, 부록)을 실제 코드에 반영.
- Precompute -> Candidate -> Teacher -> QA까지 `subject_mode/policy_id`가 닫히는 운영 경로를 구성.
- 기존 파이프라인 호환성을 유지하면서, routed 산출물을 기본 다운스트림 입력으로 사용하도록 e2e를 확장.

## 2) 주요 구현 요약
- 신규 모듈 `src/routing/subject_mode_router.py`
  - `subject_mode` 추론(`portrait_single/group`, `object_single/multi`, `scene_landscape`, `background_texture_copyspace`, `text_document`, `other_ambiguous`)
  - `policy_id` 매핑
  - `c2_seg` Top-N/importance 산출 로직
    - `area_ratio`, `center_dist_norm`, `border_touch`, `bg_like`, `importance_score`, `importance_comp`, `source`
  - `subject_set` 생성
    - `num_person`, `num_subject_inst`, `union_box_xyxy`, `primary_idx`, `c2_primary_area_ratio`, `c2_primary_bg_like`, `multi_subject`

- 신규 스크립트 `src/scripts/enrich_subject_mode_jsonl.py`
  - 입력: merged precompute jsonl + filtered parquet
  - 출력: routed precompute jsonl
  - 주입 필드:
    - `routing`
    - `c2_primary_idx`
    - `c2_union_box_xyxy`
    - `c2_topn`
    - `c2_stats`
    - 확장된 `c2_seg`(Top-N + importance)

- `src/generate_candidates.py` 확장
  - routed feature의 `routing/c2_primary_idx/c2_union_box_xyxy`를 입력으로 사용.
  - `subject_prior` 구성 시 `subject_mode` 기반 union/none-subject 처리 적용.
  - Candidate 출력에 `routing`, `subject` 메타를 추가.
  - overview에 `subject_mode_counts`, `policy_id_counts` 추가.

- `src/score_teacher.py` 확장
  - candidate/feature routing 힌트를 읽어 모드별 정책 오버라이드 적용.
  - `policy_id` 기반 `lambdas/tau_improve/w_area` 스케줄 조정.
  - scene/copyspace/text에서 HR/LR 축소, context/copyspace/text 비중 강화.
  - teacher output(`routing`, `route_global`)에 `subject_mode/policy_id/subject_set` 반영.

- `src/scripts/qa_teacher_report.py` 확장
  - 모드 KPI 추가:
    - `subject_mode_counts`, `policy_id_counts`
    - `subject_mode_conf` 통계/충돌율
    - `candidate_volume`(num_input/c2_num_inst)
    - `subject_mode_kpi`(`bg_selected_rate`, `multi_subject_detect_rate`, `union_used_rate`)
  - 리포트 축 확장:
    - `by_subject_mode`
    - `by_subject_mode_ar`

- `src/scripts/run_phaseA_to_teacher_e2e.sh` 확장
  - Subject-Mode 단계 추가:
    - `07a_enrich_subject_mode`
  - 신규 옵션:
    - `--run_subject_routing`
    - `--subject_routing_top_n`
    - `--subject_routing_union_top_m`
    - `--subject_routing_allow_det_proxy`
  - routed 파일(`feats_*_routed.jsonl`)을 Candidate/Teacher 다운스트림 기본 입력으로 사용.

- `src/scripts/run_teacher_scorer.sh` 보완
  - feature 입력 자동 탐색 시 routed jsonl 우선 사용.

- 문서 업데이트
  - `README_SSTK_Curation.md`에 Subject-Mode enrich 단계/옵션/권장 산출물 반영.

## 3) 변경 파일
- 신규
  - `src/routing/__init__.py`
  - `src/routing/subject_mode_router.py`
  - `src/scripts/enrich_subject_mode_jsonl.py`
  - `ImplementPlan_Docs/SSTK_SubjectMode_Routing_Patch_Implementation_KO.md` (본 문서)

- 수정
  - `src/generate_candidates.py`
  - `src/score_teacher.py`
  - `src/scripts/qa_teacher_report.py`
  - `src/scripts/run_phaseA_to_teacher_e2e.sh`
  - `src/scripts/run_teacher_scorer.sh`
  - `README_SSTK_Curation.md`

## 4) 검증 결과
- 문법/정적 검증
  - `py_compile` (신규/수정 Python 파일): 통과
  - `bash -n` (`run_phaseA_to_teacher_e2e.sh`, `run_teacher_scorer.sh`): 통과

- 단위 테스트
  - `tests/test_generate_candidates.py`: 통과
  - `tests/test_teacher_scorer.py`: 통과

- 스모크 E2E(10K_local, max_images=20, proxy expensive)
  - subject-mode enrich: 성공
    - 출력: `data/SSTK/10K_local/artifacts/precompute/feats_c2c3c5_v2_strict_enriched_routed.jsonl`
  - candidate: 성공
    - 출력: `data/SSTK/10K_local/artifacts/candidates/candidates_ar_subject_mode_smoke.jsonl`
  - teacher scorer: 성공
    - 출력: `data/SSTK/10K_local/artifacts/teacher/scores/teacher_scores_ar_subject_mode_smoke.jsonl`
  - QA: 성공
    - 출력: `data/SSTK/10K_local/artifacts/teacher/qa/teacher_scores_qa_report_subject_mode_smoke.json`
  - QA JSON에 `by_subject_mode`, `by_subject_mode_ar`, `subject_mode_kpi` 생성 확인.

## 5) 운영 가이드(핵심)
- e2e 기본값으로 Subject-Mode 단계가 활성화됨(`--run_subject_routing 1`).
- 다운스트림 입력은 자동으로 routed precompute를 우선 사용.
- 기존 산출물과 호환되며, routing 미존재 파일도 fallback 경로로 동작.

