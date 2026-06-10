# SSTK Routing v2 통합 의사결정 및 구현 브리프

- 작성일: 2026-06-09
- 최신 상태: `v17_precompute_native_nofood_petstrict_cropstrict_nms_v5`
- 목적: routing_v2 taxonomy, label 생성, visual audit, 남은 의사결정을 한 문서에서 확인한다.
- 최신 canonical 문서는 `Implement_Docs/SSTK_MultiMode_TrainingLabels_Enhanced_Onboarding_KO_v2_0.md`이다. 이 문서는 빠른 의사결정 브리프와 run 이력으로만 사용한다.

## 1. 최종 결정 요약

| 항목 | 최신 결정 |
|---|---|
| image-level primary target | `image_route_name_no_placement` 6-class |
| food class | 제외. food/tableware evidence는 audit/probe metadata로만 남기고 active route는 `scene`으로 접는다. |
| query/crop v16 mode | `route_family_v2 + person_shot_type + placement_intent` 12-class |
| positive crop vote | image-level route family/shot/context/confidence를 override하지 않는다. audit 참고값으로만 저장한다. |
| placement target | image-level route에서 제외. query/crop intent와 crop scorer conditioning에만 사용한다. |
| single person shot | `face_headshot`, `upper_half_body`, `full_body` |
| upper/half body | `upper_half_body` 하나로 통합 |
| environmental portrait | 독립 route/shot class 없음. `context_intent=environmental` auxiliary로만 남김 |
| pet route | `pet_dogcat`만 활성화. dog/cat 외 animal/object는 hard negative/audit 대상 |
| pet positive gate | 유효 AP-10K dog/cat pose/head precompute subset에서만 positive 허용 |
| full_body policy | 기존 metadata가 full_body여도 reliable lower-body pose validator를 통과해야 유지 |
| flat vs hierarchical | image-route 또는 hierarchical head를 primary로 두고 flat class는 auxiliary/ablation |
| v16 retrofit labels | audit/회귀분석 이력으로만 유지. 최신 학습에는 v17 precompute-native 산출물 사용 |
| candidate bank 누락 | label factory 내부 대체 금지. 먼저 candidate-bank completion artifact 생성 |

가장 중요한 학습 지침은 다음이다.

1. image-level route classifier target은 `image_route_name_no_placement` 또는 `image_route_id_no_placement`다.
2. image-level route classifier에 `v16_mode_name`을 target으로 쓰지 않는다.
3. `placement_intent`는 image-level target이 아니라 query/crop mode 및 crop scorer용 auxiliary다.
4. `food`는 현재 route class가 아니다.

## 2. 최신 산출물

| 산출물 | path |
|---|---|
| v17 training_labels | `data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529/artifacts/training_labels/260609_v17_routing_v2_precompute_native_nofood_petstrict_cropstrict_nms_v5/` |
| v17 training_labels_multimode | `data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529/artifacts/training_labels_multimode/260609_v17_routing_v2_precompute_native_nofood_petstrict_cropstrict_nms_v5/` |
| v17 result brief | `data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529/artifacts/training_labels_multimode/260609_v17_routing_v2_precompute_native_nofood_petstrict_cropstrict_nms_v5/PRECOMPUTE_NATIVE_V17_RESULTS_KO.md` |
| v17 validation | `data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529/artifacts/training_labels_multimode/260609_v17_routing_v2_precompute_native_nofood_petstrict_cropstrict_nms_v5/validation_summary.json` |
| v17 crop review panels | `data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529/artifacts/training_labels_multimode/260609_v17_routing_v2_precompute_native_nofood_petstrict_cropstrict_nms_v5/crop_review_routing_v2_v17_precompute_native/panels_unified_220/` |
| v17 audit | `data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529/artifacts/training_labels_multimode/260609_v17_routing_v2_precompute_native_nofood_petstrict_cropstrict_nms_v5/crop_review_routing_v2_v17_precompute_native/panels_unified_220/audit_jy_260609_v17/` |
| completed candidate bank | `data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529/artifacts/candidates/candidates_ar_260529_v11_photo_primary_fullsstk_phaseA_strict_intent_subjectsafe_split9010_v17_complete_260609.jsonl` |
| simple source labels | `data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529/artifacts/training_labels/260609_routing_v2_simple_v8_nofood_petstrict_shotstrict_nms_v2/` |
| simple source multimode | `data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529/artifacts/training_labels_multimode/260609_routing_v2_simple_v8_nofood_petstrict_shotstrict_nms_v2/` |
| v16 labels (legacy/audit-only) | `data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529/artifacts/training_labels/260609_v16_routing_v2_mode_catalog_nofood_petstrict_shotstrict_nms_v2/` |
| v16 multimode | `data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529/artifacts/training_labels_multimode/260609_v16_routing_v2_mode_catalog_nofood_petstrict_shotstrict_nms_v2/` |
| v16 catalog | `.../260609_v16_routing_v2_mode_catalog_nofood_petstrict_shotstrict_nms_v2/v16_mode_catalog.json` |
| image route catalog | `.../260609_v16_routing_v2_mode_catalog_nofood_petstrict_shotstrict_nms_v2/image_route_no_placement_catalog.json` |
| contract validation | `.../260609_v16_routing_v2_mode_catalog_nofood_petstrict_shotstrict_nms_v2/routing_v2_contract_validation.json` |
| visual catalog | `.../260609_v16_routing_v2_mode_catalog_nofood_petstrict_shotstrict_nms_v2/routing_v2_unified_visual_catalog/` |
| crop review panels | `.../260609_v16_routing_v2_mode_catalog_nofood_petstrict_shotstrict_nms_v2/crop_review_routing_v2_v16_nofood_petstrict_shotstrict/panels_unified_220/` |
| auto triage decision pack | `.../260609_v16_routing_v2_mode_catalog_nofood_petstrict_shotstrict_nms_v2/routing_v2_auto_triage_decision_pack_260609/` |
| AP-10K NMS precompute | `data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529/artifacts/precompute/animal_pose_260609_routing_v2_simple_gpu_full_ap10k_nms_v2/` |

## 2.1 v17 Precompute-Native 검증 결과

| 지표 | 값 |
|---|---:|
| images | 10,000 |
| multimode annotations | 201,396 |
| positive annotations | 47,415 |
| candidate bank empty AR | 0 |
| missing candidate images | 0 |
| validation state | `passed` |
| selected full-body mismatch rate | 0.0 |
| full positive full-body mismatch rate | 0.0 |

v17에서 수행한 핵심 보정은 다음이다.

- 원본 candidate bank의 9:16 empty bucket 4건을 별도 completion artifact로 생성했다. label factory 내부 대체는 금지했다.
- `face` query가 existing `shot_type=full_body`에 오염되지 않도록 face mode 우선순위를 core inference에서 보정했다.
- full_body crop positive는 relaxed support/bottom-margin 후보를 demote한다.
- image-level shot이 `face_headshot` 또는 `upper_half_body`로 신뢰되는 single-person 이미지에서는 full_body crop positive를 허용하지 않는다.

## 3. 최신 Taxonomy

### 3.1 Image-Level Route Catalog

| id | label | full images |
|---:|---|---:|
| 0 | `scene` | 6,084 |
| 1 | `person_single_face_headshot` | 473 |
| 2 | `person_single_upper_half_body` | 2,136 |
| 3 | `person_single_full_body` | 223 |
| 4 | `person_group` | 761 |
| 5 | `pet_dogcat` | 323 |

### 3.2 Query/Crop v16 Mode Catalog

| id | v16 mode |
|---:|---|
| 0 | `scene_center` |
| 1 | `scene_rot` |
| 2 | `person_single_face_headshot_center` |
| 3 | `person_single_face_headshot_rot` |
| 4 | `person_single_upper_half_body_center` |
| 5 | `person_single_upper_half_body_rot` |
| 6 | `person_single_full_body_center` |
| 7 | `person_single_full_body_rot` |
| 8 | `person_group_center` |
| 9 | `person_group_rot` |
| 10 | `pet_dogcat_center` |
| 11 | `pet_dogcat_rot` |

### 3.3 Hierarchical Axes

| axis | values | 정책 |
|---|---|---|
| `route_family_v2` | `scene`, `person_single`, `person_group`, `pet_dogcat` | image route 또는 hierarchical primary |
| `person_shot_type` | `face_headshot`, `upper_half_body`, `full_body` | `person_single`에서만 masked CE |
| `context_intent` | `tight_subject`, `balanced`, `environmental` | auxiliary |
| `placement_intent` | `center`, `rot` | image-level target 제외, query/crop mode에만 사용 |
| `mode_feasible` | `true`, `false` | query/no-positive calibration |

`flat_route_class`는 observed 17개 class가 있다. 희소 class collapse 위험이 있으므로 primary target이 아니라 auxiliary/ablation로만 사용한다.

## 4. 원인 분석 및 수정 내역

### 4.1 `004_person_single_full_body.jpg`

문제: `sstk_image_89594566`은 실제 상반신/허리 위주 이미지인데 `full_body`로 라벨링됐다.

원인:

- 기존 `infer_person_shot_type()`이 metadata의 `shot_type=full_body` 또는 약한 lower-body keypoint를 우선 신뢰했다.
- `has_ankle=True`만으로 full_body 승격이 가능했다.
- 해당 샘플은 ankle confidence가 낮고 얼굴/몸 비율이 전신 구도와 맞지 않았다.

수정:

- `full_body`는 `lower_body_reliable=True`일 때만 유지한다.
- 기존 metadata가 `full_body`여도 validator 실패 시 `upper_half_body`로 demote한다.
- validator 기준은 ankle confidence, face/body ratio, lower-body visibility를 함께 본다.

최종 결과:

| image | final image route | reason |
|---|---|---|
| `sstk_image_89594566` | `person_single_upper_half_body` | `shot_upper_half_body:pose_partial_lower_body_upper_half_body` |

### 4.2 `006_pet_dogcat.jpg`

문제: `sstk_image_676581775`는 사람 2명의 group portrait인데 dog/cat이 없고 `pet_dogcat`으로 라벨링됐다.

원인:

- COCO dog detector가 낮은 score `0.293703`의 false dog box를 냈다.
- 기존 pet override가 `portrait_group`에도 적용됐다.
- AP-10K gate가 row 존재/weak head score에 너무 관대했다.
- `c2_seg`와 `c2_det` 중복으로 dog/cat count와 area가 부풀 수 있었다.

수정:

- dog/cat detector evidence는 primary C2 stream만 사용해 중복 count를 줄인다.
- `dogcat_present` score 하한을 `0.45`로 상향했다.
- legacy `portrait_group`은 pet override로 뒤집지 않는다.
- AP-10K gate는 `species_confidence >= 0.50`, `pose_confidence >= 0.30`, `visible_groups.head >= 0.45`를 요구한다.

최종 결과:

| image | final image route | reason |
|---|---|---|
| `sstk_image_676581775` | `person_group` | `legacy_portrait_group` |

### 4.3 Food Exclusion

문제: `bigstock_image_349630243`처럼 음식/접시/손 중심 이미지가 `person_single`로 들어갔다.

결정:

- food route는 다시 제외했다.
- 다만 food/tableware evidence가 강하고 person area가 작으며 legacy portrait가 불확실하면 `scene`으로 접는다.

최종 결과:

| image | final image route | reason |
|---|---|---|
| `bigstock_image_349630243` | `scene` | `food_evidence_over_legacy_portrait_folded_to_scene_no_food_route` |

### 4.4 Pet 유지 사례

`bigstock_image_286519432`는 개가 중앙 주피사체이며 AP-10K pose/head gate를 통과하므로 `pet_dogcat` 유지가 맞다.

| image | final image route | reason |
|---|---|---|
| `bigstock_image_286519432` | `pet_dogcat` | `dogcat_evidence_over_legacy_portrait` |
| `sstk_image_80436817` | `pet_dogcat` | `dogcat_evidence`, 단 person+pet co-primary audit 대상 |

## 5. 검증 결과

| 검증 | 결과 |
|---|---|
| unit test | `tests/test_routing_v2_simple.py`, `tests/test_routing_v2_contract.py`: 23 passed |
| contract validator | `ok=true`, image routes 6, v16 modes 12, invalid rows 0 |
| food route scan | `food`, `food_center`, `food_rot`, `route_family_v2=food` match 없음 |
| pet pose gate | `pet_dogcat` 323건 모두 `pet_pose_head_available=True` |
| positive vote override | validator 기준 override rows 0 |
| visual catalog | image-route cards 6, unified contact sheet 1 |
| crop review | 220 panels, missing 0 |
| AP-10K NMS | raw dog/cat boxes 1,203 -> NMS suppressed 654 -> final pose rows 549 |

전면 감사 결과:

| audit item | count | 해석 |
|---|---:|---|
| `full_body_lower_body_not_reliable` | 0 | 기존 full_body 우회 문제 제거 |
| `full_body_face_ratio_high` | 0 | close-up full_body 오분류 제거 |
| `food_route` | 0 | food active route 없음 |
| `pet_pose_gate_not_true` | 0 | pet route는 모두 pose/head gate 통과 |
| `pet_score_lt_055` | 17 | 낮은 detector score지만 pose/head gate 통과. auto triage 검수 대상 |
| `pet_over_legacy_portrait` | 3 | portrait override pet. 대표 예: 실제 개 중심 `bigstock_image_286519432` |
| `food_present_non_scene` | 144 | 사람+음식/테이블웨어 혼합. active food route는 아니며 false-person boundary audit 대상 |

## 6. Visual 산출물 해석

| 산출물 | 의미 |
|---|---|
| `routing_v2_unified_visual_catalog/unified_image_route_contact_sheet.jpg` | placement 없는 image route 6-class 대표 카드 |
| `routing_v2_unified_visual_catalog/unified_image_route_manifest.csv` | 대표 카드의 image route, v16 summary aux, flat aux 분리 기록 |
| `crop_review_routing_v2_v16_nofood_petstrict_shotstrict/panels_unified_220/` | route/shot/pet/scene 및 강제 issue sample 포함 crop review |
| `routing_v2_auto_triage_decision_pack_260609/contact_sheets/auto_needs_user_review_top.jpg` | 사람이 먼저 볼 자동 검수 상위 후보 |

visual title 정책:

- image-level card title은 `IMAGE ROUTE: <image_route_name_no_placement>`만 쓴다.
- `placement_intent`는 `image_placement_aux`, `v16_summary_aux`, crop/query panel에서만 표시한다.
- `positive route/vote`는 crop annotation metadata 집계이며 image-level route를 override하지 않는다.

## 7. Auto Triage Decision Pack

기존 decision pack은 CSV에 수동으로 채워야 할 column이 너무 많았다. 최신 pack은 자동 권고를 먼저 제공한다.

| 항목 | 값 |
|---|---:|
| total images | 10,000 |
| bucket max | 80 |
| union `needs_user_review` ids in pack | 247 |
| `auto_needs_user_review_top` | 80 |
| `co_primary_person_pet_review` | 18 |

주요 column:

| column | 의미 |
|---|---|
| `auto_suggested_action` | `keep`, `review_low_confidence`, `review_person_vs_scene_object`, `review_pet_or_demote_scene_person` 등 |
| `needs_user_review` | 사람이 확인해야 할지 여부 |
| `auto_review_reason` | 자동 권고 이유 |
| `reviewer_*` | 자동 권고가 틀린 경우에만 채운다 |

사용 순서:

1. `contact_sheets/auto_needs_user_review_top.jpg`를 먼저 본다.
2. 같은 이름의 CSV에서 `needs_user_review=true` 행만 확인한다.
3. 자동 권고가 틀린 행에만 `reviewer_*` column을 채운다.
4. 아래 의사결정 항목을 확정하면 다음 label build rule/loss config에 반영한다.

## 8. 남은 의사결정

내가 구현/검증으로 닫은 항목:

| 항목 | 상태 |
|---|---|
| food class 제외 | 완료 |
| positive vote image route override 제거 | 완료 |
| placement image-level target 제외 | 완료 |
| `person_single_environmental_portrait` 제거 | 완료 |
| pet AP-10K NMS 및 gate 강화 | 완료 |
| weak full_body demotion | 완료 |
| training_labels / training_labels_multimode 동시 재생성 | 완료 |
| visual catalog / crop review / auto triage pack 생성 | 완료 |

사용자가 반드시 결정해야 하는 항목:

| id | 검토 산출물 | 결정 질문 | 허용 결정 |
|---|---|---|---|
| D1 | `pet_hard_negative_review.jpg`, `pet_hard_negative_review.csv` | dog/cat pet positive와 non-dog/cat hard negative 경계 | `keep_pet`, `hard_negative_scene`, `non_dogcat_object`, `unsure` |
| D2 | `co_primary_person_pet_review.jpg`, `co_primary_person_pet_review.csv` | 사람+pet 공동 주피사체를 single dominant로 둘지 multi-label/joint ablation을 열지 | `single_dominant`, `allow_multilabel`, `add_person_pet_joint_ablation`, `unsure` |
| D3 | `shot_full_body_boundary_review.jpg`, `shot_full_body_boundary_review.csv` | strict full_body policy를 유지할지 VLM/visual audit로 recall 보강할지 | `keep_strict`, `add_vlm_audit`, `relax_threshold`, `unsure` |
| D4 | `low_confidence_calibration_review.jpg`, `low_confidence_calibration_review.csv` | low-confidence hard label을 그대로 쓸지 low-weight/calibration-only로 돌릴지 | `hard_label_ok`, `low_weight_only`, `calibration_only`, `relabel_needed` |

권장 기본값:

| id | 권장 |
|---|---|
| D1 | `keep_pet`, 단 `pet_score_lt_055` 17건은 hard-negative seed로 육안 확인 |
| D2 | `single_dominant` 유지, `person_pet_joint`는 별도 ablation만 열기 |
| D3 | `keep_strict`, full_body recall은 VLM audit로 보강 |
| D4 | `low_weight_only` 또는 `calibration_only`를 route head 학습에서 ablation |

## 9. 학습 적용 계획

### 9.1 Route Head

| head | target | loss |
|---|---|---|
| `image_route_head` | `image_route_id_no_placement` 6-class | primary CE, class-balanced/focal ablation |
| `route_family_head` | `route_family_v2` 4-class | auxiliary 또는 hierarchical primary |
| `person_shot_head` | `person_shot_type` 3-class | `person_single` row masked CE |
| `context_head` | `context_intent` | auxiliary |
| `feasibility_head` | `mode_feasible` | query/no-positive calibration |
| `flat_head` | `flat_route_class_id` | auxiliary/ablation only |

### 9.2 Crop Model

| component | 사용 신호 |
|---|---|
| image encoder | image route embedding |
| candidate/query encoder | v16 query/crop mode embedding |
| crop scorer | route-conditioned score branch |
| pet branch | animal head/body visibility auxiliary, AP-10K precompute teacher |
| calibration | `routing_confidence`, `mode_feasible`, no-positive reasons |

### 9.3 다음 실험

| 우선순위 | 작업 |
|---|---|
| P1 | auto triage top 80 육안 검수 후 rule threshold 미세 조정 |
| P1 | 6-class image-route pretrained backbone route expert 재실행 |
| P2 | hierarchical multi-head vs image-route single-head 비교 |
| P2 | full_body strict policy로 인한 recall 저하를 VLM teacher 또는 visual audit seed로 보강 |
| P3 | route-conditioned crop scorer에서 v16 mode embedding gain 평가 |
| P3 | pet-specific crop score component와 hard-negative mining |

## 10. 구현 변경 파일

| 파일 | 변경 |
|---|---|
| `src/routing/routing_v2_simple.py` | food route 제거, pet detector/pose gate 강화, C2 dog/cat 중복 방지, full_body validator/demotion |
| `src/routing/routing_v2_v16_modes.py` | 6 image routes, 12 v16 modes |
| `src/scripts/build_routing_v2_simple_labels.py` | AP-10K valid head id 판정 강화, v8 default tag |
| `src/scripts/build_routing_v2_v16_mode_labels.py` | nofood v16 default tag |
| `src/scripts/build_routing_v2_v16_visual_catalog.py` | nofood visual default, route axis 4-class |
| `src/scripts/select_routing_v2_multimode_review_samples.py` | nofood crop review default |
| `src/scripts/build_routing_v2_followup_decision_pack.py` | auto triage, food decision 제거, shot/pet/confidence review 중심 |
| `src/scripts/train_mobilecropnet_v4_route_expert.py` | default route family fallback 4-class |
| `tests/test_routing_v2_simple.py` | food fold, weak full_body demotion, pet/v16 tests |
| `tests/test_routing_v2_contract.py` | 6 route / 12 mode contract |

## 11. 과거 문서와의 관계

| 문서 | 상태 |
|---|---|
| `SSTK_Routing_Mode_Extension_Person_Pet_Research_Plan_KO_2026-06-04.md` | 연구 배경/문헌용. food-open 결정은 최신 아님 |
| `SSTK_Routing_v2_Simple_Implementation_and_Validation_Report_KO_2026-06-08.md` | v6/v7 구현 이력용. 최신 target count는 이 문서 기준 |
| `SSTK_Routing_v2_v16_Mode_Catalog_and_Label_Visual_Audit_KO_2026-06-09.md` | v16 visual audit 이력용. 최신 nofood/strict 정책은 이 문서 기준 |

## 12. 260609 Crop Review 후속 Audit 업데이트

사용자 crop review 메모 `CROP_REVIEW_JY_260609.md`를 기준으로 별도 전수 audit를 수행했다. 결과적으로 현재 `260609_v16_routing_v2_mode_catalog_nofood_petstrict_shotstrict_nms_v2` run은 schema/contract는 통과했지만, `person_shot_type` teacher label 품질 측면에서는 추가 재작업이 필요하다.

상세 문서:

- `data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529/artifacts/training_labels_multimode/260609_v16_routing_v2_mode_catalog_nofood_petstrict_shotstrict_nms_v2/crop_review_routing_v2_v16_nofood_petstrict_shotstrict/panels_unified_220/CROP_REVIEW_JY_260609_ANALYSIS_AND_FOLLOWUP_PLAN.md`

생성 audit 산출물:

| 산출물 | 용도 |
|---|---|
| `audit_jy_260609/audit_summary.json` | crop/image route audit 요약 |
| `audit_jy_260609/user_marked_issue_reproduction.csv` | 사용자 지적 샘플 재현 |
| `audit_jy_260609/crop_full_body_mismatch_selected_220.csv` | selected 220 panel의 full-body crop mismatch 후보 |
| `audit_jy_260609/crop_full_body_mismatch_full_annotations.csv` | 전체 positive annotation의 full-body crop mismatch 후보 |
| `audit_jy_260609/image_group_single_person_suspicious.csv` | group route지만 single-person evidence인 후보 |
| `audit_jy_260609/image_face_headshot_fullbody_pose_suspicious.csv` | face/headshot route지만 full-body pose evidence인 후보 |

핵심 수치:

| 항목 | 결과 |
|---|---:|
| selected panel full-body positive | 529 |
| selected panel full-body mismatch 후보 | 193 |
| selected panel full-body mismatch rate | 36.48% |
| 전체 full-body positive | 6,467 |
| 전체 full-body mismatch 후보 | 3,567 |
| 전체 full-body mismatch rate | 55.16% |
| image route `person_group` 중 single-person evidence 의심 | 640 / 761 |
| image route `person_single_face_headshot` 중 full-body pose evidence 의심 | 195 / 473 |

원인:

| 문제 | 원인 |
|---|---|
| 얼굴/상반신 crop이 `person_single_full_body`로 라벨링 | annotation route가 bbox 내부 pose visibility를 보지 않고 image/query-level shot을 승계 |
| full-body 사진이 image route `face_headshot`으로 라벨링 | `existing_shot_type=headshot`에 대한 pose 기반 promotion/validation 부재 |
| 1인 사진이 image route `person_group`으로 라벨링 | `legacy_portrait_group`을 person-count/pose-count로 검증하지 않음 |

업데이트된 후속 우선순위:

| 단계 | 작업 |
|---|---|
| P0 | 현재 run의 crop-level `person_shot_type`을 route head 학습 고정 라벨로 사용하지 않음 |
| P1 | annotation bbox 기준 crop-level shot inference를 구현하고 full-body crop mismatch를 relabel/suppress |
| P1 | COCO image width/height를 feature row에 주입하여 lower-body reliability calibration 수정 |
| P2 | `existing_shot_type=headshot`의 pose validation/promotion 추가 |
| P2 | `legacy_portrait_group`의 person-count/member-evidence validation 및 low-confidence/demotion 정책 추가 |
| P3 | 새 run tag로 `training_labels`와 `training_labels_multimode` 동시 재생성 |
| P4 | 동일 audit 재실행 후 selected full-body mismatch rate를 10% 이하로 낮추는지 확인 |
