# SSTK Routing v2 Simple 구현 및 검증 리포트

- 빠른 의사결정/실행 기준은 통합본 `Implement_Docs/SSTK_Routing_v2_Unified_Decision_and_Implementation_Brief_KO_2026-06-09.md`를 우선 참조한다. 최신 정책은 `v8_nofood_petstrict_shotstrict_nms_v2`이며, food는 active route class에서 제외됐다.
- 작성일: 2026-06-08
- 최종 반영 버전: `routing_v2_simple_v6`
- 기준 문서: `Implement_Docs/SSTK_Routing_Mode_Extension_Person_Pet_Research_Plan_KO_2026-06-04.md`
- 목적: 향후 `training_labels` 및 `training_labels_multimode` 생성과 MobileCropNet route head 학습에 사용할 사진 의도 routing 라벨, AP-10K animal pose precompute, Flat vs Hierarchical route 비교 실험, 정성 검증 시각화를 실제 산출물로 고정한다.

## 1. 최종 변경 요약

2026-06-09 v6 최종 패치는 image-level route teacher의 신뢰도를 우선해 두 가지를 바꿨다. 첫째, image-level route family/shot/context/confidence는 positive crop vote로 override하지 않는다. positive vote는 `positive_vote_route_*` audit field와 reasons의 `*_observed_only`로만 남긴다. 둘째, `person_single_environmental_portrait`는 희소하고 boundary noise가 커서 독립 shot/class에서 제거한다. 환경성은 `context_intent=environmental` auxiliary로만 둔다. 기존 `upper_body`와 `half_body`는 계속 `upper_half_body`로 통합한다.

| 축 | 최종 값 |
|---|---|
| `route_family_v2` | `scene`, `person_single`, `person_group`, `pet_dogcat`, `food` |
| `person_shot_type` | `face_headshot`, `upper_half_body`, `full_body` |
| `placement_intent` | `center`, `rot` |
| `context_intent` | `tight_subject`, `balanced`, `environmental` |
| feasibility/confidence | `mode_feasible`, `routing_confidence`, `teacher_reliability` |
| flat target | `flat_route_class`, `flat_route_class_id` |
| hierarchical target | `hierarchical_targets.{route_family_v2_id,person_shot_type_id,placement_intent_id,context_intent_id,mode_feasible_id}` |
| v16 image-level target | `image_route_name_no_placement`, `image_route_id_no_placement` |

`upper_half_body` 통합 사유는 다음이다.

| 항목 | 판단 |
|---|---|
| 라벨 안정성 | precompute pose만으로 shoulder-only upper와 hip/waist half를 고신뢰도로 나누기 어렵다. |
| 데이터 균형 | 기존 `upper_body` 희소 문제가 별도 class로 유지될 경우 route head가 쉽게 collapse한다. |
| crop score 정책 | 둘 다 full-body support grounding을 강하게 요구하지 않고, head/torso 보존과 자연스러운 하단 절단을 우선한다. |
| 학습 계획 | full-body와 headshot 사이의 중간 portrait로 통합한 뒤, 리뷰 seed가 충분해지면 세분화 여부를 재검토한다. |

## 2. 구현 파일

| 파일 | 내용 |
|---|---|
| `src/routing/routing_v2_simple.py` | simple routing teacher, environmental portrait evidence, `upper_half_body` 통합, flat/hier target 생성 |
| `src/scripts/build_routing_v2_simple_labels.py` | v14 multimode 산출물에서 `training_labels`와 `training_labels_multimode`를 동시에 재생성 |
| `src/scripts/compare_routing_v2_flat_vs_hier.py` | flat class head와 hierarchical head를 precompute feature 기반 probe로 비교 |
| `src/scripts/build_animal_pose_precompute.py` | AP-10K RTMPose-M checkpoint 다운로드, SHA 검증, dog/cat detection NMS, pose JSONL 생성 |
| `src/scripts/visualize_routing_v2_simple_labels.py` | routing label card, representative contact sheet, AP-10K pose overlay 생성 |
| `src/scripts/visualize_multimode_crop_review.py` | v14 crop review 형식에 `routing_v2_simple` header/footer와 route 분포 summary를 추가 |
| `src/scripts/select_routing_v2_multimode_review_samples.py` | route/shot/context/feasibility/AR-mode/co-primary bucket을 통합 포함하는 multimode 정성 검토 샘플 선별 |
| `src/scripts/train_mobilecropnet_v4_route_expert.py` | `routing_v2_route_family`, `routing_v2_flat`, `routing_v2_image_route_no_placement` label source 지원 |
| `tests/test_routing_v2_simple.py` | scene/person/pet/environmental/flat-hier target 단위 테스트 |

## 3. Multimode Mode List 및 Routing 매핑

v14 multimode의 10개 기본 mode는 유지하고, 각 image/query/annotation에 `routing_v2_simple` metadata를 붙인다. simple taxonomy는 object 일반 category를 아직 독립 route family로 열지 않으며, dog/cat evidence가 있는 object만 `pet_dogcat`으로 승격한다.

| v14 mode | 설명 | simple routing 매핑 |
|---|---|---|
| `landscape` | 장면, 배경, 풍경, 공간 중심 crop | `route_family_v2=scene`, `context_intent=balanced` |
| `single_person_center` | 단일 인물 중앙 배치 query | `person_single`, shot evidence에 따라 `face_headshot`/`upper_half_body`/`full_body`; 환경성은 `context_intent=environmental` |
| `single_person_rot` | 단일 인물 rule-of-thirds 배치 query | `person_single`, `placement_intent=rot` |
| `group_center` | 복수 인물 중앙 배치 query | `person_group`, shot type은 `na` |
| `group_rot` | 복수 인물 rule-of-thirds 배치 query | `person_group`, `placement_intent=rot` |
| `face` | 얼굴/headshot 중심 query | `person_single`, `person_shot_type=face_headshot` |
| `object_single_center` | 단일 object 중앙 배치 query | dog/cat evidence가 있으면 `pet_dogcat`, 아니면 `scene`으로 접힘 |
| `object_single_rot` | 단일 object rule-of-thirds 배치 query | dog/cat evidence가 있으면 `pet_dogcat`, 아니면 `scene`으로 접힘 |
| `object_multi_center` | 다중 object 중앙 배치 query | dog/cat evidence가 있으면 `pet_dogcat`, 아니면 `scene`으로 접힘 |
| `object_multi_rot` | 다중 object rule-of-thirds 배치 query | dog/cat evidence가 있으면 `pet_dogcat`, 아니면 `scene`으로 접힘 |

### 3.1 라벨 삽입 위치

| 산출물 | 필드 |
|---|---|
| `training_labels/*.jsonl` | image-level `routing_v2_simple`, `flat_route_class`, `flat_route_class_id`, `hierarchical_targets`, `label_weight`, `probe_features` |
| `training_labels_multimode/label_json/*.json` image row | image-level `routing_v2_simple`, `flat_route_class`, `flat_route_class_id`, `routing_v2_hierarchical_targets` |
| `training_labels_multimode/label_json/*.json` annotation attributes | query/annotation-level `routing_v2_simple`, `flat_route_class`, `flat_route_class_id`, `routing_v2_hierarchical_targets` |
| `mode_query_status_routing_v2_simple.jsonl` | query-level `routing_v2_simple`, `flat_route_class`, `flat_route_class_id`, `routing_v2_hierarchical_targets` |
| `routing_v2_simple_vocabs.json` | flat/hier vocab, `schema_version=routing_v2_simple_v6` |

## 4. 최종 산출물

최신 생성 run tag는 `260609_routing_v2_simple_v6_image_route_novote_noenv_food_petpose_v16prep`이다.

| 산출 계열 | 경로 |
|---|---|
| multimode | `data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529/artifacts/training_labels_multimode/260609_routing_v2_simple_v6_image_route_novote_noenv_food_petpose_v16prep/` |
| training_labels | `data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529/artifacts/training_labels/260609_routing_v2_simple_v6_image_route_novote_noenv_food_petpose_v16prep/` |
| vocab | `.../routing_v2_simple_vocabs.json` |
| query status | `.../mode_query_status_routing_v2_simple.jsonl` |
| v16 promoted labels | `.../training_labels/260609_v16_routing_v2_mode_catalog_novote_noenv_food_petpose/` |
| v16 promoted multimode | `.../training_labels_multimode/260609_v16_routing_v2_mode_catalog_novote_noenv_food_petpose/` |
| multimode crop review viz | `.../training_labels_multimode/260609_v16_routing_v2_mode_catalog_novote_noenv_food_petpose/crop_review_routing_v2_v16_novote_noenv/panels_unified_220/` |

Schema validation 결과는 다음이다.

| split | rows | schema_version | missing schema field |
|---|---:|---|---:|
| full | 10,000 | `routing_v2_simple_v6` 10,000 rows | 0 |
| train | 9,000 | `routing_v2_simple_v6` 9,000 rows | 0 |
| val | 1,000 | `routing_v2_simple_v6` 1,000 rows | 0 |

Image-level route 분포는 다음이다.

| split | scene | person_single | person_group | pet_dogcat | food |
|---|---:|---:|---:|---:|---:|
| full | 5,097 | 2,830 | 756 | 409 | 908 |
| train | 4,620 | 2,527 | 678 | 372 | 803 |
| val | 477 | 303 | 78 | 37 | 105 |

Person shot/context 분포는 다음이다.

| split | `face_headshot` | `upper_half_body` | `full_body` | `context=environmental` |
|---|---:|---:|---:|---:|
| full | 472 | 1,564 | 794 | 193 |
| train | 428 | 1,402 | 697 | 170 |
| val | 44 | 162 | 97 | 23 |

Routing confidence 요약은 min 0.25, max 0.99, mean 0.692274, `routing_confidence >= 0.65` 비율 49.49%이다. route head primary supervision은 고신뢰도 row를 중심으로 두고, low-confidence row는 낮은 loss weight 또는 calibration/hard-negative 용도로 분리해야 한다.

## 5. Hierarchical Label List

Hierarchical head는 축별 loss를 분리한다. `person_shot_type_id=-1`은 `route_family_v2 != person_single`인 non-applicable row를 의미하며, 학습 시 shot loss mask에서 제외해야 한다.

| head | id/value | 설명 |
|---|---|---|
| `route_family_v2` | 0 `scene` | 장면/배경/일반 object folded route |
| `route_family_v2` | 1 `person_single` | 단일 인물 intent |
| `route_family_v2` | 2 `person_group` | 복수 인물 intent |
| `route_family_v2` | 3 `pet_dogcat` | dog/cat pet intent |
| `route_family_v2` | 4 `food` | food/tableware intent |
| `person_shot_type` | 0 `face_headshot` | 얼굴/머리 중심, tight subject 가능 |
| `person_shot_type` | 1 `upper_half_body` | upper/half body 통합, shoulder/torso/hip 계열 중간 portrait |
| `person_shot_type` | 2 `full_body` | 발/하체/support 보존이 필요한 전신 |
| `placement_intent` | 0 `center` | 중앙 배치 |
| `placement_intent` | 1 `rot` | rule-of-thirds 배치 |
| `context_intent` | 0 `tight_subject` | 피사체 tight crop |
| `context_intent` | 1 `balanced` | 피사체와 배경 균형 |
| `context_intent` | 2 `environmental` | 환경 맥락 보존 |
| `mode_feasible` | 0 `False` | no-positive/infeasible |
| `mode_feasible` | 1 `True` | positive feasible |

## 6. Flat Class Label List

Flat head는 observed Cartesian class를 하나의 id로 예측한다. 최신 v6 full vocab은 17 class이며 실제 학습/검증 시에는 `260609_routing_v2_simple_v6_image_route_novote_noenv_food_petpose_v16prep/routing_v2_simple_vocabs.json`의 `flat_route_class`를 기준으로 한다. 아래 표는 v3 구현 당시의 legacy 예시이므로 최신 primary target으로 사용하지 않는다.

| id | flat_route_class | 설명 |
|---:|---|---|
| 0 | person_group\|full_body\|center\|balanced\|feasible | 복수 인물; 전신; 중앙 배치; 균형 맥락; positive 가능 |
| 1 | person_group\|full_body\|rot\|balanced\|feasible | 복수 인물; 전신; rule-of-thirds 배치; 균형 맥락; positive 가능 |
| 2 | person_group\|na\|center\|balanced\|feasible | 복수 인물; 중앙 배치; 균형 맥락; positive 가능 |
| 3 | person_group\|na\|center\|balanced\|infeasible | 복수 인물; 중앙 배치; 균형 맥락; no-positive/불가능 |
| 4 | person_group\|na\|center\|tight_subject\|feasible | 복수 인물; 중앙 배치; 타이트 피사체; positive 가능 |
| 5 | person_group\|na\|rot\|balanced\|feasible | 복수 인물; rule-of-thirds 배치; 균형 맥락; positive 가능 |
| 6 | legacy v3 `person_single\|environmental_portrait\|center\|environmental\|feasible` | v6에서는 제거. `person_single|full_body/upper_half_body|center|environmental|feasible`로 흡수 |
| 7 | legacy v3 `person_single\|environmental_portrait\|center\|environmental\|infeasible` | v6에서는 제거. environmental은 context auxiliary로만 유지 |
| 8 | person_single\|face_headshot\|center\|balanced\|feasible | 단일 인물; 얼굴/헤드샷; 중앙 배치; 균형 맥락; positive 가능 |
| 9 | person_single\|face_headshot\|center\|tight_subject\|feasible | 단일 인물; 얼굴/헤드샷; 중앙 배치; 타이트 피사체; positive 가능 |
| 10 | person_single\|face_headshot\|center\|tight_subject\|infeasible | 단일 인물; 얼굴/헤드샷; 중앙 배치; 타이트 피사체; no-positive/불가능 |
| 11 | person_single\|face_headshot\|rot\|balanced\|feasible | 단일 인물; 얼굴/헤드샷; rule-of-thirds 배치; 균형 맥락; positive 가능 |
| 12 | person_single\|full_body\|center\|balanced\|feasible | 단일 인물; 전신; 중앙 배치; 균형 맥락; positive 가능 |
| 13 | person_single\|full_body\|center\|balanced\|infeasible | 단일 인물; 전신; 중앙 배치; 균형 맥락; no-positive/불가능 |
| 14 | person_single\|full_body\|rot\|balanced\|feasible | 단일 인물; 전신; rule-of-thirds 배치; 균형 맥락; positive 가능 |
| 15 | person_single\|upper_half_body\|center\|balanced\|feasible | 단일 인물; 상반신/반신 통합; 중앙 배치; 균형 맥락; positive 가능 |
| 16 | person_single\|upper_half_body\|center\|balanced\|infeasible | 단일 인물; 상반신/반신 통합; 중앙 배치; 균형 맥락; no-positive/불가능 |
| 17 | person_single\|upper_half_body\|rot\|balanced\|feasible | 단일 인물; 상반신/반신 통합; rule-of-thirds 배치; 균형 맥락; positive 가능 |
| 18 | pet_dogcat\|na\|center\|balanced\|feasible | 개/고양이 pet; 중앙 배치; 균형 맥락; positive 가능 |
| 19 | pet_dogcat\|na\|center\|tight_subject\|feasible | 개/고양이 pet; 중앙 배치; 타이트 피사체; positive 가능 |
| 20 | pet_dogcat\|na\|center\|tight_subject\|infeasible | 개/고양이 pet; 중앙 배치; 타이트 피사체; no-positive/불가능 |
| 21 | pet_dogcat\|na\|rot\|balanced\|feasible | 개/고양이 pet; rule-of-thirds 배치; 균형 맥락; positive 가능 |
| 22 | scene\|na\|center\|balanced\|feasible | 장면/배경; 중앙 배치; 균형 맥락; positive 가능 |
| 23 | scene\|na\|center\|balanced\|infeasible | 장면/배경; 중앙 배치; 균형 맥락; no-positive/불가능 |
| 24 | scene\|na\|rot\|balanced\|feasible | 장면/배경; rule-of-thirds 배치; 균형 맥락; positive 가능 |

## 7. Flat vs Hierarchical 비교

`compare_routing_v2_flat_vs_hier.py`는 deploy image model이 아니라 teacher/precompute feature 기반의 빠른 separability probe다. 학습 row는 `routing_confidence >= 0.65` 4,123장으로 제한했고, 평가 row는 전체 val 및 high-confidence val을 각각 보았다.

| eval set | train rows | val rows | flat acc/bal/macroF1 | hier route_family acc/bal/macroF1 | hier shot masked acc/bal/macroF1 | hier composed exact acc/bal/macroF1 |
|---|---:|---:|---:|---:|---:|---:|
| 전체 val | 4,123 | 1,000 | 0.486 / 0.324 / 0.195 | 0.887 / 0.947 / 0.821 | 0.659 / 0.642 / 0.506 | 0.377 / 0.213 / 0.164 |
| val confidence >= 0.65 | 4,123 | 480 | 0.619 / 0.449 / 0.299 | 0.996 / 0.996 / 0.996 | 0.667 / 0.676 / 0.519 | 0.587 / 0.383 / 0.314 |

해석은 다음과 같다.

- `route_family_v2`는 hierarchical head로 매우 안정적으로 분리된다.
- flat exact는 observed class 25개를 한 번에 맞추므로 rare class 및 no-positive class가 섞인다.
- hierarchical composed exact는 route/shot/context/feasible argmax를 조합하기 때문에 축별 오류가 누적된다.
- 권장 학습 구조는 hierarchical multi-head를 primary로 두고, flat head는 auxiliary consistency 또는 ablation head로 유지하는 것이다.
- v6에서는 `environmental_portrait` 독립 class를 제거했다. 기존 희소 sample은 body shot으로 흡수하고 `context_intent=environmental` reviewer seed로 검증한다.

## 8. Animal Pose 체크포인트 및 산출물

AP-10K RTMPose-M checkpoint를 로컬과 원격 공유 스토리지에 세팅했다.

| 항목 | 값 |
|---|---|
| 로컬 경로 | `weights/mmpose/rtmpose-m_simcc-ap10k_pt-aic-coco_210e-256x256-7a041aa1_20230206.pth` |
| 원격 경로 | `/group-volume/users/jaden.ju/Sources/ImageCropping/weights/mmpose/rtmpose-m_simcc-ap10k_pt-aic-coco_210e-256x256-7a041aa1_20230206.pth` |
| SHA256 | `896e3665d849ef7eb9b6ec0995955796cc9810f024fa0aa0bdc18acb0d68bf52` |
| 공식 URL | `https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-ap10k_pt-aic-coco_210e-256x256-7a041aa1_20230206.pth` |
| fallback URL | `https://huggingface.co/waveydaveygravy/animal_openpose/resolve/d82ce5b822d4168bf6cbdbb85d6417338ec8d71e/rtmpose-m_simcc-ap10k_pt-aic-coco_210e-256x256-7a041aa1_20230206.pth` |

원격 GPU 환경은 `10.12.12.2`, `NVIDIA A100-SXM4-80GB MIG 3g.40gb`, `torch 2.5.1+cu121`, `mmpose 1.3.2`로 확인했다. 원격 `transformers 4.57.6`과 `mmpretrain 1.2.0` import 호환성 문제는 package downgrade 없이 `build_animal_pose_precompute.py` runtime shim으로 해결했다.

| run | 위치 | 결과 |
|---|---|---|
| 로컬 smoke | `.../animal_pose_260608_routing_v2_simple_local_smoke3_ap10k/` | 3 images, 9 rows, 전부 AP-10K pose |
| 원격 GPU smoke | 원격 `animal_pose_260608_routing_v2_simple_gpu_smoke50_ap10k/` | 50 images, 134 rows, dog 100/cat 34 |
| 원격 GPU full | `data/SSTK/.../animal_pose_260608_routing_v2_simple_gpu_full_ap10k/` | 469 images, 1,203 rows, dog 911/cat 292 |

Full 결과는 `pose_available=True` 1,203 rows이며, 핵심 파일은 `animal_pose.jsonl`, `summary.json`, `status.json`이다.

## 9. 시각화 산출물 및 정성 검증

시각화 스크립트는 다음 정보를 한 카드에 합성한다.

| 시각화 | 포함 정보 |
|---|---|
| Routing card | 원본 이미지, subject/person/dogcat detection box, positive crop 후보 box, `routing_v2_simple` field, flat class, confidence, support policy, environmental evidence |
| Animal pose card | AP-10K bbox, head/body box, keypoint skeleton, species hint, pose confidence, visible group score |
| Contact sheet | bucket별 대표 샘플을 한 장으로 묶어 route taxonomy와 overlay 품질을 빠르게 리뷰 |
| Manifest | representative sample id, class, confidence, card path를 CSV로 저장 |

산출 경로는 다음이다. `training_labels`와 `training_labels_multimode` 양쪽 모두 별도 시각화 산출물을 갖도록 고정했다.

| 산출물 | 경로 |
|---|---|
| `training_labels` routing cards | legacy v3: `.../training_labels/260608_routing_v2_simple_v3_env_upperhalf_from_v14_reviewfix/qualitative_viz/routing_cards/` |
| `training_labels` animal pose cards | legacy v3: `.../training_labels/260608_routing_v2_simple_v3_env_upperhalf_from_v14_reviewfix/qualitative_viz/animal_pose_cards/` |
| `training_labels` contact sheets | `.../qualitative_viz/routing_contact_sheet.jpg`, `.../qualitative_viz/animal_pose_contact_sheet.jpg` |
| `training_labels_multimode` routing/animal cards | legacy v3: `.../training_labels_multimode/260608_routing_v2_simple_v3_env_upperhalf_from_v14_reviewfix/qualitative_viz_routing_v2_simple_v3/` |
| multimode AR/mode stratified crop review | legacy v3: `.../crop_review_routing_v2_simple_v3/stratified_200_best_by_ar_mode/` |
| multimode routing-aware crop review | legacy v3: `.../crop_review_routing_v2_simple_v3/routing_v2_route_stratified/` |
| v16 unified crop review | latest v6: `.../training_labels_multimode/260609_v16_routing_v2_mode_catalog_novote_noenv_food_petpose/crop_review_routing_v2_v16_novote_noenv/panels_unified_220/` |

최종 생성 결과는 다음이다. contact sheet와 대표 panel을 직접 열어 이미지, crop box, person/pet overlay, `routing_v2_simple` text, AP-10K skeleton이 nonblank으로 표시되는 것을 확인했다.

| 산출 세트 | 생성 결과 | 검증 포인트 |
|---|---:|---|
| `training_labels/qualitative_viz` | routing card 35장, animal pose card 24장 | image-level route, flat class, confidence, support policy, animal pose overlay |
| `training_labels_multimode/qualitative_viz_routing_v2_simple_v3` | routing card 35장, animal pose card 24장 | multimode annotation overlay와 AP-10K pose를 multimode run 아래에서 직접 추적 가능 |
| `stratified_200_best_by_ar_mode` | panel 200장, contact sheet 4장, missing 0 | 기존 v14 crop review와 동일한 AR/mode 균형 검토 |
| `routing_v2_route_stratified` | panel 160장, contact sheet 2장, missing 0 | route/shot/context/feasibility 희소 bucket을 강제 포함 |
| `crop_review_routing_v2_v16_novote_noenv/panels_unified_220` | panel 220장, contact sheet 2장, missing 0 | AR/mode 균형과 route-aware 희소 bucket, co-primary person+pet bucket을 하나로 통합 |

Multimode crop review panel에는 image-level routing line과 positive annotation routing group이 header에 표시되고, 각 crop card footer에는 crop/query 기준 `crop_mode=`와 `crop_route=<route_family>/<shot>/<placement>/<context>`가 표시된다. image-level route target은 `IMAGE ROUTE TARGET(no placement)`로 별도 표기하며, `scene_center`, `person_single_full_body_rot` 같은 placement 포함 값은 crop/query mode로만 해석한다. manifest CSV/JSON에는 `image_route_name_no_placement`, `image_v16_summary_mode_aux`, `image_flat_route_class_aux`, `routing_v2_counts_json`, `flat_route_counts_json`을 추가했다.

`stratified_200_best_by_ar_mode`의 image-level route 분포는 `scene=77`, `person_single=60`, `person_group=52`, `pet_dogcat=11`이다. positive annotation routing group은 `scene/center/balanced=1196`, `scene/rot/balanced=591`, `person_single/full_body/center/balanced=700`, `person_single/full_body/rot/balanced=321`, `person_group/center/balanced=266`, `person_group/rot/balanced=190`, `pet_dogcat/center/tight_subject=96`, `pet_dogcat/rot/balanced=86` 등으로 확인된다.

최신 `routing_v2_v16_novote_noenv` unified selection은 220장을 뽑았고 image route 분포는 `food=72`, `person_group=28`, `person_single_face_headshot=25`, `person_single_full_body=27`, `person_single_upper_half_body=21`, `pet_dogcat=17`, `scene=30`이다. person shot은 `face_headshot=25`, `full_body=27`, `upper_half_body=21`, context는 `balanced=158`, `environmental=21`, `tight_subject=41`이다.

2026-06-09 이후 운영 기준은 두 세트를 별도로 유지하지 않고 v16 unified crop review 하나로 통합하는 것이다. `stratified_200_best_by_ar_mode`는 `positive_ar_mode:*` bucket으로, `routing_v2_route_stratified`는 `image_route:*`, `person_shot:*`, `context:*`, `feasible:*` bucket으로 흡수했다. selection manifest에는 `primary_selection_bucket`과 `selection_reasons_json`을 추가해 어떤 bucket 때문에 선택됐는지 추적할 수 있다.

대표 샘플은 다음처럼 선정했다.

| bucket | 샘플 | 관찰 |
|---|---|---|
| scene | `sstk_image_81655729`, `sstk_image_1606989652` | 일반 scene/object folded route에서 positive crop 후보와 scene/balanced context가 일치한다. |
| face/headshot | `sstk_image_607328609`, `bigstock_image_186539722` | face/headshot은 `tight_subject` 또는 balanced context로 분리되며 support grounding이 inactive다. |
| upper_half_body | `bigstock_image_306278578`, `pond5_image_50763469` | shoulder/torso/hip 계열 중간 portrait가 `upper_half_body`로 통합된다. |
| full_body | `bigstock_image_124620903`, `bigstock_image_183099842` | lower-body/ankle evidence가 있는 경우 `full_body`와 support-active 정책으로 분리된다. |
| environmental context | `bigstock_image_278715973`, `sstk_image_1061424497`, boundary `pond5_image_141826930` | 최신 v6에서는 독립 shot이 아니라 `context_intent=environmental`로만 남기며, boundary 샘플은 동물성 object/조형물 때문에 오탐 위험을 드러낸다. |
| person_group | `bigstock_image_115747316`, `bigstock_image_144626189` | 복수 인물은 shot type을 `na`로 두고 family/placement/context만 학습한다. |
| pet_dogcat | `bigstock_image_331768912`, `sstk_image_2138069669` | dog/cat evidence와 animal pose overlay가 routing card에 함께 표시된다. |

정성 검증에서 확인된 중요한 리스크는 `pet_dogcat` upstream 후보가 비반려동물 또는 일부 오검출을 포함할 수 있다는 점이다. `pond5_image_141826930`처럼 조형물/동물성 object가 `pet_dogcat` positive group과 environmental context를 동시에 유발하는 사례가 있고, AP-10K overlay에서도 wild animal 또는 비반려동물 후보에 dog/cat species hint가 붙는 사례가 보인다. AP-10K pose 자체는 skeleton을 생성하지만, dog/cat route teacher의 정밀도는 C2 species confidence, metadata/tag, CLIP/VLM species evidence, animal pose head/eye confidence를 함께 보아야 한다. 따라서 pet scoring에는 `animal_pose_required`만이 아니라 `species_confidence`, dog/cat tag/CLIP 보강, non-dog/cat hard negative 수집을 추가하는 것이 필요하다.

추가로 `bigstock_image_41939056`에서 실제 개 한 마리에 대해 AP-10K pose가 5개 그려지는 duplicate 문제가 확인됐다. 원인은 top-down AP-10K에 들어간 upstream dog/cat bbox가 NMS 없이 중복 전달된 것이다. `build_animal_pose_precompute.py`에 `det_nms_iou=0.65`, `det_nms_containment=0.85` 기준 NMS를 추가했고, 해당 샘플은 raw 5개에서 keep 1개/suppress 4개로 줄어든다. 기존 full artifact는 NMS 적용 전 결과이므로 다음 pet label 재생성 전에 full animal pose precompute를 재실행해야 한다.

## 10. Route Expert Smoke

`train_mobilecropnet_v4_route_expert.py`의 dynamic vocab 로딩을 v3 labels로 검증했다. 이 smoke는 random-init 64 train / 32 val, 1 epoch CPU run이므로 성능 판단용이 아니다.

| label_source | output | 결과 |
|---|---|---|
| `routing_v2_route_family` | `artifacts/mobilecropnet_v4/routing_v2_simple_v3_route_family_image_smoke_260608/` | completed, `best.pt`, `last.pt`, `summary.json`, route-balanced sampler 확인 |
| `routing_v2_flat` | `artifacts/mobilecropnet_v4/routing_v2_simple_v3_flat_image_smoke_260608/` | completed, 25-class vocab 로딩과 checkpoint/status 산출 확인 |

실제 head 성능 비교는 pretrained backbone, full train split, route-balanced sampler, high-confidence weighting, rare-class oversampling으로 별도 GPU run을 잡아야 한다.

## 11. P0-P5 적용 상태

| 단계 | 적용 내용 | 상태 |
|---|---|---|
| P0 문서/현황 고정 | v14 review 보완 제안과 simple taxonomy v3 고정 | 완료 |
| P1 routing_v2 audit/builder | image/query/annotation 라벨, flat/hier target, confidence, support policy 생성 | 완료 |
| P2 AnimalPose smoke/full | AP-10K checkpoint 직접 다운로드, 로컬/원격 SHA 검증, 로컬 smoke, 원격 smoke/full precompute | 완료 |
| P3 pilot label generation | `training_labels_multimode`와 `training_labels` 양쪽에 v3 라벨 생성 | 완료 |
| P4 route-only smoke | feature probe, Flat vs Hier 비교, image-only route expert smoke | 완료 |
| P5 full crop model integration 준비 | route expert label source 추가. MobileCropNet v4 본체의 장기 학습/배포 checkpoint 교체는 별도 GPU 실험으로 남김 | 부분 완료 |

P5의 남은 핵심은 crop model 본체에 hierarchical route heads와 route-conditioned proposal/action branch를 추가하는 것이다. 기존 MobileCropNet v4 본체는 legacy `SUBJECT_MODE_VOCAB` 7-class route를 기준으로 하므로, 이번 작업에서는 기존 checkpoint compatibility를 깨지 않기 위해 standalone route expert와 label artifacts까지 연결했다.

## 12. 학습 적용 방안

권장 target 구성은 다음이다.

| 학습 항목 | 권장 방식 |
|---|---|
| primary image route | v16/v6부터 `image_route_name_no_placement` 7-class 또는 hierarchical `route_family_v2/person_shot_type`, `routing_confidence >= 0.65` row 중심 |
| person shot | `route_family_v2=person_single` row에만 masked CE 적용 |
| placement | image-level target에서 제외. `training_labels_multimode` query/crop mode 또는 별도 ablation에서만 사용 |
| context | 전체 row에 적용하되 `tight_subject`, `environmental`은 class-balanced/focal loss 사용 |
| feasibility | query-level no-positive supervision 유지 |
| flat class | auxiliary consistency head 또는 ablation 비교용 |
| pet auxiliary | `animal_pose.jsonl`의 head/body/keypoint visible group으로 pet head/pose auxiliary target 생성 |
| label weight | `label_weight = routing_confidence`, low confidence는 낮은 weight 또는 calibration split |

Full model 학습 시 필요한 추가 구현은 다음이다.

1. MobileCropNet v4 backbone 뒤에 `image_route_head`, `route_family_head`, `person_shot_head`, `context_head`, `feasibility_head`를 분리한다.
2. `person_shot_head`는 applicable mask를 사용하고 non-person row는 loss에서 제외한다.
3. route embedding을 proposal/action decoder 또는 crop score head에 주입한다.
4. `pet_dogcat`은 animal pose-derived head/body/paw/tail cut auxiliary를 추가하되, upstream species confidence가 낮은 row는 hard positive로 쓰지 않는다.
5. batch sampler는 scene/person/pet/person_group 균형, person 내부 shot 균형, environmental/pet oversampling을 동시에 적용한다.
6. 검증은 route balanced accuracy, person shot confusion, pet precision/recall, no-positive calibration, mode별 proposal recall@K, final crop review pass rate를 분리 보고한다.

## 13. 검증 결과

| 검증 | 결과 |
|---|---|
| local pytest | `/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/python -m pytest tests/test_routing_v2_simple.py -q` -> 13 passed |
| local py_compile | routing/simple label/compare/route expert/animal pose/crop review/visualization scripts 통과 |
| label schema | full/train/val schema missing 0, `schema_version=routing_v2_simple_v6` |
| animal pose checkpoint SHA | 로컬, CPU 공유 스토리지, GPU 런 모두 `896e3665...bf52` 일치 |
| animal pose full | 469 images, 1,203 rows, 전부 `mmpose_rtmpose_ap10k` |
| animal pose NMS diagnostic | `bigstock_image_41939056` raw 5 -> keep 1/suppress 4, diagnostic image 생성 |
| `training_labels` qualitative viz | routing cards 35, animal pose cards 24, contact sheet nonblank 확인 |
| `training_labels_multimode` qualitative viz | routing cards 35, animal pose cards 24, AP-10K contact sheet nonblank 확인 |
| multimode crop review viz | AR/mode stratified 200 panels/4 contact sheets, routing-aware 160 panels/2 contact sheets, missing 0 |
| route expert smoke | route_family/flat v3 label source checkpoint/summary 생성. 2026-06-09에는 `routing_v2_image_route_no_placement` label source 로딩/라벨 매핑 단위 테스트 추가 |

## 14. 후속 연구/개발/실험 계획

후속 작업의 최우선 기준은 `routing_v2_simple_v6` teacher label의 precision을 높이고, 그 신뢰도가 MobileCropNet route head 성능으로 실제 전이되는지 검증하는 것이다.

| 우선순위 | 작업 | 구체 내용 | 완료 기준 |
|---|---|---|---|
| P0 | teacher label audit set 고정 | 최신 기준은 `crop_review_routing_v2_v16_novote_noenv/panels_unified_220` 220장이다. route family, person shot, context, feasible, crop/query mode를 사람이 독립 검수한다. | class별 reviewer agreement, teacher precision/recall 추정치 산출 |
| P0 | v16 unified crop review audit | `crop_review_routing_v2_v16_novote_noenv` 220장을 기준 reviewer seed로 사용한다. 기존 AR/mode 세트와 route-aware 세트를 별도로 보지 않는다. | selection bucket별 pass/fail, 수정 label CSV |
| P0 | pet hard negative 수집 | 조형물, 말/야생동물, 장난감, 털 소재, 동물 그림/간판 등 dog/cat 오탐 후보를 별도 CSV로 모은다. | `pet_dogcat` false-positive bucket별 최소 50장 이상 |
| P0 | animal pose NMS full rerun | NMS 적용 precompute로 `animal_pose.jsonl`을 재생성하고 pet pose/head gate를 재검증한다. | duplicate pose 감소, pose/head image ids 및 pet positives 일관성 |
| P0 | person+pet co-primary audit | 23장 co-primary 후보와 forced sample `sstk_image_80436817`을 검수한다. | single dominant route 유지 vs multi-label/`person_pet_joint` ablation 결정 |
| P1 | pet species evidence 보강 | C2 dog/cat score, segmentation area, metadata/tag, CLIP text similarity, AP-10K head/eye/torso visibility를 feature로 합쳐 pet teacher를 재보정한다. | pet route precision 개선, wild/non-pet false positive 감소 |
| P1 | environmental context 흡수 검증 | person area, face/body visibility, scene score, foreground/background mass, crop candidate의 context 보존율을 reviewer label과 맞춰 threshold calibration한다. | `context_intent=environmental` auxiliary precision 개선 |
| P1 | confidence calibration | `routing_confidence`를 단순 휴리스틱 점수가 아니라 reviewer agreement와 probe error를 반영한 calibrated confidence로 재산정한다. | reliability diagram/ECE 및 confidence bucket별 accuracy 보고 |
| P1 | `upper_half_body` 유지 검증 | `upper_body`/`half_body` 재분리는 보류하고, 통합 class 내부의 pose/crop 분포만 monitoring한다. | full-body/headshot과의 confusion matrix가 안정적일 것 |
| P2 | hierarchical full route head 학습 | pretrained backbone으로 `image_route_no_placement`, `route_family`, `person_shot`, `context`, `feasible` masked multi-head를 full train split에서 학습한다. placement는 query/crop ablation으로 분리한다. | route family balanced acc, image route macroF1, shot macroF1, context macroF1 보고 |
| P2 | flat auxiliary ablation | 동일 backbone/seed/sampler에서 flat-only, hier-only, hier+flat auxiliary를 비교한다. | rare class collapse 여부와 crop downstream metric 차이 보고 |
| P2 | sampler/loss ablation | high-confidence only, confidence-weighted, class-balanced, focal loss, pet/food sampling을 비교한다. | class별 recall과 no-positive calibration trade-off 보고 |
| P3 | route-conditioned crop model 통합 | MobileCropNet v4 본체에 route embedding을 crop proposal/action/score branch에 주입한다. | proposal recall@K, mode별 final score, crop review pass rate 개선 |
| P3 | animal pose auxiliary | pet crop에서 head/body/paw/tail cut penalty 또는 auxiliary visibility head를 붙인다. low species confidence row는 hard positive에서 제외한다. | pet crop에서 head/body 보존율 개선 |
| P3 | multimode scoring 재보정 | mode별 support grounding, tight/balanced/environmental context, no-positive threshold를 route별로 다르게 둔다. | query-level positive/no-positive precision 개선 |
| P4 | 정성 QA 자동화 | 이번 crop review 산출물을 정기 생성 job으로 만들고, reviewer 코멘트/수정 라벨을 다음 label build에 피드백한다. | 매 run 동일 schema의 manifest, summary, contact sheet 생성 |
| P4 | 원격 full experiment 운영 | GPU run은 bounded job으로 `status.json`, `train.log`, `summary.json`, `best.pt`를 남기고, CPU 서버에서 summary/report를 후처리한다. | 재현 가능한 run directory와 exact command 기록 |
| P5 | taxonomy 확장 재검토 | food는 v6/v16에서 열었으므로 food precision audit을 먼저 수행한다. product/object macro, vehicle, non-dog/cat animal route는 reviewer evidence와 downstream gain이 확인된 뒤 확장한다. | 현 taxonomy 대비 유의미한 downstream 개선 |

실험 순서는 `teacher audit -> confidence/species calibration -> hierarchical full route head -> flat/hier ablation -> route-conditioned crop model -> reviewer feedback loop`가 적절하다. 현재 단계에서 class를 더 늘리면 rare class와 teacher noise가 동시에 커지므로, `upper_half_body` 통합과 `pet_dogcat` 최소 분리는 유지하고, 다음 확장은 reviewer seed와 full training 결과로 결정한다.

## 15. 2026-06-09 v6/v16 최종 Addendum

사용자 검토 의견에 따라 v5/v16 결과를 다시 고쳤다. 최신 의사결정 기준은 통합 문서 `Implement_Docs/SSTK_Routing_v2_Unified_Decision_and_Implementation_Brief_KO_2026-06-09.md`이다.

| 항목 | v5 문제 | v6/v16 최종 변경 |
|---|---|---|
| image route vote | positive crop vote가 image-level route를 scene/pet/food로 override할 수 있었음 | image-level route family/shot/context/confidence는 positive vote 영향 제거. vote는 audit field로만 저장 |
| environmental portrait | `person_single_environmental_portrait` 독립 class가 11장으로 희소 | image route와 v16 mode에서 제거. `context_intent=environmental`로 흡수 |
| v16 mode catalog | 16 modes | 14 modes |
| image route catalog | 8-class | 7-class |
| pet over legacy person | `bigstock_image_286519432`는 positive vote 제거 시 person으로 회귀 가능 | dog/cat image evidence가 충분하면 feature-only `dogcat_evidence_over_legacy_portrait` rule로 pet 지정 |
| placement_intent | image-level target처럼 오해 가능 | image route target 제외. query/crop mode와 crop scorer에서만 사용 |
| visual title | flat/v16 label title이 placement 포함 | unified image route catalog만 사용. card title은 `IMAGE ROUTE: <placement 없는 route>` |

최신 run은 다음이다.

| 산출물 | run/path |
|---|---|
| v6 source labels | `260609_routing_v2_simple_v6_image_route_novote_noenv_food_petpose_v16prep` |
| v16 mode labels | `260609_v16_routing_v2_mode_catalog_novote_noenv_food_petpose` |
| visual catalog | `data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529/artifacts/training_labels_multimode/260609_v16_routing_v2_mode_catalog_novote_noenv_food_petpose/routing_v2_unified_visual_catalog/` |
| unified crop review | `data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529/artifacts/training_labels_multimode/260609_v16_routing_v2_mode_catalog_novote_noenv_food_petpose/crop_review_routing_v2_v16_novote_noenv/panels_unified_220/` |
| follow-up decision pack | `data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529/artifacts/training_labels_multimode/260609_v16_routing_v2_mode_catalog_novote_noenv_food_petpose/routing_v2_followup_decision_pack_260609/` |
| animal pose NMS diagnostic | `data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529/artifacts/precompute/animal_pose_260608_routing_v2_simple_gpu_full_ap10k/nms_diagnostics/` |

Full split image route 7-class 분포는 `scene=5,097`, `person_single_face_headshot=472`, `person_single_upper_half_body=1,564`, `person_single_full_body=794`, `person_group=756`, `pet_dogcat=409`, `food=908`이다. v16 multimode query/crop mode catalog는 14개이며, image-level route classifier 학습 target은 `v16_mode_name`이 아니라 `image_route_name_no_placement`다.

v5 대비 v6의 주요 전이는 다음이다. v5에서 `positive_vote_scene_override`로 scene이 됐던 1,179장은 v6에서 `person_single_upper_half_body=561`, `person_single_full_body=333`, `person_group=206`, `person_single_face_headshot=79`로 복원됐다. 기존 `person_single_environmental_portrait` 11장은 `person_single_full_body=10`, `person_single_upper_half_body=1`로 흡수됐다.

검증 결과는 `tests/test_routing_v2_simple.py` 14 passed, v16 `mode_count=14`, `image_route_no_placement_count=7`, unified visual cards 7장, crop-review panels 220장/missing 0이다.

## 16. 외부 근거

- Adobe Research, [Automatic Image Cropping using Visual Composition, Boundary Simplicity and Content Preservation Models](https://research.adobe.com/publication/automatic-image-cropping-using-visual-composition-boundary-simplicity-and-content-preservation-models/)
- arXiv, [Aesthetics-Aware Reinforcement Learning for Image Cropping](https://arxiv.org/abs/1709.04595)
- arXiv, [AP-10K: A Benchmark for Animal Pose Estimation in the Wild](https://arxiv.org/abs/2108.12617)
- MMPose Documentation, [Animal 2D Keypoint Model Zoo](https://mmpose.readthedocs.io/zh-cn/latest/model_zoo/animal_2d_keypoint.html)
- MMPose Documentation, [2D Animal Keypoint Dataset](https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_animal_keypoint.html)
- NFI, [Portrait Shots](https://mail.nfi.edu/portrait-shots/)
- Adobe, [Pet photography guide](https://www.adobe.com/creativecloud/photography/discover/pet-photography.html)
# 문서 상태 공지

이 문서는 routing_v2 simple/v16 구현 및 검증의 중간 이력으로 보존한다. 2026-06-09 이후 최신 canonical 운영 기준과 v17 precompute-native 산출물은 `Implement_Docs/SSTK_MultiMode_TrainingLabels_Enhanced_Onboarding_KO_v2_0.md`의 "부록 E. Routing v2 Precompute-Native v17 운영 기준"을 우선한다.

최신 핵심 변경: v14/v16 retrofit 라벨은 학습용 canonical에서 제외, precompute/candidate-bank/native factory 재생성으로 전환, full_body crop mismatch 0.0 검증, food/environmental_portrait class 제거.
