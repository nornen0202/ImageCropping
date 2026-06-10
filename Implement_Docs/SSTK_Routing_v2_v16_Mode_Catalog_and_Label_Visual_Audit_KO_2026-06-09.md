# SSTK Routing v2 v16 Mode Catalog 및 Label Visual Audit

- 빠른 의사결정/실행 기준은 통합본 `Implement_Docs/SSTK_Routing_v2_Unified_Decision_and_Implementation_Brief_KO_2026-06-09.md`를 우선 참조한다. 최신 정책은 `v8_nofood_petstrict_shotstrict_nms_v2`이며, food는 active route class에서 제외됐다.
- 작성일: 2026-06-09
- 기준 run: `260609_v16_routing_v2_mode_catalog_novote_noenv_food_petpose`
- source run: `260609_routing_v2_simple_v6_image_route_novote_noenv_food_petpose_v16prep`
- 목적: v14 multimode mode list를 그대로 쓰지 않고, query/crop용 `route_family_v2 + person_shot_type + placement_intent` v16 mode catalog와 image-level용 placement 없는 route catalog를 분리해 고정한다.

2026-06-09 v6 보강 사항: image-level route family/shot/context/confidence는 positive crop vote로 override하지 않는다. positive vote는 `positive_vote_route_*` audit field와 reasons의 `*_observed_only`로만 남긴다. 또한 `person_single_environmental_portrait`는 image route와 v16 mode에서 제거했고, 환경성은 `context_intent=environmental` auxiliary로만 저장한다.

## 1. placement_intent 재검토 결론

원본 이미지의 `placement_intent`를 추정하는 이유는 "원본 사진이 center 구도인가, ROT 구도인가"를 미학적으로 판정하려는 것이 아니다. 실제 목적은 crop/query 생성과 학습에서 다음 효과를 얻는 것이다.

| 목적 | 얻는 효과 |
|---|---|
| query/crop intent 분리 | 같은 피사체라도 중앙 배치 crop과 rule-of-thirds crop은 좋은 crop 후보가 다르다. |
| candidate diversity | center crop만 학습하면 rot 후보가 사라지고, rot만 학습하면 제품/인물 증명사진 계열이 흔들린다. |
| score component 분리 | placement error와 subject recall/context error를 분리해 reject reason을 해석할 수 있다. |
| route-conditioned model | route embedding 또는 mode head가 crop score branch에 배치 의도를 전달할 수 있다. |
| reviewer audit | center/rot 후보가 모두 가능한 애매한 이미지를 별도 bucket으로 모을 수 있다. |

하지만 원본 이미지 image-level `placement_intent`를 classification target으로 쓰는 것은 제외한다. 시각화 결과에서 보듯 원본 구도가 애매한데 `center`로 라벨링된 경우가 많다. 이는 현재 image-level placement가 원본 사진 자체의 고유 속성이 아니라 positive crop/query vote의 요약값이기 때문이다. 따라서 v16 정책은 다음처럼 둔다.

1. `placement_intent`는 image의 본질적 사진 타입이 아니라 crop/query intent로 해석한다.
2. `training_labels_multimode` annotation/query에서는 `center`/`rot`를 적극 사용한다.
3. image-level route head primary target은 `image_route_name_no_placement`만 사용한다.
4. placement는 image-level loss에서 제외하고, v16 query/crop mode 또는 명시적 ablation에서만 별도 분석한다.
5. 최종 crop 모델에는 image-level placement classifier보다 mode-conditioned crop scorer가 더 중요하다.

`review_bigstock_image_286519432`는 이 문제를 잘 보여준다. positive crop vote는 `pet_dogcat`으로 강하지만 center/rot vote share가 약 `0.558`로 애매하다. v6에서는 positive vote override 없이 dog/cat image evidence가 legacy portrait를 이겨 route family가 `pet_dogcat`이 됐고, `placement_intent=center`는 routing metadata와 v16 query/crop mode에만 남긴다. image-level 학습 target은 `pet_dogcat` 하나로 고정한다.

## 2. positive crop routing groups의 의미

시각화 panel 상단의 `positive crop routing groups`는 각 positive crop image를 별도 vision classifier로 다시 라우팅한 결과가 아니다. v6/v16 pipeline에서 각 annotation/query에 `routing_v2_simple` metadata를 새로 붙였고, 시각화 스크립트가 `gt_flag=1`인 positive annotations의 `routing_v2_simple` 값을 집계해 보여주는 것이다. 예전 패널의 `positive routing_v2 groups`와 같은 값이며, 최신 패널에서는 crop/query 기준 집계임을 명확히 하기 위해 이름을 바꿨다.

따라서 의미는 다음과 같다.

| 항목 | 의미 |
|---|---|
| image routing line | image row에 붙은 image-level route summary |
| positive crop routing groups | positive annotation/query들의 route family, shot, placement, context, feasible 집계 |
| crop card footer `crop_route=` | 해당 crop annotation/query의 routing metadata |

기존 v14에는 crop별 routing classifier가 없었다. 이번 작업에서 crop annotation 자체를 다시 분류한 것이 아니라, query/mode 이름과 image feature evidence를 조합해 annotation-level routing metadata를 추가한 것이다.

## 3. 오분류 사례 원인 및 수정

### 3.1 `review_bigstock_image_286519432`

관찰: 개가 중앙에 있고 사람은 우측 하반신 위주로 존재한다. 기존 v3 image routing은 `person_single`이었다.

원인:

| 원인 | 내용 |
|---|---|
| legacy route 우선순위 | `routing.subject_mode=portrait_single`을 image-level route에서 우선했다. |
| dog evidence 미사용 우선순위 | C2 dog score가 높고 positive object query들이 `pet_dogcat`인데도 legacy person이 먼저 적용됐다. |
| placement ambiguity | pet positive query는 center/rot가 거의 비슷해 image-level placement가 애매했다. |

v6 수정:

| 항목 | 결과 |
|---|---|
| route_family_v2 | `pet_dogcat` |
| person_shot_type | `null` |
| placement_intent | `center`, 단 low-confidence/ambiguous placement로 해석 |
| reasons | `dogcat_evidence_over_legacy_portrait`, positive vote는 `positive_vote_route_agreement_observed_only` |

보완 방안:

1. route family는 dog/cat image evidence가 충분할 때 legacy person보다 우선한다.
2. pet positive는 AP-10K animal pose/head precompute가 있는 subset에서만 허용한다.
3. placement는 pet route 학습의 primary target으로 쓰지 않고, center/rot candidate scoring용 intent로 유지한다.

### 3.2 `review_bigstock_image_349630243`

관찰: 음식/접시가 주 피사체이고 사람 손만 보인다. 기존 v3 image routing은 `person_single/environmental_portrait`이었다. v6에서는 `environmental_portrait` 독립 class가 제거됐고, 이 샘플은 food/tableware evidence로 `food`가 된다.

원인:

| 원인 | 내용 |
|---|---|
| object route 부재 | simple taxonomy에 일반 object/food route가 없어 scene 또는 person으로 접혔다. |
| person false positive | C2 person/pose가 손/팔/접시 영역을 `portrait_single`처럼 해석했다. |
| food evidence 미사용 | fork/tableware C2 evidence가 route family 결정에 쓰이지 않았다. |

v6 수정:

| 항목 | 결과 |
|---|---|
| route_family_v2 | `food` |
| person_shot_type | `null` |
| placement_intent | `center` |
| support policy | `food_object_required` |
| reasons | `food_evidence_over_legacy_portrait_single` |

보완 방안:

1. object 확장 중 food를 먼저 연다.
2. COCO food class와 tableware class를 food evidence로 사용하되, 사람과 음식이 같이 있는 party/chef/cafe 이미지의 경계는 reviewer audit으로 보정한다.
3. food route는 product/vehicle보다 먼저 열되, food 정밀도 검증 전에는 downstream crop model의 primary route로 과신하지 않는다.

## 4. v16 Mode Catalog 및 Image-Level Route Catalog

v16 mode는 v14의 10개 mode를 그대로 따르지 않는다. `context_intent`와 `mode_feasible`은 mode name에 넣지 않고, query/crop mode name에는 `route_family_v2 + person_shot_type + placement_intent`만 사용한다. 반면 image-level route target은 placement를 제거한 `image_route_name_no_placement`를 사용한다.

### 4.1 Image-Level Route Catalog

`training_labels/*.jsonl`와 `training_labels_multimode/label_json/*.json` image row에는 다음 필드를 추가했다.

| 필드 | 의미 |
|---|---|
| `image_route_schema_version` | `routing_v2_image_route_no_placement_v1` |
| `image_route_name_no_placement` | image-level route head primary class, placement 제외 |
| `image_route_id_no_placement` | placement 없는 image route id |
| `v16_mode_name` | query/crop mode 호환용 summary, placement 포함 |

Image-level route label list는 다음 7개다.

| id | image route | 설명 |
|---:|---|---|
| 0 | `scene` | 장면/배경/일반 object folded route |
| 1 | `person_single_face_headshot` | 단일 인물 얼굴/headshot |
| 2 | `person_single_upper_half_body` | 단일 인물 upper/half 통합 |
| 3 | `person_single_full_body` | 단일 인물 전신 |
| 4 | `person_group` | 복수 인물 |
| 5 | `pet_dogcat` | dog/cat pet |
| 6 | `food` | food/tableware object |

Full image-level route 분포는 다음이다.

| image route | full images |
|---|---:|
| `scene` | 5,097 |
| `person_single_face_headshot` | 472 |
| `person_single_upper_half_body` | 1,564 |
| `person_single_full_body` | 794 |
| `person_group` | 756 |
| `pet_dogcat` | 409 |
| `food` | 908 |

### 4.2 Query/Crop v16 Mode Catalog

| id | v16 mode | 설명 |
|---:|---|---|
| 0 | `scene_center` | 장면/일반 배경/일반 object folded route, center intent |
| 1 | `scene_rot` | 장면/일반 배경/일반 object folded route, ROT intent |
| 2 | `person_single_face_headshot_center` | 단일 인물 얼굴/headshot, center intent |
| 3 | `person_single_face_headshot_rot` | 단일 인물 얼굴/headshot, ROT intent |
| 4 | `person_single_upper_half_body_center` | 단일 인물 upper/half 통합, center intent |
| 5 | `person_single_upper_half_body_rot` | 단일 인물 upper/half 통합, ROT intent |
| 6 | `person_single_full_body_center` | 단일 인물 전신, center intent |
| 7 | `person_single_full_body_rot` | 단일 인물 전신, ROT intent |
| 8 | `person_group_center` | 복수 인물, center intent |
| 9 | `person_group_rot` | 복수 인물, ROT intent |
| 10 | `pet_dogcat_center` | dog/cat pet, center intent |
| 11 | `pet_dogcat_rot` | dog/cat pet, ROT intent |
| 12 | `food_center` | food/tableware object, center intent |
| 13 | `food_rot` | food/tableware object, ROT intent |

Full multimode annotation 분포는 다음이다.

| v16 mode | annotations |
|---|---:|
| `scene_center` | 67,682 |
| `scene_rot` | 25,648 |
| `person_single_full_body_center` | 20,632 |
| `person_single_full_body_rot` | 5,514 |
| `person_single_upper_half_body_center` | 5,147 |
| `person_single_upper_half_body_rot` | 895 |
| `person_single_face_headshot_center` | 482 |
| `person_single_face_headshot_rot` | 79 |
| `person_group_center` | 2,778 |
| `person_group_rot` | 1,486 |
| `pet_dogcat_center` | 4,483 |
| `pet_dogcat_rot` | 4,445 |
| `food_center` | 13,135 |
| `food_rot` | 14,981 |

Full image-level family 분포는 `scene=5,097`, `person_single=2,830`, `food=908`, `person_group=756`, `pet_dogcat=409`이다.

## 5. Visual Catalog 산출물

대표 샘플은 현재 teacher label 기준의 representative이며 사람이 확정한 gold label이 아니다. contact sheet는 taxonomy coverage와 label noise를 동시에 보기 위한 audit 도구로 사용한다.

2026-06-09 후속 수정으로 `flat_class_cards`와 `v16_mode_cards`를 더 이상 별도 대표 이미지 산출물로 보지 않는다. 두 세트가 같은 원본 이미지를 거의 중복으로 보여주므로, image-level visual audit은 `routing_v2_unified_visual_catalog` 하나로 통합했다. 각 카드 제목은 `IMAGE ROUTE: <image_route_name_no_placement>`로만 표시하고, placement 포함 값은 side panel의 `v16 summary aux`, `flat aux`, `placement aux` 또는 crop review panel에서만 확인한다.

| 산출물 | 경로 | 결과 |
|---|---|---:|
| unified image route cards | `data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529/artifacts/training_labels_multimode/260609_v16_routing_v2_mode_catalog_novote_noenv_food_petpose/routing_v2_unified_visual_catalog/unified_image_route_cards/` | 7 |
| unified image route contact sheet | `.../routing_v2_unified_visual_catalog/unified_image_route_contact_sheet.jpg` | 1 |
| unified image route manifest | `.../routing_v2_unified_visual_catalog/unified_image_route_manifest.csv` | 7 rows |
| hierarchical axis manifest | `.../routing_v2_unified_visual_catalog/hierarchical_axis_manifest.csv` | 15 rows |
| deprecated split visuals | `.../routing_v2_v16_visual_catalog/flat_class_cards/`, `.../routing_v2_v16_visual_catalog/v16_mode_cards/` | 보존만 하고 신규 기준에서는 사용하지 않음 |
| image route catalog | `.../image_route_no_placement_catalog.json` | 7 routes |

### 5.1 unified image route contact sheet

![](../data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529/artifacts/training_labels_multimode/260609_v16_routing_v2_mode_catalog_novote_noenv_food_petpose/routing_v2_unified_visual_catalog/unified_image_route_contact_sheet.jpg)

### 5.2 unified crop review

query/crop v16 mode 대표성은 separate card catalog가 아니라 crop review panel에서 확인한다. header와 crop card 표기는 다음처럼 분리했다.

| 표기 | 의미 |
|---|---|
| `IMAGE ROUTE TARGET(no placement)` | image-level classifier primary target |
| `image metadata` | image row의 route family/shot/context/confidence metadata |
| `v16 summary aux` | image row의 보조 summary. primary target 아님 |
| `positive crop/query modes` | positive annotation/query의 `mode_name` 집계 |
| `crop_mode=` | 해당 crop annotation/query의 v16 mode |
| `crop_route=` | 해당 crop annotation/query의 `route_family/person_shot/placement/context` metadata |

| 산출물 | 결과 |
|---|---:|
| selected panels | 220 |
| contact sheets | 2 |
| missing source images | 0 |
| selection manifest | `.../crop_review_routing_v2_v16_novote_noenv/unified_selection/routing_v2_review_selection_manifest.csv` |
| panel manifest | `.../crop_review_routing_v2_v16_novote_noenv/panels_unified_220/routing_v2_v16_novote_noenv_crop_review_manifest.csv` |

![](../data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529/artifacts/training_labels_multimode/260609_v16_routing_v2_mode_catalog_novote_noenv_food_petpose/crop_review_routing_v2_v16_novote_noenv/panels_unified_220/routing_v2_v16_novote_noenv_crop_review_contact_sheet_part001.jpg)

## 6. Pet Pose Gate 확인

v6 builder는 `animal_pose_260608_routing_v2_simple_gpu_full_ap10k/animal_pose.jsonl`을 읽어 AP-10K head visible group이 있는 image id만 pet positive로 허용한다.

| 항목 | 결과 |
|---|---:|
| AP-10K pose rows | 1,203 |
| pose/head image ids | 469 |
| v6 image-level `pet_dogcat` | 409 |
| positive pet annotations | 4,259 |
| positive pet annotations with pose/head | 4,259 |
| pose gate suppressed positives | 0 |

현재 산출물에서는 모든 pet positive가 pose/head subset에 포함된다. 중요한 점은 v3/v4에서는 이것이 코드상 강제 조건이 아니었고, v5 이후 명시적으로 강제된다는 것이다. 향후 AP-10K가 실패한 dog/cat 후보는 pet positive로 쓰지 않고 no-positive 또는 scene/food/object fallback으로 내려야 한다.

### 6.1 Animal Pose duplicate/NMS 보완

`bigstock_image_41939056` overlay에서는 실제 개는 한 마리인데 dog pose가 5개 그려졌다. 원인은 AP-10K top-down inference 전 입력 dog/cat bbox가 중복돼 있었기 때문이다. 해당 raw box들에는 IoU 0.982, IoU 1.000 중복과 큰 containment box가 섞여 있었다.

`build_animal_pose_precompute.py`에 dog/cat detection NMS를 추가했다. 기본값은 `--det_nms_iou 0.65`, `--det_nms_containment 0.85`이며, 같은 샘플에서 raw 5개가 keep 1개/suppress 4개로 줄어드는 것을 단위 테스트와 diagnostic image로 확인했다.

![](../data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529/artifacts/precompute/animal_pose_260608_routing_v2_simple_gpu_full_ap10k/nms_diagnostics/bigstock_image_41939056_raw5_to_nms1.jpg)

기존 `animal_pose_260608_routing_v2_simple_gpu_full_ap10k` full artifact는 NMS 적용 전 결과다. 다음 label build 전에는 동일 checkpoint로 full animal pose precompute를 재실행하고, `raw_instances_seen`, `nms_suppressed_instances`, pose/head image ids, positive pet annotation gate를 다시 검증해야 한다.

### 6.2 Person+Pet co-primary 처리

`sstk_image_80436817`은 사람과 pet이 모두 주피사체로 보이는 boundary다. 현재 image route는 `pet_dogcat`이지만 positive crop/query 집계는 `person_single=14`, `pet_dogcat=10`, `scene=2`다. full 10K 기준으로 `person_single` positive와 `pet_dogcat` positive가 각각 2개 이상인 후보는 23장이다.

현재는 `person_pet_joint`를 즉시 9번째 image route class로 추가하지 않는다. 샘플 수가 작고, 사람+pet 동시 등장과 공동 주피사체를 teacher만으로 분리하기 어렵기 때문이다. 대신 unified crop review selector에 `co_primary:person_single+pet_dogcat` bucket을 추가해 reviewer audit seed로 고정했다. 후속 reviewer 결과에서 precision과 downstream crop gain이 확인되면 multi-label route head 또는 `person_pet_joint` class를 별도 ablation으로 연다.

## 7. 남은 의사결정 항목

| 항목 | 권장 결정 |
|---|---|
| image-level placement target | 제외 완료. primary route head target은 `image_route_name_no_placement` 7개 class만 사용한다. |
| food route scope | food/tableware를 v16에 열되, party/chef/person-with-food 경계는 reviewer audit 대상에 넣는다. |
| pet pose gate strictness | AP-10K head visible이 없으면 pet positive 금지. 단 inference에서는 pet detector fallback을 둘 수 있다. |
| animal pose NMS full regeneration | NMS 적용 전 full artifact를 그대로 쓰지 말고, 다음 pet teacher 재생성 전 full AP-10K precompute를 다시 실행한다. |
| person+pet co-primary | 즉시 route class 추가는 보류. 23장 co-primary 후보를 reviewer audit 후 multi-label head 또는 `person_pet_joint` ablation 여부 결정. |
| context_intent mode 승격 | v16 mode name에는 넣지 않는다. context는 auxiliary head 또는 score policy로 유지한다. |
| mode_feasible mode 승격 | v16 mode name에는 넣지 않는다. no-positive/feasible은 별도 feasibility head로 유지한다. |
| flat class head | observed 17 class는 placement/context/feasible까지 포함하므로 primary image route가 아니라 auxiliary/ablation으로 유지한다. |
| food vs scene ambiguity | food route가 열린 만큼 scene 감소가 발생하므로 food precision review 없이는 downstream metric을 단정하지 않는다. |
| 다음 object 확장 | product/vehicle/general object는 food/pet 성능이 확인된 뒤 별도 route로 연다. |

정리하면 남은 의사결정의 초점은 더 이상 image-level placement 사용 여부가 아니다. 이 항목은 제외로 결정됐다. 후속 의사결정은 `food` precision audit, `pet_dogcat` pose/head gate의 inference fallback, `context_intent=environmental` 흡수 검증, flat auxiliary의 실제 효용, 그리고 route-conditioned crop scorer 통합 순서에 집중한다.
# 문서 상태 공지

이 문서는 v16 mode catalog와 visual audit 산출물의 중간 이력으로 보존한다. 2026-06-09 이후 최신 canonical 운영 기준과 v17 precompute-native 산출물은 `Implement_Docs/SSTK_MultiMode_TrainingLabels_Enhanced_Onboarding_KO_v2_0.md`의 "부록 E. Routing v2 Precompute-Native v17 운영 기준"을 우선한다.

최신 핵심 변경: v16 retrofit 방식은 full_body crop 오염 가능성이 있어 audit-only로 낮췄고, v17은 perception precompute + completed GAIC/CGS/CACNet candidate bank + AP-10K dog/cat pose/head gate를 통해 `training_labels`와 `training_labels_multimode`를 새로 생성했다.
