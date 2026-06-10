# SSTK Routing Mode Extension: Person Shot-Type and Pet Mode Research/Implementation Plan

- 빠른 의사결정/실행 기준은 통합본 `Implement_Docs/SSTK_Routing_v2_Unified_Decision_and_Implementation_Brief_KO_2026-06-09.md`를 우선 참조한다. 최신 정책은 `v8_nofood_petstrict_shotstrict_nms_v2`이며, food는 active route class에서 제외됐다.
- 작성일: 2026-06-04
- 목적: 입력 이미지의 사진 타입, 즉 사용자의 촬영 의도에 가까운 routing 모드를 확장하여 향후 `training_labels` 및 `training_labels_multimode` 생성과 MobileCropNet 학습에 반영하기 위한 조사, 정책, 구현 계획을 정리한다.
- 구현/검증 업데이트: `Implement_Docs/SSTK_Routing_v2_Simple_Implementation_and_Validation_Report_KO_2026-06-08.md`에 simple taxonomy 라벨 생성, AP-10K animal pose checkpoint 세팅, flat-vs-hier 비교, route expert smoke, 정성 시각화 결과를 별도로 정리했다.
- 2026-06-08 최종 simple v3 결정: 원안은 `upper_body`와 `half_body`를 별도 shot type으로 검토했지만, 실제 `routing_v2_simple_v3` 구현/라벨/검증에서는 두 class를 `upper_half_body`로 통합했다. 이 문서의 세분화안은 future ontology 후보로만 남기며, 현재 학습 라벨의 유효 taxonomy는 `face_headshot`, `upper_half_body`, `full_body`, `environmental_portrait`이다.
- 2026-06-09 v6 최종 결정: image-level route는 positive crop vote로 override하지 않는다. `person_single_environmental_portrait`는 image route와 v16 mode에서 제거하고, 환경성은 `context_intent=environmental` auxiliary로만 둔다. 최신 유효 `person_shot_type`은 `face_headshot`, `upper_half_body`, `full_body` 3개다.
- 검토 대상 문서:
  - `Implement_Docs/SSTK_MultiMode_TrainingLabels_Enhanced_Onboarding_KO_v1_0.md`
  - `Implement_Docs/MobileCropNet_v4_0_Implementation_Master_Report_KO_2026-04-27.md`
  - `Implement_Docs/MobileCropNet_Training_Data_Generation_Master_Report_KO_2026-04-24.md`
  - `data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529/artifacts/training_labels_multimode/260601_v14_photo_primary_fullsstk_phaseA_scene_subjectsafe_residual_patch_split9010/crop_review_v14_photo_primary/review_jy_260604_followup/review_jy_260604_followup_report_KO.md`의 `10. 향후 연구/개발`

## 1. 핵심 결론

현재 v14 multimode 라벨은 `image_task`, `mode_id`, `entity_id`, `target_ar`의 query 단위로 후보 crop을 평가한다. 이 구조는 라벨 생성과 학습 모두에 유리하지만, `single_person_*`와 `object_*`가 너무 넓은 의미를 갖기 때문에 실제 사진 의도별 score 정책이 충돌한다. 특히 단일 인물 사진은 headshot, upper-body, half-body, full-body, environmental portrait가 모두 다른 crop 규칙을 갖고, 반려동물 사진은 일반 object보다 사람 portrait에 가까운 눈, 얼굴, 몸, 시선, 움직임 여백 정책이 필요하다.

권장 방향은 v14의 10개 기본 mode를 즉시 폐기하지 않고, `routing_v2` 속성 레이어를 추가해 계층형 intent를 먼저 학습/검증하는 것이다. `mode_id`는 기존 호환성을 유지하되, 라벨 row와 query metadata에 `family`, `person_shot_type`, `pet_species`, `pet_shot_type`, `placement_intent`, `context_intent`, `feasibility`를 병행 저장한다. 이후 class balance, no-positive rate, review pass rate가 확인되면 v16 수준에서 확장 mode catalog로 승격한다.

1차 확장 범위는 다음처럼 잡는 것이 안전하다.

| 축 | 최소 확장안 | 이유 |
|---|---|---|
| Single person | simple v3: `face_headshot`, `upper_half_body`, `full_body`, `environmental_portrait`; future 후보: `upper_body`/`half_body` 재분리 | support grounding, headroom, body leakage, context preservation 정책이 shot type별로 다르다. 현재는 teacher label 안정성을 위해 upper/half를 통합한다. |
| Placement | `center`, `rot`, `wide_context` | 현재 center/rot의 장점은 유지하되 environmental portrait와 pet action에서 context 여백을 분리한다. |
| Object | `pet_dogcat` 우선 분리 | dog/cat는 AP-10K/MMPose류 pose precompute를 붙일 수 있고, SSTK PhaseA에도 충분한 후보 수가 있다. |
| Pet | `pet_single_center`, `pet_single_rot`, `pet_multi_center`, `pet_multi_rot`부터 시작 | 일반 object single/multi와 호환되며 score policy를 빠르게 분리할 수 있다. |

## 2. 기존 문서/코드에서 확인한 현 상태

### 2.1 v14 multimode의 장점과 병목

`SSTK_MultiMode_TrainingLabels_Enhanced_Onboarding_KO_v1_0.md`와 현 코드 기준 v14 mode catalog는 다음 10개가 기본이다.

| 계열 | mode |
|---|---|
| Scene | `landscape` |
| Single person | `single_person_center`, `single_person_rot` |
| Group | `group_center`, `group_rot` |
| Face | `face` |
| Object single | `object_single_center`, `object_single_rot` |
| Object multi | `object_multi_center`, `object_multi_rot` |

이 체계는 mode-query 기반이라 한 이미지에서 여러 의도를 동시에 라벨링할 수 있다. 또한 `q_subj`, `q_scale`, `q_place`, `q_axis`, `q_ar`, `q_ctx`, `q_scene`, `q_head`, `q_look`, `face_recall`, `joint_cut_score`, `intrusion_ratio`, `face_body_leakage`, `object_recall` 등 설명 가능한 score component를 이미 축적하고 있다.

하지만 `single_person_center/rot`의 의미가 너무 넓어 다음 문제가 생긴다.

- Headshot/upper-body crop은 발, 지면, support line을 보존하지 않아도 좋은 crop일 수 있다.
- Half-body crop은 무릎/발 절단을 허용할 수 있지만, 머리/상체/골반 안정성이 중요하다.
- Full-body crop은 발/지면/support grounding이 중요하고, 이 정책은 headshot과 정면으로 충돌한다.
- Environmental portrait는 사람을 크게 채우는 것보다 장면 맥락을 살리는 것이 목적이므로 scale/placement의 최적점이 다르다.

`review_jy_260604_followup_report_KO.md`의 10장도 같은 결론을 제시한다. 특히 `sstk_image_1197510367`류 사례에서 좋은 upper/half-body crop이 `support_grounding_miss`, `center_support_grounding_miss`, `rot_support_grounding_miss`로 탈락할 수 있으며, 이는 single-person shot type 분리가 필요하다는 직접 증거다.

### 2.2 PhaseA v14 통계가 시사하는 점

PhaseA PhotoPrimary 10K v14 산출물 기준 전체 규모는 이미지 10,000장, query 217,890개, positive query 86,554개, annotation 167,387개다. mode별 positive query rate는 person과 object 사이에 큰 차이가 있다.

| mode | query | positive query | positive query rate |
|---|---:|---:|---:|
| `single_person_center` | 26,196 | 2,518 | 0.096 |
| `single_person_rot` | 26,196 | 3,136 | 0.120 |
| `face` | 22,152 | 8,397 | 0.379 |
| `object_*` 계열 | - | - | 대략 0.45-0.57 |

이 수치는 단일 인물 mode가 실제 person crop을 과소수용하고 있거나, 너무 엄격한 공통 gate가 좋은 shot type을 섞어서 탈락시키고 있음을 시사한다. 특히 review report의 face 3:4/9:16 실패 분석에는 `body_leakage_strict`, `head_top_cut`, `face_center_placement_miss_strict` 등이 나타나며, 이는 face/headshot과 upper-body를 한 덩어리로 다루면 label policy가 불안정해진다는 근거다.

### 2.3 코드의 준비 상태

현 코드에는 확장 기반이 일부 존재한다.

- `src/portrait_composition.py`는 `_infer_single_shot_type()`으로 `headshot`, `half`, `three_quarter`, `full`을 추론한다.
- 같은 파일에는 eye target, torso/pelvis anchor, support active policy가 이미 있고, `three_quarter/full`에서 support line을 강하게 보는 구조가 있다.
- `src/multimode/mode_scorer.py`에는 half/full support grounding 완화 helper와 `support_grounding_miss` 계열 reason이 존재한다.
- `src/saliency_semantic.py`에는 `ANIMAL_CLASS_IDS={14,15,16,17,18,19,20,21,22,23,77}`가 있고, `src/multimode/entity_atoms.py`는 animal instance를 `family="animal"`인 object atom으로 만든다.

반면 아직 없는 것은 다음이다.

- `upper_half_body`와 `environmental_portrait`의 명시적 shot type.
- animal/pet 전용 head, eye, nose, body, paw, tail, gaze/lookroom feature.
- dog/cat species subtype을 routing label로 안정적으로 내보내는 경로.
- pet 전용 positive threshold, hard gate, no-positive reason 체계.

### 2.4 SSTK PhaseA animal 후보 규모

PhaseA precompute feature를 간단히 집계하면 animal class instance가 있는 이미지는 1,130장, animal C2 instance는 3,404개였다. dog/cat 또는 pet 태그로 볼 수 있는 후보 이미지는 약 572장 수준으로 확인되었다. route 분포는 `object_single` 662장, `object_multi` 233장, `portrait_single` 88장, `scene_general` 68장, `background_texture_copyspace` 42장, `portrait_group` 37장으로, pet mode를 독립 pilot으로 열기에 충분하다.

단, 현재 animal은 일반 object로만 scoring되므로 pet mode를 열기 전 animal pose/head precompute가 필요하다. dog/cat 이미지에서는 일반 object center crop보다 눈/얼굴 보존, 몸통 및 발/꼬리 절단, 시선/동작 방향의 여백이 crop 품질을 좌우한다.

## 3. 관련 문헌/사이트 조사 요약

### 3.1 일반 image cropping 연구

Adobe Research의 WACV 2022 논문 [Automatic Image Cropping using Visual Composition, Boundary Simplicity and Content Preservation Models](https://research.adobe.com/publication/automatic-image-cropping-using-visual-composition-boundary-simplicity-and-content-preservation-models/)는 crop 품질을 단일 aesthetic score만이 아니라 composition, boundary simplicity, content preservation의 결합 문제로 본다. 이는 v14의 `q_subj`, `q_ctx`, cut penalty, recall 계열 component와 방향이 맞고, routing별로 content preservation 대상이 달라져야 한다는 근거가 된다.

[Aesthetics-Aware Reinforcement Learning for Image Cropping](https://arxiv.org/abs/1709.04595)은 crop이 순차적 decision/action 문제로도 모델링될 수 있음을 보인다. MobileCropNet v4.0 문서의 route, subject box, action/top-return 문제가 단순 score regression만으로 해결되지 않는다는 결론과 연결된다.

최근 [ProCrop: Learning Aesthetic Image Cropping from Professional Compositions](https://arxiv.org/abs/2505.22490)와 [Learning Subject-Aware Cropping by Outpainting Professional Photos](https://arxiv.org/abs/2312.12080)는 전문 사진의 composition과 subject-aware crop을 더 직접적으로 학습하려는 방향이다. 두 흐름은 한 이미지에 단 하나의 crop 정답만 있다고 보기 어렵고, subject/intent 조건에 따라 복수의 합리적 crop이 존재한다는 multimode 설계를 지지한다.

### 3.2 사람 중심 crop과 portrait shot type

[Human-Centric Image Cropping](https://github.com/CodeMonsterPHD/Human-Centric-Image-Cropping) 계열 연구는 사람 crop에서 pose, body preservation, aesthetic prior를 일반 object와 다르게 다뤄야 함을 보여준다. 현 v14의 single-person positive rate가 낮고 support gate 충돌이 발생하는 것도 같은 맥락이다.

사진 실무 자료에서도 portrait는 headshot, bust/upper-body, half-body, three-quarter/full-body, environmental portrait로 구분된다. 예를 들어 [NFI의 portrait shot type 정리](https://www.nfi.edu/portrait-shots/)는 프레이밍 범위 자체가 촬영 의도를 구성한다는 점을 설명한다. headroom, lookroom/leadroom은 [Adobe의 photography guide](https://www.adobe.com/creativecloud/photography/discover.html)류 실무 자료에서 반복적으로 언급되는 composition 원칙이며, crop 라벨에서는 단순 rule-of-thirds보다 shot type별 허용 범위로 구현하는 편이 낫다.

### 3.3 Pet photography와 animal pose

Pet photography 실무 자료는 눈 초점, 낮은 시점, 동물의 움직임 방향, 얼굴/몸 전체의 보존을 강조한다. [Adobe pet photography guide](https://www.adobe.com/creativecloud/photography/discover/pet-photography.html), [B&H beginner's guide to pet photography](https://www.bhphotovideo.com/explora/photography/buying-guide/a-beginners-guide-to-pet-photography), [Nikon pet photography tips](https://www.nikonusa.com/learn-and-explore/c/tips-and-techniques/pet-photography-tips)는 모두 동물 사진이 일반 물체 사진과 다르게 eye/face, pose, motion/context를 고려해야 함을 보여준다.

기술적으로는 [AP-10K](https://arxiv.org/abs/2108.12617)가 다양한 사족동물 keypoint dataset을 제공하고, [AP-10K GitHub](https://github.com/AlexTheBad/AP-10K)와 [MMPose](https://mmpose.readthedocs.io/) model zoo를 통해 animal pose estimation을 pipeline에 붙일 수 있다. [Animal Kingdom](https://sutdcv.github.io/Animal-Kingdom/)은 더 넓은 동물 행동/pose 연구 축이고, dog-specific 보조 자료로는 [Stanford Dogs/StanfordExtra 계열](http://vision.stanford.edu/aditya86/ImageNetDogs/)이 있다. 따라서 dog/cat pet mode는 일반 object saliency만으로 처리할 것이 아니라 animal keypoint precompute를 붙이는 것이 합리적이다.

## 4. 제안하는 routing_v2 ontology

### 4.1 설계 원칙

`routing_v2`는 기존 `mode_id`를 대체하는 단일 flat label이 아니라, 다음 축을 가진 계층형 intent label로 둔다.

| 필드 | 값 예시 | 설명 |
|---|---|---|
| `route_family_v2` | `scene`, `person_single`, `person_group`, `pet_dogcat`, `object_single`, `object_multi`, `text_document`, `copyspace_background` | 입력 이미지의 주 피사체/의도 계열 |
| `person_shot_type` | simple v3: `face_headshot`, `upper_half_body`, `full_body`, `environmental_portrait`; future 후보: `upper_body`, `half_body`, `unknown` | single person일 때 프레이밍 의도 |
| `pet_species` | `dog`, `cat`, `dogcat_mixed`, `unknown_pet` | pet mode일 때 species subtype |
| `pet_shot_type` | `pet_head`, `pet_upper_body`, `pet_full_body`, `pet_environmental_action`, `unknown` | 초기에는 optional, pose 품질이 확인된 후 사용 |
| `placement_intent` | `center`, `rot`, `wide_context`, `free` | query/crop 배치 의도. image-level primary target에서는 제외 |
| `context_intent` | `tight_subject`, `balanced`, `environmental`, `copyspace` | scale/context 선호 |
| `mode_feasible` | `true`, `false` | 해당 query가 의미 있는 positive crop을 가질 수 있는지 |
| `routing_confidence` | 0.0-1.0 | rule/precompute 기반 routing 신뢰도 |

이 구조의 장점은 mode catalog explosion을 늦추면서도 학습 라벨에는 촬영 의도 정보를 넣을 수 있다는 점이다. MobileCropNet의 image-level route head는 `family -> shot_type`을 중심으로 학습하고, placement는 query/crop mode와 route-conditioned scorer에만 전달한다. 원본 이미지의 placement는 positive crop vote 요약값일 수 있어 primary route classification target에서 제외한다.

### 4.1.1 이번 구현에 적용한 simple v3 taxonomy

아래 taxonomy가 2026-06-08 산출물 `260608_routing_v2_simple_v3_env_upperhalf_from_v14_reviewfix`의 실제 학습 라벨 기준이다. 이 표가 `training_labels`와 `training_labels_multimode` 생성 시 우선한다.

| 필드 | 값 | 비고 |
|---|---|---|
| `route_family_v2` | `scene`, `person_single`, `person_group`, `pet_dogcat` | object 일반 route는 simple 단계에서 `scene`으로 접고 dog/cat만 pet으로 승격 |
| `person_shot_type` | `face_headshot`, `upper_half_body`, `full_body`, `environmental_portrait` | `upper_body`와 `half_body`는 통합 |
| `placement_intent` | `center`, `rot` | v14 center/rot query/crop 호환. image-level primary target 제외 |
| `context_intent` | `tight_subject`, `balanced`, `environmental` | environmental portrait 전용 context 추가 |
| `mode_feasible` | `true`, `false` | no-positive supervision 유지 |
| `routing_confidence` | 0.0-1.0 | primary supervision은 0.65 이상 권장 |

### 4.2 v14 호환 확장 mode catalog

Pilot 단계에서는 기존 10개 mode를 유지하고 `routing_v2` 속성을 붙인다. 이후 v16 mode catalog로 승격할 때는 다음 후보를 권장한다.

| 계열 | 확장 mode 후보 |
|---|---|
| Person headshot | `single_person_headshot_center`, `single_person_headshot_rot` |
| Person upper body | `single_person_upper_body_center`, `single_person_upper_body_rot` |
| Person half body | `single_person_half_body_center`, `single_person_half_body_rot` |
| Person full body | `single_person_full_body_center`, `single_person_full_body_rot` |
| Environmental portrait | `single_person_environmental_center`, `single_person_environmental_rot`, 필요 시 `single_person_environmental_wide` |
| Pet single | `pet_dogcat_single_center`, `pet_dogcat_single_rot` |
| Pet multi | `pet_dogcat_multi_center`, `pet_dogcat_multi_rot` |

단, 모든 확장 mode를 한 번에 category id로 만들면 route class imbalance와 no-positive query가 급증할 수 있다. 따라서 권장 순서는 다음과 같다.

1. v15 pilot: `mode_name`은 v14 유지, `routing_v2` metadata와 shot-specific score/reason만 추가한다.
2. v15.1: person shot type별 `positive_threshold`와 hard gate를 분리하되, output category는 기존 single-person/face와 호환되게 둔다.
3. v15.2: pet dog/cat mode를 object에서 별도 query family로 열고, animal pose precompute를 hard dependency로 둔다.
4. v16: class balance와 review 품질이 통과한 shot type만 독립 category/mode로 승격한다.

## 5. Single Person shot-type scoring 정책

### 5.1 공통 utility 형태

Person shot type별 crop score는 같은 component를 공유하되 weight와 hard gate가 달라져야 한다.

$$U_{\text{person},s}(c)=w_{\text{subj},s}q_{\text{subj}}+w_{\text{scale},s}q_{\text{scale}}+w_{\text{place},s}q_{\text{place}}+w_{\text{head},s}q_{\text{head}}+w_{\text{look},s}q_{\text{look}}+w_{\text{ctx},s}q_{\text{ctx}}+w_{\text{ar},s}q_{\text{ar}}-P_{\text{cut},s}-P_{\text{intrusion},s}$$

핵심은 `P_cut`, `q_scale`, `q_ctx`, support grounding을 shot type별로 다르게 적용하는 것이다. 특히 `support_grounding`은 full-body/three-quarter에서는 중요하지만, headshot/upper-body에서는 강한 reject reason이 되면 안 된다.

### 5.2 shot type별 보존 대상과 gate

| Shot type | 보존 대상 | 권장 scale/context | Hard gate | 완화해야 할 항목 |
|---|---|---|---|---|
| `face_headshot` | face, head top, chin, hair/hat margin, eyes | 매우 tight, context 낮음 | face recall, head top margin, eye visibility, severe face cut reject | body leakage는 일정 수준 허용, support grounding 비활성 |
| `upper_half_body` | head, face, neck, shoulders, torso, pelvis/hip line 일부, arms | tight-balanced/balanced | head/face recall, shoulder/torso/pelvis severe cut, headroom/lookroom | knee/feet cut 허용, support grounding 비활성 |
| `full_body` | full person, feet, support/ground contact | balanced-wide | full body recall, foot/ankle cut, support grounding, joint cut | face가 작아도 허용, context 부족 penalty 완화 |
| `environmental_portrait` | person identity plus meaningful scene/context | wide/environmental | subject not tiny, face/head or body anchor 유지, context relevance | strict center/scale 완화, support는 shot evidence에 따라 optional |

### 5.3 shot type 추론 feature

현재 `portrait_composition.py`의 `headshot`, `half`, `three_quarter`, `full` 추론을 다음처럼 확장한다.

| Feature | 계산 방향 |
|---|---|
| `face_to_body_area_ratio` | headshot/upper-body 분리 |
| `visible_keypoint_groups` | head, shoulder, elbow, torso, pelvis, knee, ankle group별 visibility |
| `crop_bottom_body_anchor` | crop bottom이 chest, waist, hip, thigh, foot 중 어디를 지나는지 |
| `support_line_visible` | foot/ground contact 또는 standing support evidence |
| `subject_scale_in_crop` | environmental portrait 분리 |
| `scene_context_score` | 배경/장면 semantic, copy-space, saliency context |
| `lookroom_vector` | face/gaze 방향과 여백 균형 |

simple v3에서는 shoulder와 upper torso가 충분히 보이는 경우, 또는 pelvis/hip 또는 upper thigh까지 보이고 knee/ankle은 없어도 되는 경우를 모두 `upper_half_body`로 둔다. `environmental_portrait`는 사람이 주 피사체지만 crop 내 subject area ratio가 낮고 scene context가 명확한 경우다.

### 5.4 threshold와 reason 정책

초기 threshold는 v14의 single-person 계열 positive threshold 0.62를 기준으로 하되, shot type별로 다음처럼 다르게 둔다.

| Shot type | 초기 positive threshold | 주요 reject reason |
|---|---:|---|
| `face_headshot` | 0.66-0.70 | `face_recall_low`, `head_top_cut`, `eye_anchor_bad`, `chin_cut` |
| `upper_half_body` | 0.60-0.64 | `head_recall_low`, `shoulder_cut`, `torso_anchor_bad`, `torso_pelvis_recall_low`, `waist_cut_bad`, `arms_cut_bad` |
| `full_body` | 0.62-0.66 | `foot_cut`, `support_grounding_miss`, `full_body_recall_low` |
| `environmental_portrait` | 0.58-0.63 | `subject_too_tiny`, `context_irrelevant`, `identity_anchor_low` |

Review에서 이미 관찰된 `support_grounding_miss` 계열 reason은 `full_body` 전용 또는 support-active shot 전용으로 내려야 한다. `face_headshot`, `upper_half_body`에는 `support_grounding_not_applicable` 상태를 명시적으로 기록해 downstream 분석에서 gate 미적용과 실패를 구분한다.

## 6. Pet dog/cat mode scoring 및 precompute 보완

### 6.1 pet mode가 일반 object와 다른 이유

일반 object crop은 object box recall, saliency overlap, center/rot placement, intrusion 제어가 중심이다. 그러나 dog/cat pet 사진은 다음 요소가 더 중요하다.

- 눈/얼굴/코가 crop 안에서 안정적으로 보이는지.
- 몸 전체 사진이면 발, 꼬리, 귀가 부자연스럽게 잘리지 않는지.
- 앉음/서있음/뛰는 동작에서 진행 방향 또는 시선 방향의 lead room이 있는지.
- 사람과 함께 있는 pet이면 companion 관계를 깨지 않는지.
- 여러 마리이면 primary pet와 secondary pet를 구분하고, multi-subject crop으로 처리하는지.

따라서 `pet_dogcat`은 object subtype이 아니라 별도 route family로 보는 편이 낫다. 다만 초기에는 dog/cat에 한정하고, 말/새/야생동물은 `object_animal_general` 또는 기존 object mode로 남긴다.

### 6.2 animal pose precompute 제안

새 precompute stage를 `C3P AnimalPose`로 추가한다. 입력은 C2 semantic/instance stage에서 `family="animal"`이고, species가 dog/cat일 가능성이 높은 bbox다. 출력은 JSONL로 저장한다.

권장 산출 경로 예시는 다음과 같다.

| Artifact | 경로 예시 |
|---|---|
| log | `data/SSTK/.../artifacts/precompute/animal_pose_<run_tag>/run.log` |
| progress | `data/SSTK/.../artifacts/precompute/animal_pose_<run_tag>/status.json` |
| per-instance output | `data/SSTK/.../artifacts/precompute/animal_pose_<run_tag>/animal_pose.jsonl` |
| summary | `data/SSTK/.../artifacts/precompute/animal_pose_<run_tag>/summary.json` |
| QA overlays | `data/SSTK/.../artifacts/precompute/animal_pose_<run_tag>/overlays/` |

Per-instance schema 초안은 다음과 같다.

```json
{
  "image_id": "sstk_image_...",
  "instance_id": "animal_0",
  "species_hint": "dog",
  "species_confidence": 0.82,
  "bbox_xyxy": [100, 80, 520, 640],
  "pose_model": "rtmpose_ap10k",
  "pose_confidence": 0.74,
  "keypoints": {
    "left_eye": [240, 150, 0.91],
    "right_eye": [280, 150, 0.88],
    "nose": [260, 180, 0.93],
    "neck": [260, 240, 0.70],
    "tail_base": [470, 430, 0.55]
  },
  "head_box_xyxy": [210, 110, 315, 220],
  "body_box_xyxy": [150, 180, 520, 620],
  "head_direction": "left",
  "motion_direction": "unknown",
  "visible_groups": {
    "head": 0.91,
    "torso": 0.72,
    "front_paws": 0.48,
    "hind_paws": 0.35,
    "tail": 0.55
  }
}
```

모델 후보는 AP-10K 기반 MMPose/RTMPose checkpoint를 우선 검토한다. `MAX_IMAGES=100-300`의 smoke precompute에서 overlay review를 먼저 수행하고, dog/cat head/eye keypoint 품질이 낮은 경우에는 pet mode를 `object_animal_general` 수준으로 제한한다.

### 6.3 pet crop utility

Pet mode score는 object score를 상속하되 head/eye/pose/lookroom을 추가한다.

$$U_{\text{pet}}(c)=w_o q_{\text{object\_recall}}+w_h q_{\text{animal\_head}}+w_e q_{\text{animal\_eye}}+w_b q_{\text{animal\_body}}+w_l q_{\text{pet\_lookroom}}+w_p q_{\text{placement}}+w_c q_{\text{context}}-P_{\text{paw/tail\_cut}}-P_{\text{intrusion}}$$

초기 hard gate는 다음처럼 둔다.

| Pet subtype | Hard gate | 주요 score component |
|---|---|---|
| `pet_single_center` | primary pet recall, head/face recall, severe head cut reject | object recall, animal head, eye/nose visibility, center placement, mild context |
| `pet_single_rot` | primary pet recall, head/face recall, lookroom direction consistency | animal head, pet lookroom, rot placement, motion/lead room |
| `pet_multi_center` | primary+secondary pet coverage, severe animal cut reject | multi animal recall, group compactness, intrusion |
| `pet_multi_rot` | multi animal coverage plus directional/context room | group recall, context, lookroom/motion room |
| `pet_environmental_action` | optional phase 2 | body/action pose, lead room, scene relevance |

Dog/cat full-body pet crop에서는 paw/tail cut penalty를 적용한다. 반대로 pet head portrait에서는 tail/paw cut을 reject로 쓰지 않는다. 이 점은 person shot type과 동일하게 pet shot type별 gate가 달라져야 한다는 뜻이다.

## 7. training_labels / training_labels_multimode 반영안

### 7.1 `training_labels` 단일 라벨 계열

기존 단일 라벨 row에 다음 필드를 추가한다.

| 필드 | 설명 |
|---|---|
| `route_family_v2` | teacher가 판단한 주 route family |
| `route_family_v2_confidence` | routing 신뢰도 |
| `person_shot_type` | single person이면 shot type |
| `pet_species`, `pet_shot_type` | pet이면 species/shot type |
| `placement_intent` | center/rot/wide/free |
| `route_feasibility` | runtime에서 해당 route로 crop할 가치가 있는지 |
| `routing_v2_source` | rule, precompute, VLM, reviewer override 등 |

단일 라벨에서는 top crop 하나만 남더라도 routing supervision은 별도 head에 쓰인다. 특히 MobileCropNet v4.0 문서가 지적한 route collapse를 막기 위해, `final_score`가 높은 crop만 남기는 것이 아니라 route decision을 명시 supervision으로 분리해야 한다.

### 7.2 `training_labels_multimode` query 계열

Multimode query row에는 다음 정보를 추가한다.

| 필드 | 설명 |
|---|---|
| `routing_v2` | query intent 전체 dict |
| `shot_type_applicable` | 해당 shot type이 이미지 evidence상 가능한지 |
| `support_grounding_policy` | `active`, `weak`, `inactive`, `not_applicable` |
| `animal_pose_available` | pet query에서 pose evidence 사용 여부 |
| `positive_threshold_v2` | shot/pet subtype별 threshold |
| `no_positive_reason_v2` | `shot_type_impossible`, `pose_missing`, `support_only_fullbody`, `pet_head_uncertain` 등 |

No-positive row를 반드시 유지해야 한다. route classifier는 positive crop만으로는 학습되지 않고, 어떤 이미지에서 어떤 mode가 성립하지 않는지도 배워야 한다. 특히 `person_full_body`와 `person_upper_half_body`는 같은 사람 이미지에서 서로 동시에 가능할 수도 있고, 한쪽만 가능할 수도 있다.

### 7.3 query builder 변경

Query opening rule은 다음처럼 바꾼다.

| Evidence | 열 query |
|---|---|
| face/head 크고 body 거의 없음 | `face_headshot` |
| face+shoulder+upper torso | `upper_half_body` |
| face+torso+pelvis/hip | `upper_half_body` |
| ankle/foot/support visible | `full_body` |
| person area 작고 scene context 강함 | `environmental_portrait` |
| dog/cat bbox와 animal pose/head evidence | `pet_dogcat_single_*` 또는 `pet_dogcat_multi_*` |
| animal bbox는 있으나 pose/head 불확실 | 기존 `object_*` 유지, pet query는 low-confidence/no-positive로 기록 |

현재 `object` query builder에 있는 animal companion 예외는 유지하되, dog/cat primary subject가 확실한 이미지는 pet query를 우선 생성한다. 사람과 pet이 같이 있는 사진은 `person_*`와 `pet_*` query를 동시에 열 수 있으며, 최종 route head는 image-level primary intent와 candidate-level mode score를 따로 학습한다.

### 7.4 candidate bank 변경

Shot type별 candidate seed를 분리한다.

| Query | Candidate seed |
|---|---|
| `face_headshot` | face/head box expansion, eye-line y target, chin/headroom margin |
| `upper_half_body` | face+shoulder+upper torso 또는 head+torso+pelvis union, waist/hip cut-safe seed |
| `full_body` | full person box, feet/support-safe seed |
| `environmental_portrait` | person+scene context seed, wider scale, copy-space-aware seed |
| `pet_single_*` | animal head+body union, eye/nose anchor, lookroom seed |
| `pet_multi_*` | multiple animal union, primary/secondary balance seed |

이렇게 해야 scorer에서만 shot type을 분리하는 것이 아니라 proposal recall 자체가 올라간다. MobileCropNet 학습에서도 proposal/action target이 routing과 일관된다.

## 8. MobileCropNet 학습 반영안

### 8.1 Hierarchical route head

Flat 20-30 class route head 하나로 바로 학습하면 class imbalance와 confusion이 커질 가능성이 높다. 대신 다음 head를 분리한다.

| Head | Target |
|---|---|
| `route_family_head` | scene/person_single/person_group/pet_dogcat/object/text/copyspace |
| `person_shot_head` | face_headshot/upper_half_body/full_body/environmental_portrait/none |
| `pet_species_head` | dog/cat/dogcat_mixed/none |
| `placement_head` | center/rot/wide/free |
| `feasibility_head` | mode feasible/no-positive |

Loss는 다음처럼 구성한다.

$$L=L_{\text{crop}}+\lambda_{\text{route}}L_{\text{route}}+\lambda_{\text{shot}}L_{\text{shot}}+\lambda_{\text{feas}}L_{\text{feas}}+\lambda_{\text{subject}}L_{\text{subject}}+\lambda_{\text{rank}}L_{\text{rank}}$$

`route_family_head`는 image-level supervision을, `feasibility_head`는 query-level supervision을 받는다. `person_shot_head`와 `pet_species_head`는 applicable mask가 true인 row에서만 loss를 걸어야 한다.

### 8.2 Route-conditioned proposal/action

MobileCropNet v4.0 문서의 핵심 실패는 no-prior runtime에서 route, subject box, action/top-return이 분리되어 collapse가 생긴다는 점이다. 따라서 routing_v2는 classification head로 끝나면 안 되고, proposal/action branch에 조건으로 들어가야 한다.

권장 구조는 다음이다.

- Shared backbone + lightweight FPN/feature pyramid.
- Route token 또는 route embedding을 proposal/action decoder에 주입.
- Person branch는 face/head/torso/pelvis/support map을 auxiliary target으로 학습.
- Pet branch는 animal bbox/head box/keypoint-valid heatmap을 auxiliary target으로 학습.
- Candidate score head는 mode-conditioned score와 global final score를 모두 예측.
- Top-return policy는 route feasibility와 mode-conditioned score를 함께 사용한다.

Runtime에는 precompute feature를 입력하지 않는다. Precompute는 teacher label과 auxiliary supervision 생성에만 사용한다. 이는 MobileCropNet의 deploy-aligned no-prior 원칙과 맞다.

### 8.3 Sampling과 loss balancing

확장 routing에서 가장 큰 위험은 frequent class로의 route collapse다. 다음 sampling 정책이 필요하다.

| 정책 | 내용 |
|---|---|
| Route-balanced sampler | batch 안에 scene/person/pet/object를 균형 배치 |
| Shot-balanced sampler | person batch에서 headshot/upper_half/full/environmental 비율 제어 |
| Pet oversampling | 초기 dog/cat 후보가 적으므로 pet query와 hard negative를 oversample |
| No-positive sampling | infeasible query를 일정 비율 유지 |
| Hard negative mining | object로 보이지만 pet이 아닌 animal, 사람 일부가 잘린 full-body, close-up object를 별도 hard negative로 유지 |
| Class-balanced/focal loss | rare route의 gradient를 보존 |

### 8.4 평가 지표

Routing 확장은 crop score만으로 평가하면 실패한다. 다음 지표를 별도로 보고해야 한다.

| 평가 | 지표 |
|---|---|
| Route classification | balanced accuracy, macro F1, per-class precision/recall |
| Person shot type | headshot/upper_half/full/environmental confusion matrix |
| Pet mode | dog/cat pet precision/recall, pose-available subset metric |
| Feasibility | feasible/no-positive AUROC, no-positive calibration |
| Proposal | mode별 proposal recall@K, oracle crop score |
| Final crop | Product-AR score, direct policy top-1, public benchmark compatibility |
| Review | shot type별 human review pass rate, reason distribution |

Review artifact는 기존 crop review 형식에 shot type facet을 추가한다. 특히 `support_grounding_miss`가 headshot/upper-body에서 발생하지 않는지, pet mode에서 eye/head cut failure가 줄었는지를 별도 table로 확인해야 한다.

## 9. 구현 단계 제안

### P0. 문서/현황 고정

이 문서의 결론을 기준으로 v14 라벨과 review report를 baseline으로 고정한다. 현 v14 mode stats, animal candidate stats, 대표 실패 샘플을 `routing_v2_baseline_summary.json`으로 남긴다.

### P1. routing_v2 audit builder

새 스크립트 예시: `src/scripts/build_routing_v2_audit.py`

입력:

- PhaseA image manifest
- v14 multimode labels
- C2 saliency/semantic/entity atoms
- C6 face/gaze/pose/portrait composition feature

출력:

- `artifacts/routing_v2_audit/<run_tag>/routing_v2_candidates.jsonl`
- `artifacts/routing_v2_audit/<run_tag>/mode_shot_stats.csv`
- `artifacts/routing_v2_audit/<run_tag>/summary.json`
- representative sample pack

이 단계에서는 category를 바꾸지 않는다. shot type 추론과 pet 후보량, no-positive 예상량만 검증한다.

### P2. AnimalPose smoke precompute

dog/cat 후보 100-300장에 대해 AP-10K/MMPose 기반 animal pose를 실행한다. 결과 overlay를 검토해 다음 기준을 통과해야 한다.

- dog/cat head/eye/nose keypoint가 crop scoring에 쓸 수 있을 정도로 안정적인가.
- bbox가 작은 이미지, occlusion, side-view, fur/low-light에서 실패율이 감당 가능한가.
- pet mode를 열었을 때 object mode보다 review pass rate가 실제로 좋아지는가.

통과하지 못하면 pet mode는 `pet_dogcat_candidate` metadata만 붙이고 score policy 승격은 보류한다.

### P3. v15 pilot label generation

기존 v14 label generator에 다음 patch를 적용한다.

- `routing_v2` metadata 필드 추가.
- person shot type별 threshold/gate 분리.
- support grounding policy를 `active/weak/inactive/not_applicable`로 기록.
- pet query는 animal pose가 있는 subset에서만 positive 가능하게 열고, pose가 없으면 no-positive reason을 명시한다.

Pilot 규모는 PhaseA 10K 전체 또는 animal/person subset 2K로 시작한다. 산출물은 v14와 같은 durable 구조를 따른다.

### P4. MobileCropNet routing-only smoke

Crop regression 전체를 바로 학습하지 않고, 먼저 routing head만 학습한다.

- 입력: deploy-aligned RGB image only.
- target: `route_family_v2`, `person_shot_type`, `pet_species`, `context_intent`, `feasibility`. `placement_intent`는 image-level target에서 제외하고 query/crop ablation으로만 둔다.
- 평가: balanced accuracy, macro F1, confusion matrix.

이 smoke에서 person shot type과 pet dog/cat가 구분되지 않으면 full crop model로 확장해도 route collapse가 반복될 가능성이 높다.

### P5. Full crop model integration

Routing-only smoke가 통과하면 route-conditioned proposal/action head에 반영한다.

- route embedding 추가.
- person/pet auxiliary subject map 추가.
- route-balanced sampler 적용.
- v14 baseline, v15 routing_v2 pilot, pet-enabled variant를 비교한다.

성공 기준은 Product-AR top-1 metric뿐 아니라 shot type별 review pass rate와 no-positive calibration 개선이다.

## 10. 데이터/스키마 변경 초안

### 10.1 query metadata

```json
{
  "mode_name": "single_person_center",
  "routing_v2": {
    "route_family_v2": "person_single",
    "person_shot_type": "upper_half_body",
    "placement_intent": "center",
    "context_intent": "tight_subject",
    "mode_feasible": true,
    "routing_confidence": 0.78
  },
  "support_grounding_policy": "inactive",
  "positive_threshold_v2": 0.62
}
```

### 10.2 pet query metadata

```json
{
  "mode_name": "object_single_center",
  "routing_v2": {
    "route_family_v2": "pet_dogcat",
    "pet_species": "dog",
    "pet_shot_type": "pet_full_body",
    "placement_intent": "center",
    "context_intent": "balanced",
    "mode_feasible": true,
    "routing_confidence": 0.74
  },
  "animal_pose_available": true,
  "animal_pose_instance_id": "animal_0",
  "positive_threshold_v2": 0.60
}
```

### 10.3 score explanation

```json
{
  "score_components_v2": {
    "q_animal_head": 0.92,
    "q_animal_eye": 0.88,
    "q_pet_lookroom": 0.71,
    "q_paw_tail_cut": 0.84,
    "q_object_recall": 0.93
  },
  "reject_reasons_v2": []
}
```

## 11. 위험 요소와 방어책

| 위험 | 방어책 |
|---|---|
| mode category 폭증 | `routing_v2` metadata pilot 후 통과 class만 mode 승격 |
| person shot type ambiguity | hard label만 쓰지 말고 confidence와 applicable mask 저장 |
| support grounding 오적용 | support policy를 shot type별로 명시하고 reason 통계로 검증 |
| pet pose model 실패 | pose/head-available subset에서만 pet positive 허용, 실패 시 no-positive 또는 scene/food/object fallback |
| dog/cat 외 animal 오분류 | 초기 scope를 dog/cat로 제한하고 species confidence threshold 적용 |
| animal pose duplicate | AP-10K top-down 입력 전 dog/cat bbox NMS를 적용하고, raw/kept/suppressed instance 수를 summary에 기록 |
| person+pet 공동 주피사체 | 즉시 class 증설하지 않고 co-primary audit flag/bucket으로 먼저 검수 |
| route collapse | route-balanced sampler, focal/class-balanced loss, no-positive supervision |
| training/inference 불일치 | precompute는 teacher label 생성 전용, runtime 모델 입력은 RGB only 유지 |

## 12. 권장 의사결정

1. v15에서는 `routing_v2`를 반드시 metadata layer로 먼저 추가한다.
2. `single_person`은 simple v3에서는 `face_headshot`, `upper_half_body`, `full_body`, `environmental_portrait`로 분리한다. `upper_body`/`half_body` 재분리는 reviewer seed와 pose evidence가 충분해진 후 future ontology에서만 검토한다.
3. `support_grounding_miss`는 full-body/support-active shot에만 강한 reject로 사용한다.
4. object 확장은 `pet_dogcat`과 `food`를 먼저 연다. food는 COCO food/tableware evidence를 쓰되, party/chef/person-with-food 경계는 reviewer audit 대상으로 둔다. product/vehicle/general object는 food/pet precision이 확인된 뒤 연다.
5. pet mode는 animal pose/head precompute가 붙은 subset에서만 positive를 허용한다. 2026-06-09 v5 builder부터 이 조건은 코드상 강제된다. 단 animal pose precompute는 dog/cat bbox NMS 적용 버전으로 재생성해야 한다.
6. MobileCropNet은 flat route head가 아니라 hierarchical route head와 route-conditioned proposal/action으로 학습한다.
7. 최종 성공 기준은 crop final score가 아니라 route accuracy, shot type confusion, pet review pass rate, no-positive calibration을 함께 포함해야 한다.
8. 원본 image-level `placement_intent`는 강한 사진 타입 target이 아니라 query/crop intent 요약값으로 해석한다. image-level placement loss는 기본 학습에서 제외하고, v16 query/crop mode 또는 명시적 ablation에서만 별도 평가한다.
9. 사람과 pet이 함께 주피사체인 이미지는 당장 `person_pet_joint` class를 추가하지 않고, `co_primary_person_pet` audit bucket으로 먼저 고정한다. reviewer precision과 downstream gain이 확인되면 multi-label route head 또는 joint class ablation으로 확장한다.

## 13. 즉시 실행 가능한 작업 목록

| 우선순위 | 작업 | 산출물 |
|---|---|---|
| P0 | v14 person/pet 실패 샘플 pack 고정 | `routing_v2_baseline_summary.json` |
| P1 | `build_routing_v2_audit.py` 작성 | shot type/pet 후보 통계 |
| P1 | `portrait_composition.py` shot type 확장 | simple v3 기준 `upper_half_body`, `environmental_portrait` |
| P2 | animal pose smoke precompute | `animal_pose.jsonl`, overlay review |
| P2 | pet scoring prototype | pet-specific score components/reasons |
| P3 | v15 pilot label generation | `training_labels_multimode` with `routing_v2` |
| P4 | routing-only MobileCropNet smoke | route/shot/pet metrics |
| P5 | route-conditioned crop model 학습 | v14 vs v15/v16 비교 report |

## 13.1 2026-06-08 구현 반영 및 후속 정리

2026-06-08 기준으로 simple taxonomy는 `routing_v2_simple_v3`로 구현되었고, `training_labels`와 `training_labels_multimode` 양쪽에 동일한 route metadata가 생성되었다. 최신 구현/검증 상세는 `Implement_Docs/SSTK_Routing_v2_Simple_Implementation_and_Validation_Report_KO_2026-06-08.md`를 기준 문서로 둔다.

최신 결정 사항은 다음이다.

| 항목 | 결정 |
|---|---|
| person shot | `face_headshot`, `upper_half_body`, `full_body`, `environmental_portrait` |
| upper/half body | `upper_body`와 `half_body`를 분리하지 않고 `upper_half_body`로 통합 |
| pet scope | object 일반 route는 보류하고 dog/cat evidence가 있는 경우만 `pet_dogcat` |
| context | `tight_subject`, `balanced`, `environmental` |
| head 비교 | hierarchical route head를 primary, flat class head를 auxiliary/ablation으로 유지 |
| animal pose | AP-10K RTMPose-M checkpoint를 로컬/원격에 세팅하고 full precompute 469 images/1,203 rows 생성 |

멀티모드 정성 검증 산출물도 추가되었다.

| 산출물 | 경로/결과 |
|---|---|
| card형 route/animal pose viz | `.../training_labels_multimode/260608_routing_v2_simple_v3_env_upperhalf_from_v14_reviewfix/qualitative_viz_routing_v2_simple_v3/`, routing card 35장, animal pose card 24장 |
| v14 방식 AR/mode crop review | `.../crop_review_routing_v2_simple_v3/stratified_200_best_by_ar_mode/`, 200 panels, 4 contact sheets |
| routing-aware crop review | `.../crop_review_routing_v2_simple_v3/routing_v2_route_stratified/`, 160 panels, 2 contact sheets |

후속 작업은 teacher label 신뢰도 개선을 최우선으로 둔다. `pet_dogcat`은 AP-10K pose만으로 확정하지 말고 C2 dog/cat score, metadata/tag, CLIP/VLM species evidence, pose head/eye visibility를 결합해야 한다. `environmental_portrait`는 full 57장으로 희소하므로 reviewer seed 확장과 oversampling이 필요하다. 본체 학습은 `teacher audit -> confidence/species calibration -> hierarchical full route head -> flat/hier ablation -> route-conditioned crop model -> reviewer feedback loop` 순서로 진행한다.

## 13.2 2026-06-09 v5/v16 반영 및 추가 의사결정

2026-06-09 기준으로 `routing_v2_simple_v5`와 v16 mode catalog가 추가되었다. 상세 문서는 `Implement_Docs/SSTK_Routing_v2_v16_Mode_Catalog_and_Label_Visual_Audit_KO_2026-06-09.md`를 기준으로 둔다.

최신 변경 사항은 다음이다.

| 항목 | 결정/결과 |
|---|---|
| source run | `260609_routing_v2_simple_v5_food_petpose_v16prep` |
| v16 run | `260609_v16_routing_v2_mode_catalog_from_v5_food_petpose` |
| route family | `scene`, `person_single`, `person_group`, `pet_dogcat`, `food` |
| food route | COCO food/tableware evidence 기반으로 open |
| pet positive gate | AP-10K animal pose/head image id subset에서만 positive 허용 |
| v16 mode catalog | `route_family_v2 + person_shot_type + placement_intent`, 16 modes |
| image-level route catalog | `route_family_v2 + person_shot_type`, placement 제외 8 classes |
| visual catalog | split cards를 deprecated하고 unified image route cards 8장으로 통합 |
| unified crop review | AR/mode와 routing-aware 선별을 `crop_review_routing_v2_v16_unified_audit` 220 panels로 통합 |

검증 결과 `review_bigstock_image_286519432`는 `pet_dogcat_center`, `review_bigstock_image_349630243`는 `food_center`로 수정되었다. v5 full split의 image-level route 분포는 `scene=6,188`, `person_single=1,906`, `food=931`, `person_group=551`, `pet_dogcat=424`이다. positive pet annotations 4,259개는 모두 AP-10K pose/head subset에 포함되며, pose gate suppressed positive는 0건이다.

추가로 의사결정해야 할 항목은 다음이다.

| 항목 | 권장 방향 |
|---|---|
| placement supervision | image-level target에서는 제외 완료. query/crop intent와 mode-conditioned scorer에서만 유지한다. |
| food precision | food/tableware route를 열었지만 사람+음식 경계가 많으므로 reviewer seed와 hard negative를 별도 수집한다. |
| food scoring | plate/food recall, utensil/table context, hand/person intrusion tolerance를 food-specific score component로 분리한다. |
| pet strict gate | training positive는 pose/head required를 유지한다. inference fallback은 별도 정책으로 설계한다. |
| animal pose NMS regeneration | `bigstock_image_41939056`에서 raw dog/cat box 5개가 AP-10K pose 5개를 만든 문제가 확인됐다. NMS 적용 full precompute를 재실행하고 pet gate summary를 갱신한다. |
| person+pet co-primary | full 10K에서 person/pet positive가 함께 강한 후보는 23장이다. 즉시 class 추가는 보류하고 reviewer audit 후 multi-label 또는 `person_pet_joint` ablation 여부를 결정한다. |
| v16 mode label | context/feasible은 mode name에서 제외하고 별도 head/loss로 유지한다. |
| flat vs hier | v16 flat observed class는 auxiliary/ablation으로 두고 hierarchical route/mode를 primary로 유지한다. |

2026-06-09 후속 결정으로 image-level route classifier의 primary label은 `image_route_name_no_placement`로 고정한다. `v16_mode_name`은 placement를 포함하므로 image classifier target이 아니라 multimode query/crop catalog target이다. 이에 따라 후속 의사결정은 placement threshold가 아니라 `food` precision, `pet_dogcat` inference fallback, `environmental_portrait` 희소성, flat auxiliary 효용, route-conditioned crop scorer 통합 우선순위로 이동한다.

## 13.3 2026-06-09 unified visual/crop audit 및 AP-10K NMS 반영

사용자 추가 검토에 따라 visual audit 산출물과 animal pose precompute 정책을 다음처럼 보완했다.

| 항목 | 반영 |
|---|---|
| image-level visual | `flat_class_cards`와 `v16_mode_cards`를 통합해 `routing_v2_unified_visual_catalog/unified_image_route_contact_sheet.jpg`를 기준 산출물로 사용 |
| crop review | 기존 `stratified_200_best_by_ar_mode`와 `routing_v2_route_stratified`를 `crop_review_routing_v2_v16_unified_audit`로 통합 |
| crop notation | image target은 `IMAGE ROUTE TARGET(no placement)`, crop/query label은 `crop_mode=` 및 `crop_route=`로 분리 |
| co-primary person+pet | `sstk_image_80436817`은 image route `pet_dogcat`이지만 positive route vote가 `person_single=14`, `pet_dogcat=10`으로 나뉘므로 co-primary audit seed로 강제 포함 |
| AP-10K duplicate | `bigstock_image_41939056` raw dog/cat input 5개 -> NMS keep 1/suppress 4 진단 이미지 생성 |

AP-10K duplicate 진단 산출물:

![](../data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529/artifacts/precompute/animal_pose_260608_routing_v2_simple_gpu_full_ap10k/nms_diagnostics/bigstock_image_41939056_raw5_to_nms1.jpg)

## 14. 참고 자료

### Local project

- `Implement_Docs/SSTK_MultiMode_TrainingLabels_Enhanced_Onboarding_KO_v1_0.md`
- `Implement_Docs/MobileCropNet_v4_0_Implementation_Master_Report_KO_2026-04-27.md`
- `Implement_Docs/MobileCropNet_Training_Data_Generation_Master_Report_KO_2026-04-24.md`
- `data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529/artifacts/training_labels_multimode/260601_v14_photo_primary_fullsstk_phaseA_scene_subjectsafe_residual_patch_split9010/crop_review_v14_photo_primary/review_jy_260604_followup/review_jy_260604_followup_report_KO.md`
- `src/multimode/mode_catalog.py`
- `src/multimode/query_builder.py`
- `src/multimode/mode_scorer.py`
- `src/portrait_composition.py`
- `src/saliency_semantic.py`

### External literature/sites

- Adobe Research, [Automatic Image Cropping using Visual Composition, Boundary Simplicity and Content Preservation Models](https://research.adobe.com/publication/automatic-image-cropping-using-visual-composition-boundary-simplicity-and-content-preservation-models/)
- arXiv, [Aesthetics-Aware Reinforcement Learning for Image Cropping](https://arxiv.org/abs/1709.04595)
- arXiv, [ProCrop: Learning Aesthetic Image Cropping from Professional Compositions](https://arxiv.org/abs/2505.22490)
- arXiv, [Learning Subject-Aware Cropping by Outpainting Professional Photos](https://arxiv.org/abs/2312.12080)
- GitHub, [Human-Centric Image Cropping](https://github.com/CodeMonsterPHD/Human-Centric-Image-Cropping)
- arXiv, [AP-10K: A Benchmark for Animal Pose Estimation in the Wild](https://arxiv.org/abs/2108.12617)
- GitHub, [AP-10K dataset repository](https://github.com/AlexTheBad/AP-10K)
- MMPose Documentation, [MMPose](https://mmpose.readthedocs.io/)
- Animal Kingdom, [Project page](https://sutdcv.github.io/Animal-Kingdom/)
- Stanford Vision Lab, [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/)
- Adobe, [Pet photography guide](https://www.adobe.com/creativecloud/photography/discover/pet-photography.html)
- B&H Photo, [A Beginner's Guide to Pet Photography](https://www.bhphotovideo.com/explora/photography/buying-guide/a-beginners-guide-to-pet-photography)
- Nikon, [Pet Photography Tips](https://www.nikonusa.com/learn-and-explore/c/tips-and-techniques/pet-photography-tips)
- NFI, [Portrait Shots](https://www.nfi.edu/portrait-shots/)
# 문서 상태 공지

이 문서는 routing_v2 person/pet 확장 연구 배경과 v16 이전 의사결정 이력으로 보존한다. 2026-06-09 이후 최신 canonical 운영 기준과 v17 precompute-native 산출물은 `Implement_Docs/SSTK_MultiMode_TrainingLabels_Enhanced_Onboarding_KO_v2_0.md`의 "부록 E. Routing v2 Precompute-Native v17 운영 기준"을 우선한다.

최신 핵심 변경: food route 제외, `person_single_environmental_portrait` class 제거, image-level placement target 제외, candidate bank 누락 시 factory 내부 대체 금지 및 별도 completion artifact 생성, full_body crop strict gate 적용.
