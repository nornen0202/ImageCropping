# SSTK Multi-Mode Full 10K 학습/검증 라벨 생성 결과

작성일: 2026-05-06

## 0. 2026-06-01 v14 residual scene-subject 보완 업데이트

2026-06-01 기준 최신 full 10k multi-mode 학습 라벨 후보는 v14이다.

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260601_v14_multimode_full10k_scene_subjectsafe_residual_patch_split9010
```

v14는 v13의 strict mode-intent gate, landscape subject-safe GAIC-Qteach 정책, `group_all` fallback, low GAIC-Qteach scene/interior no-label 정책을 유지하면서, 큰 `distributed_attention` guidance envelope, low-reliability `multi_subject`, `raw_anchor`-saliency mismatch scene pseudo-subject가 `SUBJ` overlay와 subject-safe gate/score에 남는 문제를 보완했다. annotation JSON은 기존과 동일하게 `label_json/multimode_labels_*.json` 및 `label_json/multimode_labels_target_ar_only_*.json` 체계를 유지하고, train/val split은 9:1이다.

v14는 10,000장 전체 커버리지, 9:1 train/val split, strict validation, quality audit risk 0, strict intent audit suspicious 0, residual subject-safe active 0을 통과했다. 같은 정책을 PhaseA PhotoPrimary 10K에도 적용했다. 상세 내용은 다음 문서를 우선 확인한다.

```text
Implement_Docs/SSTK_MultiMode_Quality_Gated_Label_Audit_KO_2026-05-08.md
data/SSTK/Full_10000/artifacts/training_labels_multimode/260601_v14_multimode_full10k_scene_subjectsafe_residual_patch_split9010/QUALITY_GATED_LABELS_REPORT_KO.md
data/SSTK/Full_10000/artifacts/training_labels_multimode/260601_v14_multimode_full10k_scene_subjectsafe_residual_patch_split9010/compare_v13_v14/COMPARE_V13_V14_REVIEW_GUIDE_KO.md
data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529/artifacts/training_labels_multimode/260601_v14_photo_primary_fullsstk_phaseA_scene_subjectsafe_residual_patch_split9010/QUALITY_GATED_LABELS_REPORT_KO.md
```

아래 v3/v5/v8/v9/v10/v11/v12/v13 설명은 historical baseline 또는 이전 후보 기록이다.

## 0.1. 2026-06-02 explainability checklist/why-tag 라벨 보강

2026-06-02에는 새 v15를 만들지 않고 기존 v14 label JSON을 in-place 보강했다. 대상은 Full v14와 PhaseA v14의 아래 파일이다.

```text
label_json/multimode_labels_full.json
label_json/multimode_labels_train.json
label_json/multimode_labels_val.json
```

추가된 attributes:

- `checklist_scores`: `final_score`와 score component regression target.
- `checklist_labels`: threshold로 파생한 설명 가능 label.
- `checklist_applicable`: mode별 loss mask.
- `why_tags`: 사람이 읽는 crop 선택/거절 이유 tag.
- `teacher_checklist_labels`, `teacher_checklist_scores`, `teacher_why_tags`: MobileCropNet v4 loader 호환 alias.
- `explainability_schema_version`, `explainability_label_policy_version`, `explainability`: schema/policy meta.

중요한 운영 원칙은 regression-first다. 즉 모델은 `checklist_scores`의 연속값을 먼저 예측하고, `headroom_ok`, `lookroom_insufficient`, `center_comp_strong` 같은 label은 예측 score에 고정 threshold를 적용해 후처리로 만든다. `label_json/multimode_labels_target_ar_only_*.json`은 호환성을 위해 계속 `attributes={"target_ar": ...}`만 유지한다.

Full v14 explainability 검증 요약:

| split | image | annotation | schema coverage | target_ar_only 구조 | missing score components |
|---|---:|---:|---:|---|---:|
| full | 10,000 | 150,744 | 1.0 | ok | 0 |
| train | 9,000 | 135,178 | 1.0 | ok | 0 |
| val | 1,000 | 15,566 | 1.0 | ok | 0 |

PhaseA v14 explainability 검증 요약:

| split | image | annotation | schema coverage | target_ar_only 구조 | missing score components |
|---|---:|---:|---:|---|---:|
| full | 10,000 | 167,387 | 1.0 | ok | 0 |
| train | 9,000 | 151,322 | 1.0 | ok | 0 |
| val | 1,000 | 16,065 | 1.0 | ok | 0 |

관련 파일:

```text
src/multimode/explainability.py
src/scripts/enrich_multimode_labels_with_explainability.py
src/scripts/validate_multimode_explainability_labels.py
data/SSTK/Full_10000/artifacts/training_labels_multimode/260601_v14_multimode_full10k_scene_subjectsafe_residual_patch_split9010/explainability_enrichment_summary.json
data/SSTK/Full_10000/artifacts/training_labels_multimode/260601_v14_multimode_full10k_scene_subjectsafe_residual_patch_split9010/explainability_validation_summary.json
data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529/artifacts/training_labels_multimode/260601_v14_photo_primary_fullsstk_phaseA_scene_subjectsafe_residual_patch_split9010/explainability_enrichment_summary.json
data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529/artifacts/training_labels_multimode/260601_v14_photo_primary_fullsstk_phaseA_scene_subjectsafe_residual_patch_split9010/explainability_validation_summary.json
```

v14 Full_10000 생성/검증 요약:

| 항목 | 값 |
|---|---:|
| source image | 10,000 |
| train/val image | 9,000 / 1,000 |
| mode query | 211,057 |
| annotation | 150,744 |
| positive annotation | 82,437 |
| negative annotation | 68,307 |
| missing source/feature/candidate | 0 |
| quality audit risk | 0 |
| strict intent audit suspicious | 0 |
| residual subject-safe active | 0 |

v13/v14 비교는 v3 시각화 200장 전체에 대해 생성했고, crop thumbnail과 subject/support overlay를 포함한 pair 이미지로 저장했다.

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260601_v14_multimode_full10k_scene_subjectsafe_residual_patch_split9010/compare_v13_v14/expanded_all_v3_visualize_with_subject_support_crops/
```

## 0.2. 2026-05-29 v11 strict-intent 업데이트

v11은 v12 직전 full 10k multi-mode 학습 라벨 후보다.

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260529_v11_multimode_full10k_strict_intent_landscape_subjectsafe_split9010
```

v11은 v10의 support grounding 완화, `group_all` fallback atom, low GAIC-Qteach scene/interior no-label 정책을 유지하고, 이후 정성 검수에서 확인된 center/ROT 의도 불일치와 subject-cut landscape 문제를 보완한 산출물이다. annotation JSON은 `label_json/multimode_labels_*.json` 및 `label_json/multimode_labels_target_ar_only_*.json` 체계를 유지하고, train/val split은 9:1이다.

v11은 10,000장 전체 커버리지, 9:1 train/val split, strict validation, quality audit risk 0, strict intent audit suspicious 0을 통과했다. 단, 이후 landscape/scene 이미지의 low-reliability pseudo-subject 문제가 확인되어 v13/v14로 후속 보완했다.

```text
Implement_Docs/SSTK_MultiMode_Quality_Gated_Label_Audit_KO_2026-05-08.md
data/SSTK/Full_10000/artifacts/training_labels_multimode/260529_v11_multimode_full10k_strict_intent_landscape_subjectsafe_split9010/QUALITY_GATED_LABELS_REPORT_KO.md
data/SSTK/Full_10000/artifacts/training_labels_multimode/260529_v11_multimode_full10k_strict_intent_landscape_subjectsafe_split9010/compare_v3_v11/COMPARE_V3_V11_REVIEW_GUIDE_KO.md
```

v11 생성/검증 요약:

| 항목 | 값 |
|---|---:|
| source image | 10,000 |
| train/val image | 9,000 / 1,000 |
| mode query | 211,057 |
| annotation | 151,915 |
| positive annotation | 80,167 |
| negative annotation | 71,748 |
| missing source/feature/candidate | 0 |

v3/v11 비교는 v3 시각화 200장 전체에 대해 생성했고, notable 72장은 crop thumbnail을 포함한 pair 이미지로 별도 저장했다.

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260529_v11_multimode_full10k_strict_intent_landscape_subjectsafe_split9010/compare_v3_v11/expanded_all_v3_visualize_with_subject_support_crops/
data/SSTK/Full_10000/artifacts/training_labels_multimode/260529_v11_multimode_full10k_strict_intent_landscape_subjectsafe_split9010/compare_v3_v11/notable_with_subject_support_crops/
```

## 0.3. 2026-05-29 v10 정책 패치 업데이트

v10은 v11 직전 후보이며, v9의 landscape `prefer_gaic + teacher_only + gaic scope` 정책을 유지하고, v9 정성 리뷰에서 확인된 support grounding miss, merged-human group 누락, 낮은 GAIC-Qteach scene/interior positive 문제를 보완한 산출물이다.

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260529_v10_multimode_full10k_policy_patch_landscape_gaic_qteach_split9010
```

## 0.4. 2026-05-29 v9 landscape GAIC/Q_teach 업데이트

v9은 v8의 portrait/group/object/face quality gate는 유지하고, landscape mode만 GAIC teacher를 더 직접 따르도록 변경한 산출물이다. 적용 옵션은 `--landscape_candidate_policy prefer_gaic`, `--landscape_score_policy teacher_only`, `--landscape_teacher_score_scope gaic`이다. 즉 FREE-form crop은 직접 `teacher:gaic` 후보를 먼저 보고, 고정 종횡비 crop은 GAIC provenance/lineage가 있는 후보를 우선 사용하며, landscape score는 기존 composition mixture가 아니라 GAIC teacher score를 정규화한 `Q_teach`만 사용한다. 단, 기존 안전 gate와 target AR gate는 그대로 유지된다.

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260529_v9_multimode_full10k_landscape_gaic_qteach_split9010
```

v9은 10,000장 전체 커버리지, 9:1 train/val split, validation, quality audit risk 0을 통과했지만, 이후 정성 리뷰에서 full-body support grounding miss와 merged-human group 누락이 확인되어 v10으로 보완했다.

## 0.5. 2026-05-08 품질 게이트 업데이트

2026-05-06의 v3 산출물은 전체 1만장/9:1 split 라벨 생성 검증을 위한 최초 full run이었다. 이후 시각화 감사에서 face/person/object 오검출성 positive가 확인되어 v4, v5 quality-gated 후보를 추가 생성했다.

당시 품질 게이트 후보:

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260508_v5_multimode_full10k_quality_gated_saliency_split9010
```

v5는 strict validation을 통과했고 object saliency residual과 center/rot hard-gate residual은 0건으로 줄었다. 다만 당시에는 `secondary_person_positive`가 전체 713장, 5,636 annotation 남아 있어 최종 승인본으로 확정하지 않았다. 이 기록은 historical baseline이며, 최신 학습 입력은 v14를 기준으로 한다.

아래 v5 보류 설명은 당시 기록이다. 이후 secondary policy 확정과 v6-v14 보완 run이 진행되어 최신 상태는 v14 기준으로 봐야 한다.

## 1. 목적

이 문서는 SSTK Full_10000 이미지 1만장 전체에 대해 multi-mode 학습 라벨을 COCO-format JSON으로 재생성하고, train/validation split을 9:1 비율로 구성한 결과를 설명한다. 여기서 COCO는 라벨 저장 스키마를 의미하며, 별도의 COCO 데이터셋 이미지를 사용했다는 뜻이 아니다. 기존 산출물 `260413_v1_multimode_multimode_v1`은 생성 옵션이 `max_images=200`으로 제한되어 있어 전체 1만장용 라벨로 사용하기에는 부적절했다.

2026-05-06 최초 full-run 산출물 경로:

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260506_v3_multimode_full10k_split9010
```

이 경로는 v3 baseline이다. 이후 v5-v14 품질 게이트 후보가 순차적으로 생성됐고, 최신 학습 입력은 v14를 기준으로 한다. 중간에 생성되던 8:2 run은 사용자 요청에 따라 중단 처리했으며 최종 산출물로 사용하지 않는다.

## 2. 생성 명령

```bash
/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/python src/scripts/build_multimode_training_labels.py \
  --features_jsonl data/SSTK/Full_10000/artifacts/precompute/feats_c2c3c5_v2_strict_enriched_routed_final.jsonl \
  --candidates_jsonl data/SSTK/Full_10000/artifacts/candidates/candidates_ar_260413_v1_multimode.jsonl \
  --image_root data/SSTK/Full_10000/images \
  --out_dir data/SSTK/Full_10000/artifacts/training_labels_multimode/260506_v3_multimode_full10k_split9010 \
  --target_ars FREE,1:1,9:16,16:9,3:4,4:3 \
  --max_images 0 \
  --include_optional_negatives 1 \
  --write_split_outputs 1 \
  --train_ratio 0.9 \
  --split_seed 20260506 \
  --write_debug_viz 0 \
  --progress 1
```

`--max_images 0`은 입력 전체를 처리한다는 의미다. split은 source image id를 기준으로 SHA1 정렬을 적용해 결정적으로 생성했다.

## 3. 코드 변경 요약

주요 변경 파일:

- `src/scripts/build_multimode_training_labels.py`
- `src/scripts/validate_multimode_training_labels.py`
- `tests/test_multimode_training_label_splits.py`

변경 내용:

- 전체 COCO-format JSON 외에 train/val COCO-format JSON을 함께 저장하도록 추가했다.
- `--train_ratio`, `--split_seed`, `--write_split_outputs` 옵션을 추가했다.
- source image 기준 deterministic split을 적용해 annotation 누수를 방지했다.
- split별 stats JSON/CSV와 split manifest를 저장하도록 추가했다.
- `status.json`을 생성해 실행 상태, 입력, 출력, 완료/실패 정보를 durable artifact로 남기도록 했다.
- 검증 스크립트에 image root, feature, candidate, split manifest coverage 검사를 추가했다.
- split 함수와 통계 수집 로직에 대한 pytest를 추가했다.

## 4. 주요 산출물

```text
label_json/multimode_labels_full.json
label_json/multimode_labels_target_ar_only_full.json
label_json/multimode_labels_train.json
label_json/multimode_labels_target_ar_only_train.json
label_json/multimode_labels_val.json
label_json/multimode_labels_target_ar_only_val.json
summary.json
dataset_stats.json
mode_stats.csv
target_ar_stats.csv
mode_target_ar_stats.csv
mode_query_status.jsonl
status.json
validation_summary_strict.json
splits/split_manifest.json
splits/train_image_ids.txt
splits/val_image_ids.txt
splits/train_stats.json
splits/val_stats.json
splits/train_mode_stats.csv
splits/val_mode_stats.csv
splits/train_target_ar_stats.csv
splits/val_target_ar_stats.csv
splits/train_mode_target_ar_stats.csv
splits/val_mode_target_ar_stats.csv
```

파일 크기:

| 파일 | 크기 |
|---|---:|
| full annotation JSON | 636 MB |
| full target_ar_only annotation JSON | 119 MB |
| train annotation JSON | 570 MB |
| train target_ar_only annotation JSON | 107 MB |
| val annotation JSON | 67 MB |
| val target_ar_only annotation JSON | 13 MB |

## 5. 라벨 데이터 구조

이 산출물의 `label_json/*.json` 파일들은 annotation JSON 포맷을 따른다. 즉 top-level key는 `images`, `annotations`, `categories`이며, 이미지는 SSTK 원본 이미지를 참조하고 crop 라벨은 `annotations[]`에 저장된다.

### 5.1 `multimode_labels_*.json`

원본 multi-mode 라벨 파일이다.

대상 파일:

- `label_json/multimode_labels_full.json`
- `label_json/multimode_labels_train.json`
- `label_json/multimode_labels_val.json`

`images[]`는 source image 단위 엔트리다. 하나의 source image에 여러 target aspect ratio와 mode query가 붙을 수 있으므로, `images[]` 엔트리 수와 query/annotation 수는 서로 다르다. 이번 full 산출물의 `images[]` 엔트리 수는 10,000이며, 이는 SSTK source image 1만장 전체가 annotation JSON 라벨 파일에 참조됐다는 의미다.

`images[]` 주요 필드:

- `id`: annotation JSON 내부 image id
- `source_image_id`: SSTK source image id
- `file_name`: `image_root` 기준 상대 이미지 경로
- `width`, `height`: 원본 이미지 크기

`categories[]`는 10개 mode class다.

| mode id | mode name |
|---:|---|
| 0 | landscape |
| 1 | single_person_center |
| 2 | single_person_rot |
| 3 | group_center |
| 4 | group_rot |
| 5 | face |
| 6 | object_single_center |
| 7 | object_single_rot |
| 8 | object_multi_center |
| 9 | object_multi_rot |

`annotations[]`는 crop 후보 라벨이다. bbox는 COCO 표준의 pixel `xywh` 형식이며, 각 annotation은 특정 source image, target aspect ratio, mode query, entity에 연결된다.

`annotations[]` 주요 필드:

- `id`: annotation JSON 내부 annotation id
- `image_id`: `images[].id` 참조
- `category_id`: `categories[].id` 참조
- `bbox`: pixel 기준 `[x, y, width, height]`
- `area`: bbox 면적
- `iscrowd`: 항상 `0`
- `gt_flag`: `1`이면 positive crop, `0`이면 optional negative crop
- `query_id`: `source_image_id::target_ar::mode_name::entity_id` 형식의 query id
- `entity_id`: person/object/group/scene atom id
- `mode_name`: mode class 이름
- `is_best`: 해당 query에서 선택된 대표 positive crop이면 `1`
- `score_mode`: mode-specific crop score
- `source_route_mode`: routing stage의 subject mode
- `attributes`: target AR, entity/routing/candidate/score component 등 상세 메타데이터

`annotations[].attributes` 핵심 필드:

- `target_ar`: `FREE`, `1:1`, `9:16`, `16:9`, `3:4`, `4:3`
- `entity_type`: `person`, `group`, `object`, `object_multi`, `scene`
- `route_family`, `route_mode`: routing 정보
- `candidate_id`, `candidate_source`: 선택된 crop 후보 출처
- `score_components`: 세부 score component
- `negative_reason`: negative crop이면 negative 사유
- `hard_reject_reasons`: hard reject 사유 목록
- `query_has_positive`: 해당 query가 positive를 가진 경우 `1`
- `atom_family`, `atom_entity_type`: entity atom 정보

### 5.2 `multimode_labels_target_ar_only_*.json`

축약 라벨 파일이다.

대상 파일:

- `label_json/multimode_labels_target_ar_only_full.json`
- `label_json/multimode_labels_target_ar_only_train.json`
- `label_json/multimode_labels_target_ar_only_val.json`

이 파일들도 annotation JSON 구조(`images`, `annotations`, `categories`)는 유지한다. 차이는 `annotations[].attributes`를 `{"target_ar": ...}`만 남기도록 축약했다는 점이다. downstream에서 target aspect ratio만 필요하거나, 상세 score/routing/entity 메타데이터가 불필요한 경우 사용할 수 있다.

### 5.3 `mode_query_status.jsonl`

`mode_query_status.jsonl`은 mode query 단위 상태 기록이다. positive가 없는 query도 이 파일에 기록되므로, 왜 annotation이 없거나 negative만 있는지 추적할 때 사용한다.

## 6. 전체 통계

| 항목 | 값 |
|---|---:|
| source image 수 | 10,000 |
| annotation JSON `images[]` 엔트리 수 | 10,000 |
| image/AR task 수 | 59,998 |
| mode query 수 | 297,991 |
| annotation 수 | 355,679 |
| positive annotation 수 | 135,058 |
| negative annotation 수 | 220,621 |

이론상 10,000 images x 6 target AR = 60,000 image/AR task가 기대되지만 실제 생성 task는 59,998개다. 전체 source image는 모두 포함되었고, 특정 이미지의 `9:16` task 2건만 입력 candidate/query 부재로 생성되지 않았다.

누락된 image/AR task:

| source image id | 누락 target_ar |
|---|---|
| pond5_image_245616065 | 9:16 |
| sstk_image_408545911 | 9:16 |

mode query 수 297,991은 `mode_query_status.jsonl`의 row 수와 동일하다. 계산 단위는 source image 1장이나 target AR 1개가 아니라, `source_image_id + target_ar + mode_name + entity_id` 조합이다.

query 생성 방식:

- 각 source image에 대해 target AR 6종을 순회한다.
- 해당 image/AR 조합에 crop candidate가 있으면 image/AR task가 1개 생성된다.
- 각 image/AR task 안에서 entity atom과 routing 정보를 기반으로 mode query를 만든다.
- scene atom은 `landscape` query를 만든다.
- person atom은 gate 통과 시 `single_person_center`, `single_person_rot` query를 만들고, face box가 있으면 `face` query도 만든다.
- group atom은 gate 통과 시 `group_center`, `group_rot` query를 만든다.
- object atom은 gate 통과 시 `object_single_center`, `object_single_rot` query를 만든다.
- object_multi atom은 gate 통과 시 `object_multi_center`, `object_multi_rot` query를 만든다.

따라서 한 image/AR task가 항상 10개 mode query를 갖는 구조가 아니다. 이미지 안의 entity atom 개수, face 존재 여부, route gate 통과 여부에 따라 query 수가 달라진다.

이번 산출물의 mode query 총합:

```text
59,998 image/AR tasks에서 생성된 mode query row 총합
= mode_query_status.jsonl line 수
= mode_stats.csv의 query_count 합
= 297,991
```

target AR별 query count 합으로도 동일하게 확인된다.

| target_ar | mode query 수 |
|---|---:|
| FREE | 49,666 |
| 1:1 | 49,666 |
| 9:16 | 49,661 |
| 16:9 | 49,666 |
| 3:4 | 49,666 |
| 4:3 | 49,666 |
| 합계 | 297,991 |

각 mode query는 최대 1개의 positive crop을 만들고, positive가 선택된 query에 대해서 optional negative crop을 추가로 저장한다. 그래서 positive annotation 수는 positive query 수와 같고, 전체 annotation 수는 positive annotation과 negative annotation의 합이다.

## 7. Train/Validation Split

split 비율은 사용자 요청에 따라 9:1로 설정했다.

| split | source image 수 | mode query 수 | annotation 수 | positive | negative |
|---|---:|---:|---:|---:|---:|
| train | 9,000 | 266,305 | 318,748 | 121,095 | 197,653 |
| val | 1,000 | 31,686 | 36,931 | 13,963 | 22,968 |

split 기준:

- source image id 기준 분할
- seed: `20260506`
- train ratio: `0.9`
- train/val image id overlap 없음
- train/val union은 source image 10,000장 전체와 일치

## 8. Mode별 통계

| mode | mode query | positive | negative | positive image |
|---|---:|---:|---:|---:|
| landscape | 59,998 | 53,281 | 105,218 | 10,000 |
| face | 54,947 | 17,009 | 26,492 | 2,652 |
| single_person_center | 46,505 | 7,698 | 11,365 | 1,476 |
| single_person_rot | 46,505 | 7,742 | 10,401 | 1,504 |
| object_single_center | 24,546 | 16,522 | 23,827 | 2,768 |
| object_single_rot | 24,546 | 15,457 | 22,027 | 2,641 |
| group_center | 11,610 | 3,903 | 5,670 | 1,157 |
| group_rot | 11,610 | 3,629 | 5,512 | 1,068 |
| object_multi_center | 8,862 | 5,295 | 5,458 | 1,477 |
| object_multi_rot | 8,862 | 4,522 | 4,651 | 1,148 |

## 9. Target AR별 통계

| target_ar | mode query | positive | negative | positive image |
|---|---:|---:|---:|---:|
| FREE | 49,666 | 28,453 | 44,931 | 10,000 |
| 1:1 | 49,666 | 24,431 | 41,532 | 9,892 |
| 9:16 | 49,661 | 14,148 | 23,106 | 6,956 |
| 16:9 | 49,666 | 21,300 | 33,944 | 9,036 |
| 3:4 | 49,666 | 21,396 | 35,234 | 9,726 |
| 4:3 | 49,666 | 25,330 | 41,874 | 9,711 |

## 10. 검증 결과

strict validation 결과는 `validation_summary_strict.json`에 저장했다.

| 검증 항목 | 결과 |
|---|---|
| validation ok | true |
| image root count | 10,000 |
| source images missing from annotation JSON `images[]` | 0 |
| annotation JSON `images[]` entries missing from image root | 0 |
| feature unique images | 10,000 |
| annotation JSON `images[]` entries missing from features | 0 |
| candidate unique images | 10,000 |
| annotation JSON `images[]` entries missing from candidates | 0 |
| train images | 9,000 |
| val images | 1,000 |

테스트:

```bash
/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/python -m pytest tests/test_multimode_coco_writer.py tests/test_multimode_training_label_splits.py -q
```

결과:

```text
7 passed
```

## 11. 기존 260413 산출물에 대한 결론

기존 `260413_v1_multimode_multimode_v1` 산출물에서 확인된 약 7,000개 수치는 이미지 수가 아니라 annotation 수에 가깝다. 해당 산출물은 source image 200장만 처리된 partial run이므로 전체 1만장 학습 라벨로 사용하면 안 된다.

2026-05-06 시점의 전체 1만장/9:1 split baseline은 다음 경로였다.

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260506_v3_multimode_full10k_split9010
```

2026-05-08 시점의 quality-gated 후보는 다음 경로였다. 단, `secondary_person_positive` residual이 남아 있어 당시 최종 승인본으로는 보류했다.

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260508_v5_multimode_full10k_quality_gated_saliency_split9010
```

## 12. 2026-05-29 v9 full 10k 결과

v9은 v8 이후 landscape mode가 GAIC teacher cropper 결과를 더 강하게 따르도록 만든 full 10k 산출물이다. 경로와 파일명에는 `coco`를 쓰지 않지만, `label_json/multimode_labels_*.json`의 bbox는 기존과 동일하게 COCO-style `xywh` schema를 따른다.

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260529_v9_multimode_full10k_landscape_gaic_qteach_split9010
```

생성 옵션 핵심:

```text
--landscape_candidate_policy prefer_gaic
--landscape_score_policy teacher_only
--landscape_teacher_score_scope gaic
--train_ratio 0.9
--split_seed 20260506
```

`prefer_gaic`는 FREE crop에서는 직접 `teacher:gaic` 후보를 우선 배치하고, 고정 종횡비 crop에서는 GAIC provenance/lineage가 있는 teacher 후보를 우선 사용한다. `teacher_only + gaic scope`는 landscape score를 GAIC teacher raw score에서 정규화한 `Q_teach`로만 계산한다는 뜻이다. `candidate_source`가 `teacher:jitter`, `teacher:cgs`, `teacher:cacnet`으로 보이는 경우도 해당 후보의 lineage/provenance에 GAIC가 있으면 GAIC score만 사용해 선택될 수 있다.

v9 생성/검증 요약:

| 항목 | 값 |
|---|---:|
| validation ok | true |
| source image 수 | 10,000 |
| train image 수 | 9,000 |
| val image 수 | 1,000 |
| mode query 수 | 210,481 |
| annotation 수 | 171,625 |
| positive annotation 수 | 104,609 |
| negative annotation 수 | 67,016 |
| image root 누락 | 0 |
| feature/candidate 누락 | 0 |

v9 split:

| split | source image 수 | mode query 수 | annotation 수 | positive | negative |
|---|---:|---:|---:|---:|---:|
| train | 9,000 | 189,151 | 153,967 | 93,941 | 60,026 |
| val | 1,000 | 21,330 | 17,658 | 10,668 | 6,990 |

v9 mode별 positive:

| mode | mode query | positive |
|---|---:|---:|
| landscape | 59,998 | 53,153 |
| face | 23,285 | 9,242 |
| object_single_center | 23,292 | 12,978 |
| object_single_rot | 23,292 | 12,802 |
| object_multi_center | 8,526 | 4,399 |
| object_multi_rot | 8,526 | 3,907 |
| single_person_center | 27,017 | 2,642 |
| single_person_rot | 27,017 | 3,301 |
| group_center | 4,764 | 1,305 |
| group_rot | 4,764 | 880 |

landscape positive 53,153건의 선택 tier는 `gaic_lineage=48,598`, `direct_gaic_free_preferred=4,555`다. target AR별 landscape positive는 `FREE=9,674`, `1:1=9,650`, `9:16=6,230`, `16:9=8,555`, `3:4=9,411`, `4:3=9,633`이다. `Q_teach` 평균은 `0.673351`이고 raw GAIC teacher score 평균은 `2.540106`이다.

v1 대비 비교 시각화:

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260529_v9_multimode_full10k_landscape_gaic_qteach_split9010/compare_v1_v9/expanded_common_with_crops/
```

공통 v1-positive 샘플 160장을 선별했으며, 각 샘플은 좌측 v1, 우측 v9 overlay와 positive crop thumbnail strip을 포함한다. 전체 contact sheet는 JPEG 차원 한계를 피하기 위해 `v1_v9_common_overlay_crop_comparison_contact_sheet_part001.jpg`, `v1_v9_common_overlay_crop_comparison_contact_sheet_part002.jpg`로 분할 저장했다.

v3 대비 전체 200개 subject/support overlay 비교:

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260529_v9_multimode_full10k_landscape_gaic_qteach_split9010/compare_v3_v9/expanded_all_v3_visualize_with_subject_support_crops/
```

이 비교는 v3 시각화 디렉터리에 있던 200개 샘플 전체를 대상으로 한다. v9 우측 overlay에는 mode별 subject 영역이 반투명으로 추가 표시된다. 기존 v9 label JSON에는 query subject bbox가 저장돼 있지 않았기 때문에, 해당 overlay는 `feats_c2c3c5_v2_strict_enriched_routed_final.jsonl`에서 subject를 복원해 생성했다. 이후 새로 생성되는 라벨에는 `subject_debug`가 annotation attributes에 저장되도록 코드가 보완됐다.

정성 리뷰 후속 결과:

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260529_v9_multimode_full10k_landscape_gaic_qteach_split9010/compare_v3_v9/notable_with_crops/review_260529_jy.md
data/SSTK/Full_10000/artifacts/training_labels_multimode/260529_v9_multimode_full10k_landscape_gaic_qteach_split9010/compare_v3_v9/notable_with_crops/review_260529_jy_policy_patch_smoke/
```

`sstk_image_1767719213`의 wide face crop 쏠림 문제는 face center hard gate 추가 후 리뷰 5장 smoke run에서 제거됨을 확인했다. 다만 이 smoke run은 full 10k v9 재생성이 아니라 코드 보완 검증용 부분 실행이다.

## 13. 2026-05-29 v10 full 10k 결과

v10은 v9 정성 리뷰와 `review_260529_jy.md` 후속 계획을 반영한 full 10k 산출물이다. 경로와 파일명에는 `coco`를 쓰지 않지만, `label_json/multimode_labels_*.json`의 bbox는 기존과 동일하게 COCO-style `xywh` annotation schema를 따른다. 현재 최신 후보는 v14이다.

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260529_v10_multimode_full10k_policy_patch_landscape_gaic_qteach_split9010
```

v10에서 추가 반영한 핵심 정책:

- full-body primary 인물의 support point가 keypoint와 bbox bottom 사이에서 어긋나는 경우 robust support point와 `portrait_bbox_bottom` seed를 사용한다.
- strict recall/face/intrusion 조건을 만족하는 full-body primary 인물에 한해 support grounding hard gate를 완화한다.
- `portrait_group` route에서 raw trusted person이 2명 이상이나 dedupe로 group atom이 사라지는 경우 `group_all` fallback atom을 생성한다.
- raw person 1명뿐인 single-person 오분류는 group fallback 대상에서 제외한다.
- scene/interior landscape query에서 GAIC scope `Q_teach`가 positive threshold 미만이면 `landscape_low_gaic_qteach_scene_no_label`로 no-label 처리한다.
- v10 annotation attributes에는 mode별 subject/support 검토용 `subject_debug`가 저장된다.

v10 생성/검증 요약:

| 항목 | 값 |
|---|---:|
| validation ok | true |
| source image 수 | 10,000 |
| train image 수 | 9,000 |
| val image 수 | 1,000 |
| image/AR task 수 | 59,998 |
| mode query 수 | 211,057 |
| annotation 수 | 173,546 |
| positive annotation 수 | 105,627 |
| negative annotation 수 | 67,919 |
| image root/feature/candidate 누락 | 0 |
| quality audit risk | 0 |

v10 split:

| split | source image 수 | mode query 수 | annotation 수 | positive | negative |
|---|---:|---:|---:|---:|---:|
| train | 9,000 | 189,631 | 155,645 | 94,831 | 60,814 |
| val | 1,000 | 21,426 | 17,901 | 10,796 | 7,105 |

v10 mode별 positive:

| mode | mode query | positive |
|---|---:|---:|
| landscape | 59,998 | 53,153 |
| face | 23,285 | 9,194 |
| object_single_center | 23,292 | 12,978 |
| object_single_rot | 23,292 | 12,802 |
| object_multi_center | 8,526 | 4,399 |
| object_multi_rot | 8,526 | 3,907 |
| single_person_center | 27,017 | 3,008 |
| single_person_rot | 27,017 | 3,861 |
| group_center | 5,052 | 1,382 |
| group_rot | 5,052 | 943 |

v9 대비 v10 변화:

| 항목 | v9 | v10 | delta |
|---|---:|---:|---:|
| positive annotation | 104,609 | 105,627 | +1,018 |
| non-landscape positive | 51,456 | 52,474 | +1,018 |
| landscape positive | 53,153 | 53,153 | 0 |

정책 계수:

| 항목 | 값 |
|---|---:|
| group fallback positive annotation | 140 |
| full-body support relaxed positive annotation | 638 |
| low GAIC-Qteach scene/interior no-label query | 690 |
| approved secondary exception annotation/image | 907 / 170 |

v3 대비 전체 200개 subject/support overlay 비교:

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260529_v10_multimode_full10k_policy_patch_landscape_gaic_qteach_split9010/compare_v3_v10/expanded_all_v3_visualize_with_subject_support_crops/
```

이 비교는 v3 시각화 디렉터리에 있던 200개 샘플 전체를 대상으로 한다. v10 우측 overlay에는 mode별 subject/support 영역이 반투명으로 표시되고, 각 비교 이미지는 양쪽 positive crop thumbnail strip을 함께 포함한다. 대표 40개는 다음 경로에 별도 묶음으로 저장했다.

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260529_v10_multimode_full10k_policy_patch_landscape_gaic_qteach_split9010/compare_v3_v10/notable_with_crops/
```

v10의 상세 보고서는 다음 문서를 확인한다.

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260529_v10_multimode_full10k_policy_patch_landscape_gaic_qteach_split9010/QUALITY_GATED_LABELS_REPORT_KO.md
data/SSTK/Full_10000/artifacts/training_labels_multimode/260529_v10_multimode_full10k_policy_patch_landscape_gaic_qteach_split9010/compare_v3_v10/COMPARE_V3_V10_REVIEW_GUIDE_KO.md
```

## 14. 2026-05-29 v11 full 10k strict-intent 결과

v11은 v10 이후 정성 검수에서 확인된 center/ROT 의도 불일치와 subject-cut landscape 문제를 보완했던 v12 직전 full 10k 산출물이다.

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260529_v11_multimode_full10k_strict_intent_landscape_subjectsafe_split9010
```

v11에서 추가 반영한 핵심 정책:

- `--mode_intent_policy strict_v11`로 center/ROT/face/object mode별 hard gate를 강화했다.
- `--landscape_score_policy teacher_subject_safe`로 GAIC scope `Q_teach`를 우선하되, subject core recall/`q_subj` 미달 crop은 no-label 처리한다.
- `--landscape_candidate_policy prefer_gaic_subject_safe`로 GAIC lineage 후보를 우선하면서 subject-safe 검사를 적용한다.
- strict intent audit 스크립트 `src/scripts/audit_multimode_intent_alignment.py`를 추가해 full positive annotation을 전수 검사한다.

v11 생성/검증 요약:

| 항목 | 값 |
|---|---:|
| validation ok | true |
| source image 수 | 10,000 |
| train image 수 | 9,000 |
| val image 수 | 1,000 |
| image/AR task 수 | 59,998 |
| mode query 수 | 211,057 |
| annotation 수 | 151,915 |
| positive annotation 수 | 80,167 |
| negative annotation 수 | 71,748 |
| image root/feature/candidate 누락 | 0 |
| quality audit risk | 0 |
| strict intent audit suspicious | 0 |

v11 split:

| split | source image 수 | mode query 수 | annotation 수 | positive | negative |
|---|---:|---:|---:|---:|---:|
| train | 9,000 | 189,631 | 136,185 | 71,969 | 64,216 |
| val | 1,000 | 21,426 | 15,730 | 8,198 | 7,532 |

v11 mode별 positive:

| mode | mode query | positive |
|---|---:|---:|
| landscape | 59,998 | 34,057 |
| face | 23,285 | 9,060 |
| object_single_center | 23,292 | 11,200 |
| object_single_rot | 23,292 | 10,751 |
| object_multi_center | 8,526 | 3,796 |
| object_multi_rot | 8,526 | 3,192 |
| single_person_center | 27,017 | 2,733 |
| single_person_rot | 27,017 | 3,407 |
| group_center | 5,052 | 1,296 |
| group_rot | 5,052 | 675 |

v10 대비 v11 변화:

| 항목 | v10 | v11 | delta |
|---|---:|---:|---:|
| positive annotation | 105,627 | 80,167 | -25,460 |
| non-landscape positive | 52,474 | 46,110 | -6,364 |
| landscape positive | 53,153 | 34,057 | -19,096 |

v3 대비 전체 200개 subject/support overlay 비교:

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260529_v11_multimode_full10k_strict_intent_landscape_subjectsafe_split9010/compare_v3_v11/expanded_all_v3_visualize_with_subject_support_crops/
```

notable 72개 crop thumbnail 포함 비교:

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260529_v11_multimode_full10k_strict_intent_landscape_subjectsafe_split9010/compare_v3_v11/notable_with_subject_support_crops/
```

v11의 상세 보고서는 다음 문서를 확인한다.

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260529_v11_multimode_full10k_strict_intent_landscape_subjectsafe_split9010/QUALITY_GATED_LABELS_REPORT_KO.md
data/SSTK/Full_10000/artifacts/training_labels_multimode/260529_v11_multimode_full10k_strict_intent_landscape_subjectsafe_split9010/compare_v3_v11/COMPARE_V3_V11_REVIEW_GUIDE_KO.md
```

## 15. 2026-06-01 v12 full 10k scene-subject 보완 결과

v12는 v11 이후 정성 검수에서 확인된 landscape/scene 계열 pseudo-subject 시각화 및 gating 문제를 보완한 최신 full 10k 산출물이다.

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260601_v12_multimode_full10k_strict_intent_landscape_scene_subjectsafe_split9010
```

v12에서 추가 반영한 핵심 정책:

- `effective_subject_region.state == no_dominant_subject`이고 detector/saliency severe disagreement, 낮은 reliability/agreement가 함께 확인되면 scene pseudo-subject를 실제 주피사체로 승격하지 않는다.
- 해당 landscape query에서는 subject-safe core recall/`q_subj` gate와 `salient_subject_loss`를 비활성화한다.
- scene core/support는 full-frame diagnostic context로 격하하고, 원래 guidance/saliency/effective box는 meta에 남긴다.
- subject-safe가 suppression된 landscape query는 시각화에서 `SUBJ` overlay를 그리지 않아 좌상단 pseudo-subject 표시로 인한 오해를 없앤다.
- `src/scripts/audit_landscape_scene_subject_alignment.py`로 low-reliability scene query를 전수 감사한다.

v12 생성/검증 요약:

| 항목 | 값 |
|---|---:|
| validation ok | true |
| source image 수 | 10,000 |
| train image 수 | 9,000 |
| val image 수 | 1,000 |
| image/AR task 수 | 59,998 |
| mode query 수 | 211,057 |
| annotation 수 | 150,027 |
| positive annotation 수 | 79,856 |
| negative annotation 수 | 70,171 |
| image root/feature/candidate 누락 | 0 |
| quality audit risk | 0 |
| strict intent audit suspicious | 0 |

v12 split:

| split | source image 수 | mode query 수 | annotation 수 | positive | negative |
|---|---:|---:|---:|---:|---:|
| train | 9,000 | 189,631 | 134,498 | 71,684 | 62,814 |
| val | 1,000 | 21,426 | 15,529 | 8,172 | 7,357 |

v12 mode별 positive:

| mode | mode query | positive |
|---|---:|---:|
| landscape | 59,998 | 34,740 |
| face | 23,285 | 9,060 |
| object_single_center | 23,292 | 11,200 |
| object_single_rot | 23,292 | 10,751 |
| object_multi_center | 8,526 | 3,796 |
| object_multi_rot | 8,526 | 3,192 |
| single_person_center | 27,017 | 2,296 |
| single_person_rot | 27,017 | 2,850 |
| group_center | 5,052 | 1,296 |
| group_rot | 5,052 | 675 |

v11 대비 low-reliability scene-subject 감사 결과:

| 항목 | v11 | v12 | delta |
|---|---:|---:|---:|
| affected landscape query | 2,364 | 2,364 | 0 |
| positive query | 1,569 | 2,252 | +683 |
| no-positive query | 786 | 90 | -696 |
| `landscape_subject_core_recall_miss` | 728 | 0 | -728 |
| `landscape_subject_quality_miss` | 500 | 0 | -500 |
| `salient_subject_loss` | 191 | 0 | -191 |

v3 대비 전체 변화:

| 항목 | v3 | v12 | delta |
|---|---:|---:|---:|
| positive annotation | 135,058 | 79,856 | -55,202 |
| non-landscape positive | 81,777 | 45,116 | -36,661 |
| landscape positive | 53,281 | 34,740 | -18,541 |

v3 대비 전체 200개 subject/support overlay 비교:

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260601_v12_multimode_full10k_strict_intent_landscape_scene_subjectsafe_split9010/compare_v3_v12/expanded_all_v3_visualize_with_subject_support_crops/
```

notable 80개 crop thumbnail 포함 비교:

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260601_v12_multimode_full10k_strict_intent_landscape_scene_subjectsafe_split9010/compare_v3_v12/notable_with_subject_support_crops/
```

v12의 상세 보고서는 다음 문서를 확인한다.

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260601_v12_multimode_full10k_strict_intent_landscape_scene_subjectsafe_split9010/QUALITY_GATED_LABELS_REPORT_KO.md
data/SSTK/Full_10000/artifacts/training_labels_multimode/260601_v12_multimode_full10k_strict_intent_landscape_scene_subjectsafe_split9010/compare_v3_v12/COMPARE_V3_V12_REVIEW_GUIDE_KO.md
```
