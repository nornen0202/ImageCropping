# SSTK Multi-Mode Quality-Gated Label Audit

작성일: 2026-05-08

최신 업데이트: 2026-06-02. v13 정성 검수에서 남은 큰 `distributed_attention` scene guidance envelope, low-reliability `multi_subject`, `raw_anchor`-saliency mismatch pseudo-subject 문제를 반영해 v14 full 10k 및 PhaseA 10k 산출물을 생성/검증했다. 현재 최신 후보 run은 `260601_v14_multimode_full10k_scene_subjectsafe_residual_patch_split9010`이며, v13의 strict mode-intent / landscape GAIC-Qteach 정책을 유지하면서 의심 scene subject가 `SUBJ` overlay와 subject-safe gate/score에 남지 않도록 보완했다. 추가로 2026-06-02에는 새 version directory를 만들지 않고 기존 v14 label JSON에 explainability checklist/why-tag attributes를 in-place 보강했다.

## 0. 2026-06-02 explainability checklist/why-tag 보강 업데이트

보강 대상:

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260601_v14_multimode_full10k_scene_subjectsafe_residual_patch_split9010/label_json/multimode_labels_{full,train,val}.json
data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529/artifacts/training_labels_multimode/260601_v14_photo_primary_fullsstk_phaseA_scene_subjectsafe_residual_patch_split9010/label_json/multimode_labels_{full,train,val}.json
```

추가 field:

- `attributes.checklist_scores`: `final_score`와 핵심 score component regression target.
- `attributes.checklist_labels`: score threshold로 파생한 설명 가능 label.
- `attributes.checklist_applicable`: mode별 적용 가능 여부와 loss mask.
- `attributes.why_tags`: crop 선택/거절 이유를 나타내는 multi-label tag.
- `attributes.teacher_checklist_*`: 기존 MobileCropNet v4 checklist loader와의 호환 alias.
- `attributes.explainability_schema_version="multimode_explainability_v1"`.
- `attributes.explainability_label_policy_version="thresholds_20260602_v1"`.

`target_ar_only` JSON은 의도적으로 보강하지 않고 `attributes={"target_ar": ...}`만 유지했다. 이는 target AR만 필요한 loader와 기존 실험의 호환성을 유지하기 위한 것이다.

검증 요약:

| run | split | image | annotation | explainability coverage | target_ar_only 구조 | missing score components |
|---|---|---:|---:|---:|---|---:|
| Full v14 | full | 10,000 | 150,744 | 1.0 | ok | 0 |
| Full v14 | train | 9,000 | 135,178 | 1.0 | ok | 0 |
| Full v14 | val | 1,000 | 15,566 | 1.0 | ok | 0 |
| PhaseA v14 | full | 10,000 | 167,387 | 1.0 | ok | 0 |
| PhaseA v14 | train | 9,000 | 151,322 | 1.0 | ok | 0 |
| PhaseA v14 | val | 1,000 | 16,065 | 1.0 | ok | 0 |

관련 artifact:

```text
explainability_enrichment_summary.json
explainability_label_policy.json
explainability_validation_summary.json
EXPLAINABILITY_LABELS_REPORT_KO.md
```

## 1. 2026-06-01 v14 residual scene-subject 보완 업데이트

최신 full 10k multi-mode 학습 라벨 후보:

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260601_v14_multimode_full10k_scene_subjectsafe_residual_patch_split9010
```

추가 PhaseA 10K 산출물:

```text
data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529/artifacts/training_labels_multimode/260601_v14_photo_primary_fullsstk_phaseA_scene_subjectsafe_residual_patch_split9010
```

v14 적용 사항:

- v11의 `--mode_intent_policy strict_v11`, `--landscape_candidate_policy prefer_gaic_subject_safe`, `--landscape_score_policy teacher_subject_safe`, GAIC scope `Q_teach` 정책은 유지한다.
- v13에서 남은 큰 `distributed_attention` guidance envelope를 guidance area 크기와 무관하게 low-reliability scene subject로 격하한다.
- low-reliability `multi_subject + detector/saliency severe disagreement`와 `dominant_subject/raw_anchor`이지만 saliency와 거의 맞지 않는 큰 guidance envelope도 subject-safe 비활성화 대상으로 추가한다.
- 해당 landscape query에서는 subject-safe core recall/`q_subj` gate, subject bonus/penalty, `salient_subject_loss`를 비활성화한다.
- scene core/support box는 full-frame으로 격하하고, 원래 guidance/saliency/effective box와 guidance-saliency IoU는 diagnostic meta로 남긴다.
- 시각화에서는 `landscape_subject_safe_active=1`인 경우에만 `SUBJ` overlay를 그린다.
- 전수 재생성 대신 v14 영향 범위만 subset 재생성 후 v13 full 라벨에 병합했다. Full_10000은 1,999장, PhaseA는 2,429장이 replacement 대상이다.

v14 Full_10000 검증 요약:

| 항목 | 값 |
|---|---:|
| validation ok | true |
| source image | 10,000 |
| train/val image | 9,000 / 1,000 |
| mode query | 211,057 |
| annotation | 150,744 |
| positive annotation | 82,437 |
| negative annotation | 68,307 |
| quality audit risk | 0 |
| strict intent audit suspicious | 0 |
| residual subject-safe active | 0 image / 0 annotation |

v14 PhaseA 검증 요약:

| 항목 | 값 |
|---|---:|
| validation ok | true |
| source image | 10,000 |
| train/val image | 9,000 / 1,000 |
| mode query | 217,890 |
| annotation | 167,387 |
| positive annotation | 86,554 |
| negative annotation | 80,833 |
| quality audit risk | 0 |
| strict intent audit suspicious | 0 |
| residual subject-safe active | 0 image / 0 annotation |

v13/v14 비교 산출물:

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260601_v14_multimode_full10k_scene_subjectsafe_residual_patch_split9010/compare_v13_v14/expanded_all_v3_visualize_with_subject_support_crops/
data/SSTK/Full_10000/artifacts/training_labels_multimode/260601_v14_multimode_full10k_scene_subjectsafe_residual_patch_split9010/compare_v13_v14/v13_v14_full_delta_summary.json
```

v13 대비 positive annotation은 Full_10000 기준 `79,936 -> 82,437`으로 증가했다. 증가는 low-reliability scene pseudo-subject를 subject gate에서 제거하면서 GAIC-Qteach가 충분한 landscape crop이 다시 positive가 된 영향이 대부분이다. 중요한 검증 기준은 의심 scene subject가 `landscape_subject_safe_active=1`로 남아 있는지이며, v14 residual audit 결과 Full_10000/PhaseA 모두 0건이다.

## 2. 작업 범위

`260506_v3_multimode_full10k_split9010` 시각화에서 확인된 오검출성 positive crop을 기준으로 multi-mode 라벨 생성 파이프라인을 재검토했다. 대상 이슈는 다음이다.

- `sstk_image_400750522`: 휴대폰 영역에 face mode box가 생성됨
- `bigstock_image_108250919`: 사람이 든 아령 영역에 face/person mode box가 생성됨
- `pond5_image_84875481`: `single_person_rot` crop에서 사람이 좌측에 과도하게 붙는 구도 의심
- center/rot 구도 hard gate 필요성
- landscape mode 기준 모호성
- object mode를 saliency foreground 중심으로 제한할 필요성

## 3. 수정한 코드

수정 파일:

- `src/portrait_composition.py`
- `src/multimode/entity_atoms.py`
- `src/multimode/query_builder.py`
- `src/multimode/mode_scorer.py`
- `src/scripts/audit_multimode_label_quality.py`
- `src/scripts/audit_multimode_intent_alignment.py`
- `src/scripts/visualize_multimode_labels.py`

핵심 변경:

- person atom에 C2 person segmentation/detection 일치도, face score, pose score, face area ratio를 저장하고 `person_atom_trusted`/`face_query_valid`를 계산하도록 보강했다.
- face query는 `face_query_valid=1`인 trusted person atom에서만 생성되도록 제한했다.
- single person query는 untrusted atom을 차단하고, secondary-like person atom에는 dominance/share/area 조건을 더 강하게 적용했다.
- group atom은 trusted person atom만 기반으로 생성하도록 제한했다.
- center/rot person mode에 `q_place`, `portrait_q_x`, control deviation, side margin, bottom margin, subject outside hard gate를 추가했다.
- landscape mode는 non-scene route의 경우 public teacher 후보(`teacher:*`) 또는 충분한 context가 없으면 reject하도록 보강했다.
- object mode는 saliency foreground가 crop 안에 충분히 포함되지 않는 경우 positive에서 제외하도록 보강했다.
- v5 검증 이후 재현 가능한 품질 감사 스크립트와 시각화 스크립트를 추가했다.

## 4. v5 산출물

v5 생성 경로:

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260508_v5_multimode_full10k_quality_gated_saliency_split9010
```

생성 명령:

```bash
/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/python src/scripts/build_multimode_training_labels.py \
  --features_jsonl data/SSTK/Full_10000/artifacts/precompute/feats_c2c3c5_v2_strict_enriched_routed_final.jsonl \
  --candidates_jsonl data/SSTK/Full_10000/artifacts/candidates/candidates_ar_260413_v1_multimode.jsonl \
  --image_root data/SSTK/Full_10000/images \
  --out_dir data/SSTK/Full_10000/artifacts/training_labels_multimode/260508_v5_multimode_full10k_quality_gated_saliency_split9010 \
  --target_ars FREE,1:1,9:16,16:9,3:4,4:3 \
  --include_optional_negatives 1 \
  --write_split_outputs 1 \
  --train_ratio 0.9 \
  --split_seed 20260506 \
  --progress 1
```

strict validation:

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260508_v5_multimode_full10k_quality_gated_saliency_split9010/validation_summary_strict.json
```

검증 결과:

| 항목 | 값 |
|---|---:|
| validation ok | true |
| source image 수 | 10,000 |
| train image 수 | 9,000 |
| val image 수 | 1,000 |
| mode query 수 | 237,661 |
| annotation 수 | 249,930 |
| positive annotation 수 | 106,778 |
| negative annotation 수 | 143,152 |
| image root 누락 | 0 |
| feature/candidate 누락 | 0 |

## 5. v3/v4/v5 비교

비교 산출물:

```text
artifacts/research_agent_runs/multimode_label_quality_audit_260508/post_v5_quality_gated_saliency_audit/v3_v4_v5_summary_comparison.json
artifacts/research_agent_runs/multimode_label_quality_audit_260508/post_v5_quality_gated_saliency_audit/v3_v4_v5_mode_positive_comparison.csv
```

| run | mode query | annotation | positive | negative |
|---|---:|---:|---:|---:|
| v3 | 297,991 | 355,679 | 135,058 | 220,621 |
| v4 | 237,661 | 254,859 | 108,658 | 146,201 |
| v5 | 237,661 | 249,930 | 106,778 | 143,152 |

v5는 v4 대비 object saliency foreground gate를 추가 적용한 결과다. query 수는 v4와 같고, object mode positive/negative annotation이 추가로 줄었다.

| mode | v3 positive | v4 positive | v5 positive | v5-v4 |
|---|---:|---:|---:|---:|
| face | 17,009 | 11,164 | 11,164 | 0 |
| group_center | 3,903 | 1,173 | 1,173 | 0 |
| group_rot | 3,629 | 1,577 | 1,577 | 0 |
| landscape | 53,281 | 52,761 | 52,761 | 0 |
| object_multi_center | 5,295 | 4,481 | 4,481 | 0 |
| object_multi_rot | 4,522 | 4,388 | 3,983 | -405 |
| object_single_center | 16,522 | 12,838 | 12,838 | 0 |
| object_single_rot | 15,457 | 14,137 | 12,662 | -1,475 |
| single_person_center | 7,698 | 2,883 | 2,883 | 0 |
| single_person_rot | 7,742 | 3,256 | 3,256 | 0 |

## 6. 품질 감사 결과

감사 명령:

```bash
/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/python src/scripts/audit_multimode_label_quality.py \
  --label_json data/SSTK/Full_10000/artifacts/training_labels_multimode/260508_v5_multimode_full10k_quality_gated_saliency_split9010/label_json/multimode_labels_full.json \
  --summary_json data/SSTK/Full_10000/artifacts/training_labels_multimode/260508_v5_multimode_full10k_quality_gated_saliency_split9010/summary.json \
  --out_dir artifacts/research_agent_runs/multimode_label_quality_audit_260508/post_v5_quality_gated_saliency_audit \
  --prefix post_v5 \
  --focus_image_ids artifacts/research_agent_runs/multimode_label_quality_audit_260508/v3_visualization_source_ids.txt
```

감사 산출물:

```text
artifacts/research_agent_runs/multimode_label_quality_audit_260508/post_v5_quality_gated_saliency_audit/post_v5_quality_audit_summary.json
artifacts/research_agent_runs/multimode_label_quality_audit_260508/post_v5_quality_gated_saliency_audit/post_v5_suspicious_positive_annotations.csv
artifacts/research_agent_runs/multimode_label_quality_audit_260508/post_v5_quality_gated_saliency_audit/post_v5_quality_audit_report_ko.md
artifacts/research_agent_runs/multimode_label_quality_audit_260508/post_v5_quality_gated_saliency_audit/post_v5_focus_200_residual_summary.json
```

전체 10,000장 감사 결과:

| risk signal | v4 annotations/images | v5 annotations/images |
|---|---:|---:|
| object_low_saliency_overlap | 2,701 / 696 | 0 / 0 |
| center/rot hard-gate residual | 0 / 0 | 0 / 0 |
| face low score/query invalid | 0 / 0 | 0 / 0 |
| landscape non-scene non-teacher | 0 / 0 | 0 / 0 |
| secondary_person_positive | 5,636 / 713 | 5,636 / 713 |

해석:

- 휴대폰/아령 예시처럼 person/face/object가 비인물 foreground에 붙는 문제는 trusted person/face gate와 object saliency gate로 해소됐다.
- object saliency foreground 미포함 positive는 v5에서 0건으로 줄었다.
- center/rot hard gate 기준으로는 잔여 위반이 잡히지 않았다.
- 그러나 `secondary_person_positive`가 713장, 5,636 annotation 남아 있다. 이는 detection rank 기준 2순위 이후 인물이 single/face positive로 남은 경우다. 실제로는 장면 내 유효한 보조 인물일 수 있지만, “single person은 대표 인물만 허용” 정책이면 아직 미해결이다.

## 7. 200장 시각화 재생성

기존 v3 시각화 폴더에서 200개 source image id를 추출했다.

```text
artifacts/research_agent_runs/multimode_label_quality_audit_260508/v3_visualization_source_ids.txt
```

v5 시각화 출력:

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260508_v5_multimode_full10k_quality_gated_saliency_split9010/visualize_sstk_boxes/visualize
data/SSTK/Full_10000/artifacts/training_labels_multimode/260508_v5_multimode_full10k_quality_gated_saliency_split9010/visualize_sstk_boxes/visualization_summary.json
```

시각화 요약:

| 항목 | 값 |
|---|---:|
| selected image | 200 |
| written image | 200 |
| missing image | 0 |
| top_k per category | 10 |
| focus residual image | 12 |
| focus residual annotation | 114 |

focus 200 residual은 모두 `secondary_person_positive`다. residual source image id는 `post_v5_focus_200_residual_summary.json`에 저장했다.

## 7. 사용자 지적 예시 재확인

`bigstock_image_108250919`:

- v5 positive는 landscape 6개만 남았다.
- face/person/object positive는 생성되지 않았다.

`sstk_image_400750522`:

- v5 positive는 landscape 5개만 남았다.
- 휴대폰 영역 face/person/object positive는 생성되지 않았다.

`pond5_image_84875481`:

- 기존에 문제로 보였던 `single_person_rot` 1:1, 3:4 positive는 v5에서 제거됐다.
- `single_person_rot` 16:9, 4:3 positive는 남아 있으며 자동 gate 기준은 통과한다.
- 자동 gate 통과가 곧 최종 시각 품질 승인이라는 뜻은 아니므로, 이 케이스는 추가 human review 대상이다.

## 8. 현재 결론

v5는 v3 대비 명확히 개선됐고, object saliency 및 비인물 face/person 오검출 예시는 해결됐다. 하지만 `secondary_person_positive`가 전체 713장에 남아 있어 “완벽히 개선”됐다고 판정하지 않았다.

당시 사용자 지시에 따라 v6 생성 또는 추가 재생성은 진행하지 않고 작업을 보류했다. v5는 2026-05-08 기준 quality-gated candidate였으며, 이후 secondary person 정책과 v6-v14 보완 run이 진행되어 최신 후보는 v14이다.

## 9. 후속 작업 계획

1. `secondary_person_positive` 정책 확정

현재 single/face mode가 detection rank 2순위 이후 인물도 일부 허용한다. 다음 중 하나를 선택해야 한다.

- strict primary-only: `person_rank == 0`만 single/face positive 허용
- dominant-secondary-only: rank > 0이어도 face score, area share, crop dominance가 매우 높을 때만 허용
- mode 분리: secondary single을 별도 mode 또는 hard negative mining 대상으로 분리

2. 713장 residual human review

`post_v5_suspicious_positive_annotations.csv`를 기준으로 713장 중 실제 문제와 허용 가능한 보조 인물을 분리한다. 200장 focus set의 12장은 우선 검토 대상이다.

3. `pond5_image_84875481` rot borderline 검토

현재 hard gate는 통과하지만 4:3 rot의 control placement가 시각적으로 타이트하게 느껴질 수 있다. public cropper와 비교해 `rot_control_thirds_deviation`, `portrait_rx_ctrl`, side-margin threshold를 추가 조정할지 결정한다.

## 10. 2026-05-28 v6 정책 반영

사용자 결정에 따라 secondary person 정책을 다음처럼 확정했다.

- 기본값은 primary-only다. `person_rank == 0`인 대표 인물만 single/face mode positive 후보로 허용한다.
- rank > 0 secondary person은 `rank == 1`이고 crop 내 dominance, area share, area ratio, face score가 모두 매우 높은 경우에만 예외적으로 허용한다.
- single person 예외는 dominance >= 0.92, crop 내 area share >= 0.50, 원본 대비 area ratio >= 0.060, face score >= 0.92를 요구한다.
- face 예외는 dominance >= 0.90, crop 내 area share >= 0.45, face score >= 0.94, face area ratio >= 0.020을 요구한다.
- 예외로 통과한 query도 scorer 단계에서 subject ratio, core recall, intrusion ratio, face body leakage를 다시 hard gate로 검사한다.

수정 파일:

```text
src/multimode/query_builder.py
src/multimode/mode_scorer.py
src/scripts/audit_multimode_label_quality.py
```

v6 생성 경로:

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260528_v6_multimode_full10k_primary_default_secondary_exception_split9010
```

생성 명령:

```bash
/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/python src/scripts/build_multimode_training_labels.py \
  --features_jsonl data/SSTK/Full_10000/artifacts/precompute/feats_c2c3c5_v2_strict_enriched_routed_final.jsonl \
  --candidates_jsonl data/SSTK/Full_10000/artifacts/candidates/candidates_ar_260413_v1_multimode.jsonl \
  --image_root data/SSTK/Full_10000/images \
  --out_dir data/SSTK/Full_10000/artifacts/training_labels_multimode/260528_v6_multimode_full10k_primary_default_secondary_exception_split9010 \
  --target_ars FREE,1:1,9:16,16:9,3:4,4:3 \
  --include_optional_negatives 1 \
  --write_split_outputs 1 \
  --train_ratio 0.9 \
  --split_seed 20260506 \
  --progress 1
```

## 11. v6 strict validation 결과

strict validation 산출물:

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260528_v6_multimode_full10k_primary_default_secondary_exception_split9010/validation_summary_strict.json
```

| 항목 | 값 |
|---|---:|
| validation ok | true |
| source image 수 | 10,000 |
| train image 수 | 9,000 |
| val image 수 | 1,000 |
| mode query 수 | 202,633 |
| annotation 수 | 236,768 |
| positive annotation 수 | 101,175 |
| negative annotation 수 | 135,593 |
| image root 누락 | 0 |
| feature/candidate 누락 | 0 |

mode query 수는 생성된 mode query 단위의 총합이다. 즉 각 source image에 대해 target AR 6종과 mode/entity 조합별 query를 만들고, hard gate 이전에 실제 평가 대상으로 남은 query를 mode별로 합산한 값이다. v6에서는 secondary person 기본 차단 때문에 face/single_person query 자체가 줄어 v5의 237,661개에서 202,633개로 감소했다.

v6 mode별 positive:

| mode | query | positive |
|---|---:|---:|
| landscape | 59,998 | 52,761 |
| face | 15,521 | 7,651 |
| object_single_center | 23,130 | 12,838 |
| object_single_rot | 23,130 | 12,662 |
| object_multi_center | 8,736 | 4,481 |
| object_multi_rot | 8,736 | 3,983 |
| single_person_center | 23,285 | 1,862 |
| single_person_rot | 23,285 | 2,187 |
| group_center | 8,406 | 1,173 |
| group_rot | 8,406 | 1,577 |

## 12. v6 품질 감사 결과

감사 산출물:

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260528_v6_multimode_full10k_primary_default_secondary_exception_split9010/quality_audit/v6_quality_audit_summary.json
data/SSTK/Full_10000/artifacts/training_labels_multimode/260528_v6_multimode_full10k_primary_default_secondary_exception_split9010/quality_audit/v6_suspicious_positive_annotations.csv
data/SSTK/Full_10000/artifacts/training_labels_multimode/260528_v6_multimode_full10k_primary_default_secondary_exception_split9010/quality_audit/v6_quality_audit_report_ko.md
```

| 항목 | 값 |
|---|---:|
| risk annotation counts | 0 |
| risk image counts | 0 |
| focus 200 risk annotation counts | 0 |
| focus 200 risk image counts | 0 |
| approved secondary exception annotation | 33 |
| approved secondary exception image | 13 |

v6에서는 v5의 `secondary_person_positive` 5,636 annotation / 713 image가 current audit 기준 residual risk 0으로 줄었다. 단, policy상 허용한 dominant secondary 예외 33 annotation / 13 image는 별도 계수로 남겨 추적한다.

v5/v6 비교 산출물:

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260528_v6_multimode_full10k_primary_default_secondary_exception_split9010/quality_audit/v5_v6_secondary_policy_comparison.json
data/SSTK/Full_10000/artifacts/training_labels_multimode/260528_v6_multimode_full10k_primary_default_secondary_exception_split9010/quality_audit/v5_v6_mode_positive_comparison.csv
```

| 항목 | v5 | v6 | v6-v5 |
|---|---:|---:|---:|
| mode query | 237,661 | 202,633 | -35,028 |
| annotation | 249,930 | 236,768 | -13,162 |
| positive annotation | 106,778 | 101,175 | -5,603 |
| negative annotation | 143,152 | 135,593 | -7,559 |

## 13. v1/v6 시각 비교 산출물

사용자가 복사해 둔 v1 visualization과 v6 focus 200 visualization을 같은 source image id 기준으로 좌우 비교했다.

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260528_v6_multimode_full10k_primary_default_secondary_exception_split9010/compare_v1_v6
data/SSTK/Full_10000/artifacts/training_labels_multimode/260528_v6_multimode_full10k_primary_default_secondary_exception_split9010/compare_v1_v6/curated_clear_improvements
```

대표 contact sheet:

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260528_v6_multimode_full10k_primary_default_secondary_exception_split9010/compare_v1_v6/curated_clear_improvements/curated_v1_v6_clear_improvements_contact_sheet.jpg
```

2026-05-28 추가 확장 비교:

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260528_v6_multimode_full10k_primary_default_secondary_exception_split9010/compare_v1_v6/expanded_with_crops/v1_v6_overlay_crop_comparison_contact_sheet.jpg
data/SSTK/Full_10000/artifacts/training_labels_multimode/260528_v6_multimode_full10k_primary_default_secondary_exception_split9010/compare_v1_v6/expanded_with_crops/pairs_with_crops/
data/SSTK/Full_10000/artifacts/training_labels_multimode/260528_v6_multimode_full10k_primary_default_secondary_exception_split9010/compare_v1_v6/expanded_with_crops/EXPANDED_V1_V6_IMPLEMENTATION_DIFF_REPORT_KO.md
```

확장 비교는 기존 curated 4개에서 28개로 샘플을 늘렸고, overlay 아래에 JSON positive bbox로 원본 이미지를 실제 crop한 thumbnail strip을 함께 붙였다. 선택 28개 샘플 합산 기준 positive total은 v1 791개에서 v6 330개로 줄었고, human/face positive는 440개에서 108개로, object positive는 101개에서 37개로 줄었다. 이 차이는 주로 trusted person/face gate, secondary person primary-only default, center/rot hard gate, object saliency foreground gate, landscape teacher/context gate에서 발생했다.

대표 판단 포인트:

- `bigstock_image_108250919`: v1의 아령/손 주변 face 및 single_person 오탐이 v6에서 제거되고 landscape만 남았다.
- `sstk_image_400750522`: v1의 휴대폰 주변 face-mode 오탐 박스가 v6에서 제거됐다.
- `bigstock_image_184019338`: v1의 옷걸이/상단 영역 face/person 오탐이 v6에서 제거됐다.
- `pond5_image_84875481`: rot 구도 허용성 확인 샘플이다. v6 audit은 통과하지만 최종 구도 정책상 너무 타이트하다고 판단되면 rot threshold를 추가로 좁혀야 한다.

## 14. v6 결론 및 v7 전환 사유

v6는 10,000장 전체 커버리지와 9:1 train/val split을 통과했고, v5의 가장 큰 미해결 항목이던 secondary person residual을 current audit 기준 0으로 줄였다. 그러나 이후 `bigstock_image_106621058`, `bigstock_image_124431041`처럼 원본 자체가 상반신 portrait 또는 group portrait인 케이스에서 face/person/group positive가 과도하게 빠지는 문제가 확인됐다.

사진학 관점에서 이 문제는 full-body margin gate를 모든 portrait에 동일하게 적용한 데서 발생한다. head-and-shoulders, upper-body, three-quarter portrait는 full-body portrait와 다른 유효 구도이며, 원본 source edge가 이미 몸통을 자르는 경우 그 하단 경계만으로 crop 실패라고 판단하면 정상 crop까지 제거한다.

참고 자료:

- ePHOTOzine, [A Practical Photography Guide To Portrait Lengths](https://www.ephotozine.com/article/a-practical-photography-guide-to-portrait-lengths-29169)
- Photofocus, [A guide to cropping portraits](https://photofocus.com/photography/shooting-photography/a-guide-to-cropping-portraits/)
- SLR Lounge, [Portrait Cropping Guide: Bad Portrait Crops & How to Fix Them](https://www.slrlounge.com/portrait-cropping-guide-bad-portrait-crops-how-to-fix-them/)
- Wikipedia, [Headroom (photographic framing)](https://en.wikipedia.org/wiki/Headroom_%28photographic_framing%29)
- Tu et al., [Image Cropping with Composition and Saliency Aware Aesthetic Score Map](https://ojs.aaai.org/index.php/AAAI/article/view/6889), AAAI 2020
- Microsoft Research, [Learning the Change for Automatic Image Cropping](https://www.microsoft.com/en-us/research/publication/learning-the-change-for-automatic-image-cropping/), CVPR 2013

위 자료들을 기준으로 v7에서는 다음 정책을 반영했다.

- portrait source-edge relaxation: crop edge, source edge, subject/core envelope가 함께 맞물린 경우에만 bottom/side margin hard gate를 완화한다.
- group portrait secondary exception: primary-only를 기본값으로 유지하되, `route_mode == "portrait_group"`이고 trusted rank > 0 인물이 crop 내 dominance/face score/area share/area ratio를 높은 수준으로 만족하면 group portrait 한정 예외를 허용한다.
- duplicate group dedupe: 같은 사람의 중복 person/pose atom이 group mode를 만들지 않도록 bbox containment, face IoU/containment, center distance로 group member를 정리한다.

## 15. 2026-05-28 v7 정책 반영

수정 파일:

```text
src/multimode/query_builder.py
src/multimode/entity_atoms.py
src/multimode/mode_scorer.py
src/scripts/audit_multimode_label_quality.py
src/scripts/build_multimode_v1_v6_crop_comparison.py
tests/test_multimode_quality_gate_policy.py
```

v7 생성 경로:

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260528_v7_multimode_full10k_photographic_source_edge_group_secondary_split9010
```

생성 명령:

```bash
/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/python src/scripts/build_multimode_training_labels.py \
  --features_jsonl data/SSTK/Full_10000/artifacts/precompute/feats_c2c3c5_v2_strict_enriched_routed_final.jsonl \
  --candidates_jsonl data/SSTK/Full_10000/artifacts/candidates/candidates_ar_260413_v1_multimode.jsonl \
  --image_root data/SSTK/Full_10000/images \
  --out_dir data/SSTK/Full_10000/artifacts/training_labels_multimode/260528_v7_multimode_full10k_photographic_source_edge_group_secondary_split9010 \
  --target_ars FREE,1:1,9:16,16:9,3:4,4:3 \
  --train_ratio 0.9 \
  --split_seed 20260506 \
  --write_debug_viz 0 \
  --progress 1
```

v7 strict validation:

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260528_v7_multimode_full10k_photographic_source_edge_group_secondary_split9010/validation_summary_strict.json
```

| 항목 | 값 |
|---|---:|
| validation ok | true |
| source image 수 | 10,000 |
| train image 수 | 9,000 |
| val image 수 | 1,000 |
| mode query 수 | 207,169 |
| annotation 수 | 241,193 |
| positive annotation 수 | 103,212 |
| negative annotation 수 | 137,981 |
| image root 누락 | 0 |
| feature/candidate 누락 | 0 |

v7 mode별 positive:

| mode | query | positive |
|---|---:|---:|
| landscape | 59,998 | 52,761 |
| face | 17,909 | 8,194 |
| object_single_center | 23,130 | 12,838 |
| object_single_rot | 23,130 | 12,662 |
| object_multi_center | 8,736 | 4,481 |
| object_multi_rot | 8,736 | 3,983 |
| single_person_center | 27,017 | 2,492 |
| single_person_rot | 27,017 | 3,301 |
| group_center | 5,748 | 1,479 |
| group_rot | 5,748 | 1,021 |

## 16. v7 품질 감사 및 v6/v7 비교

감사 산출물:

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260528_v7_multimode_full10k_photographic_source_edge_group_secondary_split9010/quality_audit/v7_quality_audit_summary.json
data/SSTK/Full_10000/artifacts/training_labels_multimode/260528_v7_multimode_full10k_photographic_source_edge_group_secondary_split9010/quality_audit/v7_suspicious_positive_annotations.csv
data/SSTK/Full_10000/artifacts/training_labels_multimode/260528_v7_multimode_full10k_photographic_source_edge_group_secondary_split9010/quality_audit/v7_quality_audit_report_ko.md
```

| 항목 | 값 |
|---|---:|
| risk annotation counts | 0 |
| risk image counts | 0 |
| approved secondary exception annotation | 813 |
| approved secondary exception image | 155 |

v6/v7 비교 산출물:

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260528_v7_multimode_full10k_photographic_source_edge_group_secondary_split9010/compare_v6_v7/v6_v7_policy_delta_summary.json
data/SSTK/Full_10000/artifacts/training_labels_multimode/260528_v7_multimode_full10k_photographic_source_edge_group_secondary_split9010/compare_v6_v7/v6_v7_mode_delta.csv
data/SSTK/Full_10000/artifacts/training_labels_multimode/260528_v7_multimode_full10k_photographic_source_edge_group_secondary_split9010/compare_v6_v7/v6_v7_selected_image_delta.csv
data/SSTK/Full_10000/artifacts/training_labels_multimode/260528_v7_multimode_full10k_photographic_source_edge_group_secondary_split9010/compare_v6_v7/expanded_with_crops/v6_v7_overlay_crop_comparison_contact_sheet.jpg
data/SSTK/Full_10000/artifacts/training_labels_multimode/260528_v7_multimode_full10k_photographic_source_edge_group_secondary_split9010/compare_v6_v7/expanded_with_crops/pairs_with_crops/
```

| 항목 | v6 | v7 | v7-v6 |
|---|---:|---:|---:|
| mode query | 202,633 | 207,169 | +4,536 |
| annotation | 236,768 | 241,193 | +4,425 |
| positive annotation | 101,175 | 103,212 | +2,037 |
| negative annotation | 135,593 | 137,981 | +2,388 |

mode별 핵심 변화:

- face: positive +543. group portrait secondary face 예외로 정상 보조 얼굴 crop 일부 회복.
- single_person_center: positive +630. 상반신 source-edge portrait crop 일부 회복.
- single_person_rot: positive +1,114. rot에서도 source-edge upper-body 구도 일부 회복.
- group_center: query -2,658, positive +306. duplicate group query는 줄었지만 source-edge group crop은 회복.
- group_rot: query -2,658, positive -556. duplicate 기반 false group rot query/positive 감소.
- landscape/object modes: 변화 없음. v5/v6의 landscape teacher/context gate와 object saliency foreground gate 유지.

사용자 지적 샘플 결과:

- `bigstock_image_106621058`: v6 landscape-only에서 v7 face 4개, group_center 2개, single_person_rot 1개가 회복됐다.
- `bigstock_image_124431041`: v6 face/landscape-only에서 v7 single_person_center 2개, single_person_rot 2개가 회복됐다. duplicate atom 기반 group mode는 생성되지 않는다.
- `bigstock_image_108250919`: v6/v7 모두 landscape 6개만 남아 아령/손 영역 오검출은 재발하지 않았다.
- `sstk_image_400750522`: v6/v7 모두 landscape 5개만 남아 휴대폰 face 오검출은 재발하지 않았다.
- `pond5_image_84875481`: v6/v7 positive count는 동일하다. source-edge relaxation이 이 rot 샘플을 추가로 느슨하게 만들지 않았다.

## 17. v8 후속 보완 결과

v8 산출물:

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260529_v8_multimode_full10k_group_hardgate_animal_facefallback_sourceaxis_landscape_teacher_split9010
```

v8은 v6 수동 리뷰 문서와 v7 시각화 검토에서 남은 후속 이슈를 반영했던 full 10k 라벨이다. 최신 후보는 상단 v14 섹션을 따른다. 구현상 주요 차이는 다음이다.

- `entity_atoms.py`: `portrait_single` route의 group atom 생성에 multi-human hard gate를 추가했다.
- `entity_atoms.py`: high-confidence large face box에 한해 face fallback box를 만들고 `face_query_fallback_used`를 기록한다.
- `query_builder.py`: human route animal companion object는 허용하고, human/text route non-animal object pair는 기본 차단한다.
- `mode_scorer.py`: source bottom edge에 붙은 half/three-quarter portrait의 `single_person_center` axis hard gate를 제한적으로 완화한다.
- `build_multimode_training_labels.py`: landscape query에 teacher 후보 우선 정책을 추가했고 v8은 `prefer_teacher`로 생성했다. v9에서는 이 축이 `prefer_gaic + teacher_only + gaic scope`로 확장됐다.
- `build_multimode_route_label_mismatch_manifest.py`: route와 최종 positive label 간 mismatch를 진단용 JSON/CSV로 저장한다.
- `build_multimode_v1_v6_crop_comparison.py`: crop thumbnail 크기/컬럼 수와 `max_crops <= 0` 전체 표시 옵션을 추가했다.

v8 생성/검증 요약:

| 항목 | 값 |
|---|---:|
| validation ok | true |
| source image 수 | 10,000 |
| train image 수 | 9,000 |
| val image 수 | 1,000 |
| mode query 수 | 210,481 |
| annotation 수 | 186,597 |
| positive annotation 수 | 104,140 |
| negative annotation 수 | 82,457 |
| image root 누락 | 0 |
| feature/candidate 누락 | 0 |

v8 mode별 positive:

| mode | query | positive |
|---|---:|---:|
| landscape | 59,998 | 52,684 |
| face | 23,285 | 9,242 |
| object_single_center | 23,292 | 12,978 |
| object_single_rot | 23,292 | 12,802 |
| object_multi_center | 8,526 | 4,399 |
| object_multi_rot | 8,526 | 3,907 |
| single_person_center | 27,017 | 2,642 |
| single_person_rot | 27,017 | 3,301 |
| group_center | 4,764 | 1,305 |
| group_rot | 4,764 | 880 |

v7/v8 비교 산출물:

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260529_v8_multimode_full10k_group_hardgate_animal_facefallback_sourceaxis_landscape_teacher_split9010/compare_v7_v8/v7_v8_policy_delta_summary.json
data/SSTK/Full_10000/artifacts/training_labels_multimode/260529_v8_multimode_full10k_group_hardgate_animal_facefallback_sourceaxis_landscape_teacher_split9010/compare_v7_v8/v7_v8_mode_delta.csv
data/SSTK/Full_10000/artifacts/training_labels_multimode/260529_v8_multimode_full10k_group_hardgate_animal_facefallback_sourceaxis_landscape_teacher_split9010/compare_v7_v8/v7_v8_selected_image_delta.csv
data/SSTK/Full_10000/artifacts/training_labels_multimode/260529_v8_multimode_full10k_group_hardgate_animal_facefallback_sourceaxis_landscape_teacher_split9010/compare_v7_v8/expanded_with_crops/v7_v8_overlay_crop_comparison_contact_sheet.jpg
data/SSTK/Full_10000/artifacts/training_labels_multimode/260529_v8_multimode_full10k_group_hardgate_animal_facefallback_sourceaxis_landscape_teacher_split9010/compare_v7_v8/expanded_with_crops/pairs_with_crops/
```

| mode | v7 positive | v8 positive | delta | 해석 |
|---|---:|---:|---:|---|
| face | 8,194 | 9,242 | +1,048 | large face fallback으로 누락 얼굴 회복 |
| single_person_center | 2,492 | 2,642 | +150 | source-edge half portrait center 회복 |
| group_center | 1,479 | 1,305 | -174 | false group hard gate 강화 |
| group_rot | 1,021 | 880 | -141 | weak group rot 감소 |
| object_single_center | 12,838 | 12,978 | +140 | animal companion object 회복 |
| object_single_rot | 12,662 | 12,802 | +140 | animal companion object 회복 |
| object_multi_center | 4,481 | 4,399 | -82 | human route non-animal object pair 억제 |
| object_multi_rot | 3,983 | 3,907 | -76 | human route non-animal object pair 억제 |
| landscape | 52,761 | 52,684 | -77 | teacher 후보 우선 정책 |

사용자 지적 샘플 결과:

- `bigstock_image_141285011`: v7의 group false positive 제거.
- `bigstock_image_133089656`: v7의 group false positive 제거.
- `bigstock_image_137810084`: 강아지 animal object_single center/rot 복구.
- `bigstock_image_145156133`: face fallback 및 `single_person_center` 복구.
- `bigstock_image_13604147`: face fallback 복구.
- `bigstock_image_149283662`: human route object_multi 제거.
- `bigstock_image_108250919`, `sstk_image_400750522`: 비인물 face/person/object 오검출 재발 없음.

v8 quality audit:

| 항목 | 값 |
|---|---:|
| risk annotation counts | 0 |
| risk image counts | 0 |
| approved secondary exception annotation | 853 |
| approved secondary exception image | 164 |

route-label mismatch manifest:

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260529_v8_multimode_full10k_group_hardgate_animal_facefallback_sourceaxis_landscape_teacher_split9010/route_label_mismatch_manifest.json
data/SSTK/Full_10000/artifacts/training_labels_multimode/260529_v8_multimode_full10k_group_hardgate_animal_facefallback_sourceaxis_landscape_teacher_split9010/route_label_mismatch_manifest.csv
```

flagged image는 3,364장이다. 이는 strict validation 실패가 아니라 route precompute와 최종 mode-specific positive 사이의 진단 목록이다. 대부분은 route가 portrait/object/group이어도 해당 mode crop이 hard gate를 통과하지 못해 landscape만 남은 케이스다.

## 18. v8 결론

아래 v8 결론은 2026-05-29 v9 생성 전 기준 기록이다. v8은 v7 대비 사용자가 지적한 false group, human-route object_multi, face 누락, animal object 누락, half portrait center 누락을 보완했고, 10,000장 전체 커버리지, 9:1 train/val split, validation, audit risk 0을 만족했다.

남은 판단은 route-label mismatch manifest에 기록된 진단 목록을 학습 투입에서 그대로 둘지, 별도 exclude/route-recompute variant로 분리할지다.

## 19. v7 결론

아래 v7 결론은 2026-05-28 기준 기록이다. 최신 후보는 상단 v14 섹션을 따른다.

당시 학습 라벨 후보는 v7이었다. v7은 v6 대비 사진학적으로 타당한 상반신/그룹 포트레이트를 회복하면서도, 기존 사용자 지적 비인물 face/person/object 오검출을 다시 열지 않았다. 10,000장 전체 커버리지, 9:1 train/val split, strict validation, current audit risk 0을 모두 만족했다.

남은 판단은 자동 감사 실패가 아니라 정책 강도다. group portrait secondary exception은 155개 이미지로 제한되어 있으나, 최종 학습 투입 전에는 `compare_v6_v7/expanded_with_crops/` contact sheet에서 해당 예외가 서비스 목표와 맞는지 human review하는 것이 좋다.

## 20. 검증

```bash
/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/python -m py_compile \
  src/multimode/entity_atoms.py \
  src/multimode/query_builder.py \
  src/multimode/mode_scorer.py \
  src/scripts/build_multimode_training_labels.py \
  src/scripts/validate_multimode_training_labels.py \
  src/scripts/audit_multimode_label_quality.py \
  src/scripts/visualize_multimode_labels.py
```

```bash
/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/python -m pytest \
  tests/test_multimode_coco_writer.py \
  tests/test_multimode_training_label_splits.py \
  tests/test_multimode_quality_gate_policy.py -q
```

결과:

```text
11 passed
```

## 21. 2026-05-29 v9 landscape GAIC/Q_teach 보완

v9은 v8의 person/face/group/object quality gate를 유지하고, landscape mode 생성 방식만 GAIC teacher를 더 직접 따르도록 변경한 full 10k 산출물이다.

최신 산출물:

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260529_v9_multimode_full10k_landscape_gaic_qteach_split9010
```

구현 변경:

- `build_multimode_training_labels.py`: `--landscape_candidate_policy prefer_gaic`, `--landscape_score_policy teacher_only`, `--landscape_teacher_score_scope gaic` 옵션을 추가했다.
- `mode_scorer.py`: landscape에 한해 `Q_teach` 기반 scorer를 추가했다. `teacher_only` 정책에서도 기존 safety gate, target AR gate, subject/context loss gate는 유지된다.
- `prefer_gaic`: FREE crop은 직접 `teacher:gaic` 후보를 1순위로 두고, 고정 종횡비 crop은 GAIC provenance/lineage가 있는 후보를 우선 선택한다.
- `gaic scope`: `teacher:cgs` 또는 `teacher:cacnet` 후보라도 lineage/provenance에 GAIC가 있으면 GAIC raw score만 `Q_teach` 계산에 사용한다. 따라서 `candidate_source`는 최종 후보의 surface source이고, `landscape_teacher_score_scope=gaic`는 scoring에 사용한 teacher score source다.
- `build_multimode_v1_v6_crop_comparison.py`: v1/v9처럼 대량 샘플을 비교할 때 contact sheet가 JPEG 차원 제한을 넘으면 자동으로 part 파일로 분할 저장한다.
- `select_multimode_v1_vn_improvement_samples.py`: v1 대비 개선을 실제로 볼 수 있도록 `--require_v1_positive` 옵션을 추가했다.

v9 생성 옵션:

```bash
/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/python src/scripts/build_multimode_training_labels.py \
  --features_jsonl data/SSTK/Full_10000/artifacts/precompute/feats_c2c3c5_v2_strict_enriched_routed_final.jsonl \
  --candidates_jsonl data/SSTK/Full_10000/artifacts/candidates/candidates_ar_260413_v1_multimode.jsonl \
  --image_root data/SSTK/Full_10000/images \
  --out_dir data/SSTK/Full_10000/artifacts/training_labels_multimode/260529_v9_multimode_full10k_landscape_gaic_qteach_split9010 \
  --target_ars FREE,1:1,9:16,16:9,3:4,4:3 \
  --max_images 0 \
  --include_optional_negatives 1 \
  --write_split_outputs 1 \
  --train_ratio 0.9 \
  --split_seed 20260506 \
  --write_debug_viz 0 \
  --landscape_candidate_policy prefer_gaic \
  --landscape_score_policy teacher_only \
  --landscape_teacher_score_scope gaic \
  --progress 1
```

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
| missing source/feature/candidate | 0 |

v8 대비 변화:

| 항목 | v8 | v9 | delta |
|---|---:|---:|---:|
| mode query | 210,481 | 210,481 | 0 |
| positive annotation | 104,140 | 104,609 | +469 |
| negative annotation | 82,457 | 67,016 | -15,441 |
| landscape positive | 52,684 | 53,153 | +469 |

v9의 non-landscape mode positive 수는 v8과 동일하다. 즉 이번 변경의 효과는 landscape 후보 선택과 landscape negative 후보 풀 축소에 국한된다.

v9 landscape 세부 통계:

| 항목 | 값 |
|---|---:|
| landscape positive | 53,153 |
| `direct_gaic_free_preferred` | 4,555 |
| `gaic_lineage` | 48,598 |
| `Q_teach` mean | 0.673351 |
| raw GAIC teacher score mean | 2.540106 |

target AR별 landscape positive:

| target_ar | positive |
|---|---:|
| FREE | 9,674 |
| 1:1 | 9,650 |
| 9:16 | 6,230 |
| 16:9 | 8,555 |
| 3:4 | 9,411 |
| 4:3 | 9,633 |

quality audit 결과:

| 항목 | 값 |
|---|---:|
| risk annotation counts | 0 |
| risk image counts | 0 |
| approved secondary exception annotation | 853 |
| approved secondary exception image | 164 |

v1/v9 비교 시각화:

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260529_v9_multimode_full10k_landscape_gaic_qteach_split9010/compare_v1_v9/expanded_common_with_crops/
```

선별 기준은 v1에 실제 positive가 있는 공통 샘플 중, v1 suspicious positive 수, 제거된 non-landscape positive 수, landscape box 변화, v9 GAIC landscape 수를 조합한 score다. 총 160장을 생성했고, 각 개별 파일은 `pairs_with_crops/compare_*_with_crops.jpg`에 저장된다. contact sheet는 다음 2개 파일이다.

```text
v1_v9_common_overlay_crop_comparison_contact_sheet_part001.jpg
v1_v9_common_overlay_crop_comparison_contact_sheet_part002.jpg
```

대표 확인 샘플:

- `bigstock_image_108250919`: v1의 face/person 오검출이 v9에서 제거되어 landscape만 남는다.
- `sstk_image_400750522`: 기존 휴대폰 face/person 오검출 재발이 없다.
- `bigstock_image_149283662`: v1의 human-route object_multi positive가 v9에서 제거된다.
- `bigstock_image_133089656`: v1의 group false positive가 v9에서 제거되어 landscape만 남는다.
- `bigstock_image_124431041`, `bigstock_image_106621058`: 상반신 portrait에 대한 v7/v8 source-edge relaxation은 유지되며, v6의 과도한 bottom-margin 제거 문제는 재발하지 않는다.

사진학 기준은 v7/v8 보완 때 정리한 portrait length/crop rule 자료를 그대로 따른다. head-and-shoulders, upper-body, three-quarter portrait는 full-body crop과 다른 유효 구도이며, 원본 source edge가 이미 몸통을 자르는 경우 하단 margin만으로 실패 처리하지 않는다. 참고 자료는 ePHOTOzine의 portrait length guide, Photofocus의 portrait cropping guide, SLR Lounge의 portrait crop guide다.

v9 정성 리뷰 후속 작업:

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260529_v9_multimode_full10k_landscape_gaic_qteach_split9010/compare_v3_v9/notable_with_crops/review_260529_jy.md
data/SSTK/Full_10000/artifacts/training_labels_multimode/260529_v9_multimode_full10k_landscape_gaic_qteach_split9010/compare_v3_v9/notable_with_crops/review_260529_jy_analysis_summary.json
data/SSTK/Full_10000/artifacts/training_labels_multimode/260529_v9_multimode_full10k_landscape_gaic_qteach_split9010/compare_v3_v9/expanded_all_v3_visualize_with_subject_support_crops/
```

해당 리뷰에서 `sstk_image_1767719213`의 16:9 face crop이 face-center 구도라기보다 rot/lookroom crop처럼 보이는 문제가 확인됐다. 후속 코드에는 face mode center placement hard gate와 향후 label JSON용 `subject_debug` bbox 저장을 추가했다. 리뷰 5장 smoke run에서는 이 샘플의 face positive가 3개에서 2개로 감소했고, 문제 16:9 face crop은 positive에서 제거됐다.

2026-05-29 추가 정책 패치:

- `sstk_image_1197510367` 계열: full-body support point가 ankle keypoint와 person bbox bottom 사이에서 크게 어긋나는 경우를 처리하기 위해 bbox-bottom 기반 robust support point와 `portrait_bbox_bottom` seed를 추가했다. primary full-body, 높은 recall/face score, 낮은 intrusion 조건에서만 `support_grounding_miss` hard gate를 완화한다.
- `sstk_image_1484424431` 계열: `portrait_group` route에서 raw trusted person이 2개 이상이나 dedupe로 distinct member가 1개가 되는 merged-human 케이스에 대해 `group_all` fallback atom을 생성한다. raw person이 1개뿐인 single-person group-route 오분류는 fallback 대상에서 제외한다.
- `sstk_image_2315697813` 계열: scene/interior landscape query에서 GAIC scope `Q_teach`가 positive tau 미만이면 `landscape_low_gaic_qteach_scene_no_label` hard reject로 명시 no-label 처리한다.

추가 smoke 산출물:

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260529_v9_multimode_full10k_landscape_gaic_qteach_split9010/compare_v3_v9/notable_with_crops/review_260529_jy_policy_patch_smoke/
```

smoke 결과는 `sstk_image_1197510367`의 person mode 전 AR 복구, `sstk_image_1484424431`의 group mode 전 AR 생성, `sstk_image_2315697813`의 low-Qteach no-label 유지를 확인했다.

## 22. 2026-05-29 v10 full 10k 정책 패치 산출물

v10 생성 경로:

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260529_v10_multimode_full10k_policy_patch_landscape_gaic_qteach_split9010
```

v10은 v9의 `prefer_gaic + teacher_only + gaic scope` landscape 정책을 유지한다. 변경점은 non-landscape 품질 보완과 no-label 명시화다.

- full-body primary 인물에서 support point가 ankle/keypoint보다 bbox bottom에 가까운 경우 robust support point와 `portrait_bbox_bottom` seed를 사용한다.
- strict recall/face/intrusion 조건을 만족할 때만 `support_grounding_miss`를 완화한다.
- `portrait_group` route에서 raw trusted person이 2명 이상이나 dedupe로 group atom이 사라지는 경우 `group_all` fallback atom을 생성한다.
- raw person 1명뿐인 single-person group-route 오분류는 fallback 대상에서 제외한다.
- scene/interior landscape query에서 GAIC scope `Q_teach`가 positive threshold 미만이면 `landscape_low_gaic_qteach_scene_no_label`로 no-label 처리한다.
- v10 annotation attributes에는 mode별 subject/support 확인용 `subject_debug`가 저장된다.

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
| missing source/feature/candidate | 0 |
| quality audit risk | 0 |

v9/v10 변화:

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

v3/v10 비교 산출물:

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260529_v10_multimode_full10k_policy_patch_landscape_gaic_qteach_split9010/compare_v3_v10/COMPARE_V3_V10_REVIEW_GUIDE_KO.md
data/SSTK/Full_10000/artifacts/training_labels_multimode/260529_v10_multimode_full10k_policy_patch_landscape_gaic_qteach_split9010/compare_v3_v10/expanded_all_v3_visualize_with_crops/
data/SSTK/Full_10000/artifacts/training_labels_multimode/260529_v10_multimode_full10k_policy_patch_landscape_gaic_qteach_split9010/compare_v3_v10/expanded_all_v3_visualize_with_subject_support_crops/
data/SSTK/Full_10000/artifacts/training_labels_multimode/260529_v10_multimode_full10k_policy_patch_landscape_gaic_qteach_split9010/compare_v3_v10/notable_with_crops/
```

v3 기존 시각화 디렉터리에 있던 200개 샘플 전체를 비교했고, notable 40개는 별도 contact sheet로 묶었다. full 10k 기준 positive annotation은 v3 `135,058`에서 v10 `105,627`로 `29,431`건 감소했다. 감소 대부분은 non-landscape false positive 및 중복 positive 축소에 해당한다.

검증:

```bash
/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/python -m pytest \
  tests/test_multimode_coco_writer.py \
  tests/test_multimode_training_label_splits.py \
  tests/test_multimode_quality_gate_policy.py -q
```

결과:

```text
22 passed
```

## 23. 2026-05-29 v11 strict-intent / subject-safe landscape 산출물

v11 생성 경로:

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260529_v11_multimode_full10k_strict_intent_landscape_subjectsafe_split9010
```

v11은 v10의 support grounding 완화, `group_all` fallback, low-Qteach scene no-label 정책을 유지하면서, 다음 후속 보완을 추가했다.

- center mode가 ROT처럼 보이는 crop을 제거하기 위해 center deviation과 thirds proximity hard gate를 강화했다.
- ROT mode가 center처럼 보이는 crop을 제거하기 위해 thirds deviation과 center-like reject를 추가했다.
- landscape mode는 GAIC scope `Q_teach`를 우선하되, subject core recall/`q_subj`가 낮아 사람이 잘리는 crop을 no-label 처리한다.
- object mode는 saliency foreground overlap 기준을 강화했다.
- strict intent audit 스크립트로 full positive annotation을 전수 검사한다.

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
| missing source/feature/candidate | 0 |
| quality audit risk | 0 |
| strict intent audit suspicious | 0 |

v10/v11 변화:

| 항목 | v10 | v11 | delta |
|---|---:|---:|---:|
| positive annotation | 105,627 | 80,167 | -25,460 |
| non-landscape positive | 52,474 | 46,110 | -6,364 |
| landscape positive | 53,153 | 34,057 | -19,096 |

v3/v11 비교 산출물:

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260529_v11_multimode_full10k_strict_intent_landscape_subjectsafe_split9010/compare_v3_v11/COMPARE_V3_V11_REVIEW_GUIDE_KO.md
data/SSTK/Full_10000/artifacts/training_labels_multimode/260529_v11_multimode_full10k_strict_intent_landscape_subjectsafe_split9010/compare_v3_v11/expanded_all_v3_visualize_with_subject_support_crops/
data/SSTK/Full_10000/artifacts/training_labels_multimode/260529_v11_multimode_full10k_strict_intent_landscape_subjectsafe_split9010/compare_v3_v11/notable_with_subject_support_crops/
```

v3 기존 시각화 디렉터리에 있던 200개 샘플 전체를 비교했고, notable 72개는 별도 contact sheet와 per-image pair 이미지로 묶었다. full 10k 기준 positive annotation은 v3 `135,058`에서 v11 `80,167`로 `54,891`건 감소했다. 감소의 주된 원인은 weak/misaligned positive를 no-label 처리한 것이다.

검증:

```bash
/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/python -m pytest \
  tests/test_multimode_coco_writer.py \
  tests/test_multimode_training_label_splits.py \
  tests/test_multimode_quality_gate_policy.py -q
```

결과:

```text
26 passed
```

## 24. 2026-06-01 v12 scene-subject 보완 산출물

v12 생성 경로:

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260601_v12_multimode_full10k_strict_intent_landscape_scene_subjectsafe_split9010
```

v12는 v11의 strict-intent / subject-safe landscape 산출물을 기반으로 하되, landscape 타입으로 분류된 scene/interior 이미지에서 subject 영역이 좌상단 등 의미 없는 위치에 표시되던 문제를 해결했다. 원인은 saliency foreground 자체가 그 위치를 강하게 예측했다기보다, `no_dominant_subject` 상태의 detector/saliency 불일치 diagnostic box가 scene atom의 core/support와 visualization subject overlay로 승격되던 구현상의 policy gap이었다.

v12 보완 사항:

- low-reliability scene pseudo-subject 조건을 감지한다. 조건은 `no_dominant_subject`, detector/saliency severe disagreement, `subject_reliability <= 0.25`, `subject_agreement_iou <= 0.05`의 조합이다.
- 해당 scene atom의 core/support를 full-frame으로 격하하고, 원래 guidance/effective/saliency bbox는 diagnostic meta로 보존한다.
- landscape subject-safe gate와 `salient_subject_loss`는 해당 조건에서 비활성화한다. scene/interior처럼 명확한 주피사체가 없는 이미지는 subject box recall로 crop을 reject하지 않는다.
- visualization은 subject-safe suppressed landscape query에 대해 `SUBJ` overlay를 표시하지 않는다.
- `src/scripts/audit_landscape_scene_subject_alignment.py`로 v11/v12의 low-reliability scene query 영향 범위를 비교한다.

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
| missing source/feature/candidate | 0 |
| quality audit risk | 0 |
| strict intent audit suspicious | 0 |

v11/v12 low-reliability scene-subject 감사:

| 항목 | v11 | v12 | delta |
|---|---:|---:|---:|
| low-reliability scene image | 394 | 394 | 0 |
| affected landscape query | 2,364 | 2,364 | 0 |
| positive query | 1,569 | 2,252 | +683 |
| no-positive query | 786 | 90 | -696 |
| `landscape_subject_core_recall_miss` | 728 | 0 | -728 |
| `landscape_subject_quality_miss` | 500 | 0 | -500 |
| `salient_subject_loss` | 191 | 0 | -191 |
| `landscape_low_gaic_qteach_scene_no_label` | 84 | 84 | 0 |
| `landscape_no_gaic_teacher_score` | 6 | 6 | 0 |

이전 v3/v12 비교 산출물(현재 검수는 200장의 v3/v13 비교 산출물을 우선 확인):

```text
data/SSTK/Full_10000/artifacts/training_labels_multimode/260601_v12_multimode_full10k_strict_intent_landscape_scene_subjectsafe_split9010/compare_v3_v12/COMPARE_V3_V12_REVIEW_GUIDE_KO.md
data/SSTK/Full_10000/artifacts/training_labels_multimode/260601_v12_multimode_full10k_strict_intent_landscape_scene_subjectsafe_split9010/compare_v3_v12/expanded_all_v3_visualize_with_subject_support_crops/
data/SSTK/Full_10000/artifacts/training_labels_multimode/260601_v12_multimode_full10k_strict_intent_landscape_scene_subjectsafe_split9010/compare_v3_v12/notable_with_subject_support_crops/
```

v3 기존 시각화 디렉터리에 있던 200개 샘플 전체를 비교했고, notable 80개는 별도 contact sheet와 per-image pair 이미지로 묶었다. full 10k 기준 positive annotation은 v3 `135,058`에서 v12 `79,856`로 `55,202`건 감소했다.

검증:

```bash
/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/python -m pytest \
  tests/test_multimode_coco_writer.py \
  tests/test_multimode_training_label_splits.py \
  tests/test_multimode_quality_gate_policy.py -q
```

결과:

```text
31 passed
```

## 25. 최신 결론

2026-06-01 현재 최신 학습 라벨 후보는 v14이다. v14는 v13의 support/group fallback, strict mode-intent gate, subject-safe landscape GAIC-Qteach 정책을 유지하면서 residual scene pseudo-subject active 문제를 보완했고, 10,000장 전체 커버리지, 9:1 split, strict validation, quality audit risk 0, strict intent audit suspicious 0, residual subject-safe active 0을 만족한다. 신규 MobileCropNet/multi-mode 학습 입력은 v14 `label_json/multimode_labels_*.json`을 우선 사용한다.
