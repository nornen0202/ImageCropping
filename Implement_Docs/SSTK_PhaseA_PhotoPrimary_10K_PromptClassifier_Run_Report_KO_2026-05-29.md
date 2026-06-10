# SSTK Phase A Photo-Primary 10K Prompt Classifier Run Report

작성일: 2026-05-29

이 문서는 SSTK curation/media-type 보완과 새 Phase A photo-primary 10K pool 생성, 그리고 해당 pool에 대한 data factory 및 multi-mode v11 학습 라벨 생성/검증 결과만 다룬다. Lookroom/gaze QA 내용은 별도 문서로 분리한다.

## 1. 결론

기존 `Full_10000`에 사후 필터를 적용한 것이 아니라, 전체 SSTK mapped cache를 입력으로 다시 Phase A 후보를 구성했다. 이후 metadata/pixel heuristic과 SigLIP2 prompt classifier를 결합해 camera photo-primary만 남기고, 그 안에서 category quota를 다시 refill해 정확히 10,000장을 구성했다.

최종 pool은 다음 조건을 만족한다.

| 항목 | 결과 |
|---|---:|
| 최종 source image | 10,000 |
| `media_decision=keep_photo_primary` | 10,000 |
| final image 파일 수 | 10,000 |
| missing image | 0 |
| monochrome/desaturated flag | 0 |
| SigLIP2 classified rows | 40,000 / 40,000 |
| SigLIP2 classifier error | 0 |

이 pool에 대해 data factory를 이어서 수행했고, multi-mode v11 strict-intent 라벨도 생성 및 검증했다.

| 항목 | 결과 |
|---|---:|
| source image entries | 10,000 |
| mode query | 217,890 |
| annotation | 163,443 |
| positive annotation | 80,519 |
| negative annotation | 82,924 |
| train / val split | 9,000 / 1,000 |
| strict validation | pass |
| quality audit risk | 0 |
| strict intent audit suspicious | 0 |

## 2. 기존 문제 원인

기존 curated pool은 aesthetic/category sampling 중심이었고, 실제 카메라 촬영 사진인지에 대한 강한 media-type gate가 sampling 전에 충분히 걸리지 않았다. 따라서 다음 유형이 high-aesthetic 또는 category quota를 통해 들어올 수 있었다.

- 일러스트, vector, render, CGI, 생성 이미지
- 제품/동물/오브젝트가 흰색 또는 단색 배경에 고립된 packshot
- 배경, texture, copyspace, 단색 studio backdrop
- 흑백 또는 채도가 극단적으로 낮은 사진

`full10000_siglip2_gpu_20260529`에서 흑백 이미지가 다수 보인 핵심 원인은 SigLIP2 prompt classifier가 "실제 카메라 사진"의 의미적 유사도를 잘 잡는 대신, 흑백 사진도 semantic camera photo로 볼 수 있다는 점이다. 즉 prompt classifier만으로는 흑백/저채도/단색 low-context를 안정적으로 배제하기 어렵다. 그래서 v5 정책에서는 pixel 기반 `monochrome_score`, saturation, entropy, dominant color, edge density를 별도 hard review gate로 두고, `monochrome_flag >= 0.70`을 `review_monochrome_desaturated`로 보내 최종 `keep_photo_primary`에서 제외했다.

## 3. 보완 설계

이번 구현은 prompt classifier를 단독 판정기로 쓰지 않고, metadata/pixel heuristic과 결합했다.

| 단계 | 목적 | 구현 |
|---|---|---|
| full cache candidate sampling | 전체 SSTK에서 category-balanced 후보 40K 구성 | `src/scripts/sample_sstk_phaseA_candidates_from_cache.py` |
| metadata/pixel media audit | render/graphic/packshot/background/monochrome 사전 감지 | `src/sstk_media_type_audit.py` |
| SigLIP2 prompt classifier | heuristic이 놓친 graphic/packshot/background/studio low-context demotion | `PromptMediaClassifier` |
| final 10K refill | `keep_photo_primary` 내부에서 category quota 재구성 | `src/scripts/finalize_sstk_photo_primary_pool.py` |
| data factory e2e | 새 pool에 precompute/candidates/teacher/multimode 수행 | `src/scripts/run_phaseA_to_teacher_e2e.sh` |

Prompt classifier는 bad row를 photo로 승격하지 않고, heuristic상 `keep_photo_primary`였던 row를 review로 내리는 방향으로만 사용했다. 이 방향성이 중요하다. prompt model의 zero-shot 오류가 들어와도 최종 pool을 더 느슨하게 만들지 않는다.

## 4. 구현 변경

추가/수정한 주요 파일:

- `src/scripts/sample_sstk_phaseA_candidates_from_cache.py`: full mapped cache streaming, category별 top aesthetic percentile threshold, weighted reservoir sampling, tar image export, durable status/summary.
- `src/sstk_media_type_audit.py`: v5 media-type policy, SigLIP2/CLIP optional prompt classifier, monochrome/desaturated review gate, prompt demotion logic.
- `src/scripts/finalize_sstk_photo_primary_pool.py`: audited parquet에서 `keep_photo_primary`만 category-balanced 10K로 refill.
- `src/scripts/run_phaseA_to_teacher_e2e.sh`: media-type audit/prompt classifier 옵션, v11 multi-mode 옵션 passthrough, `c7_saliency_priority=opencv` 지원.
- `src/extract_features/c7_saliency.py`, `src/scripts/augment_saliency_subject_features.py`: HF saliency model 다운로드 timeout을 피하기 위한 OpenCV saliency backend 선택.
- 테스트: `tests/test_sample_sstk_phaseA_candidates_from_cache.py`, `tests/test_finalize_sstk_photo_primary_pool.py`, 기존 split test 재실행.

## 5. Phase A 생성 결과

Remote base:

```text
data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529
```

### 5.1 Full SSTK candidate 40K

입력은 전체 SSTK mapped cache다.

```text
data/SSTK/Full_10000/cache/filter/df_mapped_cache_sstk_100.parquet
```

이 cache는 원격 기준 27,681,570 rows였고, category별 aesthetic 상위 20% 후보에서 weighted reservoir sampling으로 40,000장을 추출했다.

| 항목 | 결과 |
|---|---:|
| candidate size | 40,000 |
| selected rows | 40,000 |
| top percentile | 0.20 |
| tar export saved | 40,000 |
| tar export missing | 0 |
| export failed jobs | 0 |

candidate category count는 12개 category에 거의 균등하게 배분됐다.

| category | count |
|---|---:|
| animals | 3,333 |
| architecture_exterior | 3,333 |
| documents_text | 3,333 |
| food | 3,333 |
| indoor_interior | 3,333 |
| landscape_nature | 3,337 |
| other_ambiguous | 3,333 |
| people_multi | 3,333 |
| people_single | 3,333 |
| product_object | 3,333 |
| sports | 3,333 |
| transportation | 3,333 |

### 5.2 SigLIP2 media-type audit

SigLIP2 snapshot:

```text
weights/hf/google_siglip2-base-patch16-224
```

Audit output:

```text
data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529/artifacts/media_type_audit_candidate40000_siglip2_v5_r2
```

| decision | count |
|---|---:|
| keep_photo_primary | 14,376 |
| reject_nonphoto | 1,335 |
| review_background_copyspace | 10,224 |
| review_graphic_ambiguous | 7,579 |
| review_monochrome_desaturated | 2,045 |
| review_packshot_isolated | 3,319 |
| review_studio_low_context | 1,122 |

Prompt classifier 요약:

| 항목 | 결과 |
|---|---:|
| mode | siglip2 |
| model | `weights/hf/google_siglip2-base-patch16-224` |
| device | cuda |
| batch size | 64 |
| classified | 40,000 |
| errors | 0 |
| prompt adjusted rows | 11,101 |

### 5.3 Final photo-primary 10K

Final output:

```text
data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529/filtered_sstk_100.parquet
data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529/images
data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529/final_photo_primary_10k_summary.json
data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529/contact_sheets/keep_photo_primary.jpg
```

| 항목 | 결과 |
|---|---:|
| photo_primary available | 14,376 |
| selected rows | 10,000 |
| `keep_photo_primary` selected | 10,000 |
| copied/existing images | 10,000 |
| missing images | 0 |
| monochrome flag | 0 |

Final category count:

| category | count |
|---|---:|
| animals | 852 |
| architecture_exterior | 853 |
| documents_text | 692 |
| food | 854 |
| indoor_interior | 852 |
| landscape_nature | 856 |
| other_ambiguous | 852 |
| people_multi | 852 |
| people_single | 852 |
| product_object | 853 |
| sports | 779 |
| transportation | 853 |

## 6. Data Factory 실행 결과

Run tag:

```text
260529_v11_photo_primary_fullsstk_phaseA_strict_intent_subjectsafe_split9010
```

Run logs:

```text
data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529/logs/e2e_260529_v11_photo_primary_fullsstk_phaseA_strict_intent_subjectsafe_split9010
```

### 6.1 Precompute

| stage | 결과 |
|---|---:|
| unified C2/C3/C5/C6 features | 10,000 rows |
| C3 enriched records | 3,967 |
| C7 saliency processed | 10,000 |
| C7 backend | `opencv_saliency` 10,000 |
| C7 elapsed | 940.974 sec |

C4는 원격 환경에 `paddle` package가 없어 disabled path로 진행됐다. 이번 목적은 curation/pool 및 v11 label generation 검증이므로 C4 부재는 terminal blocker로 보지 않았다.

Subject routing after C7:

| route | count |
|---|---:|
| scene_general | 2,074 |
| portrait_single | 3,050 |
| background_texture_copyspace | 612 |
| object_single | 2,113 |
| portrait_group | 816 |
| object_multi | 1,333 |
| other_ambiguous | 2 |

### 6.2 Public teacher / candidates / teacher

| stage | 결과 |
|---|---:|
| public teacher raw rows | 30,000 |
| public teacher proposals | 10,000 |
| public proposal injection gate | pass, rate 1.0 |
| candidate rows | 10,000 |
| average total candidates/image | 762.6359 |
| teacher score merged JSONL size | 63,727,222,906 bytes |
| teacher shards | 8 shards, each 1,250 images |

Teacher scoring은 1-worker run이 지나치게 느려 8 shard run으로 전환했다. merged teacher score JSONL은 생성됐지만, merged overview/QA 재생성은 63GB JSONL을 반복 파싱해야 해서 `teacher_auto_repair=0`으로 중복 복구 단계를 끄고 downstream multi-mode label generation을 진행했다. Shard overview와 merged score JSONL은 보존했다.

## 7. Multi-Mode v11 결과

Output:

```text
data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529/artifacts/training_labels_multimode/260529_v11_photo_primary_fullsstk_phaseA_strict_intent_subjectsafe_split9010
```

생성 옵션 핵심:

- `--target_ars FREE,1:1,9:16,16:9,3:4,4:3`
- `--include_optional_negatives 1`
- `--write_split_outputs 1`
- `--train_ratio 0.9`
- `--split_seed 20260506`
- `--landscape_candidate_policy prefer_gaic_subject_safe`
- `--landscape_score_policy teacher_subject_safe`
- `--landscape_teacher_score_scope gaic`
- `--mode_intent_policy strict_v11`

### 7.1 Label generation summary

| 항목 | 결과 |
|---|---:|
| selected source images | 10,000 |
| image tasks | 59,996 |
| mode queries | 217,890 |
| annotations | 163,443 |
| positives | 80,519 |
| negatives | 82,924 |
| elapsed | 5,715.802 sec |

Mode positive counts:

| mode | positives |
|---|---:|
| landscape | 24,994 |
| object_single_rot | 15,401 |
| object_single_center | 15,298 |
| face | 8,397 |
| object_multi_center | 5,227 |
| object_multi_rot | 4,343 |
| single_person_rot | 2,629 |
| single_person_center | 2,108 |
| group_center | 1,380 |
| group_rot | 742 |

### 7.2 Split

| split | images | queries | annotations | positives | negatives |
|---|---:|---:|---:|---:|---:|
| train | 9,000 | 196,488 | 147,823 | 72,771 | 75,052 |
| val | 1,000 | 21,402 | 15,620 | 7,748 | 7,872 |

### 7.3 Strict validation

Validation output:

```text
validation_summary.json
```

| 항목 | 결과 |
|---|---:|
| ok | true |
| source image entries | 10,000 |
| query count | 217,890 |
| annotation count | 163,443 |
| image root count | 10,000 |
| source images missing from label JSON | 0 |
| label images missing from image root | 0 |
| feature unique images | 10,000 |
| label images missing from features | 0 |
| candidate unique images | 10,000 |
| label images missing from candidates | 0 |
| split train/val | 9,000 / 1,000 |

### 7.4 Quality and strict intent audit

Quality audit:

```text
quality_audit/phaseA_v11_quality_audit_summary.json
```

| 항목 | 결과 |
|---|---:|
| source images | 10,000 |
| annotations | 163,443 |
| positive annotations | 80,519 |
| risk annotation counts | 0 |
| risk image counts | 0 |
| approved secondary exception annotations | 898 |
| approved secondary exception images | 177 |

Strict intent audit:

```text
intent_alignment_audit/phaseA_v11_intent_alignment_summary.json
```

| 항목 | 결과 |
|---|---:|
| source images | 10,000 |
| positive annotations | 80,519 |
| suspicious annotations | 0 |
| intent flag annotation counts | 0 |
| intent flag image counts | 0 |

## 8. 로컬 복사

원격 산출물은 다음 로컬 경로로 복사 완료했다.

```text
data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529
```

복사 완료 요약:

| 항목 | 결과 |
|---|---:|
| rsync regular files transferred | 10,117 |
| copied total file size | 73,339,334,319 bytes |
| local final image files | 10,000 |
| local merged teacher score JSONL | 63,727,222,906 bytes |
| local validation artifact | `artifacts/research_agent_runs/phaseA_photo_primary_10k_siglip2_v5_20260529_local_validation.json` |

복사 범위:

- 포함: final 10K parquet/images/contact sheet, media audit 결과, candidate JSONL/parquet/log, precompute outputs, public teacher outputs, merged teacher score JSONL, multi-mode labels/splits/validation/audits/logs.
- 제외: `candidate_40000/images/` scratch images, `candidate_40000/cache/` symlink cache, teacher shard raw JSONL. Shard overview JSON과 merged teacher score JSONL은 유지한다.

이 제외는 final 학습 입력과 검증 산출물을 잃지 않으면서, 중복 raw intermediate와 로컬 filesystem symlink 문제를 피하기 위한 것이다.

## 9. 검증

로컬 테스트:

```bash
/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/python -m pytest tests/test_sample_sstk_phaseA_candidates_from_cache.py tests/test_finalize_sstk_photo_primary_pool.py tests/test_multimode_training_label_splits.py -q
```

결과:

```text
7 passed
```

Remote validation/audit:

- `validate_multimode_training_labels.py`: pass
- `audit_multimode_label_quality.py`: risk 0
- `audit_multimode_intent_alignment.py`: suspicious 0

Local copy validation:

| check | 결과 |
|---|---:|
| final selected rows | 10,000 |
| final parquet rows | 10,000 |
| `media_decision=keep_photo_primary` | 10,000 |
| monochrome flag | 0 |
| final image files | 10,000 |
| multi-mode validation ok | true |
| label images missing from features | 0 |
| label images missing from candidates | 0 |
| quality audit risk | 0 |
| strict intent suspicious | 0 |
| merged teacher score size match | true |

## 10. 근거 자료

이번 설계는 대규모 web/image dataset curation에서 metadata, visual statistics, vision-language model score를 조합하는 방식에 맞췄다.

- CLIP은 image-text contrastive pretraining으로 zero-shot image classification이 가능하다는 기준점을 제공한다. 단, 서비스 curation에서는 zero-shot score를 단독 hard accept로 쓰기보다 auxiliary signal로 쓰는 것이 안전하다. https://arxiv.org/abs/2103.00020
- OpenAI CLIP 소개 글은 자연어 prompt로 visual concept을 분류하는 운영 감각을 설명한다. https://openai.com/index/clip/
- SigLIP2는 SigLIP 계열을 개선한 multilingual vision-language encoder로, 이번 작업에서는 CLIP보다 modern한 prompt classifier 후보로 채택했다. https://arxiv.org/abs/2502.14786
- 사용한 모델 card: `google/siglip2-base-patch16-224`. https://huggingface.co/google/siglip2-base-patch16-224
- LAION-5B와 LAION-Aesthetics는 대규모 web image-text data에서 score/filtering 기반 curation이 필요하다는 배경을 제공한다. https://papers.nips.cc/paper/2022/hash/a1859debfb3b59d094f3504d5ebb6c25-Abstract-Datasets_and_Benchmarks.html, https://laion.ai/blog/laion-aesthetics/
- DataComp는 dataset filtering/selection 자체가 downstream 성능에 큰 영향을 준다는 실험적 근거를 제공한다. https://papers.nips.cc/paper_files/paper/2023/hash/56332d41d55ad7ad8024aac625881be7-Abstract-Datasets_and_Benchmarks.html
- photograph vs graphics 분류 문헌은 color/edge/texture/entropy 같은 low-level 통계가 graphic/background 구분에 여전히 유용함을 뒷받침한다. https://www.cecs.uci.edu/~papers/icme06/pdfs/0000973.pdf
- CNN-generated/AI-generated image detector 문헌은 생성 이미지 탐지가 데이터 분포 변화에 취약할 수 있음을 보여준다. 따라서 AI/render 배제는 prompt classifier 단독이 아니라 metadata, pixel statistics, review bucket을 함께 써야 한다. https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_CNN-Generated_Images_Are_Surprisingly_Easy_to_Spot..._for_Now_CVPR_2020_paper.html, https://arxiv.org/abs/2406.19435

## 11. 남은 운영 주의점

1. `documents_text` category는 photo-primary만 남겨도 문서/간판/종이 촬영 이미지가 섞일 수 있다. 현재는 실제 촬영 사진이면 유지했지만, 학습 목표가 자연 사진 중심이면 별도 downstream quota를 둘 수 있다.
2. Teacher merged overview/QA를 완전 재생성하려면 63GB score JSONL 재파싱 비용이 크다. 현재 라벨 검증은 multi-mode label, candidate, feature, split coverage를 직접 확인했으므로 학습 입력 검증에는 충분하다.
3. `candidate_40000/images/`는 final 10K 이미지가 아니므로 로컬 복사에서 제외했다. 필요하면 원격 tar에서 재생성 가능하다.
