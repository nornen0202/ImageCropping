# SSTK Curated Media-Type Audit 및 Phase A Gate 보완 보고서

작성일: 2026-05-29  
대상 경로: `data/SSTK/Full_10000/images`, `data/SSTK/Full_10000/filtered_sstk_100.parquet`

## 1. 결론

기존 SSTK Curated pool은 `sstk_type == "Photo"`, aesthetic score, category balancing 중심의 선별기였다. 실제 카메라 사진인지, 3D render/illustration/vector인지, 흰 배경 packshot인지, 단색 copy-space인지 판별하는 별도 media-type QC가 없어서 `bigstock_image_1858029`, `bigstock_image_5806405`, `bigstock_image_10568546` 같은 render/illustration 계열이 통과할 수 있었다.

이번 보완은 네 단계로 정리된다.

1. deterministic v4 media-type audit: metadata keyword와 pixel 통계를 결합해 `keep_photo_primary`, review, reject split을 만든다.
2. optional SigLIP2/CLIP prompt classifier: deterministic audit 이후 `keep_photo_primary`만 보수적으로 review로 demote한다. review/reject를 keep으로 되돌리는 promote는 하지 않는다.
3. Phase A pre-sampling media-type gate: category quota sampling 전에 metadata-only gate를 넣어, category quota가 photo-primary 후보 안에서 다시 refill되도록 했다.
4. deterministic v5 monochrome gate: 흑백/강한 저채도 사진을 `review_monochrome_desaturated`로 분리해 color natural-photo 중심의 main 학습 pool에서 제외한다.

## 2. 기존 Curated Pool 생성 로직

기존 `src/filter_sstk_dataset.py`의 Phase A filter 흐름은 다음이다.

1. SDP metadata에서 width/height, aesthetic, dedup, `sstk_type`을 확인한다. 핵심 gate는 `aesthetic_score_center/pad >= 5.0`, `image_dedup is not True`, `sstk_type == "Photo"`이다.
2. `sstk_tags`와 TRAIN metadata의 `merged_tags`를 합쳐 `tags` 컬럼을 만든다.
3. seed keyword와 `SentenceTransformer('all-MiniLM-L6-v2')` tag embedding으로 `super_cat`을 배정한다.
4. category별 aesthetic percentile cut과 stratified sampling으로 curated pool size를 맞춘다.
5. `--save_curated_images_dir`가 지정되면 이미 선택된 row의 원본 payload만 flat image dir로 export한다.

따라서 기존 로직은 "photo metadata가 붙은 aesthetically acceptable sample을 category-balanced로 뽑는 절차"이지, camera-captured photo를 보장하는 QC가 아니다.

## 3. 문제 원인

| 원인 | 설명 | 영향 |
| --- | --- | --- |
| Metadata type 과신 | Shutterstock의 `sstk_type=Photo`가 실제 촬영 사진을 보장하지 않음 | render/illustration/AI-like image 유입 |
| Aesthetic-only filtering | graphic, isolated object, copy-space도 aesthetic score가 높으면 통과 가능 | MobileCropNet의 자연사진 crop prior와 다른 domain 혼입 |
| Export 단계 QC 부재 | export는 선택된 row 저장만 수행 | feature/teacher/training이 오염된 pool을 그대로 사용 |
| Sampling 이후 사후 audit 의존 | audit만 후처리하면 quota가 오염 후보까지 포함해 이미 소진됨 | photo-primary 내부 refill이 일어나지 않음 |

## 4. Deterministic v4/v5 Camera-Primary Audit

구현 파일은 `src/sstk_media_type_audit.py`이고, CLI wrapper는 `src/scripts/audit_sstk_media_type.py`이다. 기존 v4 policy id는 `sstk_media_type_v4_20260529_camera_primary`였고, 흑백/저채도 제외를 반영한 현재 policy id는 `sstk_media_type_v5_20260529_camera_primary_no_monochrome`이다.

주요 정책은 다음과 같다.

- `render`, `3d`, `vector`, `clipart`, `cartoon`, `cgi`, AI-generated term은 hard non-photo로 reject한다.
- `illustration`, `graphic`, `symbol`, `logo`, `icon`은 soft non-photo로 보고 pixel 통계와 결합해 reject 또는 review한다.
- `isolated`, `studio`, `white background`, `clipping path`는 실제 사진 packshot일 수 있으므로 hard reject가 아니라 `review_packshot_isolated`로 분리한다.
- `background`, `copy space`, `backdrop`, `blank`, `empty`, `black/white/gray background`는 copy-space/background review 후보로 본다.
- pixel 통계는 entropy, quantized color count, dominant color ratio, white ratio, near-black ratio, edge density, saturation 통계를 사용한다.
- 단색 흰/검은 배경, 낮은 edge density, 낮은 entropy, 지배색 비율이 높은 studio-low-context 이미지는 primary pool에서 review로 이동한다.
- v5에서는 `saturation_mean`, `saturation_std`, `quant_unique_count`를 결합해 흑백 또는 강한 저채도 사진을 `review_monochrome_desaturated`로 이동한다.

v4 full 10K audit 결과:

| Decision | Count | Ratio |
| --- | ---: | ---: |
| `keep_photo_primary` | 6,743 | 67.43% |
| `review_background_copyspace` | 1,413 | 14.13% |
| `review_packshot_isolated` | 817 | 8.17% |
| `review_graphic_ambiguous` | 421 | 4.21% |
| `reject_nonphoto` | 369 | 3.69% |
| `review_studio_low_context` | 237 | 2.37% |

산출물:

| 산출물 | 경로 |
| --- | --- |
| Summary JSON | `data/SSTK/Full_10000/artifacts/media_type_audit/full10000_v4_20260529_camera_primary/media_type_audit_summary.json` |
| 전체 audited parquet | `data/SSTK/Full_10000/artifacts/media_type_audit/full10000_v4_20260529_camera_primary/filtered_sstk_media_audited.parquet` |
| main 추천 split | `data/SSTK/Full_10000/artifacts/media_type_audit/full10000_v4_20260529_camera_primary/filtered_sstk_photo_primary.parquet` |
| review split | `data/SSTK/Full_10000/artifacts/media_type_audit/full10000_v4_20260529_camera_primary/filtered_sstk_media_review.parquet` |
| packshot review | `data/SSTK/Full_10000/artifacts/media_type_audit/full10000_v4_20260529_camera_primary/filtered_sstk_packshot_review.parquet` |
| nonphoto reject | `data/SSTK/Full_10000/artifacts/media_type_audit/full10000_v4_20260529_camera_primary/filtered_sstk_reject_nonphoto.parquet` |
| contact sheets | `data/SSTK/Full_10000/artifacts/media_type_audit/full10000_v4_20260529_camera_primary/contact_sheets/` |

사용자가 재검수한 v1 `keep_photo_primary.jpg` 상위 60장의 v4 재판정 결과는 다음이었다.

| v4 decision | Count |
| --- | ---: |
| `review_background_copyspace` | 33 |
| `review_studio_low_context` | 24 |
| `review_packshot_isolated` | 2 |
| `keep_photo_primary` | 1 |

남은 keep 1장은 `sstk_image_200786861`이다. 즉, 해당 sheet 기준으로 v4는 사용자가 지적한 단색 배경/저질감 studio false keep을 거의 전부 primary pool 밖으로 이동시켰다.

## 5. SigLIP2 vs CLIP Prompt Classifier 분석

CLIP은 자연어 prompt를 category label로 사용해 zero-shot image classification을 수행할 수 있는 대표적인 vision-language encoder이다. 원 논문은 4억 image-text pair로 학습하고, 30개 이상 benchmark에서 zero-shot transfer를 평가했다. OpenAI 설명 자료도 CLIP이 natural language supervision으로 visual concept를 학습하고 category name prompt만으로 classification benchmark에 적용될 수 있음을 설명한다.

SigLIP2는 SigLIP의 후속 vision-language encoder 계열이다. 논문 초록 기준으로 captioning-based pretraining, self-supervised loss, masked prediction, online data curation 등을 결합했고, SigLIP 대비 zero-shot classification, image-text retrieval, VLM representation transfer, localization/dense prediction에서 모든 scale에서 개선되었다고 보고한다.

이번 task의 모델 선택 기준은 "최고 절대 성능"이 아니라 "10K curation에서 비용 대비 media-type 분리 효율"이다.

| 후보 | 장점 | 단점 | 채택 판단 |
| --- | --- | --- | --- |
| `google/siglip2-base-patch16-224` | SigLIP2 계열, zero-shot/prompt 분류에 적합, base급이라 10K audit 비용이 현실적, HF pipeline/model card 제공 | 현재 로컬 `transformers` 조합에서 processor fallback이 필요할 수 있음 | 기본 채택 |
| `openai/clip-vit-base-patch32` | 널리 검증됨, dependency 호환성이 좋음, fallback으로 안정적 | 최신 SigLIP2 대비 semantic/retrieval/transfer 성능 기대치가 낮음 | fallback/비교 옵션 |
| SigLIP2 large/so400m 계열 | 더 높은 성능 기대 | 10K 반복 curation에는 비용/메모리 부담이 큼 | default 제외 |

결론적으로 default는 `google/siglip2-base-patch16-224`, fallback은 `openai/clip-vit-base-patch32`로 두었다. `siglip2-base-patch16-224`의 HF model page는 `zero-shot-image-classification` pipeline 예시와 직접 model loading 예시를 제공하며, model file size는 약 1.54GB로 표시된다. 현재 목적에는 large 모델보다 base 모델이 더 효율적이다.

구현 원칙은 보수적으로 정했다.

- prompt classifier는 optional이다. 기본값은 꺼져 있다.
- classifier 결과는 `keep_photo_primary`를 review로 내리는 데만 쓴다.
- deterministic audit이 이미 review/reject한 sample을 camera photo로 promote하지 않는다.
- top non-camera prompt group score가 `min_score` 이상이고, camera-photo score보다 `min_margin` 이상 높을 때만 demote한다.
- prompt score는 calibration된 probability가 아니라 prompt ensemble 내부 상대 점수로 해석한다.

추가 dependency:

```text
transformers>=4.44.0
sentencepiece>=0.1.99
```

## 6. Prompt Classifier 구현

구현 위치는 `src/sstk_media_type_audit.py`이다.

추가된 핵심 구성:

- `DEFAULT_PROMPT_CLASSIFIER_MODEL = "google/siglip2-base-patch16-224"`
- `DEFAULT_CLIP_CLASSIFIER_MODEL = "openai/clip-vit-base-patch32"`
- `PROMPT_GROUPS`: `camera_photo`, `studio_low_context_photo`, `packshot_or_isolated_photo`, `background_texture_copyspace`, `render_or_graphic`, `illustration_or_render`
- `PromptMediaClassifier`: HF `AutoModel` 기반 image/text feature cosine matching
- `run_prompt_classifier()`: audit 대상 image batch classify
- `apply_prompt_classifier_result()`: keep row만 review로 demote

현재 로컬 환경에서는 SigLIP2 processor 로딩 시 `AutoProcessor`가 잘못된 tokenizer 조합을 시도할 수 있어, `AutoImageProcessor`와 `AutoTokenizer(use_fast=False)`를 분리 로드하는 fallback을 넣었다. 1-image CPU smoke에서 fallback 경로가 정상 동작했다.

standalone audit에서 SigLIP2 prompt classifier를 켜는 예:

```bash
/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/python src/scripts/audit_sstk_media_type.py --input_parquet data/SSTK/Full_10000/filtered_sstk_100.parquet --image_dir data/SSTK/Full_10000/images --output_dir data/SSTK/Full_10000/artifacts/media_type_audit/full10000_siglip2 --mapped_cache_parquet data/SSTK/Full_10000/cache/filter/df_mapped_cache_sstk_100.parquet --write_contact_sheets 1 --prompt_classifier siglip2 --prompt_classifier_device cuda --prompt_classifier_batch_size 16
```

CLIP fallback 비교 예:

```bash
/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/python src/scripts/audit_sstk_media_type.py --input_parquet data/SSTK/Full_10000/filtered_sstk_100.parquet --image_dir data/SSTK/Full_10000/images --output_dir data/SSTK/Full_10000/artifacts/media_type_audit/full10000_clip --mapped_cache_parquet data/SSTK/Full_10000/cache/filter/df_mapped_cache_sstk_100.parquet --write_contact_sheets 1 --prompt_classifier clip --prompt_classifier_device cuda --prompt_classifier_batch_size 32
```

## 7. Phase A Pre-Sampling Media-Type Gate

사후 audit만 적용하면 category quota가 이미 render/background/packshot 후보까지 포함해서 소진된다. 이번 요청의 핵심은 category quota를 photo-primary 후보 안에서 다시 refill하는 것이다.

구현 위치는 `src/filter_sstk_dataset.py`의 `apply_media_type_presampling_gate()`이다. 이 gate는 full image pixel audit이 아니라 metadata-only gate다. 이유는 Phase A sampling 전에 모든 후보 이미지 payload를 펼치면 비용이 커지기 때문이다.

동작 순서:

1. 기존 SDP/TRAIN metadata를 읽어 `tags`를 만든다.
2. `super_cat`, `sampling_weight`까지 계산한다.
3. `--media_type_presampling_gate 1`이면 `presampling_metadata_gate()`를 적용한다.
4. default로 `presample_media_decision == keep_photo_primary`만 남긴다.
5. 그 뒤 category별 aesthetic percentile cut과 stratified sampling을 수행한다.

이 순서 때문에 category quota는 gate 이전 후보가 아니라 gate 이후 photo-primary 후보에서 채워진다. 예를 들어 특정 category에서 background/copy-space 후보가 빠지면, 같은 category의 다른 photo-primary 후보가 quota를 채울 수 있다.

추가 CLI:

| 옵션 | 기본값 | 의미 |
| --- | --- | --- |
| `--media_type_presampling_gate` | `0` | Phase A sampling 전에 metadata-only media-type gate 적용 |
| `--media_type_presampling_keep_decisions` | `keep_photo_primary` | gate 후 유지할 decision CSV |
| `--media_type_presampling_report` | empty | gate summary JSON 경로 |

`src/scripts/run_filter.sh` 환경변수:

```bash
FILTER_MEDIA_TYPE_PRESAMPLING_GATE=1
FILTER_MEDIA_TYPE_PRESAMPLING_KEEP_DECISIONS=keep_photo_primary
FILTER_MEDIA_TYPE_PRESAMPLING_REPORT=data/SSTK/Full_10000/artifacts/media_type_audit/phaseA_presampling_gate_summary.json
```

E2E runner 옵션:

```bash
bash src/scripts/run_phaseA_to_teacher_e2e.sh --data_dir data/SSTK/Full_10000 --bucket sstk_100 --filter_media_type_presampling_gate 1 --filter_media_type_presampling_keep_decisions keep_photo_primary --export_curated_images 1 --curated_image_dir data/SSTK/Full_10000/images --run_media_type_audit 1 --media_type_audit_use_photo_primary 1 --media_type_audit_contact_sheets 1 --media_type_audit_prompt_classifier siglip2 --media_type_audit_prompt_device cuda
```

주의: pre-sampling gate는 metadata-only이므로 단색 배경처럼 metadata에 신호가 약한 케이스는 deterministic v5 pixel audit 또는 optional SigLIP2 audit에서 추가로 잡는다. 즉 pre-sampling gate는 빠른 1차 gate이고, v5 audit은 image 기반 2차 QC다.

## 8. 검증 결과

실행한 검증:

```bash
bash -n src/scripts/run_phaseA_to_teacher_e2e.sh
bash -n src/scripts/run_filter.sh
/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/python -m pytest tests/test_sstk_media_type_audit.py tests/test_filter_sstk_dataset_export.py -q
/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/python -m pytest tests/test_sstk_media_type_audit.py tests/test_filter_sstk_dataset_export.py tests/test_teacher_scorer.py -q
```

결과:

- shell syntax: pass
- media-type/filter focused tests: `14 passed`
- media-type/filter/teacher focused tests: `45 passed`
- deterministic audit 12-image smoke: pass
- SigLIP2 prompt classifier 1-image CPU smoke: pass
- 사용자 지적 샘플 포함 targeted SigLIP2 CPU smoke: pass
- GPU status/progress JSON smoke: pass
- SigLIP2 local-snapshot GPU smoke: pass, 8 rows classified, errors 0, adjusted rows 1
- SigLIP2 full 10K GPU run: pass, 10,000 rows classified, errors 0
- v5 monochrome gate focused tests: `16 passed`
- v5 corrected full 10K regeneration using cached GPU prompt columns: pass, 10,000 rows, errors 0, policy monochrome keep 0

smoke artifact:

| Smoke | 경로 | 결과 |
| --- | --- | --- |
| deterministic only | `artifacts/research_agent_runs/media_audit_smoke_20260529_prompt_none/` | 12 rows, missing image 0, decision split 생성 |
| SigLIP2 prompt classifier | `artifacts/research_agent_runs/media_audit_smoke_20260529_siglip2/` | 1 row classified, errors 0, adjusted rows 0 |
| targeted SigLIP2 prompt classifier | `artifacts/research_agent_runs/media_audit_targeted_siglip2_20260529/` | 5 rows classified, errors 0, adjusted rows 0 |

SigLIP2 smoke summary:

```json
{
  "enabled": true,
  "mode": "siglip2",
  "model_id": "google/siglip2-base-patch16-224",
  "device": "cpu",
  "classified": 1,
  "errors": 0,
  "adjusted_rows": 0
}
```

targeted SigLIP2 smoke는 `bigstock_image_1858029`, `bigstock_image_1868781`, `bigstock_image_5806405`, `bigstock_image_10568546`, `sstk_image_200786861`을 대상으로 실행했다. deterministic v4 판정은 reject 3장, packshot review 1장, keep 1장이었고, prompt classifier는 기존 review/reject를 keep으로 promote하지 않았다.

### 8.1 SigLIP2 Full 10K GPU 실행 결과

사용자 지정 interactive GPU run `10.12.12.52`에서 full 10K prompt classifier audit을 수행했다. GPU host는 `run1112931-gpu-spot-40gb-1`이고, 장비는 `NVIDIA A100-SXM4-80GB MIG 3g.40gb`로 확인됐다. 원격 서버의 Hugging Face 접근이 timeout되어 로컬 Hugging Face snapshot을 원격 공유 스토리지에 업로드한 뒤 local path로 로드했다.

| 항목 | 값 |
| --- | --- |
| Prompt model | `google/siglip2-base-patch16-224` |
| Remote model path | `/group-volume/users/jaden.ju/Sources/ImageCropping/weights/hf/google_siglip2-base-patch16-224` |
| Device | `cuda` |
| Batch size | `64` |
| Total rows | `10,000` |
| Classified | `10,000` |
| Classifier errors | `0` |
| Prompt-adjusted rows | `2,914` |
| Elapsed | `497.879 sec` |

실행 command:

```bash
cd /group-volume/users/jaden.ju/Sources/ImageCropping && mkdir -p data/SSTK/Full_10000/artifacts/media_type_audit/full10000_siglip2_gpu_20260529 && nohup /usr/local/bin/python3 src/scripts/audit_sstk_media_type.py --input_parquet data/SSTK/Full_10000/filtered_sstk_100.parquet --image_dir data/SSTK/Full_10000/images --output_dir data/SSTK/Full_10000/artifacts/media_type_audit/full10000_siglip2_gpu_20260529 --mapped_cache_parquet data/SSTK/Full_10000/cache/filter/df_mapped_cache_sstk_100.parquet --write_contact_sheets 1 --contact_sheet_max_per_decision 80 --prompt_classifier siglip2 --prompt_classifier_model weights/hf/google_siglip2-base-patch16-224 --prompt_classifier_device cuda --prompt_classifier_batch_size 64 --prompt_classifier_fail_on_error 1 --status_json data/SSTK/Full_10000/artifacts/media_type_audit/full10000_siglip2_gpu_20260529/status.json --progress_interval 250 > data/SSTK/Full_10000/artifacts/media_type_audit/full10000_siglip2_gpu_20260529/run.log 2>&1 &
```

최종 decision count:

| Decision | Count | v4 deterministic 대비 |
| --- | ---: | ---: |
| `keep_photo_primary` | 3,829 | -2,914 |
| `review_background_copyspace` | 2,605 | +1,192 |
| `review_graphic_ambiguous` | 2,040 | +1,619 |
| `review_packshot_isolated` | 828 | +11 |
| `reject_nonphoto` | 369 | 0 |
| `review_studio_low_context` | 329 | +92 |

prompt classifier는 설계대로 `keep_photo_primary`를 review 계열로 demote하는 방향으로만 작동했다. 따라서 `reject_nonphoto`는 deterministic v4와 동일하게 369장이고, primary pool은 6,743장에서 3,829장으로 줄었다. 이 결과는 기존 `keep_photo_primary.jpg`에서 지적된 흰/검은 단색 배경, low-context studio, graphic-like sample이 여전히 섞이던 문제를 더 강하게 줄이는 방향이다.

산출물은 원격에서 로컬로 복사 완료했다.

| 산출물 | 로컬 경로 |
| --- | --- |
| Run directory | `data/SSTK/Full_10000/artifacts/media_type_audit/full10000_siglip2_gpu_20260529/` |
| Status JSON | `data/SSTK/Full_10000/artifacts/media_type_audit/full10000_siglip2_gpu_20260529/status.json` |
| Summary JSON | `data/SSTK/Full_10000/artifacts/media_type_audit/full10000_siglip2_gpu_20260529/media_type_audit_summary.json` |
| Audited parquet | `data/SSTK/Full_10000/artifacts/media_type_audit/full10000_siglip2_gpu_20260529/filtered_sstk_media_audited.parquet` |
| Photo-primary split | `data/SSTK/Full_10000/artifacts/media_type_audit/full10000_siglip2_gpu_20260529/filtered_sstk_photo_primary.parquet` |
| Review split | `data/SSTK/Full_10000/artifacts/media_type_audit/full10000_siglip2_gpu_20260529/filtered_sstk_media_review.parquet` |
| Packshot review subset | `data/SSTK/Full_10000/artifacts/media_type_audit/full10000_siglip2_gpu_20260529/filtered_sstk_packshot_review.parquet` |
| Reject split | `data/SSTK/Full_10000/artifacts/media_type_audit/full10000_siglip2_gpu_20260529/filtered_sstk_reject_nonphoto.parquet` |
| Contact sheets | `data/SSTK/Full_10000/artifacts/media_type_audit/full10000_siglip2_gpu_20260529/contact_sheets/` |
| Local validation summary | `artifacts/research_agent_runs/full10000_siglip2_gpu_20260529_validation.json` |

로컬 검증 결과:

- `status.json`: `status=completed`, `phase=complete`
- `filtered_sstk_media_audited.parquet`: 10,000 rows
- prompt classifier: classified 10,000, errors 0
- missing images: 0
- split partition: photo 3,829 + review 5,802 + reject 369 = 10,000
- `packshot_review` 828 rows는 `media_review`의 부분집합으로 확인
- parquet의 `media_decision` count와 summary JSON의 `decision_counts` 일치
- contact sheet 6종 생성 확인

### 8.2 흑백 사진 이슈 원인 분석 및 v5 보완 결과

`full10000_siglip2_gpu_20260529/contact_sheets/keep_photo_primary.jpg`에서 흑백 사진이 대량으로 보인 직접 원인은 SigLIP2가 흑백 사진을 non-photo로 보지 않기 때문이다. SigLIP2 prompt classifier의 `camera_photo` prompt는 "실제 카메라로 촬영된 사진인지"를 주로 분리하므로, 흑백이라도 실제 사진처럼 보이면 `camera_photo` 또는 낮은 margin의 non-camera group으로 남는다. 기존 설계도 prompt classifier가 review/reject를 promote하지 않고 keep만 demote하는 구조라서, 흑백 사진을 별도로 제외하는 정책은 없었다.

v4에서 같은 문제가 덜 보였던 이유는 두 가지다.

1. v4도 흑백을 완전히 제거한 것은 아니었다. v5 기준 monochrome policy를 v4 `keep_photo_primary`에 적용하면 445장이 걸린다.
2. contact sheet는 `nonphoto_score` 내림차순으로 상위 샘플을 보여준다. v4 keep에는 컬러지만 배경/그래픽성이 큰 후보가 많이 남아 있어 상위 sheet에서 흑백 비중이 상대적으로 덜 두드러졌다. SigLIP2가 그런 컬러 non-camera 후보를 대량 review로 demote하면서, 남은 keep 상위 sheet에 흑백/저채도 사진이 더 많이 노출됐다.

수정은 prompt threshold를 더 세게 조이는 방식이 아니라 deterministic color gate를 추가하는 방식으로 했다. prompt classifier 입장에서는 흑백 사진도 "camera photo"일 수 있으므로, prompt만으로 흑백 제외를 강제하면 카메라 사진 판별과 색상 정책이 섞인다. v5에서는 다음 조건 중 하나를 만족하면 `review_monochrome_desaturated`로 이동한다.

- `saturation_mean <= 0.03`
- `saturation_mean <= 0.10`, `saturation_std <= 0.12`, `quant_unique_count <= 256`
- `saturation_mean <= 0.15`, `saturation_std <= 0.15`, `quant_unique_count <= 256`

기존 GPU run의 `prompt_*` score는 이미 로컬로 복사되어 있으므로, prompt 모델을 다시 로드하지 않고 `--reuse_prompt_classifier_columns 1`로 full 10K를 재생성했다. 즉 새 산출물은 동일한 SigLIP2 GPU score에 v5 monochrome gate를 추가 적용한 결과다.

재생성 command:

```bash
/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/python src/scripts/audit_sstk_media_type.py --input_parquet data/SSTK/Full_10000/artifacts/media_type_audit/full10000_siglip2_gpu_20260529/filtered_sstk_media_audited.parquet --image_dir data/SSTK/Full_10000/images --output_dir data/SSTK/Full_10000/artifacts/media_type_audit/full10000_siglip2_gpu_20260529_no_monochrome_v5 --write_contact_sheets 1 --contact_sheet_max_per_decision 80 --prompt_classifier siglip2 --reuse_prompt_classifier_columns 1 --prompt_classifier_device cuda --prompt_classifier_batch_size 64 --status_json data/SSTK/Full_10000/artifacts/media_type_audit/full10000_siglip2_gpu_20260529_no_monochrome_v5/status.json --progress_interval 500
```

v5 corrected full 10K decision count:

| Decision | Count | 기존 SigLIP2 run 대비 |
| --- | ---: | ---: |
| `keep_photo_primary` | 3,578 | -251 |
| `review_monochrome_desaturated` | 517 | +517 |
| `review_background_copyspace` | 2,532 | -73 |
| `review_graphic_ambiguous` | 1,854 | -186 |
| `review_packshot_isolated` | 828 | 0 |
| `review_studio_low_context` | 322 | -7 |
| `reject_nonphoto` | 369 | 0 |

검증 결과:

- 10,000 rows 처리 완료
- prompt columns reused, classified 10,000, errors 0
- missing images 0
- photo 3,578 + review 6,053 + reject 369 = 10,000
- 기존 SigLIP2 keep에서 v5 monochrome policy에 걸리던 후보 251장 제거
- 새 `filtered_sstk_photo_primary.parquet` 내 v5 monochrome policy hit 0장
- 새 `keep_photo_primary.jpg` 상위 80장 내 v5 monochrome policy hit 0장

v5 산출물:

| 산출물 | 로컬 경로 |
| --- | --- |
| Run directory | `data/SSTK/Full_10000/artifacts/media_type_audit/full10000_siglip2_gpu_20260529_no_monochrome_v5/` |
| Summary JSON | `data/SSTK/Full_10000/artifacts/media_type_audit/full10000_siglip2_gpu_20260529_no_monochrome_v5/media_type_audit_summary.json` |
| Photo-primary split | `data/SSTK/Full_10000/artifacts/media_type_audit/full10000_siglip2_gpu_20260529_no_monochrome_v5/filtered_sstk_photo_primary.parquet` |
| Review split | `data/SSTK/Full_10000/artifacts/media_type_audit/full10000_siglip2_gpu_20260529_no_monochrome_v5/filtered_sstk_media_review.parquet` |
| Monochrome review | `data/SSTK/Full_10000/artifacts/media_type_audit/full10000_siglip2_gpu_20260529_no_monochrome_v5/contact_sheets/review_monochrome_desaturated.jpg` |
| Keep contact sheet | `data/SSTK/Full_10000/artifacts/media_type_audit/full10000_siglip2_gpu_20260529_no_monochrome_v5/contact_sheets/keep_photo_primary.jpg` |
| Local validation summary | `artifacts/research_agent_runs/full10000_siglip2_gpu_20260529_no_monochrome_v5_validation.json` |

## 9. 운영 권장안

main MobileCropNet 학습 pool은 v5 corrected SigLIP2 run의 `filtered_sstk_photo_primary.parquet`를 우선 사용한다. GPU prompt classifier를 쓰지 못하는 환경에서는 v5 deterministic audit 또는 Phase A pre-sampling gate를 켠 재생성 결과를 fallback으로 사용한다.

```text
data/SSTK/Full_10000/artifacts/media_type_audit/full10000_siglip2_gpu_20260529_no_monochrome_v5/filtered_sstk_photo_primary.parquet
```

previous SigLIP2 path는 흑백/저채도 제외 전 결과이므로 더 이상 main 학습 pool로 권장하지 않는다.

```text
data/SSTK/Full_10000/artifacts/media_type_audit/full10000_siglip2_gpu_20260529/filtered_sstk_photo_primary.parquet
```

권장 단계:

1. 빠른 재생성: `--filter_media_type_presampling_gate 1`로 Phase A에서 obvious non-photo/background/packshot metadata 후보를 먼저 제거한다.
2. 이미지 기반 QC: `--run_media_type_audit 1`로 deterministic v5 audit을 수행하고 `--media_type_audit_use_photo_primary 1`로 downstream을 photo-primary split으로 전환한다.
3. 고신뢰 보강: GPU가 있으면 `--media_type_audit_prompt_classifier siglip2`를 켜서 prompt classifier demotion을 추가한다.
4. 비교 실험: `photo_primary only`, `photo_primary + packshot`, `original 10K` 세 pool의 downstream 성능과 qualitative crop 결과를 비교한다.

## 10. 한계

- Phase A pre-sampling gate는 metadata-only라서 metadata가 누락된 photorealistic render나 단색 background를 완전히 잡지는 못한다.
- SigLIP2/CLIP prompt score는 calibrated probability가 아니므로 threshold는 contact sheet 검수로 calibration해야 한다.
- SigLIP2 full 10K GPU run은 수행 완료했지만, 현재 threshold는 보수적 demotion heuristic이다. downstream MobileCropNet 성능과 qualitative crop review로 primary pool 축소가 과도하지 않은지 추가 확인해야 한다.
- v5 monochrome gate는 색상 정책이다. 흑백 사진이 composition 학습에 필요한 별도 domain이라면 삭제가 아니라 review split에서 별도 실험 pool로 운용해야 한다.
- review pool은 삭제 대상이 아니다. packshot, studio, background는 별도 product/domain robustness 실험에 쓸 수 있다.

## 11. 참고 문헌과 출처

- Radford et al., "Learning Transferable Visual Models From Natural Language Supervision", arXiv 2021. <https://arxiv.org/abs/2103.00020>
- OpenAI, "CLIP: Connecting text and images". <https://openai.com/index/clip/>
- Tschannen et al., "SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features", arXiv 2025. <https://arxiv.org/abs/2502.14786>
- Google, `google/siglip2-base-patch16-224` Hugging Face model page. <https://huggingface.co/google/siglip2-base-patch16-224>
- Schuhmann et al., "LAION-5B", NeurIPS Datasets and Benchmarks 2022. <https://papers.nips.cc/paper/2022/hash/a1859debfb3b59d094f3504d5ebb6c25-Abstract-Datasets_and_Benchmarks.html>
- LAION-Aesthetics. <https://laion.ai/blog/laion-aesthetics/>
- Gadre et al., "DataComp", NeurIPS Datasets and Benchmarks 2023. <https://papers.nips.cc/paper_files/paper/2023/hash/56332d41d55ad7ad8024aac625881be7-Abstract-Datasets_and_Benchmarks.html>
- Automatic Classification of Photographs and Graphics. <https://www.cecs.uci.edu/~papers/icme06/pdfs/0000973.pdf>
- Wang et al., "CNN-Generated Images Are Surprisingly Easy to Spot... for Now", CVPR 2020. <https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_CNN-Generated_Images_Are_Surprisingly_Easy_to_Spot..._for_Now_CVPR_2020_paper.html>
- Yan et al., "A Sanity Check for AI-generated Image Detection", arXiv 2024. <https://arxiv.org/abs/2406.19435>
