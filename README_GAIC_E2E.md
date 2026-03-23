# GAIC End-to-End 실행 가이드

이 문서는 [README_SSTK_Curation.md](/media/jyju25/T7_4TB_JY/Projects_26/Sources/ImageCropping/README_SSTK_Curation.md) 와 [run_phaseA_to_teacher_e2e.sh](/media/jyju25/T7_4TB_JY/Projects_26/Sources/ImageCropping/src/scripts/run_phaseA_to_teacher_e2e.sh) 기반 파이프라인을 `GAIC` 이미지셋에 맞게 재사용하는 운영 가이드입니다.

핵심 차이점은 세 가지입니다.

- GAIC는 `curated pool` 선별 단계가 없습니다. `data/Publics/GAIC/images` 아래의 모든 이미지를 curated pool로 간주합니다.
- GAIC는 태그/캡션 메타데이터가 없으므로 메타데이터 의존적인 `C1 + real-expensive teacher` 경로를 기본 비활성화합니다.
- 기존 파이프라인이 flat `image_dir`를 기대하므로, GAIC 이미지를 심볼릭 링크 기반의 평탄화된 `images/` 디렉토리로 준비한 뒤 기존 e2e 스크립트를 그대로 호출합니다.

## 1. 추가된 파일

- 래퍼 실행 스크립트: [src/scripts/run_gaic_to_teacher_e2e.sh](/media/jyju25/T7_4TB_JY/Projects_26/Sources/ImageCropping/src/scripts/run_gaic_to_teacher_e2e.sh)
- GAIC 준비 스크립트: [src/scripts/prepare_gaic_curated_dataset.py](/media/jyju25/T7_4TB_JY/Projects_26/Sources/ImageCropping/src/scripts/prepare_gaic_curated_dataset.py)
- GAIC synthetic caption 스크립트: [src/scripts/generate_gaic_captions.py](/media/jyju25/T7_4TB_JY/Projects_26/Sources/ImageCropping/src/scripts/generate_gaic_captions.py)
- 산출물 검증 스크립트: [src/scripts/validate_gaic_e2e_outputs.py](/media/jyju25/T7_4TB_JY/Projects_26/Sources/ImageCropping/src/scripts/validate_gaic_e2e_outputs.py)

## 2. GAIC 전용 설계

### 2.1 Phase A 대체

SSTK의 `filter` 단계 대신, GAIC에서는 준비 스크립트가 아래를 수행합니다.

- `data/Publics/GAIC/images` 를 재귀 스캔
- 모든 이미지를 flat `data/GAIC/.../images/<image_id>.<ext>` 형태로 링크
- 기존 파이프라인이 읽을 수 있는 pseudo-filtered parquet 생성
- `tar_name` 은 실제 tar가 아니라 메모리 사용량을 낮추기 위한 pseudo chunk 값(`gaic_chunk_00000.local`)으로 채움

즉, GAIC에서는 “선별”이 아니라 “전체 이미지를 curated pool 형식으로 정규화”하는 단계입니다.

### 2.2 메타데이터 부재 처리

GAIC는 이미지 외 메타데이터가 없으므로 기본값은 다음과 같습니다.

- `--run_c1 0`
- `--use_real_expensive 0`
- `tags=""`, `caption=""`, `super_cat=""` 로 parquet 작성
- subject routing은 태그 대신 `C2/C3/C4/C5` 신호를 우선 사용
- Stage-10 라벨러는 기본적으로 `heuristic` backend를 사용
- `run_c1=0` 또는 `use_real_expensive=0` 인 경우 scorer는 더 이상 `proxy expensive`를 쓰지 않습니다. 이때 `expensive=0`, `expensive_source=disabled`, `A_macro=na` 로 기록되며 최종 `score_rank` 에서 A 축은 제외됩니다.

필요 시 Qwen 기반 VLM으로 바꿀 수는 있지만, 그 경우 별도 런타임/가중치 준비가 되어 있어야 합니다.

### 2.3 GAIC Synthetic Caption 경로

GAIC에서 `run_c1=1`, `use_real_expensive=1` 을 실제로 쓰려면 텍스트 메타데이터가 필요하므로 wrapper가 synthetic caption을 먼저 생성한 뒤 parquet의 `caption` 컬럼에 병합합니다.

- 구현 스크립트: [src/scripts/generate_gaic_captions.py](/media/jyju25/T7_4TB_JY/Projects_26/Sources/ImageCropping/src/scripts/generate_gaic_captions.py)
- parquet 병합 스크립트: [src/scripts/prepare_gaic_curated_dataset.py](/media/jyju25/T7_4TB_JY/Projects_26/Sources/ImageCropping/src/scripts/prepare_gaic_curated_dataset.py)
- C1 텍스트 입력은 `tags` 와 분리되어 `caption -> tags` 순서로 우선 사용합니다.

현재 코드에 반영한 기본 preset:

- 로컬 8GB GPU: `local_efficient` -> `Salesforce/blip-image-captioning-base`
- 서버 A100-80GB: `server_quality` -> `Salesforce/blip-image-captioning-large`

조사 결과 메모:

- `Florence-2-base/large-ft` 는 caption task와 더 강한 성능 지표가 있는 모델이지만, 현재 venv에서는 `flash_attn` 의존성 때문에 바로 구동되지 않았습니다. 의존성이 준비된 서버에서는 `server_quality_florence` preset으로 선택할 수 있게 열어 두었습니다.
- `Qwen2.5-VL` 계열은 성능 상위권 후보지만 공식 모델 카드가 최신 source-build `transformers` 를 권장하므로, 현재 repo 런타임과의 안정성을 우선해 기본 preset에서는 제외했습니다.

## 3. 실행 전제

이 로컬 PC에서 모든 터미널 실행은 아래 가상환경을 활성화한 뒤 수행합니다.

```bash
source /media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/activate
```

기본 입력 경로:

- 이미지 루트: `data/Publics/GAIC/images`
- 기본 출력 루트: `data/GAIC/All`

## 4. 권장 실행

### 4.1 전체 실행

```bash
# 서버에서 사용할 venv를 먼저 활성화하거나,
# 필요하면 아래 명령에 --venv_path /your/server/venv/bin/activate 를 추가
# source /your/server/venv/bin/activate

GPU_IDS=0,1,2
N_WORKERS=$(awk -F',' '{print NF}' <<< "${GPU_IDS}")

RUN_TAG=gaic_260323_r0
DATANAME=All

bash src/scripts/run_gaic_to_teacher_e2e.sh \
  --server_mode 1 \
  --data_dir data/GAIC/${DATANAME} \
  --image_root data/Publics/GAIC/images \
  --run_tag ${RUN_TAG} \
  --skip_existing 1 \
  --run_c1 1 \
  --use_real_expensive 1 \
  --gaic_caption_preset server_quality \
  --run_c4 0 \
  --run_c6 1 \
  --run_vlm_teacher 0 \
  --run_training_labels 1 \
  --run_detailed_report 1 \
  --enable_public_teacher_proposals 1 \
  --public_teacher_setup 1 \
  --public_teacher_download_weights 1 \
  --public_teachers gaic,cacnet,cgs \
  --extract_gpu_ids ${GPU_IDS} \
  --num_workers ${N_WORKERS} \
  --teacher_gpu_ids ${GPU_IDS} \
  --teacher_num_workers ${N_WORKERS} \
  | tee src/scripts/logs/run_gaic_to_teacher_e2e_${DATANAME}_${RUN_TAG}.log
```

위 명령의 실제 동작:

- GAIC 전체 이미지를 curated pool로 준비
- `run_phaseA_to_teacher_e2e.sh` 를 `run_filter=0` 으로 호출
- 품질우선(`quality_first`) precompute로 `C2/C3/C5/C6` 수행
- `C1`, `C4`, `VLM teacher` 는 수행하지 않음
- training label export 및 GAIC-like export 생성
- 마지막에 자동 검증 실행

중요:

- 이 기본 템플릿에서는 `A_macro` 가 caption/proxy 없이 계산되지 않습니다.
- 즉 report에는 `expensive_source=disabled`, `A=na` 가 보이고 rank fusion은 `S/C/T` 축만 사용합니다.

### 4.1a Public Teacher만 기존 결과에 재반영

기존 precompute를 재사용하고 `public teacher proposal -> candidates -> teacher -> report/training labels` 만 다시 만들고 싶을 때 사용합니다.

```bash
# source /your/server/venv/bin/activate

RUN_TAG=gaic_260320_r0_public_teacher
DATANAME=All

bash src/scripts/run_gaic_to_teacher_e2e.sh \
  --server_mode 1 \
  --data_dir data/GAIC/${DATANAME} \
  --image_root data/Publics/GAIC/images \
  --run_tag ${RUN_TAG} \
  --skip_existing 0 \
  --run_c1 0 \
  --use_real_expensive 0 \
  --run_c2 0 \
  --run_c3 0 \
  --run_c4 0 \
  --run_c5 0 \
  --run_c6 0 \
  --run_merge 0 \
  --run_subject_routing 0 \
  --run_candidates 1 \
  --run_teacher 1 \
  --run_training_labels 1 \
  --run_detailed_report 1 \
  --run_vlm_teacher 0 \
  --enable_public_teacher_proposals 1 \
  --public_teacher_setup 1 \
  --public_teacher_download_weights 1 \
  --public_teachers gaic,cacnet,cgs
```

### 4.1b Public Teacher를 처음부터 반영

GAIC 전체를 처음부터 다시 만들되 public teacher proposal injection까지 포함하고 싶을 때 사용합니다.

```bash
# source /your/server/venv/bin/activate

GPU_IDS=0,1,2
N_WORKERS=$(awk -F',' '{print NF}' <<< "${GPU_IDS}")

RUN_TAG=gaic_260320_r0_public_full
DATANAME=All

bash src/scripts/run_gaic_to_teacher_e2e.sh \
  --server_mode 1 \
  --data_dir data/GAIC/${DATANAME} \
  --image_root data/Publics/GAIC/images \
  --run_tag ${RUN_TAG} \
  --run_c1 0 \
  --run_c4 0 \
  --run_c6 1 \
  --run_vlm_teacher 0 \
  --run_training_labels 1 \
  --run_detailed_report 1 \
  --enable_public_teacher_proposals 1 \
  --public_teacher_setup 1 \
  --public_teacher_download_weights 1 \
  --public_teachers gaic,cacnet,cgs \
  --extract_gpu_ids ${GPU_IDS} \
  --num_workers ${N_WORKERS} \
  --teacher_gpu_ids ${GPU_IDS} \
  --teacher_num_workers ${N_WORKERS}
```

### 4.1c C1 + Real-Expensive를 synthetic caption과 함께 실행

GAIC에서 실제 `A_macro` 를 활성화하려면 이 템플릿을 사용합니다. wrapper가 caption을 먼저 생성하고 parquet `caption` 컬럼에 병합한 뒤 C1/teacher real-expensive를 수행합니다.

서버 기본 권장:

```bash
# source /your/server/venv/bin/activate

RUN_TAG=gaic_260320_r0_realexp
DATANAME=All

bash src/scripts/run_gaic_to_teacher_e2e.sh \
  --server_mode 1 \
  --data_dir data/GAIC/${DATANAME} \
  --image_root data/Publics/GAIC/images \
  --run_tag ${RUN_TAG} \
  --run_c1 1 \
  --use_real_expensive 1 \
  --gaic_caption_preset server_quality \
  --run_c4 0 \
  --run_c6 1 \
  --run_vlm_teacher 0 \
  --run_training_labels 1 \
  --run_detailed_report 1
```

의존성이 준비된 서버에서 Florence-2를 쓰고 싶으면:

```bash
bash src/scripts/run_gaic_to_teacher_e2e.sh \
  --server_mode 1 \
  --data_dir data/GAIC/All \
  --image_root data/Publics/GAIC/images \
  --run_tag gaic_realexp_florence \
  --run_c1 1 \
  --use_real_expensive 1 \
  --gaic_caption_preset server_quality_florence
```

로컬 8GB 검증용 최소 템플릿:

```bash
source /media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/activate

bash src/scripts/run_gaic_to_teacher_e2e.sh \
  --data_dir data/GAIC/CapSmoke1 \
  --image_root data/GAIC/Smoke4/images \
  --bucket gaic_capsmoke1 \
  --run_tag capsmoke1_real \
  --prepare_max_images 1 \
  --skip_existing 0 \
  --run_c1 1 \
  --use_real_expensive 1 \
  --extract_priority high_efficiency \
  --c5_priority high_efficiency \
  --gaic_caption_preset local_efficient \
  --run_c4 0 \
  --run_c6 0 \
  --run_vlm_teacher 0 \
  --run_training_labels 1 \
  --run_detailed_report 1 \
  --align_device cpu \
  --aesthetic_device cpu \
  --exp_batch_size 1
```

VLM backend 관련 용어:

- `--vlm_backend heuristic`
  실제 vision-language model을 로드하지 않고, 이미 계산된 `teacher_scores` 와 rule-based 후처리를 이용해 `selected_topk`, `why_tags`, 설명문을 생성하는 경량 Stage-10 backend입니다.
- `--vlm_fallback_backend none`
  primary backend 실패 시 대체 backend를 쓰지 않겠다는 의미입니다.
- 따라서 `--vlm_backend heuristic --vlm_fallback_backend none` 조합은
  "Qwen 같은 VLM 없이 heuristic 로직만으로 Stage-10 결과를 생성"한다는 뜻입니다.

### 4.2 준비 단계만 먼저 실행

```bash
source /media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/activate

bash src/scripts/run_gaic_to_teacher_e2e.sh \
  --data_dir data/GAIC/All \
  --image_root data/Publics/GAIC/images \
  --prepare_only 1
```

이 경우 아래만 생성됩니다.

- `data/GAIC/All/filtered_gaic_all.parquet`
- `data/GAIC/All/images/`
- `data/GAIC/All/gaic_prepare_summary.json`
- `data/GAIC/All/gaic_reference_available.json`

### 4.3 스모크 테스트

`prepare_max_images` 는 준비 단계부터 이미지 수를 제한하므로 실제 end-to-end smoke 검증에 적합합니다.

```bash
source /media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/activate

bash src/scripts/run_gaic_to_teacher_e2e.sh \
  --data_dir data/GAIC/Smoke32 \
  --image_root data/Publics/GAIC/images \
  --run_tag smoke32 \
  --prepare_max_images 32 \
  --run_vlm_teacher 1 \
  --vlm_backend heuristic \
  --run_training_labels 1
```

## 5. 자주 쓰는 옵션

- `--prepare_max_images N`: 준비 단계부터 N장만 사용
- `--pseudo_tar_chunk_size N`: pseudo `tar_name` chunk 크기. 기본 `256`
- `--link_mode symlink|hardlink|copy`: flat image dir 생성 방식
- `--run_c6 1`: C6 gaze/headpose 경로를 켜고 싶을 때
- `--gaic_generate_captions 0|1`: synthetic caption 생성 강제 on/off. 기본은 `run_c1=1`일 때 자동 on
- `--gaic_caption_preset local_efficient|server_quality|server_quality_florence`: GAIC caption 모델 preset
- `--gaic_caption_backend blip|florence2`: preset 대신 backend 직접 지정
- `--gaic_caption_model_id MODEL_ID`: caption model override
- `--gaic_caption_prompt STR`: BLIP conditional prompt 또는 Florence task prompt override
- `--safe_leftover_policy keep_negative|ignore|promote_soft_positive`: safe high-score leftover를 기본 negative로 둘지, ignore로 분리할지, soft positive로 승격할지 선택
- `--vlm_backend qwen25_vl`: Qwen 기반 Stage-10 사용 시
- `--vlm_backend heuristic`: teacher score 기반 heuristic Stage-10 사용 시
- `--vlm_fallback_backend none`: backend 실패 시 fallback 없이 바로 종료/skip 하려는 경우
- `--vlm_fallback_backend heuristic`: Qwen 실패 시 heuristic으로 자동 대체하려는 경우
- `--skip_existing 1`: 기존 산출물이 있으면 재사용

추가 옵션은 대부분 그대로 내부 SSTK e2e 스크립트로 전달됩니다. 예를 들어 GPU 샤딩을 쓰려면 다음처럼 넘기면 됩니다.

```bash
source /media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/activate

bash src/scripts/run_gaic_to_teacher_e2e.sh \
  --data_dir data/GAIC/All \
  --image_root data/Publics/GAIC/images \
  --run_tag gaic_mgpu \
  --extract_gpu_ids 0,1 \
  --num_workers 2 \
  --teacher_gpu_ids 0,1 \
  --teacher_num_workers 2
```

## 5.1 VLM Teacher 재실행 템플릿

추후 Stage-10만 별도로 다시 돌리고 싶으면, 기존 precompute/candidate/teacher 결과를 재사용하고 `run_vlm_teacher` 만 켜면 됩니다.

### A. heuristic backend로 Stage-10만 재실행

가장 가볍고 안정적입니다. 별도 Qwen 런타임이 필요 없습니다.

```bash
# source /your/server/venv/bin/activate

RUN_TAG=gaic_260320_r0
DATANAME=All

bash src/scripts/run_gaic_to_teacher_e2e.sh \
  --server_mode 1 \
  --data_dir data/GAIC/${DATANAME} \
  --image_root data/Publics/GAIC/images \
  --run_tag ${RUN_TAG} \
  --skip_existing 1 \
  --run_candidates 0 \
  --run_teacher 0 \
  --run_training_labels 0 \
  --run_detailed_report 0 \
  --run_vlm_teacher 1 \
  --vlm_backend heuristic \
  --vlm_fallback_backend none
```

이 템플릿은 기존 `teacher_scores_ar_${RUN_TAG}.jsonl` 를 입력으로 사용해 Stage-10 산출물만 다시 만듭니다.

### B. Qwen backend로 Stage-10만 재실행

실제 VLM으로 crop 후보를 다시 고르게 하고 싶을 때 사용합니다.

```bash
# source /your/qwen_runtime_venv/bin/activate

RUN_TAG=gaic_260320_r0
DATANAME=All
GPU_IDS=0,1
N_WORKERS=$(awk -F',' '{print NF}' <<< "${GPU_IDS}")

bash src/scripts/run_gaic_to_teacher_e2e.sh \
  --server_mode 1 \
  --data_dir data/GAIC/${DATANAME} \
  --image_root data/Publics/GAIC/images \
  --run_tag ${RUN_TAG} \
  --skip_existing 1 \
  --run_candidates 0 \
  --run_teacher 0 \
  --run_training_labels 0 \
  --run_detailed_report 0 \
  --run_vlm_teacher 1 \
  --vlm_backend qwen25_vl \
  --vlm_fallback_backend heuristic \
  --vlm_model_id Qwen/Qwen3-VL-4B-Instruct \
  --vlm_multi_gpu 1 \
  --vlm_gpu_ids ${GPU_IDS} \
  --vlm_num_workers ${N_WORKERS}
```

이 경우 primary backend는 `qwen25_vl` 이고, 실패 시 `heuristic` 으로 fallback 합니다.

### C. Stage-10 재실행 후 validation까지 다시 수행

Stage-10 결과만 다시 만들고 wrapper validator까지 다시 돌리고 싶으면 아래처럼 실행하면 됩니다.

```bash
# source /your/server/venv/bin/activate

RUN_TAG=gaic_260320_r0
DATANAME=All

bash src/scripts/run_gaic_to_teacher_e2e.sh \
  --server_mode 1 \
  --data_dir data/GAIC/${DATANAME} \
  --image_root data/Publics/GAIC/images \
  --run_tag ${RUN_TAG} \
  --skip_existing 1 \
  --run_candidates 0 \
  --run_teacher 0 \
  --run_training_labels 0 \
  --run_detailed_report 0 \
  --run_vlm_teacher 1 \
  --vlm_backend heuristic \
  --vlm_fallback_backend none
```

주의:

- 현재 training label export는 `teacher_scores` 기반이라 Stage-10 결과를 직접 소비하지 않습니다.
- 즉 Stage-10 재실행은 주로 `crop_label_v1`, `meta_norm_v1`, Stage-10 summary 갱신 목적입니다.

## 5.2 Training Label Variant 재생성

safe high-score leftover는

- 최종 chosen보다 점수는 높지만
- `selected_topk` diverse positive set에는 들어가지 않았고
- severe reject도 아닌

후보를 뜻합니다.

기본 `ignore` 정책에서는 이 후보들이 `ignored_candidates`로 분리되고, negative annotation에서 제외됩니다. `keep_negative`는 이전 동작을 재현할 때만 명시적으로 사용하는 보수적 호환 옵션입니다. 분석/학습 실험을 위해 아래 variant를 만들 수 있습니다.

- `keep_negative`: 이 후보를 기존처럼 `candidate_pool`의 `negative/near_negative`로 유지
- `ignore`: 이 후보를 `ignored_candidates`로 분리하고 negative annotation에서 제외
- `promote_soft_positive`: 이 후보를 `soft_positive`로 승격하고 `matching_targets`에 포함

기존 `teacher_scores`를 재사용해 training label만 다시 뽑는 명령 템플릿:

```bash
source /media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/activate

RUN_TAG=gaic_260320_r0
BASE=data/GAIC/All/artifacts/training_labels
TEACHER_JSONL=data/GAIC/All/artifacts/teacher/scores/teacher_scores_ar_${RUN_TAG}.jsonl
IMAGE_ROOT=data/GAIC/All/images
GAIC_REF=data/GAIC/All/gaic_reference_available.json

for POLICY in ignore promote_soft_positive; do
  if [ "${POLICY}" = "ignore" ]; then
    OUT_DIR=${BASE}/${RUN_TAG}_leftover_ignore
  else
    OUT_DIR=${BASE}/${RUN_TAG}_leftover_softpos
  fi

  python src/scripts/build_finalscore_training_data.py \
    --teacher_scores_jsonl "${TEACHER_JSONL}" \
    --out_dir "${OUT_DIR}" \
    --image_root "${IMAGE_ROOT}" \
    --safe_leftover_policy "${POLICY}" \
    --strict_validation 1 \
    --report_examples 8

  python src/scripts/convert_sstk_detr_labels_to_coco.py \
    --canonical_jsonl "${OUT_DIR}/train_conditional_detr_canonical.jsonl" \
    --batch_jsonl "${OUT_DIR}/train_conditional_detr_batch.jsonl" \
    --out_dir "${OUT_DIR}/coco"

  python src/scripts/convert_sstk_detr_batch_to_gaic_like.py \
    --batch_jsonl "${OUT_DIR}/train_conditional_detr_batch.jsonl" \
    --gaic_reference_json "${GAIC_REF}" \
    --out_json "${OUT_DIR}/coco/instances_conditional_detr_batch_gaic_like.json" \
    --out_summary_json "${OUT_DIR}/coco/gaic_like_conversion_summary.json" \
    --out_guide_md "${OUT_DIR}/coco/GAIC_INSTANCES_TRAIN_FORMAT_KO.md"
done
```

전체 e2e wrapper를 재사용하고 싶으면 `--safe_leftover_policy`와 `--training_labels_dir`를 같이 넘기면 됩니다. 예:

```bash
source /media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/activate

bash src/scripts/run_gaic_to_teacher_e2e.sh \
  --server_mode 1 \
  --data_dir data/GAIC/All \
  --image_root data/Publics/GAIC/images \
  --run_tag gaic_260320_r0 \
  --skip_existing 1 \
  --run_candidates 0 \
  --run_teacher 0 \
  --run_vlm_teacher 0 \
  --run_detailed_report 0 \
  --run_training_labels 1 \
  --safe_leftover_policy ignore \
  --training_labels_dir data/GAIC/All/artifacts/training_labels/gaic_260320_r0_leftover_ignore
```

2026-03-20 기준 실제 생성/검증 완료된 경로:

- 기본값(`ignore`): [data/GAIC/All/artifacts/training_labels/gaic_260320_r0](/media/jyju25/T7_4TB_JY/Projects_26/Sources/ImageCropping/data/GAIC/All/artifacts/training_labels/gaic_260320_r0)
- keep-negative variant: [data/GAIC/All/artifacts/training_labels/gaic_260320_r0_leftover_keepneg](/media/jyju25/T7_4TB_JY/Projects_26/Sources/ImageCropping/data/GAIC/All/artifacts/training_labels/gaic_260320_r0_leftover_keepneg)
- ignore variant: [data/GAIC/All/artifacts/training_labels/gaic_260320_r0_leftover_ignore](/media/jyju25/T7_4TB_JY/Projects_26/Sources/ImageCropping/data/GAIC/All/artifacts/training_labels/gaic_260320_r0_leftover_ignore)
- soft-positive variant: [data/GAIC/All/artifacts/training_labels/gaic_260320_r0_leftover_softpos](/media/jyju25/T7_4TB_JY/Projects_26/Sources/ImageCropping/data/GAIC/All/artifacts/training_labels/gaic_260320_r0_leftover_softpos)

실제 차이 요약:

- `keep_negative`: `rows_with_higher_scored_safe_pool_candidates=3576`
- `ignore`(기본값): 위 지표가 `0`으로 내려가고, GAIC-like negative annotation이 `152245 -> 115201`로 감소
- `promote_soft_positive`: 위 지표가 `0`으로 내려가고, GAIC-like positive annotation이 `32622 -> 69666`으로 증가

## 6. 산출물

기본 출력 루트가 `data/GAIC/All`, `run_tag=gaic_full` 이면 주요 파일은 아래와 같습니다.

- parquet: `data/GAIC/All/filtered_gaic_all.parquet`
- precompute raw: `data/GAIC/All/artifacts/precompute/feats_c2c3c5_v2_strict_raw.jsonl`
- routed feats: `data/GAIC/All/artifacts/precompute/feats_c2c3c5_v2_strict_enriched_routed.jsonl`
- candidates: `data/GAIC/All/artifacts/candidates/candidates_ar_gaic_full.jsonl`
- teacher: `data/GAIC/All/artifacts/teacher/scores/teacher_scores_ar_gaic_full.jsonl`
- vlm labels: `data/GAIC/All/artifacts/vlm_teacher/labels/crop_label_v1_gaic_full.jsonl`
- training validation: `data/GAIC/All/artifacts/training_labels/gaic_full/validation_summary.json`
- wrapper validation: `data/GAIC/All/artifacts/validation/gaic_e2e_validation_gaic_full.json`

## 7. 검증 정책

래퍼는 마지막에 [src/scripts/validate_gaic_e2e_outputs.py](/media/jyju25/T7_4TB_JY/Projects_26/Sources/ImageCropping/src/scripts/validate_gaic_e2e_outputs.py) 를 자동 실행합니다.

검증 내용:

- manifest parquet 무결성
- precompute / routed / candidates / teacher 의 `image_id` 집합 검증
- VLM meta/label/summary 정합성
- training label validation 요약 상태
- GAIC-like export validation 상태

subset smoke 실행에서는 `candidate/teacher/VLM` 이 manifest 전체보다 적을 수 있는데, 이 경우 validator는 warning으로만 기록합니다.

## 8. 운영 메모

- GAIC에는 태그/캡션이 없으므로 `C1 real-expensive` 경로를 기본값으로 켜는 것은 권장하지 않습니다.
- flat image dir 는 원본 이미지를 복사하지 않고 기본적으로 심볼릭 링크를 사용합니다.
- 현재 로컬 볼륨처럼 심볼릭 링크를 지원하지 않는 파일시스템에서는 준비 스크립트가 자동으로 `copy` 로 폴백합니다.
- pseudo `tar_name` 은 실제 tar를 의미하지 않으며, 기존 precompute 로더의 메모리 사용량을 제어하기 위한 chunk key입니다.
- `run_detailed_report` 는 SSTK 중심 보고서 구조를 그대로 쓰므로 기본 비활성화했습니다.

## 9. 권장 기본값 요약

- 전체 이미지 사용
- `run_filter=0`
- `run_c1=0`
- `use_real_expensive=0`
- `vlm_backend=heuristic`
- `run_training_labels=1`
- `run_detailed_report=0`
