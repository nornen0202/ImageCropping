# Test Images End-to-End 실행 가이드

이 문서는 [README_GAIC_E2E.md](/media/jyju25/T7_4TB_JY/Projects_26/Sources/ImageCropping/README_GAIC_E2E.md), [README_SSTK_Curation.md](/media/jyju25/T7_4TB_JY/Projects_26/Sources/ImageCropping/README_SSTK_Curation.md), [run_phaseA_to_teacher_e2e.sh](/media/jyju25/T7_4TB_JY/Projects_26/Sources/ImageCropping/src/scripts/run_phaseA_to_teacher_e2e.sh) 기반 파이프라인을 `data/test_images` 용으로 재사용하는 운영 가이드입니다.

핵심 차이점은 세 가지입니다.

- 테스트 이미지는 `filter` 단계 없이 `data/test_images` 아래 모든 이미지를 curated pool로 간주합니다.
- 테스트 이미지에는 GT(GAIC annotation)가 없으므로 benchmark / saliency A-B 평가 단계는 수행하지 않습니다.
- `vlm-teacher` 단계는 제외하고, 나머지 주요 단계(`C1~C7`, merge, subject routing, candidates, teacher, training labels, detailed report)는 모두 활성화합니다.

## 1. 추가된 파일

- 테스트 이미지 래퍼 실행 스크립트: [src/scripts/run_test_images_to_teacher_e2e.sh](/media/jyju25/T7_4TB_JY/Projects_26/Sources/ImageCropping/src/scripts/run_test_images_to_teacher_e2e.sh)
- 준비/manifest 재사용 스크립트: [src/scripts/prepare_gaic_curated_dataset.py](/media/jyju25/T7_4TB_JY/Projects_26/Sources/ImageCropping/src/scripts/prepare_gaic_curated_dataset.py)
- 산출물 검증 스크립트: [src/scripts/validate_gaic_e2e_outputs.py](/media/jyju25/T7_4TB_JY/Projects_26/Sources/ImageCropping/src/scripts/validate_gaic_e2e_outputs.py)

## 2. 테스트 이미지 전용 설계

### 2.1 Phase A 대체

테스트 이미지에서는 `prepare_gaic_curated_dataset.py` 가 아래를 수행합니다.

- `data/test_images` 를 스캔
- flat `data/TestImages/All/images/<image_id>.<ext>` 디렉토리로 심볼릭 링크를 생성
- `run_phaseA_to_teacher_e2e.sh` 가 바로 읽을 수 있는 pseudo-filtered parquet 를 생성
- training label 변환용 최소 reference json(`test_images_reference.json`)을 함께 생성

즉, 테스트 이미지에서는 선별이 아니라 “기존 파이프라인이 기대하는 curated manifest 형식으로 정규화”하는 단계입니다.

### 2.2 GT 부재 처리

테스트 이미지에는 GAIC GT가 없으므로 아래 정책을 사용합니다.

- `run_gaic_benchmark_eval`, `run_gaic_subject_region_ab` 는 수행하지 않음
- training labels 는 계속 생성하지만, reference json 은 로컬 이미지 목록만 포함하는 최소 payload 이므로 split 기반 benchmark 용도가 아니라 format compatibility 용도입니다
- `vlm-teacher` 는 수행하지 않으므로 validator도 VLM 산출물은 요구하지 않습니다

### 2.3 메타데이터 부재 처리

테스트 이미지에도 태그/캡션 메타데이터가 없으므로, `C1 + real-expensive teacher` 를 켜는 경우 wrapper가 synthetic caption을 먼저 생성합니다.

- 로컬 기본 preset: `local_efficient`
- 서버 기본 preset: `server_quality`
- `use_real_expensive=1` 이면 `run_c1` 이 자동으로 1로 보정됩니다

## 3. 실행 전제

이 로컬 PC에서 코드 테스트나 파이프라인 실행을 위한 모든 터미널 명령은 아래 가상환경을 활성화한 뒤 수행합니다.

```bash
source /media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/activate
```

기본 입력/출력 경로:

- 입력 이미지 루트: `data/test_images`
- 기본 출력 루트: `data/TestImages/All`

## 4. 권장 실행

아래 명령은 `filter` 와 `vlm-teacher` 를 제외한 주요 단계를 모두 켭니다. 테스트 이미지에는 GT가 없으므로 benchmark 관련 단계는 포함하지 않습니다.

```bash
#source /media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/activate

RUN_TAG=test_images_260402_reroute_v1
GPU_IDS=0
N_WORKERS=$(awk -F',' '{print NF}' <<< "${GPU_IDS}")

bash src/scripts/run_test_images_to_teacher_e2e.sh \
  --server_mode 1 \
  --data_dir data/TestImages/All \
  --image_root data/test_images \
  --run_tag ${RUN_TAG} \
  --skip_existing 0 \
  --run_c1 1 \
  --run_c2 1 \
  --run_c3 1 \
  --run_c3_enrich 1 \
  --run_c4 1 \
  --run_c5 1 \
  --run_c6 1 \
  --run_merge 1 \
  --run_subject_routing 1 \
  --run_candidates 1 \
  --run_teacher 1 \
  --run_vlm_teacher 0 \
  --run_training_labels 1 \
  --run_detailed_report 1 \
  --use_real_expensive 1 \
  --gaic_generate_captions 1 \
  --gaic_caption_preset server_quality \
  --run_c7_saliency 1 \
  --c7_saliency_priority quality_first \
  --precompute_mode unified \
  --extract_gpu_ids ${GPU_IDS} \
  --num_workers ${N_WORKERS} \
  --teacher_gpu_ids ${GPU_IDS} \
  --teacher_num_workers ${N_WORKERS} \
  --run_training_label_debug_viz 1 \
  | tee src/scripts/logs/run_test_images_to_teacher_e2e_${RUN_TAG}.log
```

위 명령의 실제 동작:

- `data/test_images` 전체를 curated pool 형식으로 준비
- `run_phaseA_to_teacher_e2e.sh` 를 `run_filter=0` 으로 호출
- synthetic caption 생성 후 `C1 + real-expensive teacher` 경로 활성화
- `C2/C3/C4/C5/C6`, merge, subject routing, `C7 saliency`, candidates, teacher 수행
- training labels, GAIC-like export, detailed report 생성
- 기본 training label output은 `${RUN_TAG}_leftover_ignore_monotonic`
- 추가 leftover variant는 기본 비활성화이며 `--auto_leftover_variants 1`일 때만 생성
- 마지막에 validator를 자동 실행

참고:

- `--run_training_label_debug_viz 1` 옵션도 wrapper에 존재하며, test images처럼 official GAIC GT 교집합이 없으면 GT 없이 pos./neg. 크롭 박스와 score distribution만 생성합니다.

debug visualization까지 마지막 단계에서 자동으로 만들고 싶으면:

```bash
  bash src/scripts/run_test_images_to_teacher_e2e.sh \
  --server_mode 1 \
  --data_dir data/TestImages/All \
  --image_root data/test_images \
  --run_tag ${RUN_TAG} \
  --skip_existing 1 \
  --run_candidates 0 \
  --run_teacher 0 \
  --run_vlm_teacher 0 \
  --run_detailed_report 0 \
  --run_training_labels 1 \
  --run_training_label_debug_viz 1 \
  --training_label_debug_viz_sample_size 50
```

## 5. 빠른 준비/검증 스모크

전체 모델 실행 전에 wrapper 연결만 빠르게 확인하려면 준비 단계만 먼저 검증할 수 있습니다.

```bash
source /media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/activate

bash src/scripts/run_test_images_to_teacher_e2e.sh \
  --prepare_only 1 \
  --prepare_max_images 4 \
  --run_tag test_images_prepare_smoke \
  --gaic_generate_captions 0

python src/scripts/validate_gaic_e2e_outputs.py \
  --manifest_parquet data/TestImages/All/filtered_test_images_all.parquet \
  --prepare_summary_json data/TestImages/All/test_images_prepare_summary.json \
  --summary_json data/TestImages/All/artifacts/validation/test_images_prepare_smoke_validation.json
```

## 6. 기본 산출물

- manifest parquet: `data/TestImages/All/filtered_test_images_all.parquet`
- flat image dir: `data/TestImages/All/images`
- teacher scores: `data/TestImages/All/artifacts/teacher/scores/teacher_scores_ar_<run_tag>.jsonl`
- training labels: `data/TestImages/All/artifacts/training_labels/<run_tag>_leftover_ignore_monotonic`
- detailed report: `data/TestImages/All/artifacts/reports/<run_tag>_detailed`
- validation summary: `data/TestImages/All/artifacts/validation/test_images_e2e_validation_<run_tag>.json`

## 7. 검증 메모

이 문서와 스크립트 추가 후 로컬 venv 기준으로 아래를 확인했습니다.

- `bash -n src/scripts/run_test_images_to_teacher_e2e.sh`
- `python -m py_compile src/scripts/prepare_gaic_curated_dataset.py`
- `--prepare_only 1` 스모크 실행 및 `validate_gaic_e2e_outputs.py` 로 준비 산출물 검증

전체 E2E는 OCR/pose/gaze/saliency/teacher 모델 실행 비용이 커서 문서 반영 시점에는 준비 단계 스모크와 validator 기준으로 연결을 검증했습니다.
