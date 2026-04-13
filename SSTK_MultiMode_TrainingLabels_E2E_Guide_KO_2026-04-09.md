# SSTK MultiMode Training Labels E2E Guide KO (2026-04-09)

## 1. 목적

이 문서는 `SSTK_MultiMode_TrainingLabels_Framework_Proposal_KO_2026-04-09.md`의 `10.3 구현 단계`를 기준으로 실제 추가된 멀티모드 라벨 생성 경로와 데이터셋별 실행 방법을 정리한다.

이번 구현은 다음 두 경로를 모두 지원한다.

1. 기존 precompute/candidates artifact를 재사용하는 `reuse-first` 실행
2. 기존 E2E wrapper에 옵션을 추가해 multimode stage를 후단에 붙이는 실행

## 2. 구현 파일

- `src/multimode/mode_catalog.py`
- `src/multimode/entity_atoms.py`
- `src/multimode/query_builder.py`
- `src/multimode/query_guidance.py`
- `src/multimode/candidate_bank.py`
- `src/multimode/mode_scorer.py`
- `src/multimode/coco_writer.py`
- `src/scripts/build_multimode_training_labels.py`
- `src/scripts/validate_multimode_training_labels.py`

기존 wrapper 확장:

- `src/scripts/run_phaseA_to_teacher_e2e.sh`
- `src/scripts/run_gaic_to_teacher_e2e.sh`
- `src/scripts/run_test_images_to_teacher_e2e.sh`

## 3. 출력물 규약

멀티모드 출력 디렉토리 기본값:

- SSTK `data/.../artifacts/training_labels_multimode/<run_tag>_multimode_v1`
- GAIC `data/GAIC/All/artifacts/training_labels_multimode/<run_tag>_multimode_v1`
- TestImages `data/TestImages/All/artifacts/training_labels_multimode/<run_tag>_multimode_v1`

핵심 산출물:

- `summary.json`
- `mode_query_status.jsonl`
- `validation_summary.json`
- `coco/instances_multimode_training_labels.json`
- `debug_viz/` (`--multimode_write_debug_viz 1`일 때만)

정책상 `positive`가 없는 query는 `mode_query_status.jsonl`에는 남지만 COCO `annotations[]`에는 직렬화하지 않는다.  
즉 COCO output은 `positive query + optional negative`만 포함한다.

## 4. 신규 wrapper 옵션

세 wrapper 공통:

- `--run_multimode_training_labels 0|1`
- `--multimode_training_labels_dir PATH`
- `--multimode_target_ars CSV`
- `--multimode_max_images INT`
- `--multimode_write_debug_viz 0|1`
- `--multimode_debug_viz_limit INT`
- `--multimode_features_jsonl_override PATH`
- `--multimode_candidates_jsonl_override PATH`

`override` 두 개를 주면 multimode stage는 현재 run의 downstream/candidates 대신 지정한 artifact를 사용한다.  
기존 `test_images_full_260408_latentsupport_v1`, `gaic_full_260408_latentsupport_v1` 같은 산출물을 재사용할 때 이 경로를 권장한다.

## 5. 권장 실행 모드

### 5.1 Reuse-first

기존 perception precompute와 candidates를 그대로 사용한다.  
가장 빠르고, 현재 구현 검증도 이 경로를 기준으로 진행했다.

### 5.2 Full wrapper continuation

기존 wrapper를 그대로 사용하되 `--run_multimode_training_labels 1`을 켠다.  
이 경우 multimode stage는 기존 finalscore stage와 독립적이며, teacher output이 없어도 동작한다.

주의:

- `run_phaseA_to_teacher_e2e.sh`는 `run_teacher=0`이면 detailed report를 자동으로 비활성화한다.
- reuse 모드에서 multimode override를 쓰는 경우 `--run_c7_saliency 0`을 주는 편이 불필요한 재연산을 막는다.
- `run_test_images_to_teacher_e2e.sh`는 `server_mode=1`일 때 로컬 venv를 activate하지 않고 `python3`를 직접 사용한다.

## 6. 데이터셋별 실행 예시

### 6.1 TestImages 재사용 실행

현재 검증에 사용한 권장 명령:

```bash
bash src/scripts/run_test_images_to_teacher_e2e.sh \
  --run_tag test_images_multimode_reuse \
  --gaic_generate_captions 0 \
  --run_c1 0 --run_c2 0 --run_c3 0 --run_c3_enrich 0 --run_c4 0 --run_c5 0 --run_c6 0 \
  --run_merge 0 --run_subject_routing 0 \
  --run_candidates 0 --run_teacher 0 --run_vlm_teacher 0 --run_training_labels 0 \
  --run_c7_saliency 0 \
  --run_multimode_training_labels 1 \
  --multimode_features_jsonl_override data/TestImages/All/artifacts/precompute/feats_c2c3c5_v2_strict_enriched_routed_final.jsonl \
  --multimode_candidates_jsonl_override data/TestImages/All/artifacts/candidates/candidates_ar_test_images_full_260408_latentsupport_v1.jsonl \
  --multimode_write_debug_viz 1 \
  --multimode_debug_viz_limit 0 \
  --skip_validation 1
```

산출물:

- `data/TestImages/All/artifacts/training_labels_multimode/test_images_multimode_reuse_multimode_v1`

### 6.1-b TestImages full 실행

기존 precompute/candidates를 재사용하지 않고 wrapper 전체를 그대로 태우는 명령이다.

로컬:

```bash
bash src/scripts/run_test_images_to_teacher_e2e.sh \
  --run_tag test_images_multimode_full \
  --run_multimode_training_labels 1 \
  --multimode_write_debug_viz 1 \
  --multimode_debug_viz_limit 0
```

서버:

```bash
bash src/scripts/run_test_images_to_teacher_e2e.sh \
  --server_mode 1 \
  --run_tag test_images_multimode_full_server \
  --run_multimode_training_labels 1 \
  --multimode_write_debug_viz 1 \
  --multimode_debug_viz_limit 0
```

서버 모드에서는 `PYTHON_BIN="/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/python"`를 쓰지 않고 `python3`로 동작하며, venv activation도 수행하지 않는다.

### 6.2 GAIC 재사용 실행

```bash
bash src/scripts/run_gaic_to_teacher_e2e.sh \
  --run_tag gaic_multimode_reuse \
  --gaic_generate_captions 0 \
  --run_c1 0 --run_c6 0 \
  --run_vlm_teacher 0 --run_training_labels 0 \
  --run_c7_saliency 0 \
  --run_multimode_training_labels 1 \
  --multimode_features_jsonl_override data/GAIC/All/artifacts/precompute/feats_c2c3c5_v2_strict_enriched_routed_final.jsonl \
  --multimode_candidates_jsonl_override data/GAIC/All/artifacts/candidates/candidates_ar_gaic_full_260408_latentsupport_v1.jsonl \
  --multimode_write_debug_viz 1 \
  --multimode_debug_viz_limit 0 \
  --skip_validation 1
```

스모크 검증은 wrapper 대신 direct builder로 먼저 확인했다.

### 6.3 SSTK `Full_10000` 기존 산출물 재사용 실행

SSTK는 기본 wrapper가 `run_phaseA_to_teacher_e2e.sh`다. 현재 `data/SSTK/Full_10000`은 `bucket=sstk_full_10000`이 아니라 기존 파일명 규칙인 `bucket=sstk_100`을 사용한다. 따라서 wrapper 입력 parquet는 `data/SSTK/Full_10000/filtered_sstk_100.parquet`가 된다.

현재 확인된 주요 산출물:

- curated set: `filtered_sstk_100.parquet`, `images/`, `cache/actual_image_size_map.json`
- precompute: `feats_c1.jsonl`, `feats_c2c3c5_v2_strict_raw.jsonl`, `feats_c2c3c5_v2_strict_enriched.jsonl`, `feats_c2c3c5_v2_strict_enriched_routed.jsonl`
- candidates: `candidates_ar_260316_r2.jsonl`, overview JSON/CSV, candidate visualizations
- public teacher: `teacher_raw_public_260316_r2.jsonl`, `teacher_proposals_public_260316_r2.jsonl`
- teacher: `teacher_scores_ar_260316_r2.jsonl`, overview/QA, teacher visualization
- existing FinalScore labels: `artifacts/training_labels/260316_r2/*`

아직 `artifacts/training_labels_multimode/`에는 `Full_10000`용 multimode output이 없으므로, 가장 빠른 경로는 기존 routed features와 candidates를 override로 넣고 multimode builder만 실행하는 것이다.

주의: 아래 변수 정의 줄까지 함께 실행한다. `DATANAME=... BASE_TAG=... bash ... ${BASE_TAG}`처럼 같은 명령 앞에 inline assignment로 붙이면 shell 확장 시점에는 `${BASE_TAG}`가 비어 있어 `--run_tag` 값이 밀린다.

```bash
DATANAME=Full_10000
BASE_TAG=260316_r2
RUN_TAG=${BASE_TAG}_multimode_reuse

bash src/scripts/run_phaseA_to_teacher_e2e.sh \
  --data_dir data/SSTK/${DATANAME} \
  --bucket sstk_100 \
  --server_mode 1 \
  --tar_dir /sstk/20230916/sstk_100 \
  --run_tag ${RUN_TAG} \
  --run_filter 0 \
  --prefer_curated_images 1 \
  --curated_image_dir data/SSTK/${DATANAME}/images \
  --skip_existing 0 \
  --run_c1 0 \
  --run_c2 0 \
  --run_c3 0 \
  --run_c3_enrich 0 \
  --run_c4 0 \
  --run_c5 0 \
  --run_c6 0 \
  --run_merge 0 \
  --run_subject_routing 0 \
  --run_c7_saliency 0 \
  --run_candidates 0 \
  --run_teacher 0 \
  --run_vlm_teacher 0 \
  --run_training_labels 0 \
  --run_detailed_report 0 \
  --run_multimode_training_labels 1 \
  --multimode_training_labels_dir data/SSTK/${DATANAME}/artifacts/training_labels_multimode/${RUN_TAG}_multimode_v1 \
  --multimode_features_jsonl_override data/SSTK/${DATANAME}/artifacts/precompute/feats_c2c3c5_v2_strict_enriched_routed.jsonl \
  --multimode_candidates_jsonl_override data/SSTK/${DATANAME}/artifacts/candidates/candidates_ar_${BASE_TAG}.jsonl \
  --multimode_target_ars FREE,1:1,9:16,16:9,3:4,4:3 \
  --multimode_max_images 0 \
  --multimode_write_debug_viz 1 \
  --multimode_debug_viz_limit 0 \
  | tee src/scripts/logs/run_phaseA_to_teacher_${DATANAME}_${RUN_TAG}.log
```

산출물:

- `data/SSTK/Full_10000/artifacts/training_labels_multimode/260316_r2_multimode_reuse_multimode_v1/summary.json`
- `data/SSTK/Full_10000/artifacts/training_labels_multimode/260316_r2_multimode_reuse_multimode_v1/mode_query_status.jsonl`
- `data/SSTK/Full_10000/artifacts/training_labels_multimode/260316_r2_multimode_reuse_multimode_v1/coco/instances_multimode_training_labels.json`
- `data/SSTK/Full_10000/artifacts/training_labels_multimode/260316_r2_multimode_reuse_multimode_v1/debug_viz/`
- `data/SSTK/Full_10000/artifacts/training_labels_multimode/260316_r2_multimode_reuse_multimode_v1/validation_summary.json`

### 6.3-b SSTK `Full_10000` downstream 전체 completion

기존 `260316_r2` teacher score까지 재사용하면서 FinalScore training labels, COCO/GAIC-like 변환, FinalScore debug-viz, multimode labels, multimode debug-viz까지 한 번에 만들려면 아래 명령을 사용한다. 기존 `artifacts/training_labels/260316_r2`는 덮어쓰지 않고 새 디렉토리에 생성한다.

```bash
DATANAME=Full_10000
RUN_TAG=260413_r1
TRAINING_OUT=data/SSTK/${DATANAME}/artifacts/training_labels/${RUN_TAG}
MULTIMODE_OUT=data/SSTK/${DATANAME}/artifacts/training_labels_multimode/${RUN_TAG}

bash src/scripts/run_phaseA_to_teacher_e2e.sh \
  --data_dir data/SSTK/${DATANAME} \
  --bucket sstk_100 \
  --server_mode 1 \
  --tar_dir /sstk/20230916/sstk_100 \
  --run_tag ${RUN_TAG} \
  --run_filter 0 \
  --prefer_curated_images 1 \
  --curated_image_dir data/SSTK/${DATANAME}/images \
  --skip_existing 0 \
  --run_c1 1 \
  --run_c2 1 \
  --run_c3 1 \
  --run_c3_enrich 1 \
  --run_c4 0 \
  --run_c5 1 \
  --run_c6 1 \
  --run_merge 1 \
  --run_subject_routing 1 \
  --run_c7_saliency 1 \
  --run_candidates 1 \
  --run_teacher 1 \
  --run_vlm_teacher 0 \
  --run_detailed_report 0 \
  --run_training_labels 1 \
  --training_labels_dir ${TRAINING_OUT} \
  --safe_leftover_policy ignore \
  --score_profile single_stage2 \
  --run_training_label_debug_viz 1 \
  --training_label_debug_viz_sample_size 50 \
  --training_label_debug_viz_seed 42 \
  --run_multimode_training_labels 1 \
  --multimode_training_labels_dir ${MULTIMODE_OUT} \
  --multimode_features_jsonl_override data/SSTK/${DATANAME}/artifacts/precompute/feats_c2c3c5_v2_strict_enriched_routed.jsonl \
  --multimode_candidates_jsonl_override data/SSTK/${DATANAME}/artifacts/candidates/candidates_ar_${RUN_TAG}.jsonl \
  --multimode_target_ars FREE,1:1,9:16,16:9,3:4,4:3 \
  --multimode_max_images 200 \
  --multimode_write_debug_viz 1 \
  --multimode_debug_viz_limit 200 \
  | tee src/scripts/logs/run_phaseA_to_teacher_${DATANAME}_${RUN_TAG}.log
```

이 경로는 `teacher_scores_ar_260316_r2.jsonl`과 `candidates_ar_260316_r2.jsonl`을 그대로 쓰므로 가장 빠르다. 단, `C7 saliency`, VLM teacher, 새 candidates/teacher score는 만들지 않는다.

### 6.3-c SSTK `Full_10000` 최대 재사용 + 모든 downstream 옵션

기존 filter/images/precompute/public-teacher 산출물을 최대한 재사용하면서도 누락된 `C7 saliency`부터 candidates, teacher, VLM teacher, FinalScore labels/debug-viz, multimode labels/debug-viz, detailed report까지 새 run tag로 만들려면 아래 명령을 사용한다.

```bash
DATANAME=Full_10000
BASE_TAG=260413_v1
RUN_TAG=${BASE_TAG}_multimode
GPU_IDS=0,1,2,3,4,5,6,7
N_WORKERS=$(awk -F',' '{print NF}' <<< "${GPU_IDS}")

bash src/scripts/run_phaseA_to_teacher_e2e.sh \
  --data_dir data/SSTK/${DATANAME} \
  --bucket sstk_100 \
  --server_mode 1 \
  --tar_dir /sstk/20230916/sstk_100 \
  --run_tag ${RUN_TAG} \
  --run_filter 0 \
  --prefer_curated_images 1 \
  --curated_image_dir data/SSTK/${DATANAME}/images \
  --skip_existing 1 \
  --precompute_mode unified \
  --run_c1 0 \
  --run_c2 0 \
  --run_c3 0 \
  --run_c3_enrich 0 \
  --run_c4 0 \
  --run_c5 0 \
  --run_c6 1 \
  --run_merge 1 \
  --run_subject_routing 1 \
  --subject_routing_top_n 5 \
  --subject_routing_union_top_m 3 \
  --subject_routing_allow_det_proxy 1 \
  --run_c7_saliency 1 \
  --c7_saliency_priority quality_first \
  --extract_mode auto \
  --extract_multi_gpu 1 \
  --gpu_ids ${GPU_IDS} \
  --gpu_workers ${N_WORKERS} \
  --run_component_viz 1 \
  --component_viz_num_samples 120 \
  --component_viz_out_dir data/SSTK/${DATANAME}/artifacts/precompute/visualizations/components_${RUN_TAG} \
  --enable_public_teacher_proposals 1 \
  --public_teacher_setup 0 \
  --public_teacher_download_weights 0 \
  --public_teacher_raw_jsonl data/SSTK/${DATANAME}/artifacts/public_teachers/raw/teacher_raw_public_${BASE_TAG}.jsonl \
  --public_teacher_proposals_jsonl data/SSTK/${DATANAME}/artifacts/public_teachers/proposals/teacher_proposals_public_${BASE_TAG}.jsonl \
  --public_teachers gaic,cacnet,cgs \
  --public_teacher_max_images -1 \
  --public_teacher_device auto \
  --public_infer_multi_gpu 1 \
  --run_candidates 1 \
  --cand_ar_list FREE,1:1,9:16,16:9,3:4,4:3 \
  --use_actual_image_size 1 \
  --strict_actual_size 1 \
  --run_teacher 1 \
  --use_real_expensive 1 \
  --teacher_multi_gpu 1 \
  --cheap_top_m 30 \
  --top_k 5 \
  --tau_div 0.75 \
  --align_device cuda \
  --aesthetic_device cuda \
  --exp_batch_size 512 \
  --run_qa 1 \
  --run_viz 1 \
  --num_viz 120 \
  --run_vlm_teacher 0 \
  --vlm_backend qwen25_vl \
  --vlm_fallback_backend heuristic \
  --vlm_model_id Qwen/Qwen3-VL-4B-Instruct \
  --vlm_device auto \
  --vlm_top_m 12 \
  --vlm_top_k 5 \
  --vlm_multi_gpu 1 \
  --vlm_save_raw_response 1 \
  --vlm_strict_backend_init 1 \
  --run_training_labels 1 \
  --safe_leftover_policy ignore \
  --score_profile single_stage2 \
  --run_training_label_debug_viz 1 \
  --training_label_debug_viz_sample_size 50 \
  --training_label_debug_viz_seed 42 \
  --run_multimode_training_labels 1 \
  --multimode_target_ars FREE,1:1,9:16,16:9,3:4,4:3 \
  --multimode_max_images 200 \
  --multimode_write_debug_viz 1 \
  --multimode_debug_viz_limit 200 \
  --run_detailed_report 1 \
  | tee src/scripts/logs/run_phaseA_to_teacher_${DATANAME}_${RUN_TAG}.log
```

Qwen3-VL 런타임이 현재 main venv와 충돌하면 `--run_vlm_teacher 0`으로 먼저 실행하고, teacher 산출물 생성 후 `run_vlm_teacher_labeler_qwen3_env.sh`로 Stage 10만 분리 실행한다.

## 7. Direct builder 실행 예시

wrapper 없이 바로 돌릴 수도 있다.

SSTK `Full_10000` 기존 산출물만 사용해 multimode labels/debug-viz를 직접 생성:

```bash
PY=/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/python
OUT_DIR=data/SSTK/Full_10000/artifacts/training_labels_multimode/260316_r2_direct_reuse_multimode_v1

${PY} src/scripts/build_multimode_training_labels.py \
  --features_jsonl data/SSTK/Full_10000/artifacts/precompute/feats_c2c3c5_v2_strict_enriched_routed.jsonl \
  --candidates_jsonl data/SSTK/Full_10000/artifacts/candidates/candidates_ar_260316_r2.jsonl \
  --image_root data/SSTK/Full_10000/images \
  --out_dir ${OUT_DIR} \
  --target_ars FREE,1:1,9:16,16:9,3:4,4:3 \
  --max_images 0 \
  --write_debug_viz 1 \
  --debug_viz_limit 0 \
  --progress 1

${PY} src/scripts/validate_multimode_training_labels.py \
  --summary_json ${OUT_DIR}/summary.json \
  --coco_json ${OUT_DIR}/coco/instances_multimode_training_labels.json \
  --query_status_jsonl ${OUT_DIR}/mode_query_status.jsonl \
  --out_json ${OUT_DIR}/validation_summary.json \
  --progress 1
```

GAIC smoke:

```bash
/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/python \
  src/scripts/build_multimode_training_labels.py \
  --features_jsonl data/GAIC/All/artifacts/precompute/feats_c2c3c5_v2_strict_enriched_routed_final.jsonl \
  --candidates_jsonl data/GAIC/All/artifacts/candidates/candidates_ar_gaic_full_260408_latentsupport_v1.jsonl \
  --image_root data/Publics/GAIC/images \
  --out_dir tmp/multimode_gaic_smoke \
  --max_images 10 \
  --progress 1
```

검증:

```bash
/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/python \
  src/scripts/validate_multimode_training_labels.py \
  --summary_json tmp/multimode_gaic_smoke/summary.json \
  --coco_json tmp/multimode_gaic_smoke/coco/instances_multimode_training_labels.json \
  --query_status_jsonl tmp/multimode_gaic_smoke/mode_query_status.jsonl \
  --progress 1
```

## 8. 구현 단계와 코드 대응

`10.3 구현 단계` 대응:

### Phase 1

- category catalog: `src/multimode/mode_catalog.py`
- entity atom extraction: `src/multimode/entity_atoms.py`
- query builder: `src/multimode/query_builder.py`
- query status export: `src/scripts/build_multimode_training_labels.py`

### Phase 2

- query-local candidate expansion: `src/multimode/candidate_bank.py`
- per-mode candidate bank: `src/scripts/build_multimode_training_labels.py`

### Phase 3

- mode scorer: `src/multimode/mode_scorer.py`
- hard reject / threshold: `src/multimode/mode_catalog.py`, `src/multimode/mode_scorer.py`
- positive / optional negative selection: `src/multimode/mode_scorer.py`

### Phase 4

- COCO-style writer: `src/multimode/coco_writer.py`
- query status sidecar: `src/scripts/build_multimode_training_labels.py`
- mode별 debug visualization: `src/scripts/build_multimode_training_labels.py`

## 9. 이번 검증 결과

### 9.1 TestImages wrapper 재사용 검증

경로:

- `data/TestImages/All/artifacts/training_labels_multimode/test_images_multimode_reuse_multimode_v1`

요약:

- images: `33`
- image-tasks: `198`
- queries: `1974`
- positive queries: `765`
- annotations: `2102`
- landscape positives: `12`

### 9.2 GAIC direct smoke 검증

경로:

- `tmp/multimode_gaic_smoke`

요약:

- images: `10`
- image-tasks: `60`
- queries: `354`
- positive queries: `192`
- annotations: `565`

### 9.3 SSTK direct smoke 검증

경로:

- `tmp/multimode_sstk_smoke`

요약:

- images: `10`
- image-tasks: `60`
- queries: `288`
- positive queries: `219`
- annotations: `608`

입력 artifact 이슈:

- `feats_c2c3c5_v2_strict_enriched_routed.jsonl`에서 invalid row `1`
- `candidates_ar_260316_r2.jsonl`에서 invalid row `1`

현재 loader는 malformed row를 경고 후 skip하도록 구현했다.

## 10. 현재 base class

- `landscape`
- `single_person_center`
- `single_person_rot`
- `group_center`
- `group_rot`
- `face`
- `object_single_center`
- `object_single_rot`
- `object_multi_center`
- `object_multi_rot`

향후 roadmap:

- `pet_*`
- `background`
- `text_document`
