# GAIC End-to-End 실행 가이드

이 문서는 [README_SSTK_Curation.md](/media/jyju25/T7_4TB_JY/Projects_26/Sources/ImageCropping/README_SSTK_Curation.md) 와 [run_phaseA_to_teacher_e2e.sh](/media/jyju25/T7_4TB_JY/Projects_26/Sources/ImageCropping/src/scripts/run_phaseA_to_teacher_e2e.sh) 기반 파이프라인을 `GAIC` 이미지셋에 맞게 재사용하는 운영 가이드입니다.

핵심 차이점은 아래와 같습니다.

- GAIC는 `curated pool` 선별 단계가 없습니다. `data/Publics/GAIC/images` 아래의 모든 이미지를 curated pool로 간주합니다.
- GAIC는 기본 parquet에 `tags/caption/super_cat`가 비어 있으므로 메타데이터 의존적인 `C1 + real-expensive teacher` 경로를 기본 비활성화합니다.
- 대신 최신 routing은 `generated caption + public teacher geometry + C2/C3/C5/C7 perception`을 함께 사용해 `object_*`, `text_document`, `background_texture_copyspace`까지 다시 열 수 있습니다.
- 기존 파이프라인이 flat `image_dir`를 기대하므로, GAIC 이미지를 심볼릭 링크 기반의 평탄화된 `images/` 디렉토리로 준비한 뒤 기존 e2e 스크립트를 그대로 호출합니다.
- saliency 기반 subject-region 보강은 기본 off 이지만, wrapper에서 `--run_c7_saliency 1` 로 바로 활성화할 수 있습니다.
- GAIC GT가 있는 경우 wrapper에서 `--run_gaic_benchmark_eval 1` 또는 `--run_gaic_subject_region_ab 1` 로 benchmark / saliency A-B까지 연속 실행할 수 있습니다.

## 1. 추가된 파일

- 래퍼 실행 스크립트: [src/scripts/run_gaic_to_teacher_e2e.sh](/media/jyju25/T7_4TB_JY/Projects_26/Sources/ImageCropping/src/scripts/run_gaic_to_teacher_e2e.sh)
- GAIC 준비 스크립트: [src/scripts/prepare_gaic_curated_dataset.py](/media/jyju25/T7_4TB_JY/Projects_26/Sources/ImageCropping/src/scripts/prepare_gaic_curated_dataset.py)
- GAIC synthetic caption 스크립트: [src/scripts/generate_gaic_captions.py](/media/jyju25/T7_4TB_JY/Projects_26/Sources/ImageCropping/src/scripts/generate_gaic_captions.py)
- 산출물 검증 스크립트: [src/scripts/validate_gaic_e2e_outputs.py](/media/jyju25/T7_4TB_JY/Projects_26/Sources/ImageCropping/src/scripts/validate_gaic_e2e_outputs.py)
- C7 saliency augment 스크립트: [src/scripts/augment_saliency_subject_features.py](/media/jyju25/T7_4TB_JY/Projects_26/Sources/ImageCropping/src/scripts/augment_saliency_subject_features.py)
- 공개 benchmark 평가 스크립트: [src/scripts/run_gaic_benchmark_eval.py](/media/jyju25/T7_4TB_JY/Projects_26/Sources/ImageCropping/src/scripts/run_gaic_benchmark_eval.py)
- saliency/effective-subject-region A/B 스크립트: [src/scripts/run_gaic_subject_region_ab.py](/media/jyju25/T7_4TB_JY/Projects_26/Sources/ImageCropping/src/scripts/run_gaic_subject_region_ab.py)

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
- 선택적으로 `C7 saliency`를 merged/routed feature에 주입해 subject anchor 보강과 `saliency_jitter` candidate 생성을 함께 켤 수 있음
- Stage-10 라벨러는 기본적으로 `heuristic` backend를 사용
- `run_c1=0` 또는 `use_real_expensive=0` 인 경우 scorer는 더 이상 `proxy expensive`를 쓰지 않습니다. 이때 `expensive=0`, `expensive_source=disabled`, `A_macro=na` 로 기록되며 최종 `score_rank` 에서 A 축은 제외됩니다.

필요 시 Qwen 기반 VLM으로 바꿀 수는 있지만, 그 경우 별도 런타임/가중치 준비가 되어 있어야 합니다.

### 2.2.1 Caption + Public Teacher + Perception 재라우팅

2026-04-01 기준 최신 구현은 GAIC/TestImages처럼 `tags/super_cat`가 없는 데이터셋에서도 아래 보강 경로를 사용합니다.

- 1차 routing은 기존처럼 merged precompute(`C2/C3/C5`, 선택 시 `C7`)만으로 수행합니다.
- public teacher proposal 생성이 끝나면 `run_phaseA_to_teacher_e2e.sh`가 `08d_reroute_subject_mode_with_public_teacher` 단계를 추가 실행합니다.
- 이 2차 routing은 [src/scripts/enrich_subject_mode_jsonl.py](/media/jyju25/T7_4TB_JY/Projects_26/Sources/ImageCropping/src/scripts/enrich_subject_mode_jsonl.py) 에 `--caption_jsonl`, `--teacher_proposals_jsonl` 을 넘겨 다시 계산합니다.
- public teacher는 semantic category를 직접 주지 않으므로, router는 top proposal box들의 `median_area_ratio`, `consensus_iou`, `best_margin_side`, `best_margin_ratio`를 요약해 `copyspace`, `object_focus`, `scene_fullframe` geometry hint를 추론합니다.
- 단, `teacher geometry-only copyspace`는 과발화를 막기 위해 보수적으로 사용합니다. blank ratio가 충분히 크고(`>= 0.40`), caption/object/scene/text와 충돌하지 않을 때만 copyspace gate를 통과시킵니다.
- 최종 downstream feature는 `feats_c2c3c5_v2_strict_enriched_routed_final.jsonl` 이며, 이후 candidate/teacher/training label은 이 rerouted 파일을 사용합니다.

이 경로 덕분에 GAIC/TestImages는 4개 mode에 갇히지 않고 `object_single`, `object_multi`, `text_document`, `background_texture_copyspace`까지 열 수 있습니다. SSTK도 같은 reroute 로직을 사용하되, 기존 `tags/super_cat` 위에 public teacher/perception을 추가 evidence로 얹어 copyspace 과발화와 `other_ambiguous` 비중을 줄이는 방향으로 동작합니다.

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

### 3.1 공용 병렬 옵션

최신 [run_gaic_to_teacher_e2e.sh](/media/jyju25/T7_4TB_JY/Projects_26/Sources/ImageCropping/src/scripts/run_gaic_to_teacher_e2e.sh) 는 준비 단계 CPU 병렬화와 후속 GPU shard 기본값을 wrapper 레벨에서 직접 받습니다.

- `--prepare_num_workers`: flat image link/materialize + actual size probing용 CPU worker 수. `0`이면 사용 가능한 전체 코어를 사용합니다.
- `--gpu_ids`: 공용 GPU CSV. 내부적으로 [run_phaseA_to_teacher_e2e.sh](/media/jyju25/T7_4TB_JY/Projects_26/Sources/ImageCropping/src/scripts/run_phaseA_to_teacher_e2e.sh) 의 filter-tag-embed / precompute / public-teacher / teacher / VLM 기본 GPU 목록으로 전파되고, GAIC subject-region A/B에도 그대로 전달됩니다.
- `--gpu_workers`: 공용 shard worker 수. `0`이면 `gpu_ids` 개수를 사용합니다.
- `--gaic_caption_multi_gpu`, `--gaic_caption_gpu_ids`, `--gaic_caption_num_workers`: synthetic caption 단계만 별도 shard 정책을 줄 때 사용합니다. 비워 두면 wrapper 공용 GPU 설정을 상속합니다.

즉, 예전처럼 `--extract_gpu_ids`, `--teacher_gpu_ids` 등을 개별 stage override로 넘길 수는 있지만, 현재 권장 방식은 wrapper 레벨에서 `--gpu_ids`, `--gpu_workers`를 한 번만 주는 것입니다.

### 3.2 로컬 Subject-Routing 검증 명령

GAIC subject routing 보강만 빠르게 재검증하려면 아래 명령을 사용합니다.

```bash
source /media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/activate

python3 src/scripts/enrich_subject_mode_jsonl.py \
  --input_feats_jsonl data/GAIC/All/artifacts/precompute/feats_c2c3c5_v2_strict_enriched_c7_saliency.jsonl \
  --input_filtered_parquet data/GAIC/All/filtered_gaic_all.parquet \
  --caption_jsonl data/GAIC/All_olds/artifacts/metadata/gaic_captions_gaic_260324_r0.jsonl \
  --teacher_proposals_jsonl data/GAIC/All_olds/artifacts/public_teachers/proposals/teacher_proposals_public_gaic_260324_r1.jsonl \
  --output_jsonl tmp/gaic_routed_caption_teacher_v3.jsonl \
  --progress 0
```

동일한 로직은 TestImages에도 그대로 적용됩니다.

```bash
python3 src/scripts/enrich_subject_mode_jsonl.py \
  --input_feats_jsonl data/TestImages/All/artifacts/precompute/feats_c2c3c5_v2_strict_enriched_c7_saliency.jsonl \
  --input_filtered_parquet data/TestImages/All/filtered_test_images_all.parquet \
  --output_jsonl tmp/test_images_routed_caption_v3.jsonl \
  --progress 0
```

2026-04-01 로컬 검증 결과:

- GAIC baseline은 `other_ambiguous 657 / portrait_group 240 / portrait_single 204 / scene_general 135` 였고, reroute 후 `scene_general 416 / portrait_group 266 / portrait_single 245 / object_single 232 / object_multi 48 / background_texture_copyspace 23 / text_document 6` 으로 바뀌었습니다.
- TestImages baseline은 4 mode(`other_ambiguous/portrait_group/portrait_single/scene_general`)만 관측됐고, reroute 후 `object_single 4`, `object_multi 2`가 추가로 열렸습니다.
- 대표적인 변경 예시는 `plate of fruit and a key sitting on a table -> object_multi`, `man sitting on a bus reading a newspaper -> text_document`, `bench in the snow by the water -> scene_general` 입니다.

## 4. 권장 실행

서버 실행 템플릿은 아래 공통 변수를 먼저 두고 쓰는 형식으로 통일하는 것을 권장합니다.

```bash
GPU_IDS=0,1,2
N_WORKERS=$(awk -F',' '{print NF}' <<< "${GPU_IDS}")
PREPARE_WORKERS=16
DATANAME=All
RUN_TAG=gaic_260324_r1
```

### 4.1 전체 실행

```bash
# 서버에서 사용할 venv를 먼저 활성화하거나,
# 필요하면 아래 명령에 --venv_path /your/server/venv/bin/activate 를 추가
# source /your/server/venv/bin/activate

GPU_IDS=0,1,2
N_WORKERS=$(awk -F',' '{print NF}' <<< "${GPU_IDS}")
PREPARE_WORKERS=16

RUN_TAG=gaic_260324_r1
DATANAME=All

bash src/scripts/run_gaic_to_teacher_e2e.sh \
  --server_mode 1 \
  --data_dir data/GAIC/${DATANAME} \
  --image_root data/Publics/GAIC/images \
  --run_tag ${RUN_TAG} \
  --skip_existing 0 \
  --prepare_num_workers ${PREPARE_WORKERS} \
  --gpu_ids ${GPU_IDS} \
  --gpu_workers ${N_WORKERS} \
  --use_real_expensive 1 \
  --gaic_caption_preset server_quality \
  --gaic_generate_captions 0 \
  --run_filter 0 \
  --run_c1 0 \
  --run_c2 0 \
  --run_c3 0 \
  --run_c3_enrich 0 \
  --run_c4 0 \
  --run_c5 0 \
  --run_c6 0 \
  --run_c7_saliency 1 \
  --c7_saliency_priority quality_first \
  --run_merge 0 \
  --run_subject_routing 1 \
  --run_vlm_teacher 0 \
  --run_training_labels 1 \
  --run_detailed_report 1 \
  --enable_public_teacher_proposals 1 \
  --public_teacher_setup 1 \
  --public_teacher_download_weights 1 \
  --public_teachers gaic,cacnet,cgs \
  --run_gaic_benchmark_eval 1 \
  --gaic_benchmark_sample_count 8 \
  --run_gaic_subject_region_ab 1 \
  --gaic_subject_ab_run_tag gaic_eval_${RUN_TAG} \
  --gaic_subject_ab_sample_count 8 \
  | tee src/scripts/logs/run_gaic_to_teacher_e2e_${DATANAME}_${RUN_TAG}.log
```

위 명령의 실제 동작:

- GAIC 전체 이미지를 curated pool로 준비
- 준비 단계는 `--prepare_num_workers` 기준으로 CPU 병렬화되고, 이후 GPU 단계들은 `--gpu_ids`, `--gpu_workers` 기본값을 공유
- `run_phaseA_to_teacher_e2e.sh` 를 `run_filter=0` 으로 호출
- 기존 precompute/caption lane이 있으면 그것을 재사용하고, 여기서는 `C7 saliency`를 추가로 켠 실무형 재실행 preset으로 동작한다
- fresh precompute를 처음부터 다시 돌리면서 모든 옵션을 켜고 싶다면 아래 `4.1-max`를 사용한다
- `C4`, `VLM teacher` 는 수행하지 않음
- public teacher proposal injection을 켠 상태로 `candidates -> teacher` 를 수행
- public teacher proposal 생성 뒤 `08d_reroute_subject_mode_with_public_teacher`가 자동 실행되며, caption/public teacher/perception 신호를 반영한 `..._routed_final.jsonl`이 downstream input으로 승격됩니다.
- training label export 및 GAIC-like export 생성
- 기본 training label output은 `${RUN_TAG}_leftover_ignore_monotonic` 하나만 생성
- `--auto_leftover_variants 1`일 때만 `*_leftover_keepneg_monotonic`, `*_leftover_softpos_monotonic` variant를 추가 생성
- `--run_training_label_debug_viz 1`일 때 마지막 단계에서 `debug_visualizations_balanced50_bottomneg/`를 자동 생성
- training label dir에 `gaic_gt_score_cache_free.jsonl` 이 있으면 debug viz의 GT bbox 라벨에 `MOS + FREE-context training score_prob`를 함께 표기
- GAIC-like export는 main json 외에 `..._train.json`, `..._test.json`, `..._unassigned.json` 도 함께 생성
- 마지막에 자동 검증 실행

위 `leftover_*` variant는 모두 같은 `teacher_scores`에서 다시 training label만 재구성한 실험용 분기다. 차이는 `safe high-score leftover`를 어떻게 처리하느냐에 있다.

- safe high-score leftover:
  - severe reject는 아니어서 unsafe negative는 아니지만
  - 최종 chosen/matching target보다 `score_prob`는 더 높고
  - diverse positive set(`selected_topk`)에는 들지 못한 후보
- `${RUN_TAG}_leftover_keepneg_monotonic`:
  - 이 후보를 기존처럼 `candidate_pool`의 `negative/near_negative`로 유지
  - 과거 동작 재현이나 보수적 호환 비교에 적합
  - conditional-DETR 관점에서는 "safe한데 positive보다 score가 높은 negative"가 남을 수 있어 가장 공격적인 negative mining variant다
- `${RUN_TAG}_leftover_ignore_monotonic`:
  - 이 후보를 `ignored_candidates`로 분리하고 negative annotation에서 제외
  - 현재 기본/권장 정책
  - positive set과 충돌할 수 있는 safe leftover를 제거하므로, detector 학습에서 contradictory negative를 줄이는 목적에 맞다
- `${RUN_TAG}_leftover_softpos_monotonic`:
  - 이 후보를 `soft_positive`로 승격해 `matching_targets`에 포함
  - positive recall을 넓히는 실험용 variant
  - 대신 positive set이 느슨해지므로 ranking/policy 일관성보다 recall을 우선할 때 적합

중요:

- 이 4.1 템플릿은 `run_c1/c2/c3/c4/c5/c6/merge=0`을 명시한 재사용형 preset이다. 따라서 fresh full-run 설명으로 읽으면 안 된다.
- fresh server full-run에서 `run_c1=1`, `use_real_expensive=1`, `gaic_generate_captions=1`까지 모두 켠 경로는 아래 `4.1-max`다.

위 4.1은 현재 운영상 가장 무난한 권장 템플릿이다. 서버에서 가능한 boolean 옵션을 사실상 모두 켠 maximal full-run이 필요하면 아래 4.1-max를 사용한다.

참고:

- 아래 후속 예제들 중 일부는 기록 보존을 위해 예전처럼 `--extract_gpu_ids`, `--teacher_gpu_ids` 등 stage별 옵션을 직접 쓰고 있습니다.
- 현재 운영에서는 wrapper 레벨 `--gpu_ids`, `--gpu_workers`를 우선 쓰고, 특정 단계만 다르게 돌릴 때만 stage별 override를 추가하는 방식을 권장합니다.

### 4.1-max 서버 최대 옵션 full-run

아래 명령은 wrapper에서 켤 수 있는 주요 옵션을 한 번에 모두 활성화한다. 즉 `C1~C7`, merge, subject routing, public teacher, VLM teacher, training labels, leftover variants, debug-viz, detailed report, benchmark, saliency A/B까지 한 번에 수행한다.

```bash
# source /your/server/venv/bin/activate

GPU_IDS=0,1,2,3,4,5,6,7
N_WORKERS=$(awk -F',' '{print NF}' <<< "${GPU_IDS}")
PREPARE_WORKERS=32

RUN_TAG=gaic_full_260406_v1
DATANAME=All

bash src/scripts/run_gaic_to_teacher_e2e.sh \
  --server_mode 1 \
  --data_dir data/GAIC/${DATANAME} \
  --image_root data/Publics/GAIC/images \
  --run_tag ${RUN_TAG} \
  --skip_existing 0 \
  --prepare_num_workers ${PREPARE_WORKERS} \
  --gpu_ids ${GPU_IDS} \
  --gpu_workers ${N_WORKERS} \
  --run_filter 0 \
  --precompute_mode unified \
  --use_real_expensive 1 \
  --gaic_caption_preset server_quality \
  --gaic_generate_captions 1 \
  --run_c1 1 \
  --run_c2 1 \
  --run_c3 1 \
  --run_c3_enrich 1 \
  --run_c4 1 \
  --run_c5 1 \
  --run_c6 1 \
  --run_c7_saliency 1 \
  --c7_saliency_priority quality_first \
  --run_merge 1 \
  --run_subject_routing 1 \
  --run_candidates 1 \
  --run_teacher 1 \
  --run_vlm_teacher 0 \
  --vlm_backend heuristic \
  --vlm_fallback_backend none \
  --run_training_labels 1 \
  --safe_leftover_policy ignore \
  --auto_leftover_variants 0 \
  --run_training_label_debug_viz 1 \
  --training_label_debug_viz_sample_size 50 \
  --training_label_debug_viz_seed 42 \
  --run_detailed_report 1 \
  --enable_public_teacher_proposals 1 \
  --public_teacher_setup 1 \
  --public_teacher_download_weights 1 \
  --public_teachers gaic,cacnet,cgs \
  --run_gaic_benchmark_eval 1 \
  --gaic_benchmark_sample_count 16 \
  --run_gaic_subject_region_ab 1 \
  --gaic_subject_ab_run_tag gaic_eval_${RUN_TAG} \
  --gaic_subject_ab_sample_count 16 \
  | tee src/scripts/logs/run_gaic_to_teacher_e2e_${DATANAME}_${RUN_TAG}.log
```

주의:

- 이 명령은 가장 느리며 서버 디스크/GPU 사용량도 가장 크다.
- `run_vlm_teacher=1`은 별도 VLM을 강제하지 않으면 wrapper 기본값인 `heuristic` backend를 사용한다.
- `run_c4=1`까지 켜므로 OCR/runtime 준비가 덜 된 서버에서는 실패할 수 있다. 그런 서버에서는 4.1 템플릿으로 먼저 재현하고 필요한 단계만 선택적으로 켜는 편이 낫다.

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

### 4.1d C7 saliency까지 포함한 teacher/training-label 실행

`C7 saliency`는 flat image dir가 있을 때만 켤 수 있습니다. GAIC wrapper는 준비 단계에서 flat image dir를 만들기 때문에 아래처럼 바로 연결 가능합니다.

```bash
source /media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/activate

bash src/scripts/run_gaic_to_teacher_e2e.sh \
  --data_dir data/GAIC/All \
  --image_root data/Publics/GAIC/images \
  --run_tag gaic_260324_c7 \
  --skip_existing 1 \
  --run_c1 0 \
  --use_real_expensive 0 \
  --run_c4 0 \
  --run_c6 1 \
  --run_c7_saliency 1 \
  --c7_saliency_priority quality_first \
  --run_vlm_teacher 0 \
  --run_training_labels 1 \
  --run_detailed_report 1
```

이 경우 `run_phaseA_to_teacher_e2e.sh` 내부에서 `augment_saliency_subject_features.py` 가 추가 실행되고, 4차 보강 기준으로는 `c7 saliency augment -> subject reroute` 순서가 적용됩니다. 즉 중간 산출물은 `..._enriched_c7_saliency.jsonl`, 최종 downstream feature jsonl은 `..._routed_c7_saliency.jsonl` 입니다.

### 4.1e E2E 직후 benchmark / saliency A-B까지 연속 실행

GAIC GT가 있는 경우 wrapper에서 benchmark와 subject-region A-B까지 바로 이어서 실행할 수 있습니다.

```bash
source /media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/activate

bash src/scripts/run_gaic_to_teacher_e2e.sh \
  --data_dir data/GAIC/All \
  --image_root data/Publics/GAIC/images \
  --run_tag gaic_260324_eval \
  --skip_existing 1 \
  --run_c7_saliency 1 \
  --c7_saliency_priority quality_first \
  --run_training_labels 1 \
  --run_gaic_benchmark_eval 1 \
  --gaic_benchmark_sample_count 8 \
  --run_gaic_subject_region_ab 1 \
  --gaic_subject_ab_run_tag gaic_260324_eval_saliency_v1 \
  --gaic_subject_ab_sample_count 8
```

주의:

- `--run_gaic_benchmark_eval 1` 은 현재 run의 `candidates / teacher / training_labels`를 이용해 `run_gaic_benchmark_eval.py` 를 실행합니다.
- `--run_gaic_subject_region_ab 1` 은 baseline 후보/benchmark summary가 필요합니다. 기본값은 `data/GAIC/All/artifacts/...gaic_260320_r0...` 를 보지만, 다르면 `--gaic_subject_ab_baseline_candidates_jsonl`, `--gaic_subject_ab_baseline_benchmark_summary` 로 직접 넘겨야 합니다.

### 4.1f 기존 Public Teacher 산출물 재사용 + Phase-4 saliency rerun

서버에서 public teacher proposal까지만 정상 생성됐고, 로컬에서 최신 saliency / support-map / reroute / benchmark 코드로 다시 내리고 싶다면 전체 4.1을 처음부터 다시 할 필요는 없습니다. 아래 템플릿은 기존 `filtered parquet`, `feats_c1`, `merged precompute`, `public teacher proposals`를 재사용하고 `subject routing -> c7 -> reroute -> candidates -> teacher -> training labels -> benchmark`만 다시 수행합니다.

전제:

- `data/GAIC/All/artifacts/precompute/feats_c1.jsonl`
- `data/GAIC/All/artifacts/precompute/feats_c2c3c5_v2_strict_enriched.jsonl`
- `data/GAIC/All/artifacts/public_teachers/proposals/teacher_proposals_public_<run>.jsonl`
- `data/GAIC/All/filtered_gaic_all.parquet`
- `data/GAIC/All/images`

```bash
GPU_IDS=0,1,2
N_WORKERS=$(awk -F',' '{print NF}' <<< "${GPU_IDS}")

RUN_TAG=gaic_260324_r2
DATANAME=All

bash src/scripts/run_gaic_to_teacher_e2e.sh \
  --data_dir data/GAIC/${DATANAME} \
  --image_root data/Publics/GAIC/images \
  --run_tag ${RUN_TAG} \
  --skip_existing 0 \
  --gaic_generate_captions 0 \
  --run_filter 0 \
  --run_c1 0 \
  --run_c2 0 \
  --run_c3 0 \
  --run_c3_enrich 0 \
  --run_c4 0 \
  --run_c5 0 \
  --run_c6 0 \
  --run_merge 0 \
  --run_subject_routing 1 \
  --run_c7_saliency 1 \
  --c7_saliency_priority quality_first \
  --teacher_proposals_jsonl data/GAIC/All/artifacts/public_teachers/proposals/teacher_proposals_public_gaic_260324_r1.jsonl \
  --use_real_expensive 1 \
  --run_vlm_teacher 0 \
  --extract_gpu_ids ${GPU_IDS} \
  --num_workers ${N_WORKERS} \
  --teacher_gpu_ids ${GPU_IDS} \
  --teacher_num_workers ${N_WORKERS} \
  --run_detailed_report 1 \
  --run_training_labels 1 \
  --safe_leftover_policy ignore \
  --auto_leftover_variants 0 \
  --run_gaic_benchmark_eval 1 \
  --run_gaic_subject_region_ab 1 \
  --run_viz 1 \
  | tee src/scripts/logs/run_gaic_to_teacher_e2e_${DATANAME}_${RUN_TAG}.log
```

참고:

- `--run_c1 0`을 명시하면, 이제 wrapper가 기존 `feats_c1.jsonl`을 재사용합니다.
- 이 템플릿은 current phase-4 기준으로 `c7 augment 후 reroute`를 다시 수행하므로, 이전 `routed_c7` 산출물이 있어도 `skip_existing=0`으로 재생성하는 편이 안전합니다.

### 4.1g `JSONDecodeError`로 training-label build가 실패할 때

대표 로그:

```text
json.decoder.JSONDecodeError: Expecting value ...
```

이 에러가 `build_finalscore_training_data.py`에서 발생하면, 원인은 거의 항상 `teacher_scores_ar_<run>.jsonl`이 중간에서 잘렸거나 부분 복사된 경우입니다. 실제로는 row 수만 대충 맞아 보여도, 파일 중간 한 줄이 끊기면 builder가 실패합니다.

현재 코드는 다음처럼 동작합니다.

- `build_finalscore_training_data.py`는 깨진 줄의 `line / col / char`와 함께 즉시 실패합니다.
- `repair_teacher_outputs.py`는 이제 parse error가 있는 final/shard를 정상으로 간주하지 않습니다.

권장 순서:

1. `teacher_scores_ar_<run>.jsonl`과 `*.shards.*`가 완전한지 확인
2. 필요하면 `src/scripts/repair_teacher_outputs.py`를 먼저 실행
3. shard도 함께 손상됐다면, teacher stage부터 다시 내려야 함

예:

```bash
source /media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/activate

python3 src/scripts/repair_teacher_outputs.py \
  --teacher_scores_jsonl data/GAIC/All/artifacts/teacher/scores/teacher_scores_ar_gaic_260324_r0.jsonl \
  --overview_json data/GAIC/All/artifacts/teacher/overview/teacher_scores_overview_gaic_260324_r0.json \
  --overview_csv data/GAIC/All/artifacts/teacher/overview/teacher_scores_overview_by_ar_gaic_260324_r0.csv \
  --qa_json data/GAIC/All/artifacts/teacher/qa/teacher_scores_qa_report_gaic_260324_r0.json \
  --qa_csv data/GAIC/All/artifacts/teacher/qa/teacher_scores_qa_report_by_ar_gaic_260324_r0.csv \
  --candidates_jsonl data/GAIC/All/artifacts/candidates/candidates_ar_gaic_260324_r0.jsonl \
  --max_images 0 \
  --prefer_real_expensive 1 \
  --strict_expected_match 1
```

만약 이 repair 단계가 `parse_error_rows` 때문에 실패하면, 복사된 `teacher_scores` 또는 shard 자체가 손상된 것이므로 로컬에서 teacher stage를 재실행하는 것이 맞습니다.

### 4.1h 로컬 PC에서 current phase-4 + teacher proposal만 다시 반영할 때

`benchmark / training-label / report` 갱신 목적이라면, 로컬에서는 아래 proxy-expensive lane이 현실적인 기본값이다.

```bash
source /media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/activate

DATANAME=All
RUN_TAG=gaic_260324_r0_saliency_v4tp_proxy

bash src/scripts/run_gaic_to_teacher_e2e.sh \
  --data_dir data/GAIC/${DATANAME} \
  --image_root data/Publics/GAIC/images \
  --run_tag ${RUN_TAG} \
  --skip_existing 1 \
  --gaic_generate_captions 0 \
  --run_filter 0 \
  --run_c1 0 \
  --run_c2 0 \
  --run_c3 0 \
  --run_c3_enrich 0 \
  --run_c4 0 \
  --run_c5 0 \
  --run_c6 0 \
  --run_merge 0 \
  --run_subject_routing 1 \
  --run_c7_saliency 1 \
  --c7_saliency_priority quality_first \
  --teacher_proposals_jsonl data/GAIC/${DATANAME}/artifacts/public_teachers/proposals/teacher_proposals_public_gaic_260324_r0.jsonl \
  --use_real_expensive 0 \
  --run_vlm_teacher 0 \
  --run_detailed_report 0 \
  --run_training_labels 1 \
  --safe_leftover_policy ignore \
  --auto_leftover_variants 0 \
  --run_gaic_benchmark_eval 1 \
  --run_gaic_subject_region_ab 0 \
  --run_viz 0
```

설명:

- 이 lane은 `feats_c1.jsonl`, `routed_c7` feature, public teacher proposals를 재사용한다.
- 현재 benchmark evaluator는 `use_real_expensive=0/1` 둘 다 지원한다. real-expensive run에서는 기존 teacher score/training label 산출물을 그대로 읽어 후처리만 다시 수행하면 된다.
- `gaic_260324_r0_saliency_v4tp_proxy` 로컬 재실행으로 teacher proposal 주입이 반영된 training-label / benchmark report 갱신을 확인했다.

### 4.1i full expensive lane은 언제 서버에서 다시 돌려야 하는가

아래 조건이면 로컬보다 서버 재실행이 맞다.

- `README 4.1`의 `use_real_expensive=1` 전체 재현이 필요
- synthetic caption + real expensive scorer까지 포함한 최종 artifact가 필요
- local GPU에서 OpenCLIP align model이 OOM나고 CPU fallback 속도가 비현실적일 때

권장 서버 명령:

```bash
source /media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/activate

GPU_IDS=0,1,2
N_WORKERS=$(awk -F',' '{print NF}' <<< "${GPU_IDS}")
DATANAME=All
RUN_TAG=gaic_260324_r0_saliency_v4tp_full

bash src/scripts/run_gaic_to_teacher_e2e.sh \
  --data_dir data/GAIC/${DATANAME} \
  --image_root data/Publics/GAIC/images \
  --run_tag ${RUN_TAG} \
  --skip_existing 1 \
  --gaic_generate_captions 1 \
  --run_filter 0 \
  --run_c1 1 \
  --run_c2 0 \
  --run_c3 0 \
  --run_c3_enrich 0 \
  --run_c4 0 \
  --run_c5 0 \
  --run_c6 0 \
  --run_merge 0 \
  --run_subject_routing 1 \
  --run_c7_saliency 1 \
  --c7_saliency_priority quality_first \
  --teacher_proposals_jsonl data/GAIC/${DATANAME}/artifacts/public_teachers/proposals/teacher_proposals_public_gaic_260324_r0.jsonl \
  --use_real_expensive 1 \
  --run_vlm_teacher 0 \
  --run_detailed_report 0 \
  --run_training_labels 1 \
  --safe_leftover_policy ignore \
  --auto_leftover_variants 0 \
  --run_gaic_benchmark_eval 0 \
  --run_gaic_subject_region_ab 0 \
  --run_viz 0 \
  --extract_gpu_ids ${GPU_IDS} \
  --num_workers ${N_WORKERS} \
  --teacher_gpu_ids ${GPU_IDS} \
  --teacher_num_workers ${N_WORKERS}
```

서버 재실행 전에 확인할 것:

- local에서 복사해온 `teacher_scores_ar_<run>.jsonl`은 중간 truncate가 없는지 먼저 검증
- public teacher proposal JSONL이 `1236`행 완전본인지 확인
- server run 뒤에는 `repair_teacher_outputs.py`를 한 번 더 돌려 parse error가 `0`인지 확인

### 4.1j 기존 `gaic_260324_r2` real-expensive 산출물 재사용 + full benchmark/report만 다시 생성

`gaic_260324_r2` 처럼 이미 아래 산출물이 완성돼 있다면:

- `data/GAIC/All/artifacts/precompute/feats_c2c3c5_v2_strict_enriched_routed_c7_saliency.jsonl`
- `data/GAIC/All/artifacts/candidates/candidates_ar_gaic_260324_r2.jsonl`
- `data/GAIC/All/artifacts/teacher/scores/teacher_scores_ar_gaic_260324_r2.jsonl`
- `data/GAIC/All/artifacts/training_labels/gaic_260324_r2`

전체 e2e를 다시 돌릴 필요는 없습니다. 용도에 따라 아래 두 경로 중 하나를 쓰면 됩니다.

1. teacher score는 그대로 두고 benchmark/report만 다시 생성
2. 최신 `score_teacher.py` 변경까지 반영하려고 teacher -> training labels -> benchmark만 다시 수행
3. 최신 `subject_region.py + generate_candidates.py + score_teacher.py + benchmark renderer` 누적 변경을 전부 반영하려고 `subject reroute -> c7 -> candidates -> teacher -> labels -> benchmark`를 다시 수행

#### A. benchmark / sample panel만 다시 생성

이 경로는 가장 가볍습니다. 기존 `teacher_scores`와 `training_labels`를 그대로 사용하므로, scorer 재계산 없이 최신 `run_gaic_benchmark_eval.py` 렌더러와 분석 로직만 반영합니다.

```bash
# source /your/server/venv/bin/activate

DATANAME=All
RUN_TAG=gaic_260324_r2

python3 src/scripts/run_gaic_benchmark_eval.py \
  --candidates_jsonl data/GAIC/${DATANAME}/artifacts/candidates/candidates_ar_${RUN_TAG}.jsonl \
  --features_jsonl data/GAIC/${DATANAME}/artifacts/precompute/feats_c2c3c5_v2_strict_enriched_routed_c7_saliency.jsonl \
  --teacher_jsonl data/GAIC/${DATANAME}/artifacts/teacher/scores/teacher_scores_ar_${RUN_TAG}.jsonl \
  --training_label_dir data/GAIC/${DATANAME}/artifacts/training_labels/${RUN_TAG} \
  --gaic_train_json data/Publics/GAIC/annotations_json/instances_train.json \
  --gaic_test_json data/Publics/GAIC/annotations_json/instances_test.json \
  --image_dir data/GAIC/${DATANAME}/images \
  --output_dir data/GAIC/${DATANAME}/artifacts/reports/gaic_benchmark_eval_${RUN_TAG} \
  --sample_count 8 \
  --max_images 0
```

이 명령은 `use_real_expensive=1` teacher run도 지원합니다. evaluator는 `teacher_scores` 안의 config를 읽어 `A_macro` 활성 여부를 그대로 반영합니다.

#### B. 최신 scorer 변경까지 반영해 teacher -> labels -> benchmark만 다시 생성

`score_teacher.py`가 바뀌었고 `A/S/C/T` 계산 자체를 다시 하고 싶다면, 기존 feature / candidate / public teacher proposal을 재사용하고 teacher 이후 단계만 다시 내려도 됩니다.

```bash
# source /your/server/venv/bin/activate

GPU_IDS=0,1,2
N_WORKERS=$(awk -F',' '{print NF}' <<< "${GPU_IDS}")
DATANAME=All
RUN_TAG=gaic_260324_r2

bash src/scripts/run_gaic_to_teacher_e2e.sh \
  --server_mode 1 \
  --data_dir data/GAIC/${DATANAME} \
  --image_root data/Publics/GAIC/images \
  --run_tag ${RUN_TAG} \
  --skip_existing 0 \
  --gaic_generate_captions 0 \
  --run_filter 0 \
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
  --run_teacher 1 \
  --use_real_expensive 1 \
  --run_training_labels 1 \
  --run_detailed_report 1 \
  --run_vlm_teacher 0 \
  --extract_gpu_ids ${GPU_IDS} \
  --num_workers ${N_WORKERS} \
  --teacher_gpu_ids ${GPU_IDS} \
  --teacher_num_workers ${N_WORKERS} \
  --run_gaic_benchmark_eval 1 \
  --gaic_benchmark_sample_count 8 \
  --run_gaic_subject_region_ab 0 \
  --run_viz 0
```

주의:

- 이 경로는 `gaic_260324_r2`의 기존 `candidates`와 `routed_c7` feature를 입력으로 그대로 씁니다.
- 즉 extraction / saliency / reroute / candidate generation은 건드리지 않고, 최신 scorer 및 benchmark 코드만 덮어씁니다.
- 기존 `teacher_scores_ar_gaic_260324_r2.jsonl`을 덮어쓸 것이므로, 보존이 필요하면 먼저 별도 백업 run tag로 복사하는 편이 안전합니다.

#### C. 지금까지 누적된 canonical guidance / support seed / candidate-generation 보완까지 전부 반영

이 경로가 “현재 코드 기준 최신 산출물”을 만드는 표준 템플릿입니다. 즉 아래 변경들이 모두 반영됩니다.

- `subject_region.py`의 reroute / support-map / reliability / fallback 보강
- `generate_candidates.py`의 guidance seed / support-derived candidate family 변경
- `score_teacher.py`의 support-map native macro scoring 변경
- `run_gaic_benchmark_eval.py`의 최신 sample overlay / panel / 진단 로직
- `run_gaic_benchmark_eval.py`는 sample 상세 패널용 real-expensive 값을 `samples/sample_expensive_cache.jsonl`에 캐시한다.
- 처음 한 번은 sample 대상 후보(`GT MOS best / Ge GT best / Ge prod best`)에 대해 OpenCLIP/NIMA를 다시 계산할 수 있지만, 같은 `output_dir`로 재실행하면 이후에는 이 cache를 재사용하므로 OpenCLIP/NIMA를 다시 올리지 않는다.
- cache를 명시적으로 다시 만들고 싶으면 `.../samples/sample_expensive_cache.jsonl`만 지운 뒤 같은 benchmark 명령을 다시 실행하면 된다.

중요:

- 이 경우에는 **candidate generation을 반드시 다시 수행해야 합니다.**
- 이유는 최신 보완안이 scorer만 바꾼 것이 아니라 `candidate seed`와 `candidate family` 자체를 바꾸기 때문입니다.
- 따라서 `B` 경로로는 “최신 scorer on old pool”은 만들 수 있어도, “최신 candidate pool + 최신 scorer”는 만들 수 없습니다.

가장 현실적인 재사용 경로는 `base precompute`는 유지하고, `subject routing -> c7 saliency augment -> reroute -> candidates -> teacher -> training labels -> benchmark`만 다시 수행하는 것입니다.

```bash
# source /your/server/venv/bin/activate

GPU_IDS=0,1,2
N_WORKERS=$(awk -F',' '{print NF}' <<< "${GPU_IDS}")

RUN_TAG=gaic_260330_r0
DATANAME=All

bash src/scripts/run_gaic_to_teacher_e2e.sh \
  --server_mode 1 \
  --data_dir data/GAIC/${DATANAME} \
  --image_root data/Publics/GAIC/images \
  --run_tag ${RUN_TAG} \
  --skip_existing 0 \
  --gaic_generate_captions 0 \
  --run_filter 0 \
  --run_c1 0 \
  --run_c2 0 \
  --run_c3 0 \
  --run_c3_enrich 0 \
  --run_c4 0 \
  --run_c5 0 \
  --run_c6 0 \
  --run_merge 0 \
  --run_subject_routing 1 \
  --run_c7_saliency 1 \
  --c7_saliency_priority quality_first \
  --teacher_proposals_jsonl data/GAIC/${DATANAME}/artifacts/public_teachers/proposals/teacher_proposals_public_gaic_260324_r1.jsonl \
  --run_candidates 1 \
  --run_teacher 1 \
  --use_real_expensive 1 \
  --run_training_labels 1 \
  --run_detailed_report 1 \
  --run_vlm_teacher 0 \
  --extract_gpu_ids ${GPU_IDS} \
  --num_workers ${N_WORKERS} \
  --teacher_gpu_ids ${GPU_IDS} \
  --teacher_num_workers ${N_WORKERS} \
  --run_gaic_benchmark_eval 1 \
  --gaic_benchmark_sample_count 8 \
  --run_gaic_subject_region_ab 1 \
  --run_viz 1 \
  | tee src/scripts/logs/run_gaic_to_teacher_e2e_${DATANAME}_${RUN_TAG}.log
```

설명:

- `run_c1~run_c6=0`, `run_merge=0` 이므로 무거운 base extraction은 재사용합니다.
- `run_subject_routing=1` 은 최신 `subject_region.py` 기준으로 routed feature를 다시 만듭니다.
- `run_c7_saliency=1` 은 최신 saliency augment 결과를 다시 반영합니다.
- `teacher_proposals_jsonl` 은 기존 public teacher proposal 산출물을 그대로 재사용합니다.
- `run_candidates=1` 이 핵심입니다. support seed / guidance candidate family가 여기서 다시 생성됩니다.
- 이후 `run_teacher=1 -> run_training_labels=1 -> run_gaic_benchmark_eval=1` 로 최신 scorer/label/report까지 한 번에 갱신됩니다.

보존이 필요하면:

- 기존 `gaic_260324_r2`를 유지하고 새 run tag를 쓰는 편이 더 안전합니다. 예: `gaic_260324_r2_refresh1`
- 반대로 동일 경로 산출물을 업데이트하려면 위처럼 같은 `run_tag`와 `skip_existing=0`을 사용하면 됩니다.

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
- `--auto_leftover_variants 0|1`: 기본 monotonic output 외 추가 variant 자동 생성 여부. 기본 `0`
- `--leftover_variant_policies CSV`: opt-in variant 정책 목록. 기본 `keep_negative,promote_soft_positive`
- `--run_training_label_debug_viz 0|1`: training label 생성 뒤 GAIC GT 대비 debug visualization 자동 생성 여부. 기본 `0`
- `--training_label_debug_viz_sample_size N`: debug viz 균등 샘플 수. 기본 `50`
- `--training_label_debug_viz_out_dir PATH`: debug viz 출력 경로. 기본 `<training_labels_dir>/debug_visualizations_balanced50_bottomneg`
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

기본 wrapper 실행에서 `--run_training_labels 1` 이면 아래 main output만 생성됩니다.

- `${RUN_TAG}_leftover_ignore_monotonic`

즉 현재 wrapper 기본 정책은

- `safe_leftover_policy=ignore`
- monotonic score / monotonic split 사용
- 추가 variant 자동 생성 비활성화

입니다.

추가 variant는 `--auto_leftover_variants 1` 일 때만 opt-in으로 생성됩니다.

- `${RUN_TAG}_leftover_keepneg_monotonic`
- `${RUN_TAG}_leftover_softpos_monotonic`

## 5.3 GT Score Cache for Debug Visualization

`debug_visualizations_*` 에서 GT bbox 라벨에 `MOS` 뿐 아니라 `training score_prob` 도 같이 보려면, official GAIC GT crop을 현재 teacher/training-label scoring 문맥에 다시 태운 cache가 필요합니다.

- cache 파일명 기본값: `<training_labels_dir>/gaic_gt_score_cache_free.jsonl`
- 현재 구현은 official GAIC GT crop이 free-form 이라는 점을 맞춰 `FREE` 문맥 score만 계산합니다.
- by-AR debug viz에서도 GT bbox는 official free-form bbox를 그대로 reference overlay로 쓰므로, 표시되는 GT `Score` 역시 `FREE-context training score_prob` 입니다.
- score 정의는 raw full candidate pool이 아니라 **training label builder가 실제로 쓰는 deduped teacher subset(`cheap_top_m + selected_topk + hard_negatives + also_considered_rejected + best + baseline`) 기준 local normalization** 입니다. 즉 debug viz의 GT `Score`는 현행 training-label `score_prob`와 같은 축으로 해석하면 됩니다.
- 기본 GT cache 생성은 cheap + macro recompute + policy_safe + local normalization까지만 다시 태웁니다. 따라서 GT `A_macro`는 `NA`일 수 있습니다.
- GT checklist panel에서 `A_macro`까지 채우려면, GT bbox에 대해서도 expensive stage를 다시 계산한 cache를 서버에서 생성해야 합니다.

로컬에서 현재 선택된 debug-viz subset만 빠르게 계산:

```bash
source /media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/activate

python3 src/scripts/build_gaic_gt_score_cache.py \
  --candidates_jsonl data/GAIC/All/artifacts/candidates/candidates_ar_gaic_260330_r0.jsonl \
  --features_jsonl data/GAIC/All/artifacts/precompute/feats_c2c3c5_v2_strict_enriched_routed_c7_saliency.jsonl \
  --teacher_jsonl data/GAIC/All/artifacts/teacher/scores/teacher_scores_ar_gaic_260330_r0_monotonic.jsonl \
  --gaic_train_json data/Publics/GAIC/annotations_json/instances_train.json \
  --gaic_test_json data/Publics/GAIC/annotations_json/instances_test.json \
  --image_ids_csv data/GAIC/All/artifacts/training_labels/gaic_260330_r0_leftover_ignore_monotonic/debug_visualizations_balanced50_bottomneg/summary/selected_images.csv \
  --out_jsonl data/GAIC/All/artifacts/training_labels/gaic_260330_r0_leftover_ignore_monotonic/gaic_gt_score_cache_free.jsonl

python3 src/scripts/build_gaic_training_label_debug_viz.py \
  --label_json data/GAIC/All/artifacts/training_labels/gaic_260330_r0_leftover_ignore_monotonic/label_json/gaic_like_labels_full.json \
  --batch_jsonl data/GAIC/All/artifacts/training_labels/gaic_260330_r0_leftover_ignore_monotonic/train_conditional_detr_batch.jsonl \
  --gaic_gt_train_json data/Publics/GAIC/annotations_json/instances_train.json \
  --gaic_gt_test_json data/Publics/GAIC/annotations_json/instances_test.json \
  --gt_score_cache_jsonl data/GAIC/All/artifacts/training_labels/gaic_260330_r0_leftover_ignore_monotonic/gaic_gt_score_cache_free.jsonl \
  --image_root data/GAIC/All/images \
  --subject_mode_vocab data/GAIC/All/artifacts/training_labels/gaic_260330_r0_leftover_ignore_monotonic/subject_mode_vocab.json \
  --sample_size 50 \
  --seed 42 \
  --out_dir data/GAIC/All/artifacts/training_labels/gaic_260330_r0_leftover_ignore_monotonic/debug_visualizations_balanced50_bottomneg
```

GT `A_macro`까지 채워야 하면, 서버에서 expensive cache를 같이 만든 뒤 로컬에서는 그 산출물을 재사용하는 것이 맞습니다.

선택된 debug-viz subset만 빠르게 채우려면 `--image_ids_csv` 를 같이 주는 subset 템플릿을 쓰고,
전체 GAIC GT cache를 만들려면 `--image_ids_csv` 없이 full-cache 템플릿을 쓰면 됩니다.

```bash
# server: selected debug-viz subset only
#source /your/server/venv/bin/activate

RUN_TAG=gaic_260402_reroute_v1

python3 src/scripts/build_gaic_gt_score_cache.py \
  --candidates_jsonl data/GAIC/All/artifacts/candidates/candidates_ar_${RUN_TAG}.jsonl \
  --features_jsonl data/GAIC/All/artifacts/precompute/feats_c2c3c5_v2_strict_enriched_routed_c7_saliency.jsonl \
  --teacher_jsonl data/GAIC/All/artifacts/teacher/scores/teacher_scores_ar_${RUN_TAG}.jsonl \
  --gaic_train_json data/Publics/GAIC/annotations_json/instances_train.json \
  --gaic_test_json data/Publics/GAIC/annotations_json/instances_test.json \
  --image_ids_csv data/GAIC/All/artifacts/training_labels/${RUN_TAG}_leftover_ignore_monotonic/debug_visualizations_balanced50_bottomneg/summary/selected_images.csv \
  --c1_jsonl data/GAIC/All/artifacts/precompute/feats_c1.jsonl \
  --image_root data/GAIC/All/images \
  --sample_expensive_cache_jsonl data/GAIC/All/artifacts/training_labels/${RUN_TAG}_leftover_ignore_monotonic/gaic_gt_expensive_cache_free.jsonl \
  --out_jsonl data/GAIC/All/artifacts/training_labels/${RUN_TAG}_leftover_ignore_monotonic/gaic_gt_score_cache_free.jsonl
```

이때:

- `gaic_gt_score_cache_free.jsonl` 에 GT `macro_scores.A_macro` 가 함께 저장됩니다.
- `gaic_gt_expensive_cache_free.jsonl` 은 GT bbox expensive signal cache라서, 이후 재생성 시 재사용됩니다.
- 로컬은 위 두 파일만 동기화한 뒤 `build_gaic_training_label_debug_viz.py` 를 다시 실행하면 됩니다.

전체 GAIC GT cache는 서버에서 한 번 생성하고, 로컬에서는 cache만 복사해서 debug viz를 다시 그리는 방식을 권장합니다.

```bash
# server: full GAIC GT cache
#source /your/server/venv/bin/activate

RUN_TAG=gaic_260402_reroute_v1

python3 src/scripts/build_gaic_gt_score_cache.py \
  --candidates_jsonl data/GAIC/All/artifacts/candidates/candidates_ar_${RUN_TAG}.jsonl \
  --features_jsonl data/GAIC/All/artifacts/precompute/feats_c2c3c5_v2_strict_enriched_routed_c7_saliency.jsonl \
  --teacher_jsonl data/GAIC/All/artifacts/teacher/scores/teacher_scores_ar_${RUN_TAG}.jsonl \
  --gaic_train_json data/Publics/GAIC/annotations_json/instances_train.json \
  --gaic_test_json data/Publics/GAIC/annotations_json/instances_test.json \
  --c1_jsonl data/GAIC/All/artifacts/precompute/feats_c1.jsonl \
  --image_root data/GAIC/All/images \
  --sample_expensive_cache_jsonl data/GAIC/All/artifacts/training_labels/${RUN_TAG}_leftover_ignore_monotonic/gaic_gt_expensive_cache_free.jsonl \
  --out_jsonl data/GAIC/All/artifacts/training_labels/${RUN_TAG}_leftover_ignore_monotonic/gaic_gt_score_cache_free.jsonl
```

그 다음 로컬에서는 위 cache 파일을 그대로 사용해 `build_gaic_training_label_debug_viz.py` 만 다시 실행하면 됩니다.

로컬에서 teacher JSONL I/O 또는 GT cache 계산이 너무 무거우면, 아래처럼 **server에서 refined training label / GT cache / debug viz를 먼저 만든 뒤 로컬에는 산출물만 동기화**하는 방식을 권장합니다.

```bash
# server
#source /your/server/venv/bin/activate

RUN_TAG=gaic_260402_reroute_v1
OUT_DIR=data/GAIC/All/artifacts/training_labels/${RUN_TAG}_leftover_ignore_monotonic_refined

python3 src/scripts/build_finalscore_training_data.py \
  --teacher_scores_jsonl data/GAIC/All/artifacts/teacher/scores/teacher_scores_ar_${RUN_TAG}.jsonl \
  --out_dir ${OUT_DIR} \
  --image_root data/GAIC/All/images \
  --safe_leftover_policy ignore

mkdir -p ${OUT_DIR}/coco

python3 src/scripts/convert_sstk_detr_batch_to_gaic_like.py \
  --batch_jsonl ${OUT_DIR}/train_conditional_detr_batch.jsonl \
  --gaic_reference_json data/Publics/GAIC/annotations_json/instances_train.json \
  --gaic_train_reference_json data/Publics/GAIC/annotations_json/instances_train.json \
  --gaic_test_reference_json data/Publics/GAIC/annotations_json/instances_test.json \
  --out_json ${OUT_DIR}/label_json/gaic_like_labels_full.json \
  --out_train_json ${OUT_DIR}/label_json/gaic_like_labels_train.json \
  --out_test_json ${OUT_DIR}/label_json/gaic_like_labels_test.json \
  --out_unassigned_json ${OUT_DIR}/label_json/gaic_like_labels_unassigned.json \
  --out_summary_json ${OUT_DIR}/label_json/gaic_like_conversion_summary.json \
  --out_guide_md ${OUT_DIR}/label_json/GAIC_LIKE_LABEL_FORMAT_KO.md

python3 src/scripts/build_gaic_gt_score_cache.py \
  --candidates_jsonl data/GAIC/All/artifacts/candidates/candidates_ar_${RUN_TAG}.jsonl \
  --features_jsonl data/GAIC/All/artifacts/precompute/feats_c2c3c5_v2_strict_enriched_routed_c7_saliency.jsonl \
  --teacher_jsonl data/GAIC/All/artifacts/teacher/scores/teacher_scores_ar_${RUN_TAG}.jsonl \
  --gaic_train_json data/Publics/GAIC/annotations_json/instances_train.json \
  --gaic_test_json data/Publics/GAIC/annotations_json/instances_test.json \
  --c1_jsonl data/GAIC/All/artifacts/precompute/feats_c1.jsonl \
  --image_root data/GAIC/All/images \
  --sample_expensive_cache_jsonl ${OUT_DIR}/gaic_gt_expensive_cache_free.jsonl \
  --out_jsonl ${OUT_DIR}/gaic_gt_score_cache_free.jsonl

python3 src/scripts/build_gaic_training_label_debug_viz.py \
  --label_json ${OUT_DIR}/label_json/gaic_like_labels_full.json \
  --batch_jsonl ${OUT_DIR}/train_conditional_detr_batch.jsonl \
  --gaic_gt_train_json data/Publics/GAIC/annotations_json/instances_train.json \
  --gaic_gt_test_json data/Publics/GAIC/annotations_json/instances_test.json \
  --gt_score_cache_jsonl ${OUT_DIR}/gaic_gt_score_cache_free.jsonl \
  --teacher_jsonl data/GAIC/All/artifacts/teacher/scores/teacher_scores_ar_${RUN_TAG}.jsonl \
  --image_root data/GAIC/All/images \
  --subject_mode_vocab ${OUT_DIR}/subject_mode_vocab.json \
  --sample_size 50 \
  --seed 42 \
  --out_dir ${OUT_DIR}/debug_visualizations_balanced50_bottomneg
```

이 refined lane은 다음을 반영한다.

- placement family-aware composition aggregation
- `S_macro` bottleneck aggregation
- conditional safety relax
- current-score selection refresh
- `protected_secondary_positive`
- colored checklist debug panel
- refined COCO/GAIC-like export를 명시적으로 먼저 생성한 뒤 debug viz를 그림

참고:

- `teacher_scores_ar_<RUN_TAG>.jsonl` 원본이 부분 손상되었거나 truncated 된 경우가 있으면, 현재 스크립트는 같은 디렉토리의 `teacher_scores_ar_<RUN_TAG>_monotonic.jsonl` 이 존재할 때 자동 fallback 합니다.
- `A_macro` 유무 확인은 `train_conditional_detr_batch.jsonl`이 아니라 `train_conditional_detr_canonical.jsonl`의 `candidates[].macro_targets.A_macro` 기준으로 보는 것이 맞습니다. canonical export에서 non-null이면 expensive/macro 산출은 살아 있는 것입니다.
- GT checklist의 `A_macro`는 `gaic_gt_score_cache_free.jsonl` 기준입니다. cheap-only GT cache면 `A_macro=NA`가 정상이고, server-side expensive GT cache를 다시 만든 뒤에만 채워집니다.

또한 각 training label dir의 `label_json/` 아래에는 다음 split export가 함께 생성됩니다.

- `gaic_like_labels_full.json`
- `gaic_like_labels_train.json`
- `gaic_like_labels_test.json`
- `gaic_like_labels_unassigned.json`

`unassigned` 는 현재 local GAIC subset에 존재하지만 public `instances_train/test.json` 어디에도 image_id가 없는 샘플입니다.

safe high-score leftover는

- 최종 chosen보다 점수는 높지만
- `selected_topk` diverse positive set에는 들어가지 않았고
- severe reject도 아닌

후보를 뜻합니다.

기본 `ignore + monotonic` 정책에서는 이 후보들이 `ignored_candidates`로 분리되고, weak soft-positive tail도 pruning되며, high-score unsafe/hard는 `overflow_candidates`로 빠집니다. 즉 main `candidate_pool` 기준으로는 `positive > negative`가 강제됩니다.

opt-in variant 실험을 위해 아래 분기를 만들 수 있습니다.

## 5.4 Crop Score Priority Experiment Runner

우선순위 기반 scorer 실험은 아래 runner로 수행합니다.

2026-04-06 기준의 권장 채택 scorer는 `single_stage2_utility` 입니다. `current_refined`는 regression / rollback safety를 위한 baseline으로만 유지하는 것을 권장합니다.

- script: `src/scripts/run_crop_score_priority_experiments.py`
- 핵심 입력:
  - `teacher_scores_ar_<RUN_TAG>.jsonl`
  - `candidates_ar_<RUN_TAG>.jsonl`
  - routed feature jsonl
  - GAIC GT json
- 핵심 출력:
  - experiment별 training labels
  - experiment별 benchmark report
  - stage winner 비교 리포트
  - optional GT cache / debug viz

### 로컬 smoke 검증

로컬에서 전체 teacher JSONL을 그대로 쓰면 training-label build가 무겁습니다. smoke 검증은 `--teacher_max_images` 로 subset을 만든 뒤 돌리는 편이 맞습니다.

```bash
source /media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/activate

python3 src/scripts/run_crop_score_priority_experiments.py \
  --teacher_jsonl data/GAIC/All/artifacts/teacher/scores/teacher_scores_ar_gaic_260402_reroute_v1.jsonl \
  --candidates_jsonl data/GAIC/All/artifacts/candidates/candidates_ar_gaic_260402_reroute_v1.jsonl \
  --features_jsonl data/GAIC/All/artifacts/precompute/feats_c2c3c5_v2_strict_enriched_routed.jsonl \
  --gaic_train_json data/Publics/GAIC/annotations_json/instances_train.json \
  --gaic_test_json data/Publics/GAIC/annotations_json/instances_test.json \
  --image_root data/GAIC/All/images \
  --c1_jsonl data/GAIC/All/artifacts/precompute/feats_c1.jsonl \
  --out_root data/GAIC/All/artifacts/reports/single_scorer_stage2_smoke_gaic_260402_reroute_v1 \
  --experiment_ids baseline_current_refined single_stage2_utility \
  --teacher_max_images 16 \
  --benchmark_max_images 16 \
  --winner_benchmark_max_images 16 \
  --run_winner_refresh 0 \
  --run_full_expensive 1 \
  --build_debug_viz_for_all 1 \
  --debug_sample_size 12 \
  --debug_sample_size_all 12 \
  --debug_skip_by_ar 1 \
  --parallel_experiment_jobs 1 \
  --worker_cpu_threads 2 \
  --benchmark_num_workers 1 \
  --gt_cache_num_workers 1 \
  --force 1
```

이 smoke는 아래를 확인하는 용도입니다.

- `score_profile` 적용
- training label refresh / monotonic QA
- benchmark rescoring
- GT cache / debug viz 생성
- experiment summary / markdown report 생성

### 서버 full-expensive 실험

#### `single_stage2_utility` 채택 기준 full rerun

전체 GAIC overlap 대상으로 채택 scorer를 검증하려면, `baseline_current_refined`와 `single_stage2_utility` 두 개만 full rerun 하는 편이 맞습니다. stage1~4 전체 sweep은 historical comparison 용도로만 두는 것을 권장합니다.

```bash
#source /your/server/venv/bin/activate

GPU_IDS=0,1,2,3,4,5,6,7
N_WORKERS=$(awk -F',' '{print NF}' <<< "${GPU_IDS}")
PARALLEL_EXPERIMENT_JOBS=${N_WORKERS}
CPU_THREADS_PER_JOB=-1

export CUDA_VISIBLE_DEVICES=${GPU_IDS}

RUN_TAG=gaic_260402_reroute_v1
OUT_ROOT=data/GAIC/All/artifacts/reports/crop_score_priority_experiments_${RUN_TAG}_single_stage2_full

python3 src/scripts/run_crop_score_priority_experiments.py \
  --teacher_jsonl data/GAIC/All/artifacts/teacher/scores/teacher_scores_ar_${RUN_TAG}.jsonl \
  --candidates_jsonl data/GAIC/All/artifacts/candidates/candidates_ar_${RUN_TAG}.jsonl \
  --features_jsonl data/GAIC/All/artifacts/precompute/feats_c2c3c5_v2_strict_enriched_routed.jsonl \
  --gaic_train_json data/Publics/GAIC/annotations_json/instances_train.json \
  --gaic_test_json data/Publics/GAIC/annotations_json/instances_test.json \
  --image_root data/GAIC/All/images \
  --c1_jsonl data/GAIC/All/artifacts/precompute/feats_c1.jsonl \
  --out_root ${OUT_ROOT} \
  --experiment_ids baseline_current_refined single_stage2_utility \
  --benchmark_max_images 0 \
  --winner_benchmark_max_images 0 \
  --run_winner_refresh 1 \
  --run_full_expensive 1 \
  --skip_debug_viz 1 \
  --parallel_experiment_jobs ${PARALLEL_EXPERIMENT_JOBS} \
  --worker_cpu_threads ${CPU_THREADS_PER_JOB} \
  --benchmark_gpu_ids ${GPU_IDS} \
  --benchmark_num_workers ${N_WORKERS} \
  --gt_cache_gpu_ids ${GPU_IDS} \
  --gt_cache_num_workers ${N_WORKERS} \
  --force 1 \
  | tee src/scripts/logs/run_crop_score_priority_experiments_${RUN_TAG}_single_stage2_full.log
```

이 1차 full run은 기본적으로:

- baseline `current_refined`
- adopted scorer `single_stage2_utility`
- full benchmark / winner refresh / experiment summary

를 생성합니다.

#### 완전한 GT 패널 + by-AR debug viz 생성

위 1차 full run만으로는 아래 현상이 남을 수 있습니다.

- GT 우측 checklist panel의 `A_macro = NA`
- `*_by_ar` 결과 디렉터리가 비어 있음

원인:

- GT `A_macro = NA`
  - GT bbox에 대해 expensive stage를 다시 태운 `gaic_gt_score_cache_free.jsonl` 이 아직 없거나 cheap-only cache만 연결된 상태
- `*_by_ar` 비어 있음
  - `--debug_skip_by_ar 1` 로 렌더링했거나, debug viz 자체를 생략한 상태

완전한 결과를 보려면, 각 experiment에 대해 GT expensive cache를 먼저 만들고 그 다음 `build_gaic_training_label_debug_viz.py` 를 `--skip_by_ar 0` 으로 다시 실행해야 합니다.

```bash
#source /your/server/venv/bin/activate

RUN_TAG=gaic_260402_reroute_v1
OUT_ROOT=data/GAIC/All/artifacts/reports/crop_score_priority_experiments_${RUN_TAG}_single_stage2_full

python3 src/scripts/build_gaic_gt_score_cache.py \
  --candidates_jsonl data/GAIC/All/artifacts/candidates/candidates_ar_${RUN_TAG}.jsonl \
  --features_jsonl data/GAIC/All/artifacts/precompute/feats_c2c3c5_v2_strict_enriched_routed.jsonl \
  --teacher_jsonl ${OUT_ROOT}/teacher_subset.jsonl \
  --gaic_train_json data/Publics/GAIC/annotations_json/instances_train.json \
  --gaic_test_json data/Publics/GAIC/annotations_json/instances_test.json \
  --c1_jsonl data/GAIC/All/artifacts/precompute/feats_c1.jsonl \
  --image_root data/GAIC/All/images \
  --sample_expensive_cache_jsonl ${OUT_ROOT}/shared_gt_expensive_cache_free.jsonl \
  --score_profile current_refined \
  --score_profile_overrides_json ${OUT_ROOT}/baseline_current_refined/score_profile_overrides.json \
  --out_jsonl ${OUT_ROOT}/baseline_current_refined/training_labels/gaic_gt_score_cache_free.jsonl

python3 src/scripts/build_gaic_gt_score_cache.py \
  --candidates_jsonl data/GAIC/All/artifacts/candidates/candidates_ar_${RUN_TAG}.jsonl \
  --features_jsonl data/GAIC/All/artifacts/precompute/feats_c2c3c5_v2_strict_enriched_routed.jsonl \
  --teacher_jsonl ${OUT_ROOT}/teacher_subset.jsonl \
  --gaic_train_json data/Publics/GAIC/annotations_json/instances_train.json \
  --gaic_test_json data/Publics/GAIC/annotations_json/instances_test.json \
  --c1_jsonl data/GAIC/All/artifacts/precompute/feats_c1.jsonl \
  --image_root data/GAIC/All/images \
  --sample_expensive_cache_jsonl ${OUT_ROOT}/shared_gt_expensive_cache_free.jsonl \
  --score_profile single_stage2 \
  --score_profile_overrides_json ${OUT_ROOT}/single_stage2_utility/score_profile_overrides.json \
  --out_jsonl ${OUT_ROOT}/single_stage2_utility/training_labels/gaic_gt_score_cache_free.jsonl

python3 src/scripts/build_gaic_training_label_debug_viz.py \
  --label_json ${OUT_ROOT}/baseline_current_refined/training_labels/label_json/gaic_like_labels_full.json \
  --batch_jsonl ${OUT_ROOT}/baseline_current_refined/training_labels/train_conditional_detr_batch.jsonl \
  --gaic_gt_train_json data/Publics/GAIC/annotations_json/instances_train.json \
  --gaic_gt_test_json data/Publics/GAIC/annotations_json/instances_test.json \
  --gt_score_cache_jsonl ${OUT_ROOT}/baseline_current_refined/training_labels/gaic_gt_score_cache_free.jsonl \
  --teacher_jsonl ${OUT_ROOT}/teacher_subset.jsonl \
  --image_root data/GAIC/All/images \
  --subject_mode_vocab ${OUT_ROOT}/baseline_current_refined/training_labels/subject_mode_vocab.json \
  --score_profile_json ${OUT_ROOT}/baseline_current_refined/training_labels/score_profile.json \
  --sample_size 50 \
  --seed 42 \
  --skip_by_ar 0 \
  --format jpg \
  --jpg_quality 50 \
  --out_dir ${OUT_ROOT}/baseline_current_refined/training_labels/debug_visualizations_balanced50_complete

python3 src/scripts/build_gaic_training_label_debug_viz.py \
  --label_json ${OUT_ROOT}/single_stage2_utility/training_labels/label_json/gaic_like_labels_full.json \
  --batch_jsonl ${OUT_ROOT}/single_stage2_utility/training_labels/train_conditional_detr_batch.jsonl \
  --gaic_gt_train_json data/Publics/GAIC/annotations_json/instances_train.json \
  --gaic_gt_test_json data/Publics/GAIC/annotations_json/instances_test.json \
  --gt_score_cache_jsonl ${OUT_ROOT}/single_stage2_utility/training_labels/gaic_gt_score_cache_free.jsonl \
  --teacher_jsonl ${OUT_ROOT}/teacher_subset.jsonl \
  --image_root data/GAIC/All/images \
  --subject_mode_vocab ${OUT_ROOT}/single_stage2_utility/training_labels/subject_mode_vocab.json \
  --score_profile_json ${OUT_ROOT}/single_stage2_utility/training_labels/score_profile.json \
  --sample_size 50 \
  --seed 42 \
  --skip_by_ar 0 \
  --format jpg \
  --jpg_quality 50 \
  --out_dir ${OUT_ROOT}/single_stage2_utility/training_labels/debug_visualizations_balanced50_complete
```

최종적으로 확인할 핵심 파일:

- `${OUT_ROOT}/experiment_summary.json`
- `${OUT_ROOT}/baseline_current_refined/benchmark/benchmark_summary.json`
- `${OUT_ROOT}/single_stage2_utility/benchmark/benchmark_summary.json`
- `${OUT_ROOT}/single_stage2_utility/training_labels/gaic_gt_score_cache_free.jsonl`
- `${OUT_ROOT}/single_stage2_utility/training_labels/debug_visualizations_balanced50_complete/summary/summary.json`

runner와 GT cache 스크립트는 benchmark full-expensive cache와 GT expensive cache를 `<out_root>/shared_*cache*.jsonl` 로 공유하므로, baseline과 adopted scorer 간에도 동일 raw expensive signal을 재사용할 수 있습니다.

GPU 사용 관련 메모:

- teacher scorer 본체는 기존과 동일하게 `run_teacher_scorer.sh` / `run_phaseA_to_teacher_e2e.sh` 의 shard 기반 multi-GPU 실행을 사용합니다. 즉 scorer 단계는 `--teacher_gpu_ids`, `--teacher_num_workers` 로 이미 분산 가능합니다.
- stage scorer 실험에서 GPU를 쓰는 구간은 `run_gaic_benchmark_eval.py --run_full_expensive 1` 과 `build_gaic_gt_score_cache.py --c1_jsonl ... --image_root ...` 입니다.
- 이제 experiment runner는 이 두 구간을 shard precompute로 분리합니다.
  - `--benchmark_num_workers N`: benchmark full-expensive raw cache precompute shard 수
  - `--gt_cache_num_workers N`: GT expensive cache shard 수
  - `--parallel_experiment_jobs N`: training-label build/benchmark 실험 job 병렬 수
  - `--worker_cpu_threads M`: 각 job/shard subprocess가 사용할 BLAS/OpenMP thread cap. `-1` 또는 생략 시 현재 환경의 최대 CPU 코어 수를 읽고 해당 phase의 동시 worker 수로 나눈 auto 값이 적용됩니다.
- `--benchmark_gpu_ids`, `--gt_cache_gpu_ids` 를 비워 두면 우선 `CUDA_VISIBLE_DEVICES` 를 따르고, 그것도 비어 있으면 `nvidia-smi -L` 로 보이는 전체 GPU를 자동 사용합니다. 따라서 멀티-GPU 서버에서는 별도 CSV를 안 줘도 기본적으로 “보이는 GPU 수만큼 shard”가 잡힙니다.
- 반대로 특정 GPU만 쓰고 싶으면 `CUDA_VISIBLE_DEVICES=2,3` 또는 `--benchmark_gpu_ids 2,3 --gt_cache_gpu_ids 2,3` 처럼 제한하면 됩니다.
- 이 shard precompute는 raw expensive signal cache만 병렬 생성하고, 실제 benchmark metric 집계는 cache 재사용 기반 single-process로 마무리합니다. 이렇게 해야 실험 간 raw expensive signal 재사용이 가능하고 summary merge 복잡도도 낮습니다.
- benchmark sample panel이나 GT cache expensive stage 로그에 `[expensive] align model ... on cuda`, `aesthetic backend ...` 가 보이면 실제로 GPU가 사용된 것입니다.

CPU 병렬화 메모:

- `--parallel_experiment_jobs` 는 서로 다른 scorer profile 실험을 병렬로 돌립니다.
- 각 subprocess 내부에서 NumPy/BLAS가 코어를 과점유하면 오히려 느려질 수 있으므로, runner는 `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS` 를 `--worker_cpu_threads` 값으로 자동 제한합니다.
- `--worker_cpu_threads=-1` 이면 `os.cpu_count()` 기준 전체 코어 수를 읽고, phase별 동시 worker 수(`parallel_experiment_jobs`, `benchmark_num_workers`, `gt_cache_num_workers`)로 나눠 auto cap을 계산합니다.
- 고정값을 주고 싶으면 양수로 직접 지정하면 됩니다.

`--build_debug_viz_for_all 1` 을 주면 tuning candidate까지 모두 debug viz를 만들 수 있지만, GT cache와 viz 렌더링 비용이 커지므로 서버에서만 권장합니다.

### 주요 결과 파일

- `<out_root>/experiment_summary.json`
- `<out_root>/EXPERIMENT_REPORT_KO.md`
- `<out_root>/<experiment_id>/training_labels/...`
- `<out_root>/<experiment_id>/benchmark/benchmark_summary.json`
- `<out_root>/<experiment_id>/training_labels/debug_visualizations_balanced50_bottomneg`

현재 objective는 `Ge.score_policy` benchmark 지표 + production-policy GT match 지표를 합산하고, monotonic/safe-pool QA 위반에 penalty를 주는 방식입니다. 세부 식과 stage 정의는 [Implement_Docs/Crop_Score_Impact_Analysis_KO_2026-04-02.md](./Implement_Docs/Crop_Score_Impact_Analysis_KO_2026-04-02.md) 를 참고하면 됩니다.

- `keep_negative`: 이 후보를 기존처럼 `candidate_pool`의 `negative/near_negative`로 유지
- `ignore`: 이 후보를 `ignored_candidates`로 분리하고 negative annotation에서 제외
- `promote_soft_positive`: 이 후보를 `soft_positive`로 승격하고 `matching_targets`에 포함

학습 관점에서 해석하면 다음과 같습니다.

- `keep_negative`:
  - hardest negative를 가장 많이 남긴다.
  - 다만 safe leftover가 chosen/matching target보다 점수가 높은 경우도 negative로 들어가므로, conditional-DETR 분류/랭킹 관점에서 supervision 충돌이 가장 크다.
- `ignore`:
  - safe leftover를 아예 negative pool 밖으로 빼서 contradiction을 줄인다.
  - 현재 운영 기본값인 이유도 이 정책이 positive set과 negative pool 사이의 경계를 가장 안정적으로 만들기 때문이다.
- `promote_soft_positive`:
  - safe leftover를 positive recall 확장용으로 흡수한다.
  - 다만 positive set의 다양성과 허용 범위가 넓어지므로, strict한 top-choice 학습보다 multi-positive tolerance를 주는 실험에 가깝다.

기존 `teacher_scores`를 재사용해 training label만 다시 뽑고 split export까지 갱신하는 명령 템플릿:

```bash
source /media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/activate

RUN_TAG=gaic_260320_r0
BASE=data/GAIC/All/artifacts/training_labels
TEACHER_JSONL=data/GAIC/All/artifacts/teacher/scores/teacher_scores_ar_${RUN_TAG}.jsonl
IMAGE_ROOT=data/GAIC/All/images
GAIC_REF=data/GAIC/All/gaic_reference_available.json

for POLICY in ignore promote_soft_positive; do
  if [ "${POLICY}" = "ignore" ]; then
    OUT_DIR=${BASE}/${RUN_TAG}_leftover_ignore_monotonic
  else
    OUT_DIR=${BASE}/${RUN_TAG}_leftover_softpos_monotonic
  fi

  python3 src/scripts/build_finalscore_training_data.py \
    --teacher_scores_jsonl "${TEACHER_JSONL}" \
    --out_dir "${OUT_DIR}" \
    --image_root "${IMAGE_ROOT}" \
    --safe_leftover_policy "${POLICY}" \
    --strict_validation 1 \
    --report_examples 8

  python3 src/scripts/convert_sstk_detr_labels_to_coco.py \
    --canonical_jsonl "${OUT_DIR}/train_conditional_detr_canonical.jsonl" \
    --batch_jsonl "${OUT_DIR}/train_conditional_detr_batch.jsonl" \
    --out_dir "${OUT_DIR}/coco"

  python3 src/scripts/convert_sstk_detr_batch_to_gaic_like.py \
    --batch_jsonl "${OUT_DIR}/train_conditional_detr_batch.jsonl" \
    --gaic_reference_json "${GAIC_REF}" \
    --gaic_train_reference_json data/Publics/GAIC/annotations_json/instances_train.json \
    --gaic_test_reference_json data/Publics/GAIC/annotations_json/instances_test.json \
    --out_json "${OUT_DIR}/label_json/gaic_like_labels_full.json" \
    --out_train_json "${OUT_DIR}/label_json/gaic_like_labels_train.json" \
    --out_test_json "${OUT_DIR}/label_json/gaic_like_labels_test.json" \
    --out_unassigned_json "${OUT_DIR}/label_json/gaic_like_labels_unassigned.json" \
    --out_summary_json "${OUT_DIR}/label_json/gaic_like_conversion_summary.json" \
    --out_guide_md "${OUT_DIR}/label_json/GAIC_LIKE_LABEL_FORMAT_KO.md"
done
```

전체 e2e wrapper를 재사용하고 싶으면 기본 main output만 그대로 쓰거나, opt-in variant를 켜면 됩니다.

기본 monotonic main output 예:

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
  --safe_leftover_policy ignore
```

추가 variant가 필요하면 아래처럼 명시적으로 켭니다.

```bash
source /media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/activate

bash src/scripts/run_gaic_to_teacher_e2e.sh \
  --server_mode 1 \
  --data_dir data/GAIC/All \
  --image_root data/Publics/GAIC/images \
  --run_tag gaic_260330_r0 \
  --skip_existing 1 \
  --run_candidates 0 \
  --run_teacher 0 \
  --run_vlm_teacher 0 \
  --run_detailed_report 0 \
  --run_training_labels 1 \
  --auto_leftover_variants 0 \
  --leftover_variant_policies keep_negative,promote_soft_positive
```

debug visualization까지 마지막 단계에서 자동으로 만들고 싶으면:

```bash
source /media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/activate

bash src/scripts/run_gaic_to_teacher_e2e.sh \
  --server_mode 1 \
  --data_dir data/GAIC/All \
  --image_root data/Publics/GAIC/images \
  --run_tag gaic_260330_r0 \
  --skip_existing 1 \
  --run_candidates 0 \
  --run_teacher 0 \
  --run_vlm_teacher 0 \
  --run_detailed_report 0 \
  --run_training_labels 1 \
  --run_training_label_debug_viz 1 \
  --training_label_debug_viz_sample_size 50
```

2026-04-01 기준 실제 생성/검증 완료된 경로:

- 기본 monotonic output: [data/GAIC/All/artifacts/training_labels/gaic_260330_r0_leftover_ignore_monotonic](/media/jyju25/T7_4TB_JY/Projects_26/Sources/ImageCropping/data/GAIC/All/artifacts/training_labels/gaic_260330_r0_leftover_ignore_monotonic)
- monotonic debug viz: [data/GAIC/All/artifacts/training_labels/gaic_260330_r0_leftover_ignore_monotonic/debug_visualizations_balanced50_bottomneg](/media/jyju25/T7_4TB_JY/Projects_26/Sources/ImageCropping/data/GAIC/All/artifacts/training_labels/gaic_260330_r0_leftover_ignore_monotonic/debug_visualizations_balanced50_bottomneg)

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
- GAIC-like train split: `data/GAIC/All/artifacts/training_labels/gaic_full/label_json/gaic_like_labels_train.json`
- GAIC-like test split: `data/GAIC/All/artifacts/training_labels/gaic_full/label_json/gaic_like_labels_test.json`
- GAIC-like unassigned split: `data/GAIC/All/artifacts/training_labels/gaic_full/label_json/gaic_like_labels_unassigned.json`
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
