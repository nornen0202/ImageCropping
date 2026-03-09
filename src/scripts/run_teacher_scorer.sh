#!/bin/bash
# ============================================================================
# run_teacher_scorer.sh
# Teacher Scorer (Section 9): Cheap -> Expensive -> Top-K Diversity
# * C4 OCR 기반 text-preservation 항을 cheap/expensive scoring에 반영합니다.
# ============================================================================

: <<'USAGE'
# 기본 실행 (10K_local 데이터)
bash src/scripts/run_teacher_scorer.sh

# 서버 경로 기본값 사용
bash src/scripts/run_teacher_scorer.sh --server_mode 1

# 출력 경로 지정
bash src/scripts/run_teacher_scorer.sh \
  --output_jsonl data/SSTK/10K_local/artifacts/teacher/scores/teacher_scores_ar.jsonl \
  --output_overview_json data/SSTK/10K_local/artifacts/teacher/overview/teacher_scores_overview.json \
  --output_overview_csv data/SSTK/10K_local/artifacts/teacher/overview/teacher_scores_overview_by_ar.csv

# Top-K / Cheap M / Diversity IoU 조정
bash src/scripts/run_teacher_scorer.sh \
  --cheap_top_m 40 --top_k 5 --tau_div 0.75

# 실모델 Expensive 강제 (Aesthetic + cos(E_I,E_T))
bash src/scripts/run_teacher_scorer.sh \
  --use_real_expensive 1 \
  --aesthetic_backend hybrid \
  --c1_jsonl data/SSTK/10K_local/artifacts/precompute/feats_c1.jsonl \
  --align_device auto --aesthetic_device auto \
  --exp_batch_size 128 \
  --exp_preprocess_workers 8 \
  --expensive_eval_top_m 16

# 멀티 GPU 샤딩 실행 (No-Ray)
bash src/scripts/run_teacher_scorer.sh \
  --use_real_expensive 1 \
  --multi_gpu 1 \
  --gpu_ids 0,1,2,3 \
  --num_workers 4

# 시각화 설정 조정
bash src/scripts/run_teacher_scorer.sh \
  --num_viz 120 --target_ar 1:1 --decision_filter crop

# 특정 image_id만 teacher 시각화(보고서/디버그용)
bash src/scripts/run_teacher_scorer.sh \
  --run_viz 1 \
  --viz_image_ids_file data/SSTK/10K_local/artifacts/reports/rerun1_public_e2e/sample_ids_supercat12.txt \
  --image_dir data/SSTK/10K_local/images

# 시각화 생략 (스코어/통계만)
bash src/scripts/run_teacher_scorer.sh --run_viz 0

# QA 리포트만 별도 생성
bash src/scripts/run_teacher_scorer.sh --run_viz 0 --run_qa 1

# 풀런
export CUDA_VISIBLE_DEVICES=0

bash src/scripts/run_teacher_scorer.sh \
  --candidates_jsonl data/SSTK/10K_local/artifacts/candidates/candidates_ar.jsonl \
  --features_jsonl data/SSTK/10K_local/artifacts/precompute/feats_c2c3c5_v2_strict_enriched.jsonl \
  --c1_jsonl data/SSTK/10K_local/artifacts/precompute/feats_c1.jsonl \
  --parquet data/SSTK/10K_local/filtered_sstk_100.parquet \
  --tar_dir /sstk/20230916/sstk_100 \
  --output_jsonl data/SSTK/10K_local/artifacts/teacher/scores/teacher_scores_ar_p0_real.jsonl \
  --output_overview_json data/SSTK/10K_local/artifacts/teacher/overview/teacher_scores_overview_p0_real.json \
  --output_overview_csv data/SSTK/10K_local/artifacts/teacher/overview/teacher_scores_overview_by_ar_p0_real.csv \
  --qa_out_json data/SSTK/10K_local/artifacts/teacher/qa/teacher_scores_qa_report_p0_real.json \
  --qa_out_csv data/SSTK/10K_local/artifacts/teacher/qa/teacher_scores_qa_report_by_ar_p0_real.csv \
  --viz_out_dir data/SSTK/10K_local/artifacts/teacher/visualizations/teacher_scorer_p0_real \
  --use_real_expensive 1 \
  --align_device cuda \
  --aesthetic_device cuda \
  --exp_batch_size 12 \
  --run_qa 1 \
  --run_viz 1
USAGE

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$PROJECT_ROOT"

DATA_ROOT="data/SSTK/10K_local"
CANDIDATES_JSONL="${DATA_ROOT}/artifacts/candidates/candidates_ar.jsonl"
FEATURES_JSONL="${DATA_ROOT}/artifacts/precompute/feats_c2c3c5_v2_strict_enriched.jsonl"
C1_JSONL="${DATA_ROOT}/artifacts/precompute/feats_c1.jsonl"
PARQUET="${DATA_ROOT}/filtered_sstk_100.parquet"
TAR_DIR=""
IMAGE_DIR=""
SERVER_MODE=0
VENV_PATH="/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/activate"
LOCAL_TAR_DIR="/media/jyju25/T7_4TB_JY/Projects_26/Dataset/SSTK/20230916/sstk_100"
SERVER_TAR_DIR="/sstk/20230916/sstk_100"

OUTPUT_JSONL="${DATA_ROOT}/artifacts/teacher/scores/teacher_scores_ar.jsonl"
OUTPUT_OVERVIEW_JSON="${DATA_ROOT}/artifacts/teacher/overview/teacher_scores_overview.json"
OUTPUT_OVERVIEW_CSV="${DATA_ROOT}/artifacts/teacher/overview/teacher_scores_overview_by_ar.csv"

VIZ_OUT_DIR="${DATA_ROOT}/artifacts/teacher/visualizations/teacher_scorer_v1"
TARGET_AR="all"
DECISION_FILTER="all"
NUM_VIZ=120
RUN_VIZ=1
RUN_QA=1
VIZ_IMAGE_IDS=""
VIZ_IMAGE_IDS_FILE=""

CHEAP_TOP_M=30
TOP_K=5
TAU_DIV=0.75
USE_REAL_EXPENSIVE=1
HARD_HEAD_TOP_RULE=1
HEAD_TOP_FACE_EXPAND_ALPHA=0.35
HEAD_TOP_KP_EXPAND=0.06
HEAD_TOP_MIN_MARGIN=0.008
HEAD_TOP_FACE_MARGIN_ALPHA=0.20
ALIGN_MODEL_NAME=""
ALIGN_PRETRAINED=""
ALIGN_DEVICE="auto"
AESTHETIC_DEVICE="auto"
AESTHETIC_BACKEND="hybrid"
AESTHETIC_PRIOR_LAION_WEIGHT=0.15
EXP_BATCH_SIZE=24
EXPENSIVE_EVAL_TOP_M=0
EXP_PREPROCESS_WORKERS=0
EXP_PIN_MEMORY=1
AESTHETIC_MLP_PATH="weights/improved-aesthetic-predictor/sac+logos+ava1-l14-linearMSE.pth"
AESTHETIC_MLP_URL="https://raw.githubusercontent.com/christophschuhmann/improved-aesthetic-predictor/main/sac+logos+ava1-l14-linearMSE.pth"
NIMA_MODEL_PATH="weights/nima/NIMA_VGG16_ava-dc4e8265.pth"
NIMA_MODEL_URL=""
NIMA_USE_IMAGENET_BACKBONE=1
NIMA_REQUIRE_CKPT=1
MAX_IMAGES=0
SEED=42

QA_OUT_JSON="${DATA_ROOT}/artifacts/teacher/qa/teacher_scores_qa_report.json"
QA_OUT_CSV="${DATA_ROOT}/artifacts/teacher/qa/teacher_scores_qa_report_by_ar.csv"
MULTI_GPU=0
GPU_IDS=""
NUM_WORKERS=""
SHARD_TMP_DIR=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --candidates_jsonl) CANDIDATES_JSONL="$2"; shift 2 ;;
    --features_jsonl) FEATURES_JSONL="$2"; shift 2 ;;
    --c1_jsonl) C1_JSONL="$2"; shift 2 ;;
    --parquet) PARQUET="$2"; shift 2 ;;
    --tar_dir) TAR_DIR="$2"; shift 2 ;;
    --image_dir) IMAGE_DIR="$2"; shift 2 ;;
    --server_mode) SERVER_MODE="$2"; shift 2 ;;
    --venv_path) VENV_PATH="$2"; shift 2 ;;

    --output_jsonl) OUTPUT_JSONL="$2"; shift 2 ;;
    --output_overview_json) OUTPUT_OVERVIEW_JSON="$2"; shift 2 ;;
    --output_overview_csv) OUTPUT_OVERVIEW_CSV="$2"; shift 2 ;;

    --viz_out_dir) VIZ_OUT_DIR="$2"; shift 2 ;;
    --target_ar) TARGET_AR="$2"; shift 2 ;;
    --decision_filter) DECISION_FILTER="$2"; shift 2 ;;
    --num_viz) NUM_VIZ="$2"; shift 2 ;;
    --run_viz) RUN_VIZ="$2"; shift 2 ;;
    --run_qa) RUN_QA="$2"; shift 2 ;;
    --viz_image_ids) VIZ_IMAGE_IDS="$2"; shift 2 ;;
    --viz_image_ids_file) VIZ_IMAGE_IDS_FILE="$2"; shift 2 ;;

    --cheap_top_m) CHEAP_TOP_M="$2"; shift 2 ;;
    --top_k) TOP_K="$2"; shift 2 ;;
    --tau_div) TAU_DIV="$2"; shift 2 ;;
    --use_real_expensive) USE_REAL_EXPENSIVE="$2"; shift 2 ;;
    --hard_head_top_rule) HARD_HEAD_TOP_RULE="$2"; shift 2 ;;
    --head_top_face_expand_alpha) HEAD_TOP_FACE_EXPAND_ALPHA="$2"; shift 2 ;;
    --head_top_kp_expand) HEAD_TOP_KP_EXPAND="$2"; shift 2 ;;
    --head_top_min_margin) HEAD_TOP_MIN_MARGIN="$2"; shift 2 ;;
    --head_top_face_margin_alpha) HEAD_TOP_FACE_MARGIN_ALPHA="$2"; shift 2 ;;
    --align_model_name) ALIGN_MODEL_NAME="$2"; shift 2 ;;
    --align_pretrained) ALIGN_PRETRAINED="$2"; shift 2 ;;
    --align_device) ALIGN_DEVICE="$2"; shift 2 ;;
    --aesthetic_device) AESTHETIC_DEVICE="$2"; shift 2 ;;
    --aesthetic_backend) AESTHETIC_BACKEND="$2"; shift 2 ;;
    --aesthetic_prior_laion_weight) AESTHETIC_PRIOR_LAION_WEIGHT="$2"; shift 2 ;;
    --exp_batch_size) EXP_BATCH_SIZE="$2"; shift 2 ;;
    --expensive_eval_top_m) EXPENSIVE_EVAL_TOP_M="$2"; shift 2 ;;
    --exp_preprocess_workers) EXP_PREPROCESS_WORKERS="$2"; shift 2 ;;
    --exp_pin_memory) EXP_PIN_MEMORY="$2"; shift 2 ;;
    --aesthetic_mlp_path) AESTHETIC_MLP_PATH="$2"; shift 2 ;;
    --aesthetic_mlp_url) AESTHETIC_MLP_URL="$2"; shift 2 ;;
    --nima_model_path) NIMA_MODEL_PATH="$2"; shift 2 ;;
    --nima_model_url) NIMA_MODEL_URL="$2"; shift 2 ;;
    --nima_use_imagenet_backbone) NIMA_USE_IMAGENET_BACKBONE="$2"; shift 2 ;;
    --nima_require_ckpt) NIMA_REQUIRE_CKPT="$2"; shift 2 ;;
    --max_images) MAX_IMAGES="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --qa_out_json) QA_OUT_JSON="$2"; shift 2 ;;
    --qa_out_csv) QA_OUT_CSV="$2"; shift 2 ;;
    --multi_gpu) MULTI_GPU="$2"; shift 2 ;;
    --gpu_ids) GPU_IDS="$2"; shift 2 ;;
    --num_workers) NUM_WORKERS="$2"; shift 2 ;;
    --shard_tmp_dir) SHARD_TMP_DIR="$2"; shift 2 ;;

    -h|--help)
      sed -n '1,120p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

pick_latest_match() {
  local pattern="$1"
  local found
  found=$(ls -1t $pattern 2>/dev/null | head -n 1 || true)
  echo "$found"
}

resolve_input_path() {
  local primary="$1"
  shift || true
  if [ -n "$primary" ] && [ -f "$primary" ]; then
    echo "$primary"
    return
  fi
  local cand
  for cand in "$@"; do
    if [ -n "$cand" ] && [ -f "$cand" ]; then
      echo "$cand"
      return
    fi
  done
  echo "$primary"
}

jsonl_has_c1_embeddings() {
  local f="$1"
  if [ ! -f "$f" ]; then
    return 1
  fi
  python3 - "$f" <<'PY'
import json
import sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if isinstance(row.get("c1_img_embed"), list) and row.get("c1_img_embed") and \
           isinstance(row.get("c1_txt_embed"), list) and row.get("c1_txt_embed"):
            sys.exit(0)
sys.exit(1)
PY
}

resolve_c1_input_path() {
  local cand
  for cand in "$@"; do
    if [ -n "$cand" ] && [ -f "$cand" ] && jsonl_has_c1_embeddings "$cand"; then
      echo "$cand"
      return
    fi
  done
  echo "$1"
}

LATEST_CANDIDATES_ARTIFACT="$(pick_latest_match "${DATA_ROOT}/artifacts/candidates/candidates_ar*.jsonl")"
LATEST_CANDIDATES_LEGACY="$(pick_latest_match "${DATA_ROOT}/candidates_ar*.jsonl")"
LATEST_CANDIDATES_TEMP="$(pick_latest_match "${DATA_ROOT}/Temp/candidates_ar*.jsonl")"
CANDIDATES_JSONL="$(resolve_input_path \
  "$CANDIDATES_JSONL" \
  "$LATEST_CANDIDATES_ARTIFACT" \
  "$LATEST_CANDIDATES_LEGACY" \
  "$LATEST_CANDIDATES_TEMP")"

FEATURES_JSONL="$(resolve_input_path \
  "$FEATURES_JSONL" \
  "${DATA_ROOT}/artifacts/precompute/feats_c2c3c5_v2_strict_enriched_routed.jsonl" \
  "${DATA_ROOT}/artifacts/precompute/feats_c2c3c5_v2_strict_enriched.jsonl" \
  "${DATA_ROOT}/feats_c2c3c5_v2_strict_enriched.jsonl" \
  "${DATA_ROOT}/Temp/feats_c2c3c5_v2_strict_enriched.jsonl")"

C1_JSONL="$(resolve_c1_input_path \
  "$C1_JSONL" \
  "${DATA_ROOT}/artifacts/precompute/feats_c1.jsonl" \
  "${DATA_ROOT}/artifacts/precompute/feats_c2c3c5_v2_strict_raw.jsonl" \
  "${DATA_ROOT}/feats_c1.jsonl" \
  "${DATA_ROOT}/Temp/feats_c1.jsonl")"

LATEST_FILTERED_BACKUP="$(pick_latest_match "${DATA_ROOT}/Temp/filtered_sstk_100*.parquet")"
PARQUET="$(resolve_input_path \
  "$PARQUET" \
  "${DATA_ROOT}/filtered_sstk_100.parquet" \
  "$LATEST_FILTERED_BACKUP")"

if [ -z "$TAR_DIR" ]; then
  if [ "$SERVER_MODE" -eq 1 ]; then
    TAR_DIR="$SERVER_TAR_DIR"
  else
    TAR_DIR="$LOCAL_TAR_DIR"
  fi
fi

if [ "$SERVER_MODE" -ne 1 ]; then
  if [ -f "$VENV_PATH" ]; then
    # shellcheck disable=SC1090
    source "$VENV_PATH"
  else
    echo "[warn] venv not found: $VENV_PATH (using current python)"
  fi
fi

mkdir -p "$(dirname "$OUTPUT_JSONL")"
mkdir -p "$(dirname "$OUTPUT_OVERVIEW_JSON")"
mkdir -p "$(dirname "$OUTPUT_OVERVIEW_CSV")"
mkdir -p "$(dirname "$QA_OUT_JSON")"
mkdir -p "$(dirname "$QA_OUT_CSV")"
mkdir -p "$VIZ_OUT_DIR"

REQUIRED_INPUTS=("$CANDIDATES_JSONL" "$FEATURES_JSONL" "$PARQUET")
if [ "$USE_REAL_EXPENSIVE" -ne 0 ]; then
  REQUIRED_INPUTS+=("$C1_JSONL")
fi
for req in "${REQUIRED_INPUTS[@]}"; do
  if [ ! -f "$req" ]; then
    echo "[error] required input not found: $req"
    exit 1
  fi
done

echo "[config] server_mode=$SERVER_MODE tar_dir=$TAR_DIR image_dir=${IMAGE_DIR:-<none>}"
echo "[config] candidates=$CANDIDATES_JSONL"
echo "[config] features=$FEATURES_JSONL"
echo "[config] c1=$C1_JSONL"
echo "[config] parquet=$PARQUET"
echo "[config] output_jsonl=$OUTPUT_JSONL"
echo "[config] expensive_accel: batch=$EXP_BATCH_SIZE eval_top_m=$EXPENSIVE_EVAL_TOP_M preprocess_workers=$EXP_PREPROCESS_WORKERS pin_memory=$EXP_PIN_MEMORY"
echo "[config] aesthetic: backend=$AESTHETIC_BACKEND prior_laion_w=$AESTHETIC_PRIOR_LAION_WEIGHT nima_ckpt=$NIMA_MODEL_PATH require_ckpt=$NIMA_REQUIRE_CKPT"
echo "[config] portrait_safety: hard_head_top=$HARD_HEAD_TOP_RULE face_expand=$HEAD_TOP_FACE_EXPAND_ALPHA kp_expand=$HEAD_TOP_KP_EXPAND min_margin=$HEAD_TOP_MIN_MARGIN face_margin_alpha=$HEAD_TOP_FACE_MARGIN_ALPHA"
echo "[config] viz_image_ids=${VIZ_IMAGE_IDS:-<none>} viz_image_ids_file=${VIZ_IMAGE_IDS_FILE:-<none>}"

echo "[1/3] Running teacher scorer..."
run_teacher_one() {
  local out_jsonl="$1"
  local out_over_json="$2"
  local out_over_csv="$3"
  local shard_index="$4"
  local num_shards="$5"
  local progress="$6"

  python3 src/score_teacher.py \
    --candidates_jsonl "$CANDIDATES_JSONL" \
    --features_jsonl "$FEATURES_JSONL" \
    --c1_jsonl "$C1_JSONL" \
    --parquet "$PARQUET" \
    --tar_dir "$TAR_DIR" \
    --image_dir "$IMAGE_DIR" \
    --output_jsonl "$out_jsonl" \
    --output_overview_json "$out_over_json" \
    --output_overview_by_ar_csv "$out_over_csv" \
    --cheap_top_m "$CHEAP_TOP_M" \
    --top_k "$TOP_K" \
    --tau_div "$TAU_DIV" \
    --use_real_expensive "$USE_REAL_EXPENSIVE" \
    --hard_head_top_rule "$HARD_HEAD_TOP_RULE" \
    --head_top_face_expand_alpha "$HEAD_TOP_FACE_EXPAND_ALPHA" \
    --head_top_kp_expand "$HEAD_TOP_KP_EXPAND" \
    --head_top_min_margin "$HEAD_TOP_MIN_MARGIN" \
    --head_top_face_margin_alpha "$HEAD_TOP_FACE_MARGIN_ALPHA" \
    --align_model_name "$ALIGN_MODEL_NAME" \
    --align_pretrained "$ALIGN_PRETRAINED" \
    --align_device "$ALIGN_DEVICE" \
    --aesthetic_device "$AESTHETIC_DEVICE" \
    --aesthetic_backend "$AESTHETIC_BACKEND" \
    --aesthetic_prior_laion_weight "$AESTHETIC_PRIOR_LAION_WEIGHT" \
    --exp_batch_size "$EXP_BATCH_SIZE" \
    --expensive_eval_top_m "$EXPENSIVE_EVAL_TOP_M" \
    --exp_preprocess_workers "$EXP_PREPROCESS_WORKERS" \
    --exp_pin_memory "$EXP_PIN_MEMORY" \
    --aesthetic_mlp_path "$AESTHETIC_MLP_PATH" \
    --aesthetic_mlp_url "$AESTHETIC_MLP_URL" \
    --nima_model_path "$NIMA_MODEL_PATH" \
    --nima_model_url "$NIMA_MODEL_URL" \
    --nima_use_imagenet_backbone "$NIMA_USE_IMAGENET_BACKBONE" \
    --nima_require_ckpt "$NIMA_REQUIRE_CKPT" \
    --max_images "$MAX_IMAGES" \
    --seed "$SEED" \
    --num_shards "$num_shards" \
    --shard_index "$shard_index" \
    --progress "$progress"
}

if [ "$MULTI_GPU" -ne 0 ]; then
  resolve_gpu_ids() {
    local csv="$GPU_IDS"
    if [ -z "$csv" ] && [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
      csv="$CUDA_VISIBLE_DEVICES"
    fi
    if [ -z "$csv" ]; then
      local n
      n=$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)
      local arr=()
      local i
      for ((i=0; i<n; i++)); do
        arr+=("$i")
      done
      (IFS=,; echo "${arr[*]}")
      return
    fi
    csv="${csv// /}"
    echo "$csv"
  }

  GPU_CSV="$(resolve_gpu_ids)"
  IFS=',' read -r -a GPU_ARR <<< "$GPU_CSV"
  GPU_COUNT="${#GPU_ARR[@]}"
  WORKERS="${#GPU_ARR[@]}"
  if [ -n "$NUM_WORKERS" ]; then
    WORKERS="$NUM_WORKERS"
  fi
  if [ "$WORKERS" -gt "$GPU_COUNT" ]; then
    echo "[warn] num_workers($WORKERS) > gpu_ids($GPU_COUNT). enabling round-robin GPU oversubscribe."
  fi

  if [ "$WORKERS" -le 1 ]; then
    echo "[warn] multi_gpu=1 but effective workers<=1. falling back to single."
    run_teacher_one "$OUTPUT_JSONL" "$OUTPUT_OVERVIEW_JSON" "$OUTPUT_OVERVIEW_CSV" 0 1 1
  else
    if [ -z "$SHARD_TMP_DIR" ]; then
      SHARD_TMP_DIR="${OUTPUT_JSONL}.shards.$$"
    fi
    mkdir -p "$SHARD_TMP_DIR"
    echo "[multi] teacher shard mode enabled: gpu_ids=${GPU_CSV} workers=${WORKERS}"
    PIDS=()
    SHARD_JSONL=()
    i=0
    while [ "$i" -lt "$WORKERS" ]; do
      GPU_IDX=$((i % GPU_COUNT))
      GPU_ID="${GPU_ARR[$GPU_IDX]}"
      SH_OUT_JSONL="${SHARD_TMP_DIR}/teacher_scores_shard_${i}.jsonl"
      SH_OUT_OV_JSON="${SHARD_TMP_DIR}/teacher_overview_shard_${i}.json"
      SH_OUT_OV_CSV="${SHARD_TMP_DIR}/teacher_overview_by_ar_shard_${i}.csv"
      SH_LOG="${SHARD_TMP_DIR}/teacher_shard_${i}.log"
      SHARD_JSONL+=("$SH_OUT_JSONL")
      SH_PROGRESS=0
      if [ "$i" -eq 0 ]; then
        SH_PROGRESS=1
      fi
      echo "[multi] launch shard=$i/$WORKERS gpu=$GPU_ID -> $SH_OUT_JSONL"
      (
        export CUDA_VISIBLE_DEVICES="$GPU_ID"
        run_teacher_one "$SH_OUT_JSONL" "$SH_OUT_OV_JSON" "$SH_OUT_OV_CSV" "$i" "$WORKERS" "$SH_PROGRESS"
      ) >"$SH_LOG" 2>&1 &
      PIDS+=("$!")
      i=$((i + 1))
    done

    FAIL=0
    for pid in "${PIDS[@]}"; do
      if ! wait "$pid"; then
        FAIL=1
      fi
    done
    if [ "$FAIL" -ne 0 ]; then
      echo "[error] one or more teacher shards failed. logs under: $SHARD_TMP_DIR"
      exit 1
    fi

    : > "$OUTPUT_JSONL"
    for f in "${SHARD_JSONL[@]}"; do
      if [ -f "$f" ]; then
        cat "$f" >> "$OUTPUT_JSONL"
      fi
    done
    echo "[multi] merged teacher jsonl -> $OUTPUT_JSONL"

    python3 src/scripts/rebuild_teacher_overview.py \
      --teacher_scores_jsonl "$OUTPUT_JSONL" \
      --output_json "$OUTPUT_OVERVIEW_JSON" \
      --output_by_ar_csv "$OUTPUT_OVERVIEW_CSV"
    echo "[multi] rebuilt overview -> $OUTPUT_OVERVIEW_JSON"
  fi
else
  run_teacher_one "$OUTPUT_JSONL" "$OUTPUT_OVERVIEW_JSON" "$OUTPUT_OVERVIEW_CSV" 0 1 1
fi

if [ "$RUN_QA" -eq 1 ]; then
  echo "[2/3] Building QA report..."
  python3 src/scripts/qa_teacher_report.py \
    --teacher_scores_jsonl "$OUTPUT_JSONL" \
    --output_json "$QA_OUT_JSON" \
    --output_by_ar_csv "$QA_OUT_CSV"
else
  echo "[2/3] QA report skipped (--run_qa 0)."
fi

if [ "$RUN_VIZ" -eq 1 ]; then
  echo "[3/3] Rendering teacher scorer visualizations..."
  viz_args=()
  if [ -n "$IMAGE_DIR" ]; then
    viz_args+=(--image_dir "$IMAGE_DIR")
  fi
  if [ -n "$VIZ_IMAGE_IDS" ]; then
    viz_args+=(--image_ids "$VIZ_IMAGE_IDS")
  fi
  if [ -n "$VIZ_IMAGE_IDS_FILE" ]; then
    viz_args+=(--image_ids_file "$VIZ_IMAGE_IDS_FILE")
  fi
  python3 src/visualize_teacher_scores.py \
    --teacher_scores_jsonl "$OUTPUT_JSONL" \
    --features_jsonl "$FEATURES_JSONL" \
    --parquet "$PARQUET" \
    --tar_dir "$TAR_DIR" \
    --out_dir "$VIZ_OUT_DIR" \
    --target_ar "$TARGET_AR" \
    --decision_filter "$DECISION_FILTER" \
    --num_samples "$NUM_VIZ" \
    --seed "$SEED" \
    "${viz_args[@]}"
else
  echo "[3/3] Visualization skipped (--run_viz 0)."
fi

echo "[done] Teacher scorer outputs:"
echo "  - $OUTPUT_JSONL"
echo "  - $OUTPUT_OVERVIEW_JSON"
echo "  - $OUTPUT_OVERVIEW_CSV"
if [ "$RUN_QA" -eq 1 ]; then
  echo "  - $QA_OUT_JSON"
  echo "  - $QA_OUT_CSV"
fi
if [ "$RUN_VIZ" -eq 1 ]; then
  echo "  - $VIZ_OUT_DIR"
fi
