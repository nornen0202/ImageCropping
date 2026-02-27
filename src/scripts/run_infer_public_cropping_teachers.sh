#!/bin/bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: bash src/scripts/run_infer_public_cropping_teachers.sh <input_parquet> <tar_dir> <output_raw_jsonl> [extra_args...]"
  echo "       extra: [--multi_gpu -1|0|1] [--gpu_ids 0,1] [--num_workers N]"
  exit 1
fi

INPUT_PARQUET="$1"
TAR_DIR="$2"
OUTPUT_JSONL="$3"
shift 3

FALLBACK_CPU_ON_OOM=1
FALLBACK_CPU_MAX_IMAGES=3
SKIP_IF_FALLBACK_FAILED=1
DEFAULT_GAIC_WEIGHT_PATH="weights/public_cropping_teachers/gaic/shufflenet_0.682_0.641_0.607_0.566_0.858_0.825_0.805_0.778_0.850_0.872.pth"
PREFER_CURATED_IMAGES=1
CURATED_IMAGE_DIR=""
MULTI_GPU=-1
GPU_IDS=""
NUM_WORKERS=""

REQUESTED_DEVICE="auto"
RUN_SETUP=1
SETUP_DOWNLOAD_WEIGHTS=1
TEACHER_ROOT_DIR="third_party/public_cropping_teachers"
WEIGHTS_DIR="weights/public_cropping_teachers"

PASS_ARGS=()

HAS_GAIC_WEIGHT=0
TEACHERS_INCLUDE_GAIC=0
TEACHERS_SPECIFIED=0

detect_gpu_count() {
  python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
}

is_oom_log() {
  local p="$1"
  grep -Eqi "out of memory|cuda error: out of memory|oom" "$p"
}

build_args_without_pairs() {
  local remove_opts=("$@")
  local -a out=()
  local skip_next=0
  local i
  for ((i=0; i<${#PASS_ARGS[@]}; i++)); do
    if [ "$skip_next" -eq 1 ]; then
      skip_next=0
      continue
    fi
    local arg="${PASS_ARGS[$i]}"
    local matched=0
    local opt
    for opt in "${remove_opts[@]}"; do
      if [ "$arg" = "$opt" ]; then
        skip_next=1
        matched=1
        break
      fi
    done
    if [ "$matched" -eq 0 ]; then
      out+=("$arg")
    fi
  done
  printf '%s\n' "${out[@]}"
}

run_cpu_fallback() {
  local -a fallback_args_lines
  mapfile -t fallback_args_lines < <(
    build_args_without_pairs \
      --device \
      --max_images \
      --run_setup \
      --setup_download_weights \
      --num_shards \
      --shard_index
  )
  local -a fallback_args=("${fallback_args_lines[@]}")

  set +e
  python src/scripts/infer_public_cropping_teachers.py \
    --input_parquet "$INPUT_PARQUET" \
    --tar_dir "$TAR_DIR" \
    --output_jsonl "$OUTPUT_JSONL" \
    --device cpu \
    --max_images "$FALLBACK_CPU_MAX_IMAGES" \
    --run_setup 0 \
    "${fallback_args[@]}"
  local rc=$?
  set -e

  if [ "$rc" -eq 0 ]; then
    echo "[infer-wrapper] CPU fallback succeeded."
    return 0
  fi
  if [ "${SKIP_IF_FALLBACK_FAILED}" -eq 1 ]; then
    echo "[infer-wrapper][warn] CPU fallback failed (exit=$rc). skipping this step as requested."
    return 0
  fi
  echo "[infer-wrapper][error] CPU fallback failed (exit=$rc)."
  return "$rc"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --fallback_cpu_on_oom)
      FALLBACK_CPU_ON_OOM="$2"
      shift 2
      ;;
    --fallback_cpu_max_images)
      FALLBACK_CPU_MAX_IMAGES="$2"
      shift 2
      ;;
    --skip_if_fallback_failed)
      SKIP_IF_FALLBACK_FAILED="$2"
      shift 2
      ;;
    --prefer_curated_images)
      PREFER_CURATED_IMAGES="$2"
      shift 2
      ;;
    --curated_image_dir)
      CURATED_IMAGE_DIR="$2"
      shift 2
      ;;
    --multi_gpu)
      MULTI_GPU="$2"
      shift 2
      ;;
    --gpu_ids)
      GPU_IDS="$2"
      shift 2
      ;;
    --num_workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --device)
      REQUESTED_DEVICE="$2"
      PASS_ARGS+=("$1" "$2")
      shift 2
      ;;
    --run_setup)
      RUN_SETUP="$2"
      PASS_ARGS+=("$1" "$2")
      shift 2
      ;;
    --setup_download_weights)
      SETUP_DOWNLOAD_WEIGHTS="$2"
      PASS_ARGS+=("$1" "$2")
      shift 2
      ;;
    --teacher_root_dir)
      TEACHER_ROOT_DIR="$2"
      PASS_ARGS+=("$1" "$2")
      shift 2
      ;;
    --weights_dir)
      WEIGHTS_DIR="$2"
      PASS_ARGS+=("$1" "$2")
      shift 2
      ;;
    --gaic_weight)
      HAS_GAIC_WEIGHT=1
      PASS_ARGS+=("$1" "$2")
      shift 2
      ;;
    --gaic_weight_path)
      HAS_GAIC_WEIGHT=1
      PASS_ARGS+=("--gaic_weight" "$2")
      shift 2
      ;;
    --teachers)
      TEACHERS_SPECIFIED=1
      PASS_ARGS+=("$1")
      shift
      while [ "$#" -gt 0 ] && [[ "$1" != --* ]]; do
        PASS_ARGS+=("$1")
        if [ "${1,,}" = "gaic" ]; then
          TEACHERS_INCLUDE_GAIC=1
        fi
        shift
      done
      ;;
    *)
      if [ "${1,,}" = "gaic" ]; then
        TEACHERS_INCLUDE_GAIC=1
      fi
      PASS_ARGS+=("$1")
      shift
      ;;
  esac
done

if [ -z "$CURATED_IMAGE_DIR" ]; then
  PARQ_DIR="$(dirname "$INPUT_PARQUET")"
  CURATED_IMAGE_DIR="${PARQ_DIR}/images"
fi

if [ "$TEACHERS_SPECIFIED" -eq 0 ]; then
  TEACHERS_INCLUDE_GAIC=1
fi

if [ "$TEACHERS_INCLUDE_GAIC" -eq 1 ] && [ "$HAS_GAIC_WEIGHT" -eq 0 ]; then
  PASS_ARGS+=("--gaic_weight" "$DEFAULT_GAIC_WEIGHT_PATH")
fi

PASS_ARGS+=(
  --prefer_curated_images "$PREFER_CURATED_IMAGES"
  --curated_image_dir "$CURATED_IMAGE_DIR"
)

VENV_PATH="/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/activate"
if [ -f "$VENV_PATH" ]; then
  # shellcheck disable=SC1090
  source "$VENV_PATH"
fi

GPU_COUNT="$(detect_gpu_count || echo 0)"
DEVICE_LC="${REQUESTED_DEVICE,,}"
if [ "$MULTI_GPU" -lt 0 ]; then
  if [ "$DEVICE_LC" = "cpu" ]; then
    MULTI_GPU=0
  elif [ "$GPU_COUNT" -gt 1 ]; then
    MULTI_GPU=1
  else
    MULTI_GPU=0
  fi
fi
if [ "$DEVICE_LC" = "cpu" ]; then
  MULTI_GPU=0
fi

if [ "$MULTI_GPU" -eq 1 ]; then
  GPU_ARR=()
  if [ -n "$GPU_IDS" ]; then
    IFS_OLD="$IFS"
    IFS=',' read -r -a gpu_tokens <<< "$GPU_IDS"
    IFS="$IFS_OLD"
    for g in "${gpu_tokens[@]}"; do
      if [ -n "$g" ]; then
        GPU_ARR+=("$g")
      fi
    done
  else
    for ((g=0; g<GPU_COUNT; g++)); do
      GPU_ARR+=("$g")
    done
  fi

  if [ "${#GPU_ARR[@]}" -lt 2 ]; then
    echo "[infer-wrapper][warn] multi-gpu requested but available gpu_ids < 2. falling back to single run."
    MULTI_GPU=0
  fi
fi

if [ "$MULTI_GPU" -eq 1 ]; then
  WORKERS="${#GPU_ARR[@]}"
  if [ -n "$NUM_WORKERS" ] && [ "$NUM_WORKERS" -gt 0 ]; then
    WORKERS="$NUM_WORKERS"
  fi
  if [ "$WORKERS" -gt "${#GPU_ARR[@]}" ]; then
    WORKERS="${#GPU_ARR[@]}"
  fi
  if [ "$WORKERS" -lt 2 ]; then
    echo "[infer-wrapper][warn] num_workers < 2 in multi-gpu mode. falling back to single run."
    MULTI_GPU=0
  fi
fi

if [ "$MULTI_GPU" -eq 1 ]; then
  echo "[infer-wrapper] multi-gpu mode enabled: workers=$WORKERS gpu_ids=$(IFS=,; echo "${GPU_ARR[*]}")"

  if [ "$RUN_SETUP" -eq 1 ]; then
    bash src/scripts/run_setup_public_cropping_teachers.sh \
      --teacher_root_dir "$TEACHER_ROOT_DIR" \
      --weights_dir "$WEIGHTS_DIR" \
      --download_weights "$SETUP_DOWNLOAD_WEIGHTS"
  fi

  shard_base_lines=()
  mapfile -t shard_base_lines < <(
    build_args_without_pairs \
      --device \
      --run_setup \
      --setup_download_weights \
      --num_shards \
      --shard_index
  )
  shard_base_args=("${shard_base_lines[@]}")

  TMP_DIR="$(mktemp -d -t infer_public_teachers_multi.XXXXXX)"
  PIDS=()
  LOG_PATHS=()
  SHARD_PATHS=()

  for ((i=0; i<WORKERS; i++)); do
    gpu_id="${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}"
    shard_out="${TMP_DIR}/shard_${i}.jsonl"
    shard_log="${TMP_DIR}/shard_${i}.log"
    SHARD_PATHS+=("$shard_out")
    LOG_PATHS+=("$shard_log")
    (
      CUDA_VISIBLE_DEVICES="$gpu_id" \
      python src/scripts/infer_public_cropping_teachers.py \
        --input_parquet "$INPUT_PARQUET" \
        --tar_dir "$TAR_DIR" \
        --output_jsonl "$shard_out" \
        --device cuda \
        --run_setup 0 \
        --num_shards "$WORKERS" \
        --shard_index "$i" \
        "${shard_base_args[@]}" 2>&1 | tee "$shard_log"
    ) &
    PIDS+=("$!")
  done

  FAIL_COUNT=0
  OOM_FAIL=0
  for ((i=0; i<WORKERS; i++)); do
    if ! wait "${PIDS[$i]}"; then
      FAIL_COUNT=$((FAIL_COUNT + 1))
      if is_oom_log "${LOG_PATHS[$i]}"; then
        OOM_FAIL=1
      fi
    fi
  done

  if [ "$FAIL_COUNT" -eq 0 ]; then
    : > "$OUTPUT_JSONL"
    for ((i=0; i<WORKERS; i++)); do
      if [ -f "${SHARD_PATHS[$i]}" ]; then
        cat "${SHARD_PATHS[$i]}" >> "$OUTPUT_JSONL"
      fi
    done
    rm -rf "$TMP_DIR"
    echo "[infer-wrapper] multi-gpu inference succeeded -> $OUTPUT_JSONL"
    exit 0
  fi

  echo "[infer-wrapper][warn] multi-gpu inference failed: failed_workers=$FAIL_COUNT"
  rm -rf "$TMP_DIR"

  if [ "${FALLBACK_CPU_ON_OOM}" -eq 1 ] && [ "$OOM_FAIL" -eq 1 ]; then
    echo "[infer-wrapper][warn] OOM detected in multi-gpu workers. retrying CPU fallback with max_images=${FALLBACK_CPU_MAX_IMAGES}."
    run_cpu_fallback
    exit $?
  fi

  echo "[infer-wrapper][error] multi-gpu inference failed (non-OOM or fallback disabled)."
  exit 1
fi

run_infer_cmd=(
  python src/scripts/infer_public_cropping_teachers.py \
    --input_parquet "$INPUT_PARQUET" \
    --tar_dir "$TAR_DIR" \
    --output_jsonl "$OUTPUT_JSONL" \
    "${PASS_ARGS[@]}"
)

LOG_TMP="$(mktemp -t infer_public_teachers.XXXXXX.log)"
set +e
"${run_infer_cmd[@]}" 2>&1 | tee "$LOG_TMP"
RC="${PIPESTATUS[0]}"
set -e

if [ "$RC" -eq 0 ]; then
  rm -f "$LOG_TMP"
  exit 0
fi

if [ "${FALLBACK_CPU_ON_OOM}" -ne 1 ]; then
  echo "[infer-wrapper][error] inference failed and fallback disabled (exit=$RC)"
  rm -f "$LOG_TMP"
  exit "$RC"
fi

if ! is_oom_log "$LOG_TMP"; then
  echo "[infer-wrapper][error] inference failed but not detected as OOM (exit=$RC)"
  rm -f "$LOG_TMP"
  exit "$RC"
fi

echo "[infer-wrapper][warn] GPU OOM detected. retrying on CPU with max_images=${FALLBACK_CPU_MAX_IMAGES}."
rm -f "$LOG_TMP"
run_cpu_fallback
exit $?
