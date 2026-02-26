#!/bin/bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: bash src/scripts/run_infer_public_cropping_teachers.sh <input_parquet> <tar_dir> <output_raw_jsonl> [extra_args...]"
  exit 1
fi

INPUT_PARQUET="$1"
TAR_DIR="$2"
OUTPUT_JSONL="$3"
shift 3

FALLBACK_CPU_ON_OOM=1
FALLBACK_CPU_MAX_IMAGES=3
SKIP_IF_FALLBACK_FAILED=1
PASS_ARGS=()

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
    *)
      PASS_ARGS+=("$1")
      shift
      ;;
  esac
done

VENV_PATH="/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/activate"
if [ -f "$VENV_PATH" ]; then
  # shellcheck disable=SC1090
  source "$VENV_PATH"
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

if ! grep -Eqi "out of memory|cuda error: out of memory|oom" "$LOG_TMP"; then
  echo "[infer-wrapper][error] inference failed but not detected as OOM (exit=$RC)"
  rm -f "$LOG_TMP"
  exit "$RC"
fi

echo "[infer-wrapper][warn] GPU OOM detected. retrying on CPU with max_images=${FALLBACK_CPU_MAX_IMAGES}."

fallback_args=()
skip_next=0
for ((i=0; i<${#PASS_ARGS[@]}; i++)); do
  if [ "$skip_next" -eq 1 ]; then
    skip_next=0
    continue
  fi
  arg="${PASS_ARGS[$i]}"
  case "$arg" in
    --device|--max_images)
      skip_next=1
      ;;
    *)
      fallback_args+=("$arg")
      ;;
  esac
done

set +e
python src/scripts/infer_public_cropping_teachers.py \
  --input_parquet "$INPUT_PARQUET" \
  --tar_dir "$TAR_DIR" \
  --output_jsonl "$OUTPUT_JSONL" \
  --device cpu \
  --max_images "$FALLBACK_CPU_MAX_IMAGES" \
  "${fallback_args[@]}"
RC2=$?
set -e

rm -f "$LOG_TMP"

if [ "$RC2" -eq 0 ]; then
  echo "[infer-wrapper] CPU fallback succeeded."
  exit 0
fi

if [ "${SKIP_IF_FALLBACK_FAILED}" -eq 1 ]; then
  echo "[infer-wrapper][warn] CPU fallback failed (exit=$RC2). skipping this step as requested."
  exit 0
fi

echo "[infer-wrapper][error] CPU fallback failed (exit=$RC2)."
exit "$RC2"
