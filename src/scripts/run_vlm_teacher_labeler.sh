#!/bin/bash
# ==============================================================================
# run_vlm_teacher_labeler.sh
# Section 10: VLM/MLLM Teacher label generator runner
# ==============================================================================

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$PROJECT_ROOT"

DATA_ROOT="data/SSTK/10K_local"
TEACHER_SCORES_JSONL="${DATA_ROOT}/artifacts/teacher/scores/teacher_scores_ar.jsonl"
OUTPUT_JSONL="${DATA_ROOT}/artifacts/vlm_teacher/labels/crop_label_v1.jsonl"
OUTPUT_META_JSONL="${DATA_ROOT}/artifacts/vlm_teacher/meta/meta_norm_v1.jsonl"
SUMMARY_JSON="${DATA_ROOT}/artifacts/vlm_teacher/summary/vlm_teacher_summary.json"

BACKEND="qwen25_vl"
FALLBACK_BACKEND="heuristic"
MODEL_ID="Qwen/Qwen3-VL-4B-Instruct"
DEVICE="auto"
DTYPE="auto"
MAX_NEW_TOKENS=768
TEMPERATURE=0.0

IMAGE_DIR=""
TARGET_AR="all"
TOP_M=12
TOP_K=5
MAX_IMAGES=0
MAX_RETRIES=2
PROMPT_VERSION="crop_label_candidate_pick_v1"
SEED=42

SAVE_RAW_RESPONSE=0
DEBUG_DIR=""

SKIP_ON_OOM=1
FALLBACK_CPU_ON_OOM=1
FALLBACK_CPU_MAX_IMAGES=3
SKIP_IF_FALLBACK_FAILED=1
STRICT_BACKEND_INIT=1
MULTI_GPU=-1
GPU_IDS=""
NUM_WORKERS=""

SERVER_MODE=0
VENV_PATH="/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/activate"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --data_root) DATA_ROOT="$2"; shift 2 ;;
    --teacher_scores_jsonl) TEACHER_SCORES_JSONL="$2"; shift 2 ;;
    --output_jsonl) OUTPUT_JSONL="$2"; shift 2 ;;
    --output_meta_jsonl) OUTPUT_META_JSONL="$2"; shift 2 ;;
    --summary_json) SUMMARY_JSON="$2"; shift 2 ;;

    --backend) BACKEND="$2"; shift 2 ;;
    --fallback_backend) FALLBACK_BACKEND="$2"; shift 2 ;;
    --model_id) MODEL_ID="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --max_new_tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;

    --image_dir) IMAGE_DIR="$2"; shift 2 ;;
    --target_ar) TARGET_AR="$2"; shift 2 ;;
    --top_m) TOP_M="$2"; shift 2 ;;
    --top_k) TOP_K="$2"; shift 2 ;;
    --max_images) MAX_IMAGES="$2"; shift 2 ;;
    --max_retries) MAX_RETRIES="$2"; shift 2 ;;
    --prompt_version) PROMPT_VERSION="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;

    --save_raw_response) SAVE_RAW_RESPONSE="$2"; shift 2 ;;
    --debug_dir) DEBUG_DIR="$2"; shift 2 ;;

    --skip_on_oom) SKIP_ON_OOM="$2"; shift 2 ;;
    --fallback_cpu_on_oom) FALLBACK_CPU_ON_OOM="$2"; shift 2 ;;
    --fallback_cpu_max_images) FALLBACK_CPU_MAX_IMAGES="$2"; shift 2 ;;
    --skip_if_fallback_failed) SKIP_IF_FALLBACK_FAILED="$2"; shift 2 ;;
    --strict_backend_init) STRICT_BACKEND_INIT="$2"; shift 2 ;;
    --multi_gpu) MULTI_GPU="$2"; shift 2 ;;
    --gpu_ids) GPU_IDS="$2"; shift 2 ;;
    --num_workers) NUM_WORKERS="$2"; shift 2 ;;

    --server_mode) SERVER_MODE="$2"; shift 2 ;;
    --venv_path) VENV_PATH="$2"; shift 2 ;;

    -h|--help)
      sed -n '1,220p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [ -z "$IMAGE_DIR" ]; then
  IMAGE_DIR="${DATA_ROOT}/images"
fi

if [ -f "$VENV_PATH" ]; then
  # shellcheck disable=SC1090
  source "$VENV_PATH"
  echo "[info] activated venv: $VENV_PATH"
elif [ "$SERVER_MODE" -ne 1 ]; then
  echo "[warn] venv not found: $VENV_PATH (using current python)"
else
  echo "[info] server_mode=1 and venv not found: $VENV_PATH (using current python)"
fi

mkdir -p "$(dirname "$OUTPUT_JSONL")" "$(dirname "$OUTPUT_META_JSONL")" "$(dirname "$SUMMARY_JSON")"
if [ -n "$DEBUG_DIR" ]; then
  mkdir -p "$DEBUG_DIR"
fi

echo "=============================================="
echo " VLM Teacher Labeler Runner (Section 10)"
echo "=============================================="
echo "  teacher_scores_jsonl : $TEACHER_SCORES_JSONL"
echo "  output_jsonl         : $OUTPUT_JSONL"
echo "  output_meta_jsonl    : $OUTPUT_META_JSONL"
echo "  summary_json         : $SUMMARY_JSON"
echo "  backend/model        : $BACKEND / $MODEL_ID"
echo "  fallback_backend     : $FALLBACK_BACKEND"
echo "  device/dtype         : $DEVICE / $DTYPE"
echo "  image_dir            : ${IMAGE_DIR:-<none>}"
echo "  target_ar            : $TARGET_AR"
echo "  top_m/top_k          : $TOP_M / $TOP_K"
echo "  max_images/retries   : $MAX_IMAGES / $MAX_RETRIES"
echo "  oom policy           : skip=$SKIP_ON_OOM cpu_fallback=$FALLBACK_CPU_ON_OOM cpu_max_images=$FALLBACK_CPU_MAX_IMAGES"
echo "  strict_backend_init  : $STRICT_BACKEND_INIT"
echo "  multi_gpu            : $MULTI_GPU (gpu_ids=${GPU_IDS:-auto}, workers=${NUM_WORKERS:-auto})"
echo "=============================================="

if [[ "$BACKEND" == "qwen25_vl" || "$BACKEND" == "qwen25_vl_hf" ]]; then
  python - <<'PY'
import sys
print(f"[env] python={sys.executable}")
try:
    import transformers
    print(
        "[env] transformers="
        f"{transformers.__version__} "
        f"has_qwen3={hasattr(transformers, 'Qwen3VLForConditionalGeneration')} "
        f"has_qwen2_5={hasattr(transformers, 'Qwen2_5_VLForConditionalGeneration')} "
        f"has_qwen2={hasattr(transformers, 'Qwen2VLForConditionalGeneration')}"
    )
except Exception as exc:
    print(f"[env] transformers import failed: {exc}")
PY
fi

normalize_csv_ids() {
  local s="${1:-}"
  s=$(echo "$s" | tr -d ' ')
  s=$(echo "$s" | sed -E 's/^,+//; s/,+$//; s/,+/,/g')
  echo "$s"
}

count_csv_ids() {
  local s
  s=$(normalize_csv_ids "${1:-}")
  if [ -z "$s" ]; then
    echo 0
    return
  fi
  awk -F',' '{print NF}' <<< "$s"
}

detect_gpu_ids() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo ""
    return
  fi
  nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | awk '{print $1}' | paste -sd, -
}

AUTO_GPU_IDS="$(detect_gpu_ids)"
EFFECTIVE_GPU_IDS="$(normalize_csv_ids "$GPU_IDS")"
if [ -z "$EFFECTIVE_GPU_IDS" ]; then
  EFFECTIVE_GPU_IDS="$(normalize_csv_ids "$AUTO_GPU_IDS")"
fi
GPU_COUNT="$(count_csv_ids "$EFFECTIVE_GPU_IDS")"

ENABLE_MULTI=0
if [ "${MULTI_GPU}" -lt 0 ]; then
  if [[ "$BACKEND" == qwen25_vl* ]] && [[ "$DEVICE" != cpu* ]] && [ "$GPU_COUNT" -gt 1 ]; then
    ENABLE_MULTI=1
  fi
elif [ "${MULTI_GPU}" -eq 1 ]; then
  ENABLE_MULTI=1
fi

if [ "$ENABLE_MULTI" -eq 1 ] && [ "$GPU_COUNT" -le 1 ]; then
  echo "[warn] multi_gpu requested but usable gpu count is $GPU_COUNT. fallback to single."
  ENABLE_MULTI=0
fi
if [ "$ENABLE_MULTI" -eq 1 ] && [[ "$DEVICE" == cpu* ]]; then
  echo "[warn] device=$DEVICE with multi_gpu is not valid. fallback to single."
  ENABLE_MULTI=0
fi
if [ "$ENABLE_MULTI" -eq 1 ] && [[ "$BACKEND" != qwen25_vl* ]]; then
  echo "[warn] multi_gpu is optimized for qwen backend. fallback to single for backend=$BACKEND."
  ENABLE_MULTI=0
fi

WORKER_COUNT=1
if [ "$ENABLE_MULTI" -eq 1 ]; then
  if [ -z "$NUM_WORKERS" ] || [ "$NUM_WORKERS" -le 0 ]; then
    WORKER_COUNT="$GPU_COUNT"
  else
    WORKER_COUNT="$NUM_WORKERS"
  fi
  if [ "$WORKER_COUNT" -gt "$GPU_COUNT" ]; then
    echo "[warn] num_workers($WORKER_COUNT) > gpu_count($GPU_COUNT). clipping."
    WORKER_COUNT="$GPU_COUNT"
  fi
  if [ "$WORKER_COUNT" -le 1 ]; then
    ENABLE_MULTI=0
    WORKER_COUNT=1
  fi
fi

COMMON_ARGS=(
  --teacher_scores_jsonl "$TEACHER_SCORES_JSONL"
  --backend "$BACKEND"
  --fallback_backend "$FALLBACK_BACKEND"
  --model_id "$MODEL_ID"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE"
  --image_dir "$IMAGE_DIR"
  --target_ar "$TARGET_AR"
  --top_m "$TOP_M"
  --top_k "$TOP_K"
  --max_images "$MAX_IMAGES"
  --max_retries "$MAX_RETRIES"
  --prompt_version "$PROMPT_VERSION"
  --seed "$SEED"
  --save_raw_response "$SAVE_RAW_RESPONSE"
  --skip_on_oom "$SKIP_ON_OOM"
  --fallback_cpu_on_oom "$FALLBACK_CPU_ON_OOM"
  --fallback_cpu_max_images "$FALLBACK_CPU_MAX_IMAGES"
  --skip_if_fallback_failed "$SKIP_IF_FALLBACK_FAILED"
  --strict_backend_init "$STRICT_BACKEND_INIT"
)

if [ "$ENABLE_MULTI" -eq 0 ]; then
  python3 src/vlm_teacher_labeler.py \
    "${COMMON_ARGS[@]}" \
    --output_jsonl "$OUTPUT_JSONL" \
    --output_meta_jsonl "$OUTPUT_META_JSONL" \
    --summary_json "$SUMMARY_JSON" \
    --debug_dir "$DEBUG_DIR"
else
  echo "[multi] vlm shard mode enabled: gpu_ids=$EFFECTIVE_GPU_IDS workers=$WORKER_COUNT"
  IFS=',' read -r -a GPU_ID_ARR <<< "$EFFECTIVE_GPU_IDS"
  SHARD_DIR="${OUTPUT_JSONL}.shards.$$"
  mkdir -p "$SHARD_DIR"
  if [ -n "$DEBUG_DIR" ]; then
    mkdir -p "$DEBUG_DIR"
  fi

  PIDS=()
  SHARD_LOGS=()
  for ((i=0; i<WORKER_COUNT; i++)); do
    GPU_ID="${GPU_ID_ARR[$i]}"
    SHARD_OUT_JSONL="${SHARD_DIR}/labels_shard_${i}.jsonl"
    SHARD_OUT_META="${SHARD_DIR}/meta_shard_${i}.jsonl"
    SHARD_SUMMARY="${SHARD_DIR}/summary_shard_${i}.json"
    SHARD_DEBUG_DIR=""
    if [ -n "$DEBUG_DIR" ]; then
      SHARD_DEBUG_DIR="${DEBUG_DIR}/shard_${i}"
    fi
    SHARD_LOG="${SHARD_DIR}/shard_${i}.log"

    echo "[multi] launch shard=$i/$WORKER_COUNT gpu=$GPU_ID -> $SHARD_OUT_JSONL"
    CUDA_VISIBLE_DEVICES="$GPU_ID" python3 src/vlm_teacher_labeler.py \
      "${COMMON_ARGS[@]}" \
      --device "cuda:0" \
      --output_jsonl "$SHARD_OUT_JSONL" \
      --output_meta_jsonl "$SHARD_OUT_META" \
      --summary_json "$SHARD_SUMMARY" \
      --debug_dir "$SHARD_DEBUG_DIR" \
      --num_shards "$WORKER_COUNT" \
      --shard_index "$i" \
      >"$SHARD_LOG" 2>&1 &
    PIDS+=("$!")
    SHARD_LOGS+=("$SHARD_LOG")
  done

  FAILED=0
  for ((i=0; i<WORKER_COUNT; i++)); do
    PID="${PIDS[$i]}"
    if ! wait "$PID"; then
      echo "[error] shard $i failed (log=${SHARD_LOGS[$i]})"
      FAILED=1
    fi
  done
  if [ "$FAILED" -ne 0 ]; then
    for LOG_PATH in "${SHARD_LOGS[@]}"; do
      if [ -f "$LOG_PATH" ]; then
        echo "----- tail: $LOG_PATH -----"
        tail -n 80 "$LOG_PATH" || true
      fi
    done
    exit 1
  fi

  : > "$OUTPUT_JSONL"
  if [ -n "$OUTPUT_META_JSONL" ]; then
    : > "$OUTPUT_META_JSONL"
  fi
  for ((i=0; i<WORKER_COUNT; i++)); do
    cat "${SHARD_DIR}/labels_shard_${i}.jsonl" >> "$OUTPUT_JSONL"
    if [ -n "$OUTPUT_META_JSONL" ]; then
      cat "${SHARD_DIR}/meta_shard_${i}.jsonl" >> "$OUTPUT_META_JSONL"
    fi
  done

  python - "$SHARD_DIR" "$WORKER_COUNT" "$EFFECTIVE_GPU_IDS" "$SUMMARY_JSON" "$OUTPUT_JSONL" "$OUTPUT_META_JSONL" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

shard_dir = Path(sys.argv[1])
n = int(sys.argv[2])
gpu_ids = sys.argv[3]
out_summary = Path(sys.argv[4])
out_jsonl = sys.argv[5]
out_meta_jsonl = sys.argv[6]

counts = Counter()
per_ar = Counter()
num_images_seen = 0
num_images_scanned = 0
num_tasks_seen = 0
cpu_fallback_active = False
cpu_fallback_image_count = 0
shards = []
base = None

for i in range(n):
    p = shard_dir / f"summary_shard_{i}.json"
    if not p.exists():
        raise SystemExit(f"missing shard summary: {p}")
    s = json.loads(p.read_text(encoding="utf-8"))
    shards.append({"shard_index": i, "summary_json": str(p)})
    if base is None:
        base = s
    counts.update(s.get("counts", {}))
    per_ar.update(s.get("per_ar_tasks", {}))
    num_images_seen += int(s.get("num_images_seen", 0))
    num_images_scanned = max(num_images_scanned, int(s.get("num_images_scanned", 0)))
    num_tasks_seen += int(s.get("num_tasks_seen", 0))
    cpu_fallback_active = cpu_fallback_active or bool(s.get("cpu_fallback_active", False))
    cpu_fallback_image_count += int(s.get("cpu_fallback_image_count", 0))

if base is None:
    base = {}

out = dict(base)
out["schema_version"] = "vlm_teacher_summary_v1"
out["output_jsonl"] = out_jsonl
out["output_meta_jsonl"] = out_meta_jsonl
out["counts"] = dict(counts)
out["per_ar_tasks"] = dict(per_ar)
out["num_images_seen"] = num_images_seen
out["num_images_scanned"] = num_images_scanned
out["num_tasks_seen"] = num_tasks_seen
out["num_shards"] = n
out["shard_index"] = -1
out["multi_gpu"] = {"enabled": True, "gpu_ids": gpu_ids, "workers": n, "shard_dir": str(shard_dir)}
out["cpu_fallback_active"] = cpu_fallback_active
out["cpu_fallback_image_count"] = cpu_fallback_image_count
out["shards"] = shards

out_summary.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
PY

  echo "[multi] merged output_jsonl -> $OUTPUT_JSONL"
  echo "[multi] merged output_meta_jsonl -> $OUTPUT_META_JSONL"
  echo "[multi] merged summary_json -> $SUMMARY_JSON"
  echo "[multi] shard logs/outputs -> $SHARD_DIR"
fi

echo "=============================================="
echo " Done."
echo "=============================================="
