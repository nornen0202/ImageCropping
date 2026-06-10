#!/usr/bin/env bash
# ==============================================================================
# run_vlm_teacher_labeler_qwen3_env.sh
# Stage-10 VLM Teacher runner using dedicated Qwen3 runtime environment.
#
# Purpose:
# - Keep C1~C6/Teacher pipeline environment stable (transformers 4.x range)
# - Run only VLM teacher with separate Qwen3-capable environment
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

QWEN3_VENV_PATH="/group-volume/jaden.ju/Venvs/qwen3_vlm"
BASE_PYTHON="python3"
BOOTSTRAP_QWEN3_ENV=1
REINSTALL_QWEN3_RUNTIME=0
QWEN3_ALLOW_GITHUB_FALLBACK=0

FORWARD_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash src/scripts/run_vlm_teacher_labeler_qwen3_env.sh [wrapper options] [run_vlm_teacher_labeler options...]

Wrapper options:
  --qwen3_venv_path PATH              Dedicated Qwen3 venv path (default: /group-volume/jaden.ju/Venvs/qwen3_vlm)
  --base_python BIN                   Python used to create venv when missing (default: python3)
  --bootstrap_qwen3_env 0|1           Create/fix qwen3 env automatically (default: 1)
  --reinstall_qwen3_runtime 0|1       Force runtime-only reinstall in qwen3 env (default: 0)
  --qwen3_allow_github_fallback 0|1   Allow transformers GitHub fallback during runtime install (default: 0)
  -h, --help                          Show this help

Examples:
  # 1) Teacher scores가 있을 때 Stage-10만 전용 Qwen3 env로 실행
  bash src/scripts/run_vlm_teacher_labeler_qwen3_env.sh \
    --qwen3_venv_path /group-volume/jaden.ju/Venvs/qwen3_vlm \
    --teacher_scores_jsonl data/SSTK/10K_local/artifacts/teacher/scores/teacher_scores_ar_rerun1_public_e2e_subject_mode.jsonl \
    --image_dir data/SSTK/10K_local/images \
    --output_jsonl data/SSTK/10K_local/artifacts/vlm_teacher/labels/crop_label_v1_rerun1_public_e2e_subject_mode.jsonl \
    --output_meta_jsonl data/SSTK/10K_local/artifacts/vlm_teacher/meta/meta_norm_v1_rerun1_public_e2e_subject_mode.jsonl \
    --summary_json data/SSTK/10K_local/artifacts/vlm_teacher/summary/vlm_teacher_summary_rerun1_public_e2e_subject_mode.json \
    --backend qwen25_vl \
    --model_id Qwen/Qwen3-VL-4B-Instruct \
    --device auto \
    --multi_gpu 1 --gpu_ids 0,1,2,3,4,5,6,7 --num_workers 8

  # 2) 최초 1회 runtime-only 설치를 강제로 재실행 후 실행
  bash src/scripts/run_vlm_teacher_labeler_qwen3_env.sh \
    --reinstall_qwen3_runtime 1 \
    --qwen3_allow_github_fallback 1 \
    --teacher_scores_jsonl <teacher_scores.jsonl> --image_dir <images_dir>

Notes:
  - This wrapper forces run_vlm_teacher_labeler to use qwen3 venv python.
  - It disables in-run autofix to protect base/pipeline environment.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --qwen3_venv_path) QWEN3_VENV_PATH="$2"; shift 2 ;;
    --base_python) BASE_PYTHON="$2"; shift 2 ;;
    --bootstrap_qwen3_env) BOOTSTRAP_QWEN3_ENV="$2"; shift 2 ;;
    --reinstall_qwen3_runtime) REINSTALL_QWEN3_RUNTIME="$2"; shift 2 ;;
    --qwen3_allow_github_fallback) QWEN3_ALLOW_GITHUB_FALLBACK="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) FORWARD_ARGS+=("$1"); shift ;;
  esac
done

QWEN3_PYTHON="${QWEN3_VENV_PATH}/bin/python"
QWEN3_ACTIVATE="${QWEN3_VENV_PATH}/bin/activate"

if [[ ! -x "${QWEN3_PYTHON}" ]]; then
  if [[ "${BOOTSTRAP_QWEN3_ENV}" != "1" ]]; then
    echo "[error] qwen3 venv missing: ${QWEN3_VENV_PATH}" >&2
    echo "        set --bootstrap_qwen3_env 1 or create venv manually." >&2
    exit 2
  fi
  echo "[bootstrap] creating qwen3 venv: ${QWEN3_VENV_PATH}"
  "${BASE_PYTHON}" -m venv "${QWEN3_VENV_PATH}"
fi

check_qwen3_symbol() {
  "${QWEN3_PYTHON}" - <<'PY'
import sys
try:
    import transformers
except Exception as exc:
    print(f"[qwen3-env] transformers import failed: {exc}")
    sys.exit(3)

has_qwen3 = hasattr(transformers, "Qwen3VLForConditionalGeneration")
print(f"[qwen3-env] python={sys.executable}")
print(f"[qwen3-env] transformers={transformers.__version__} has_qwen3={has_qwen3}")
sys.exit(0 if has_qwen3 else 2)
PY
}

Q3_OK=0
check_qwen3_symbol || Q3_OK=$?
if [[ "${Q3_OK}" -ne 0 ]] || [[ "${REINSTALL_QWEN3_RUNTIME}" == "1" ]]; then
  if [[ "${BOOTSTRAP_QWEN3_ENV}" != "1" ]]; then
    echo "[error] qwen3 runtime missing but bootstrap disabled." >&2
    exit 2
  fi
  echo "[bootstrap] installing qwen3 runtime-only stack in dedicated env..."
  (
    cd "${PROJECT_ROOT}"
    QWEN3_RUNTIME_ONLY=1 \
    QWEN3_ALLOW_GITHUB_FALLBACK="${QWEN3_ALLOW_GITHUB_FALLBACK}" \
    PYTHON="${QWEN3_PYTHON}" \
    bash src/scripts/install_features_deps_torch251_cu121_stable.sh
  )
  check_qwen3_symbol
fi

echo "[run] invoking run_vlm_teacher_labeler.sh with dedicated qwen3 env"
(
  cd "${PROJECT_ROOT}"
  PYTHON_BIN="${QWEN3_PYTHON}" \
  bash src/scripts/run_vlm_teacher_labeler.sh \
    "${FORWARD_ARGS[@]}" \
    --server_mode 1 \
    --venv_path "${QWEN3_ACTIVATE}" \
    --autofix_qwen3_runtime 0 \
    --strict_backend_init 1
)
