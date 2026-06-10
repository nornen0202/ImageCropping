#!/usr/bin/env bash
# Qwen3-VL-4B-Instruct 오프라인 전송/설치 도우미
# - 정상 서버: prepare -> pack
# - 문제 서버: install

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$PROJECT_ROOT"

ACTION="${1:-}"
if [ $# -gt 0 ]; then
  shift
fi

REPO_ID="Qwen/Qwen3-VL-4B-Instruct"
MODEL_NAME="Qwen3-VL-4B-Instruct"
TRANSFER_ROOT_DEFAULT="$PROJECT_ROOT/artifacts/offline_qwen3_vl"
MODEL_INSTALL_DIR_DEFAULT="$PROJECT_ROOT/artifacts/models"
PYTHON_BIN_DEFAULT="python3"

TRANSFER_ROOT="$TRANSFER_ROOT_DEFAULT"
MODEL_INSTALL_DIR="$MODEL_INSTALL_DIR_DEFAULT"
PYTHON_BIN="$PYTHON_BIN_DEFAULT"
BUNDLE_PATH=""

usage() {
  cat <<USAGE
Usage:
  bash src/scripts/transfer_qwen3_vl_offline.sh <action> [options]

Actions:
  prepare   정상 서버에서 HF 모델/휠 파일 다운로드
  pack      전송용 tar.gz 번들 생성
  install   문제 서버에서 번들 해제 + 오프라인 pip 설치

Options:
  --transfer-root PATH   작업 디렉토리 (default: artifacts/offline_qwen3_vl)
  --model-install-dir P  모델 설치 루트 (default: artifacts/models)
  --bundle-path PATH     번들 경로 (default: <transfer-root>/qwen3_vl_4b_offline_bundle.tgz)
  --python-bin BIN       python 실행 파일 (default: python3)

Examples:
  # 정상 서버
  bash src/scripts/transfer_qwen3_vl_offline.sh prepare
  bash src/scripts/transfer_qwen3_vl_offline.sh pack

  # 문제 서버
  bash src/scripts/transfer_qwen3_vl_offline.sh install

  # Stage 10 실행 시 권장 환경 변수
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --transfer-root)
      TRANSFER_ROOT="$2"
      shift 2
      ;;
    --model-install-dir)
      MODEL_INSTALL_DIR="$2"
      shift 2
      ;;
    --bundle-path)
      BUNDLE_PATH="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[error] unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

if [ -z "$ACTION" ]; then
  echo "[error] action is required"
  usage
  exit 1
fi

if [ -z "$BUNDLE_PATH" ]; then
  BUNDLE_PATH="$TRANSFER_ROOT/qwen3_vl_4b_offline_bundle.tgz"
fi

MODEL_CACHE_DIR="$TRANSFER_ROOT/model/$MODEL_NAME"
WHEELHOUSE_DIR="$TRANSFER_ROOT/wheelhouse"
EXTRACT_DIR="$TRANSFER_ROOT/extracted"
TARGET_MODEL_DIR="$MODEL_INSTALL_DIR/$MODEL_NAME"

prepare() {
  mkdir -p "$MODEL_CACHE_DIR" "$WHEELHOUSE_DIR"

  echo "[prepare] download model snapshot: $REPO_ID"
  "$PYTHON_BIN" - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="${REPO_ID}",
    repo_type="model",
    local_dir="${MODEL_CACHE_DIR}",
    local_dir_use_symlinks=False,
    resume_download=True,
)
print("[ok] model snapshot completed")
PY

  echo "[prepare] build/download offline wheelhouse"
  "$PYTHON_BIN" -m pip wheel -w "$WHEELHOUSE_DIR" \
    "git+https://github.com/huggingface/transformers"
  "$PYTHON_BIN" -m pip download -d "$WHEELHOUSE_DIR" \
    "tokenizers>=0.21.0" \
    "huggingface-hub>=0.26.0" \
    "accelerate>=0.30.0" \
    "safetensors" \
    "sentencepiece"

  echo "[prepare] done"
  echo "  model    : $MODEL_CACHE_DIR"
  echo "  wheels   : $WHEELHOUSE_DIR"
}

pack_bundle() {
  if [ ! -d "$MODEL_CACHE_DIR" ]; then
    echo "[error] model directory not found: $MODEL_CACHE_DIR"
    echo "        run: bash src/scripts/transfer_qwen3_vl_offline.sh prepare"
    exit 1
  fi
  if [ ! -d "$WHEELHOUSE_DIR" ]; then
    echo "[error] wheelhouse not found: $WHEELHOUSE_DIR"
    echo "        run: bash src/scripts/transfer_qwen3_vl_offline.sh prepare"
    exit 1
  fi

  mkdir -p "$TRANSFER_ROOT"
  tar -C "$TRANSFER_ROOT" -czf "$BUNDLE_PATH" model wheelhouse

  echo "[pack] done"
  echo "  bundle: $BUNDLE_PATH"
  echo "  copy command example:"
  echo "  scp '$BUNDLE_PATH' user@<problem-server>:'$BUNDLE_PATH'"
}

install_bundle() {
  if [ ! -f "$BUNDLE_PATH" ]; then
    echo "[error] bundle not found: $BUNDLE_PATH"
    exit 1
  fi

  rm -rf "$EXTRACT_DIR"
  mkdir -p "$EXTRACT_DIR" "$MODEL_INSTALL_DIR"

  echo "[install] extract bundle: $BUNDLE_PATH"
  tar -C "$EXTRACT_DIR" -xzf "$BUNDLE_PATH"

  echo "[install] install model files -> $TARGET_MODEL_DIR"
  rm -rf "$TARGET_MODEL_DIR"
  mkdir -p "$MODEL_INSTALL_DIR"
  cp -a "$EXTRACT_DIR/model/$MODEL_NAME" "$TARGET_MODEL_DIR"

  echo "[install] offline pip install from wheelhouse"
  "$PYTHON_BIN" -m pip install -U --no-index --find-links "$EXTRACT_DIR/wheelhouse" \
    transformers tokenizers huggingface-hub accelerate safetensors sentencepiece

  echo "[install] verify transformers runtime"
  "$PYTHON_BIN" - <<'PY'
import transformers
print("transformers:", transformers.__version__)
print("has_qwen3:", hasattr(transformers, "Qwen3VLForConditionalGeneration"))
if not hasattr(transformers, "Qwen3VLForConditionalGeneration"):
    raise SystemExit("ERROR: Qwen3VLForConditionalGeneration missing after offline install")
PY

  cat <<MSG
[install] done
  local model path: $TARGET_MODEL_DIR

[run-tip] Stage 10 실행 전에 아래 환경 변수를 권장합니다.
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

[run-tip] e2e 실행 시 모델 ID를 로컬 경로로 지정하세요.
  --vlm_model_id $TARGET_MODEL_DIR
MSG
}

case "$ACTION" in
  prepare)
    prepare
    ;;
  pack)
    pack_bundle
    ;;
  install)
    install_bundle
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "[error] unknown action: $ACTION"
    usage
    exit 1
    ;;
esac
