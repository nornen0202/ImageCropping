#!/usr/bin/env bash
# ============================================================
# [권장 실행 스크립트] SSTK Feature Extraction 의존성 설치
# ============================================================
# install_features_deps_torch251_cu121.sh 는 구버전 (deprecated).
# 이 파일(_stable.sh)이 정식 진입점입니다.
#
# Key points:
#   - Torch 2.5.1 + cu121 (wheel includes CUDA runtime)
#   - OpenMMLab: mmengine==0.10.7, mmcv==2.1.0 (CUDA ops 포함)
#     mmcv 설치 3단계 전략: binary wheel → source(--no-build-isolation) → direct curl
#   - mmdet/mmpose: NON-editable install (pip 26 PEP660 호환)
#   - EfficientViT / SAM2: NON-editable install, SAM2 CUDA ext 기본 비활성화
#   - numpy<2.0.0 global constraints 적용
#   - Qwen3-VL 런타임 심볼(Qwen3VLForConditionalGeneration) 자동 검증/보정
#     (PyPI 최신 우선 시도 후, 필요 시 GitHub fallback)
#   - PaddleOCR: C4 OCR용 기본 비활성화(필요 시 ENABLE_C4_OCR=1로 활성화)
#
# ==============================================================================
# USAGE
#   # 서버(무가상환경) 기준:
#   bash src/scripts/install_features_deps_torch251_cu121_stable.sh
#   # 필요 시 python 바이너리 명시:
#   PYTHON=python3 bash src/scripts/install_features_deps_torch251_cu121_stable.sh
#   # (권장) 파이프라인 전용: Qwen3 자동 업그레이드 비활성(기본값)
#   ENABLE_QWEN3_VL=0 bash src/scripts/install_features_deps_torch251_cu121_stable.sh
#   # Qwen3-VL 전용(별도 env) 최소 설치 모드:
#   QWEN3_RUNTIME_ONLY=1 PYTHON=python3 bash src/scripts/install_features_deps_torch251_cu121_stable.sh
# ==============================================================================

set -euo pipefail

# -----------------------------
# Versions (pin for stability)
# -----------------------------
TORCH_VER="2.5.1"
TORCHVISION_VER="0.20.1"
TORCHAUDIO_VER="2.5.1"
TORCH_CUDA="cu121"

MMENGINE_VER="0.10.7"
# mmcv-lite → mmcv (CUDA 빌드 휠)로 교체
# mmpose inference 시 mmcv._ext(CUDA ops) 필요: mmcv-lite에는 미포함 → 런타임 에러 발생
# torch2.5.1+cu121 호환 빌드는 OpenMMLab wheel index에서 제공
MMCV_VER="2.1.0"
MMCV_WHEEL_INDEX="https://download.openmmlab.com/mmcv/dist/cu121/torch2.5.1/index.html"

MMDET_TAG="v3.3.0"
MMPOSE_TAG="v1.3.2"

PADDLE_GPU_VER="3.3.0"
PADDLE_CUDA_INDEX="${PADDLE_CUDA_INDEX:-https://www.paddlepaddle.org.cn/packages/stable/cu121/}"  # CUDA 12.1 우선

# Constraints (global)
NUMPY_CONSTRAINT="numpy<2.0.0"
# Qwen2.5-VL(model_type=qwen2_5_vl) 지원을 위해 transformers>=4.49 필요.
# 기본은 안정성 상한을 두되, Qwen3-VL 심볼이 없으면 아래에서 git head로 자동 보정한다.
TRANSFORMERS_CONSTRAINT="transformers>=4.49.0,<4.53.0"
# 1이면 Qwen3-VL 심볼을 강제 보장한다.
# 기본 0: C3/mmpretrain 안정성을 위해 파이프라인 환경에서 자동 업그레이드를 막는다.
ENABLE_QWEN3_VL="${ENABLE_QWEN3_VL:-0}"
# 1이면 Qwen3 심볼 누락 시 먼저 PyPI/mirror에서 최신 transformers로 재시도한다.
QWEN3_TRY_PYPI_LATEST="${QWEN3_TRY_PYPI_LATEST:-1}"
# 1이면 PyPI 최신 재시도 후에도 실패할 때 GitHub source fallback을 허용한다.
QWEN3_ALLOW_GITHUB_FALLBACK="${QWEN3_ALLOW_GITHUB_FALLBACK:-1}"
# 1이면 base pipeline deps 설치를 건너뛰고 Qwen3-VL 런타임만 설치 후 종료한다.
# 별도 환경(권장)에서 VLM teacher 전용으로 사용할 때 활성화.
QWEN3_RUNTIME_ONLY="${QWEN3_RUNTIME_ONLY:-0}"
# 1이면 C4 quality_first용 PaddleOCR(PP-OCRv5 runtime)를 설치한다. (default: OFF)
ENABLE_C4_OCR="${ENABLE_C4_OCR:-0}"
# 1이면 C4 활성 시 Paddle CUDA 런타임(컴파일+device_count>0)을 필수로 요구한다.
REQUIRE_PADDLE_CUDA_FOR_C4="${REQUIRE_PADDLE_CUDA_FOR_C4:-1}"
# 1이면 C5 quality_first용 ScaleLSD를 설치 시도한다.
ENABLE_SCALELSD="${ENABLE_SCALELSD:-1}"
# 1이면 C6 quality_first용 Gazelle(Gaze-LLE)를 설치 시도한다.
ENABLE_GAZELLE="${ENABLE_GAZELLE:-1}"
# 1이면 quality_first 모듈(C4/C5/C6) import 검증 실패 시 즉시 종료한다.
STRICT_QUALITY_IMPORTS="${STRICT_QUALITY_IMPORTS:-1}"
# 1이면 pip/setuptools/wheel을 매 실행마다 업그레이드한다. (default: skip)
UPGRADE_PIP_TOOLING="${UPGRADE_PIP_TOOLING:-0}"

# -----------------------------
PYTHON="${PYTHON:-python3}"
PIP="$PYTHON -m pip"

# -----------------------------
# Resolve project root robustly
# - If script is under src/scripts/, PROJECT_ROOT becomes repo root.
# - Otherwise fallback to current working directory.
# -----------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "${SCRIPT_DIR}/../.." ]]; then
  WORK_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
else
  WORK_DIR="$(pwd)"
fi

TP_DIR="${WORK_DIR}/third_party"
mkdir -p "${TP_DIR}"

echo "=============================================="
echo "SSTK Feature Extraction Dependencies Installer"
echo "STABLE TORCH/CUDA PLAN: torch=${TORCH_VER}+${TORCH_CUDA}"
echo "PROJECT ROOT: ${WORK_DIR}"
echo "=============================================="

# -----------------------------
# Sanity checks
# -----------------------------
$PYTHON - <<'PY'
import sys
major, minor = sys.version_info[:2]
print("Python:", sys.version.replace("\n"," "))
print("Executable:", sys.executable)
if (major, minor) < (3, 10):
    raise SystemExit("ERROR: Python>=3.10 required (SAM2 requires >=3.10).")
PY

# -----------------------------
# pip install args: 서버 무가상환경 기본. 권한 이슈 시 설치 권한이 있는 계정/컨테이너에서 실행.
PIP_INSTALL_ARGS=()
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "INFO: VIRTUAL_ENV not set. Running in system/no-venv mode."
fi

CONSTRAINTS_FILE="${WORK_DIR}/.pip_constraints_features.txt"
cat > "${CONSTRAINTS_FILE}" <<EOF
${NUMPY_CONSTRAINT}
${TRANSFORMERS_CONSTRAINT}
EOF

# Wrapper helpers (always apply constraints)
pip_install() {
  $PIP install "${PIP_INSTALL_ARGS[@]}" -c "${CONSTRAINTS_FILE}" "$@"
}

pip_uninstall() {
  $PIP uninstall -y "$@" || true
}

pip_install_noc() {
  $PIP install "${PIP_INSTALL_ARGS[@]}" "$@"
}

_VER() { $PYTHON -m pip show "$1" 2>/dev/null | awk '/^Version:/{print $2}'; }

install_if_missing() {
  local pkg_name="$1"
  shift
  local installed
  installed="$(_VER "${pkg_name}")"
  if [[ -n "${installed}" ]]; then
    echo "  [SKIP] ${pkg_name} already installed: ${installed}"
  else
    pip_install "$@"
  fi
}

check_qwen3_symbol() {
  $PYTHON - <<'PY'
import sys
try:
    import transformers
    has_qwen3 = hasattr(transformers, "Qwen3VLForConditionalGeneration")
    print(f"  transformers={transformers.__version__} has_qwen3={has_qwen3}")
    sys.exit(0 if has_qwen3 else 2)
except Exception as exc:
    print(f"  transformers import failed: {exc}")
    sys.exit(3)
PY
}

install_qwen3_runtime_stack() {
  echo "Installing Qwen3-VL runtime stack..."
  pip_install_noc -U \
    "transformers" \
    "tokenizers>=0.21.0" \
    "huggingface-hub>=0.26.0" \
    "accelerate>=0.30.0"

  local qwen_ok=0
  check_qwen3_symbol || qwen_ok=$?
  if [[ "${qwen_ok}" -ne 0 ]] && [[ "${QWEN3_ALLOW_GITHUB_FALLBACK}" == "1" ]]; then
    echo "  [Fixup] PyPI/mirror latest still missing qwen3 symbol. Trying GitHub source..."
    pip_install_noc -U \
      "git+https://github.com/huggingface/transformers" \
      "tokenizers>=0.21.0" \
      "huggingface-hub>=0.26.0" \
      "accelerate>=0.30.0"
    qwen_ok=0
    check_qwen3_symbol || qwen_ok=$?
  fi

  if [[ "${qwen_ok}" -ne 0 ]]; then
    echo "ERROR: Qwen3 runtime stack install failed (symbol missing)."
    return 1
  fi
  return 0
}

check_paddle_import() {
  $PYTHON - <<'PY'
import sys
try:
    import paddle
    print(f"  paddle={paddle.__version__}")
    sys.exit(0)
except Exception as exc:
    print(f"  paddle import failed: {exc}")
    sys.exit(2)
PY
}

check_paddleocr_textdet_import() {
  $PYTHON - <<'PY'
import sys
try:
    from paddleocr import TextDetection
    print("  paddleocr.TextDetection: OK")
    sys.exit(0)
except Exception as exc:
    print(f"  paddleocr.TextDetection import failed: {exc}")
    sys.exit(2)
PY
}

check_paddle_cuda_ready() {
  $PYTHON - <<'PY'
import sys
try:
    import paddle
except Exception as exc:
    print(f"  paddle import failed: {exc}")
    sys.exit(2)

compiled = False
try:
    compiled = bool(paddle.is_compiled_with_cuda())
except Exception:
    compiled = False

count = 0
if compiled:
    try:
        count = int(paddle.device.cuda.device_count())
    except Exception:
        count = 0

print(f"  paddle_cuda_ready: compiled={compiled} device_count={count}")
sys.exit(0 if (compiled and count > 0) else 3)
PY
}

check_xtcocotools_import() {
  $PYTHON - <<'PY'
import sys
import numpy as np
print("  numpy:", np.__version__)
try:
    from xtcocotools import _mask  # noqa: F401
    print("  xtcocotools._mask: OK")
    sys.exit(0)
except Exception as exc:
    print(f"  xtcocotools._mask import failed: {exc}")
    sys.exit(2)
PY
}

repair_xtcocotools_abi() {
  echo "  [Fixup] repairing xtcocotools binary ABI..."
  pip_install "${NUMPY_CONSTRAINT}"
  pip_uninstall xtcocotools || true
  pip_install_noc "cython>=0.29.36"
  if ! pip_install_noc --no-cache-dir --force-reinstall --no-binary :all: xtcocotools; then
    echo "  [WARN] source build failed. trying binary reinstall..."
    pip_install_noc --no-cache-dir --force-reinstall xtcocotools || true
  fi
}

install_paddle_gpu_stack() {
  # NOTE:
  # - Some mirrors expose only CPU paddlepaddle. In that case, force reinstall
  #   from Paddle CUDA index first, then try default mirrors as fallback.
  # - We always re-check runtime readiness after install attempts.
  echo "  [Fixup] reinstalling Paddle stack for CUDA runtime..."
  pip_uninstall paddlepaddle paddlepaddle-gpu paddlepaddle-cpu || true

  if ! pip_install_noc --no-cache-dir --force-reinstall -i "${PADDLE_CUDA_INDEX}" "paddlepaddle-gpu==${PADDLE_GPU_VER}"; then
    echo "  [WARN] paddlepaddle-gpu install from Paddle CUDA index failed."
    echo "  [WARN] trying default index/mirror for paddlepaddle-gpu..."
    if ! pip_install_noc --no-cache-dir --force-reinstall "paddlepaddle-gpu==${PADDLE_GPU_VER}"; then
      echo "  [WARN] paddlepaddle-gpu install from default index failed."
      echo "  [WARN] trying paddlepaddle fallback (CPU-only possible)..."
      pip_install_noc --no-cache-dir --force-reinstall "paddlepaddle==${PADDLE_GPU_VER}" || true
    fi
  fi
}

# -----------------------------
# 0) Tooling (do NOT use openmim)
# -----------------------------
echo "0) pip/setuptools/wheel check..."
if [[ "${UPGRADE_PIP_TOOLING}" == "1" ]]; then
  echo "  [RUN] UPGRADE_PIP_TOOLING=1 -> upgrading pip/setuptools/wheel"
  pip_install_noc -U pip setuptools wheel
else
  echo "  [SKIP] UPGRADE_PIP_TOOLING=${UPGRADE_PIP_TOOLING} -> keep existing pip/setuptools/wheel"
fi

echo "0a) Removing openmim/opendatalab/openxlab if present (prevents setuptools/filelock conflicts)..."
pip_uninstall openmim opendatalab openxlab || true

# -----------------------------
# 1) Torch/CUDA baseline
# -----------------------------
echo "1) Installing PyTorch ${TORCH_VER} (${TORCH_CUDA}) + matching torchvision/torchaudio..."
TORCH_INSTALLED="$($PYTHON -m pip show torch 2>/dev/null | awk '/^Version:/{print $2}' || true)"
if [[ "${TORCH_INSTALLED}" == "${TORCH_VER}+${TORCH_CUDA}" || "${TORCH_INSTALLED}" == "${TORCH_VER}" ]]; then
  echo "  [SKIP] torch already installed: ${TORCH_INSTALLED}"
else
  # Clean old torch stack first (best effort)
  pip_uninstall torch torchvision torchaudio || true
  # Install from official CUDA wheel index
  $PIP install "${PIP_INSTALL_ARGS[@]}" \
    --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" \
    "torch==${TORCH_VER}" "torchvision==${TORCHVISION_VER}" "torchaudio==${TORCHAUDIO_VER}"
fi

$PYTHON - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
PY

if [[ "${QWEN3_RUNTIME_ONLY}" == "1" ]]; then
  echo "QWEN3_RUNTIME_ONLY=1: installing Qwen3 runtime and skipping remaining pipeline deps."
  install_qwen3_runtime_stack
  echo "=============================================="
  echo "Qwen3 runtime-only installation complete."
  echo "Use this environment for Stage-10 VLM teacher only."
  echo "=============================================="
  exit 0
fi

# -----------------------------
# 2) Base ML libs
# -----------------------------
echo "2) Installing base ML libraries..."
pip_install "${NUMPY_CONSTRAINT}"

# Qwen2.5-VL 런타임 호환성 보장:
# - 기본 설치는 재설치 최소화를 위해 -U 없이 수행.
# - Qwen3 심볼이 누락된 경우에만 아래 Fixup(-U) 경로로 업그레이드 수행.
pip_install \
  "${TRANSFORMERS_CONSTRAINT}" \
  "tokenizers>=0.21.0,<0.22.0" \
  "huggingface-hub>=0.26.0" \
  "accelerate>=0.30.0"

echo "2a) Verifying Qwen3-VL runtime symbol..."
QWEN3_OK=0
check_qwen3_symbol || QWEN3_OK=$?

if [[ "${ENABLE_QWEN3_VL}" == "1" ]] && [[ "${QWEN3_OK}" -ne 0 ]]; then
  if [[ "${QWEN3_TRY_PYPI_LATEST}" == "1" ]]; then
    echo "  [Fixup-A] Qwen3-VL class missing. Trying latest transformers from PyPI/mirror..."
    # NOTE:
    # - This step intentionally bypasses constraints(-c). 사내 PyPI mirror를 통한 최신판을 우선 시도한다.
    pip_install_noc -U \
      "transformers" \
      "tokenizers>=0.21.0" \
      "huggingface-hub>=0.26.0" \
      "accelerate>=0.30.0"
    QWEN3_OK=0
    check_qwen3_symbol || QWEN3_OK=$?
  fi

  if [[ "${QWEN3_OK}" -ne 0 ]] && [[ "${QWEN3_ALLOW_GITHUB_FALLBACK}" == "1" ]]; then
    echo "  [Fixup-B] PyPI/mirror 최신판으로도 미해결. Trying transformers git head..."
    # NOTE:
    # - GitHub source fallback은 방화벽/정책 환경에서 실패할 수 있다.
    pip_install_noc -U \
      "git+https://github.com/huggingface/transformers" \
      "tokenizers>=0.21.0" \
      "huggingface-hub>=0.26.0" \
      "accelerate>=0.30.0"
    QWEN3_OK=0
    check_qwen3_symbol || QWEN3_OK=$?
  fi

  if [[ "${QWEN3_OK}" -ne 0 ]]; then
    echo "ERROR: Qwen3VLForConditionalGeneration still missing."
    echo "  - ENABLE_QWEN3_VL=${ENABLE_QWEN3_VL}"
    echo "  - QWEN3_TRY_PYPI_LATEST=${QWEN3_TRY_PYPI_LATEST}"
    echo "  - QWEN3_ALLOW_GITHUB_FALLBACK=${QWEN3_ALLOW_GITHUB_FALLBACK}"
    echo "Retry example:"
    echo "  python -m pip install -U transformers \"tokenizers>=0.21.0\" \"huggingface-hub>=0.26.0\" \"accelerate>=0.30.0\""
    echo "Or disable enforcement with ENABLE_QWEN3_VL=0 and use Qwen2.5-VL."
    exit 1
  fi
elif [[ "${ENABLE_QWEN3_VL}" != "1" ]]; then
  echo "  [Info] ENABLE_QWEN3_VL=${ENABLE_QWEN3_VL}: skipping qwen3 symbol enforcement."
  echo "  [Info] Keeping transformers in stable range (${TRANSFORMERS_CONSTRAINT}) for C3/mmpretrain compatibility."
fi

# ray
if [[ -z "$(_VER ray)" ]]; then
  pip_install "ray[default]"
else
  echo "  [SKIP] ray already installed: $(_VER ray)"
fi

# ultralytics
if [[ -z "$(_VER ultralytics)" ]]; then
  pip_install "ultralytics>=8.0.0"
else
  echo "  [SKIP] ultralytics already installed: $(_VER ultralytics)"
fi

# open_clip_torch
if [[ -z "$(_VER open_clip_torch)" ]]; then
  pip_install open_clip_torch
else
  echo "  [SKIP] open_clip_torch already installed: $(_VER open_clip_torch)"
fi

# python-json-logger (ScaleLSD 및 debug healthcheck에서 사용)
install_if_missing "python-json-logger" "python-json-logger"

# sentence-transformers, timm, webdataset, pyarrow, pandas, opencv
install_if_missing "sentence-transformers" "sentence-transformers"
install_if_missing "timm" "timm"
install_if_missing "webdataset" "webdataset"
install_if_missing "pyarrow" "pyarrow"
install_if_missing "pandas" "pandas"
install_if_missing "opencv-python-headless" "opencv-python-headless>=4.6.0"
install_if_missing "opencv-contrib-python-headless" "opencv-contrib-python-headless>=4.6.0"
pip_install "${NUMPY_CONSTRAINT}"  # re-enforce after opencv

# -----------------------------
# 3) OpenMMLab core (NO openmim)
# -----------------------------
echo "3) Installing OpenMMLab core via pip (mmengine + mmcv CUDA wheel + mmpretrain)..."
# conflicting packages only when present
MMCV_LITE_INSTALLED="$(_VER mmcv-lite)"
MMCV_FULL_INSTALLED="$(_VER mmcv-full)"
if [[ -n "${MMCV_LITE_INSTALLED}" ]] || [[ -n "${MMCV_FULL_INSTALLED}" ]]; then
  echo "  [Fixup] removing conflicting mmcv-lite/mmcv-full..."
  pip_uninstall mmcv-lite mmcv-full || true
fi

# [CRITICAL] mmengine 설치 전 setuptools 복구:
# openxlab==0.1.3이 setuptools~=60.2.0으로 다운그레이드한 경우
# pkg_resources 미등록 에러가 발생할 수 있으므로 먼저 복구
echo "  [Pre-step] Restoring setuptools >= 68 for pkg_resources availability..."
pip_install_noc "setuptools>=68" wheel 2>/dev/null || true

MMENGINE_INSTALLED="$(_VER mmengine)"
if [[ "${MMENGINE_INSTALLED}" == "${MMENGINE_VER}" ]]; then
  echo "  [SKIP] mmengine==${MMENGINE_VER} already installed."
else
  if [[ -n "${MMENGINE_INSTALLED}" ]]; then
    echo "  [Fixup] replacing mmengine ${MMENGINE_INSTALLED} -> ${MMENGINE_VER}"
    pip_uninstall mmengine || true
  fi
  pip_install "mmengine==${MMENGINE_VER}"
fi

MMPRETRAIN_INSTALLED="$(_VER mmpretrain)"
if [[ -n "${MMPRETRAIN_INSTALLED}" ]]; then
  echo "  [SKIP] mmpretrain already installed: ${MMPRETRAIN_INSTALLED}"
else
  pip_install "mmpretrain>=1.0.0"
fi

# mmcv CUDA 빌드 휠 설치 ————————————————————————————————————————
# mmpose inference 에 필요한 _ext CUDA ops 포함 버전
MMCV_INSTALLED="$($PYTHON -m pip show mmcv 2>/dev/null | awk '/^Version:/{print $2}' || true)"
if [[ "${MMCV_INSTALLED}" == "${MMCV_VER}" ]]; then
  echo "  [SKIP] mmcv==${MMCV_VER} already installed."
else
  if [[ -n "${MMCV_INSTALLED}" ]]; then
    echo "  [Fixup] replacing mmcv ${MMCV_INSTALLED} -> ${MMCV_VER}"
    pip_uninstall mmcv || true
  fi
  # 전략 1으로: --only-binary (소스 빌드 금지) + openmmlab wheel index
  # 서버 환경에서 엔터프라이즈 pip proxy가 openmmlab URL을 보통 제한하므로
  # 사내 proxy URL 만 통하는 데도 테스트 (실패하면 전략 2로)
  echo "  [Strategy 1] Installing mmcv==${MMCV_VER} binary wheel (no source build)..."
  if $PIP install "${PIP_INSTALL_ARGS[@]}" \
      --only-binary :all: \
      --extra-index-url "${MMCV_WHEEL_INDEX}" \
      "mmcv==${MMCV_VER}"; then
    echo "  [Strategy 1] mmcv wheel install: SUCCESS"
  else
    # 전략 2: --no-build-isolation으로 소스 빌드
    # pkg_resources 에러를 피하려면 격리 환경 무효화 + 현재 setuptools 사용
    echo "  [Strategy 1] FAILED. Trying Strategy 2: source build with --no-build-isolation..."
    $PIP install "${PIP_INSTALL_ARGS[@]}" \
      --no-build-isolation \
      --find-links "${MMCV_WHEEL_INDEX}" \
      "mmcv==${MMCV_VER}" || {
      # 전략 3: 직접 wheel URL 다운로드
      echo "  [Strategy 2] FAILED. Trying Strategy 3: direct wheel download..."
      MMCV_WHL="mmcv-${MMCV_VER}-cp310-cp310-manylinux1_x86_64.whl"
      MMCV_URL="https://download.openmmlab.com/mmcv/dist/cu121/torch2.5.1/${MMCV_WHL}"
      TMP_WHL="/tmp/${MMCV_WHL}"
      echo "  Downloading: ${MMCV_URL}"
      if curl -fsSL -o "${TMP_WHL}" "${MMCV_URL}"; then
        $PIP install "${PIP_INSTALL_ARGS[@]}" "${TMP_WHL}"
        rm -f "${TMP_WHL}"
      else
        echo "  [ERROR] All strategies failed for mmcv==${MMCV_VER}."
        echo "  Manual install required:"
        echo "    pip install mmcv==${MMCV_VER} --no-build-isolation --find-links ${MMCV_WHEEL_INDEX}"
        exit 1
      fi
    }
  fi
fi

$PYTHON - <<'PY'
import mmengine
import mmcv
print("MMEngine:", mmengine.__version__)
print("MMCV:", getattr(mmcv, "__version__", "unknown"))
try:
    from mmcv import _ext
    print("MMCV _ext (CUDA ops): OK")
except ImportError as e:
    print(f"MMCV _ext WARNING: {e}")
PY

# Helper: install requirements but drop mmcv/mmengine pins to avoid pulling full mmcv/mmengine
#install_filtered_requirements () {
#  local req_file="$1"
#  if [[ -f "${req_file}" ]]; then
#    echo "  - Installing filtered requirements from ${req_file}"
#    local tmp_req
#    tmp_req="$(mktemp)"
#    grep -vE '^\s*(mmcv|mmcv-lite|mmengine)\b' "${req_file}" | sed '/^\s*#/d;/^\s*$/d' > "${tmp_req}"
#    if [[ -s "${tmp_req}" ]]; then
#      pip_install -r "${tmp_req}"
#    fi
#    rm -f "${tmp_req}"
#  fi
#}
install_filtered_requirements () {
  local req_file="$1"
  if [[ -f "${req_file}" ]]; then
    echo "  - Installing filtered requirements from ${req_file}"
    local tmp_req
    tmp_req="$(mktemp)"
    # mmcv/mmengine 뿐 아니라 chumpy도 제외 (PEP517 build isolation에서 자주 깨짐)
    grep -vE '^\s*(mmcv|mmcv-lite|mmengine|chumpy)\b' "${req_file}" \
      | sed '/^\s*#/d;/^\s*$/d' > "${tmp_req}"

    if [[ -s "${tmp_req}" ]]; then
      # requirements 설치 시에도 build isolation 끄기(구형 sdist 패키지 대응)
      pip_install --no-build-isolation --prefer-binary -r "${tmp_req}"
    fi
    rm -f "${tmp_req}"
  fi
}

# -----------------------------
# 3a) MMDetection (NON-editable install for pip 26)
# -----------------------------
echo "3a) Installing MMDetection (${MMDET_TAG}) from source (NON-editable)..."
cd "${TP_DIR}"
MMDET_INSTALLED="$($PYTHON -m pip show mmdet 2>/dev/null | awk '/^Version:/{print $2}' || true)"
if [[ -n "${MMDET_INSTALLED}" ]]; then
  echo "  [SKIP] mmdet==${MMDET_INSTALLED} already installed."
else
  if [[ ! -d "mmdetection" ]]; then
    git clone --depth 1 --branch "${MMDET_TAG}" https://github.com/open-mmlab/mmdetection.git
  fi
  cd "${TP_DIR}/mmdetection"
  install_filtered_requirements "requirements/runtime.txt"
  pip_install -v . --no-deps --no-build-isolation
  cd "${TP_DIR}"
fi

# -----------------------------
# 3b) MMPose (NON-editable install for pip 26)
# -----------------------------
echo "3b) Installing MMPose (${MMPOSE_TAG}) from source (NON-editable)..."
cd "${TP_DIR}"
MMPOSE_INSTALLED="$($PYTHON -m pip show mmpose 2>/dev/null | awk '/^Version:/{print $2}' || true)"
if [[ -n "${MMPOSE_INSTALLED}" ]]; then
  echo "  [SKIP] mmpose==${MMPOSE_INSTALLED} already installed."
else
  if [[ ! -d "mmpose" ]]; then
    git clone --depth 1 --branch "${MMPOSE_TAG}" https://github.com/open-mmlab/mmpose.git
  fi
  cd "${TP_DIR}/mmpose"
  install_filtered_requirements "requirements/runtime.txt"
  pip_install -v . --no-deps --no-build-isolation
  cd "${TP_DIR}"
fi

cd "${WORK_DIR}"

# -----------------------------
# 3c) xtcocotools ABI check/fix (mmpose runtime critical)
# -----------------------------
echo "3c) Verifying xtcocotools ABI..."
if ! check_xtcocotools_import; then
  repair_xtcocotools_abi
  if ! check_xtcocotools_import; then
    echo "ERROR: xtcocotools ABI is still broken."
    echo "  - This causes: ValueError: numpy.dtype size changed"
    echo "  - Verify numpy version and rebuild xtcocotools in the same interpreter."
    exit 1
  fi
fi

# -----------------------------
# 4) PaddleOCR (default OFF, C4 quality_first)
# -----------------------------
if [[ "${ENABLE_C4_OCR}" == "1" ]]; then
  echo "4) Installing PaddlePaddle-GPU + PaddleOCR (C4 enabled)..."

  # paddleocr 구버전 transitive 의존에서 jinja2 soft_unicode 이슈가 나는 환경 방지
  pip_install_noc "jinja2>=3.1.4"

  PADDLE_IMPORT_OK=0
  check_paddle_import || PADDLE_IMPORT_OK=$?

  PADDLE_CUDA_OK=1
  if [[ "${REQUIRE_PADDLE_CUDA_FOR_C4}" == "1" ]]; then
    PADDLE_CUDA_OK=0
    check_paddle_cuda_ready && PADDLE_CUDA_OK=1 || PADDLE_CUDA_OK=0
  fi

  if [[ "${PADDLE_IMPORT_OK}" -ne 0 ]] || [[ "${PADDLE_CUDA_OK}" -ne 1 ]]; then
    install_paddle_gpu_stack
  else
    echo "  [SKIP] paddle runtime already healthy for current policy."
  fi

  if ! check_paddle_import; then
    echo "ERROR: paddle import is still failing after installation attempts."
    echo "  python executable: $($PYTHON -c 'import sys; print(sys.executable)')"
    echo "  pip executable   : $($PYTHON -m pip --version)"
    $PYTHON -m pip show paddlepaddle-gpu paddlepaddle paddleocr || true
    exit 1
  fi

  if [[ "${REQUIRE_PADDLE_CUDA_FOR_C4}" == "1" ]]; then
    echo "4a) Verifying Paddle CUDA runtime for C4..."
    if ! check_paddle_cuda_ready; then
      echo "ERROR: Paddle CUDA runtime is not ready (compiled_with_cuda/device_count check failed)."
      echo "  This server run requires GPU-backed C4 OCR to avoid CPU fallback instability."
      echo "  - REQUIRE_PADDLE_CUDA_FOR_C4=${REQUIRE_PADDLE_CUDA_FOR_C4}"
      echo "  - PADDLE_CUDA_INDEX=${PADDLE_CUDA_INDEX}"
      echo "  - python executable: $($PYTHON -c 'import sys; print(sys.executable)')"
      echo "  - pip executable   : $($PYTHON -m pip --version)"
      $PYTHON -m pip show paddlepaddle-gpu paddlepaddle paddleocr || true
      exit 1
    fi
  fi

  # PP-OCRv5 runtime API(TextDetection) 포함 버전 범위
  PADDLEOCR_INSTALLED="$(_VER paddleocr)"
  if [[ -z "${PADDLEOCR_INSTALLED}" ]]; then
    pip_install_noc "paddleocr>=3.0.0"
  else
    echo "  [SKIP] paddleocr already installed: ${PADDLEOCR_INSTALLED}"
  fi

  if ! check_paddleocr_textdet_import; then
    echo "ERROR: paddleocr.TextDetection import failed after install."
    echo "  python executable: $($PYTHON -c 'import sys; print(sys.executable)')"
    $PYTHON -m pip show paddleocr paddlepaddle-gpu paddlepaddle || true
    exit 1
  fi
else
  echo "4) Skipping PaddleOCR install (ENABLE_C4_OCR=${ENABLE_C4_OCR})"
fi

# -----------------------------
# 5) EfficientViT
# -----------------------------
echo "5) Installing EfficientViT..."
cd "${TP_DIR}"
EFFVIT_INSTALLED="$($PYTHON -m pip show efficientvit 2>/dev/null | awk '/^Version:/{print $2}' || true)"
if [[ -n "${EFFVIT_INSTALLED}" ]]; then
  echo "  [SKIP] efficientvit==${EFFVIT_INSTALLED} already installed."
else
  if [[ ! -d "efficientvit" ]]; then
    git clone --depth 1 https://github.com/mit-han-lab/efficientvit.git
  fi
  cd "${TP_DIR}/efficientvit"
  pip_install "opt-einsum>=3.3.0"
  pip_install -v . --no-build-isolation
  cd "${TP_DIR}"
fi

# -----------------------------
# 6) SAM2
# -----------------------------
echo "6) Installing SAM2..."
cd "${TP_DIR}"
SAM2_INSTALLED="$($PYTHON -m pip show SAM-2 2>/dev/null | awk '/^Version:/{print $2}' || true)"
if [[ -n "${SAM2_INSTALLED}" ]]; then
  echo "  [SKIP] SAM-2==${SAM2_INSTALLED} already installed."
else
  if [[ ! -d "sam2" ]]; then
    git clone --depth 1 https://github.com/facebookresearch/sam2.git
  fi

  cd "${TP_DIR}/sam2"
  pip_install ninja

  # SAM2_BUILD_CUDA:
  #   - GPU 추론은 SAM2_BUILD_CUDA 값에 관계없이 항상 PyTorch 통해 GPU로 동작함.
  #   - 1 이면: SAM2 자체 커스텀 CUDA 커널이 추가 컴파일됨 (메모리 효율 attention 최적화).
  #   - 0 이면: 표준 PyTorch 연산만 사용 (GPU 추론 정상 동작, cudatoolkit/nvcc 불필요).
  #   → nvcc 유무를 자동 감지하여 설정.
  if [[ -z "${SAM2_BUILD_CUDA:-}" ]]; then
    if command -v nvcc &>/dev/null; then
      NVCC_VER=$(nvcc --version | grep 'release' | awk '{print $5}' | tr -d ',')
      echo "  nvcc detected (${NVCC_VER}). Enabling SAM2 CUDA extension build (SAM2_BUILD_CUDA=1)."
      export SAM2_BUILD_CUDA=1
    else
      echo "  nvcc not found. SAM2 CUDA extension build disabled (SAM2_BUILD_CUDA=0)."
      echo "  GPU inference still works via PyTorch. Only custom CUDA kernels are skipped."
      export SAM2_BUILD_CUDA=0
    fi
  else
    echo "  SAM2_BUILD_CUDA already set by user: ${SAM2_BUILD_CUDA}"
  fi
  export SAM2_BUILD_ALLOW_ERRORS="${SAM2_BUILD_ALLOW_ERRORS:-1}"

  pip_install -v . --no-build-isolation
  cd "${TP_DIR}"
fi

cd "${WORK_DIR}"

# -----------------------------
# 7) ScaleLSD (optional, C5 quality_first)
# -----------------------------
if [[ "${ENABLE_SCALELSD}" == "1" ]]; then
  echo "7) Installing ScaleLSD (optional, C5 quality_first)..."
  cd "${TP_DIR}"
  SCALELSD_INSTALLED="$(_VER scalelsd)"
  if [[ -n "${SCALELSD_INSTALLED}" ]]; then
    echo "  [SKIP] scalelsd already installed: ${SCALELSD_INSTALLED}"
  else
    if [[ ! -d "scalelsd" ]]; then
      git clone --depth 1 https://github.com/ant-research/scalelsd.git
    fi
    cd "${TP_DIR}/scalelsd"

    if ! install_filtered_requirements "requirements.txt"; then
      echo "  [WARN] ScaleLSD requirements install failed. Continue anyway."
    fi
    if ! pip_install_noc -v -e . --no-build-isolation; then
      echo "  [WARN] ScaleLSD package install failed. C5 quality_first will fallback to Hough."
    fi
  fi
  cd "${WORK_DIR}"
else
  echo "7) Skipping ScaleLSD install (ENABLE_SCALELSD=${ENABLE_SCALELSD})"
fi

# -----------------------------
# 8) Gazelle / Gaze-LLE (optional, C6 quality_first)
# -----------------------------
if [[ "${ENABLE_GAZELLE}" == "1" ]]; then
  echo "8) Installing Gazelle (Gaze-LLE, optional, C6 quality_first)..."
  cd "${TP_DIR}"
  GAZELLE_INSTALLED="$(_VER gazelle)"
  if [[ -n "${GAZELLE_INSTALLED}" ]]; then
    echo "  [SKIP] gazelle already installed: ${GAZELLE_INSTALLED}"
  else
    if [[ ! -d "gazelle" ]]; then
      git clone --depth 1 https://github.com/fkryan/gazelle.git
    fi
    cd "${TP_DIR}/gazelle"
    if ! pip_install_noc -v -e . --no-build-isolation; then
      echo "  [WARN] Gazelle package install failed. C6 quality_first will fallback to proxy."
    fi
  fi
  GAZELLE_MODEL_NAME="${GAZELLE_MODEL_NAME:-gazelle_dinov2_vitb14_inout}"
  GAZELLE_CKPT_PATH="${GAZELLE_CKPT_PATH:-${WORK_DIR}/weights/gazelle/${GAZELLE_MODEL_NAME}.pt}"
  GAZELLE_AUTO_DOWNLOAD_CKPT="${GAZELLE_AUTO_DOWNLOAD_CKPT:-1}"
  if [[ "${GAZELLE_AUTO_DOWNLOAD_CKPT}" == "1" && ! -f "${GAZELLE_CKPT_PATH}" ]]; then
    mkdir -p "$(dirname "${GAZELLE_CKPT_PATH}")"
    GAZELLE_CKPT_URL=""
    case "${GAZELLE_MODEL_NAME}" in
      gazelle_dinov2_vitb14)
        GAZELLE_CKPT_URL="https://github.com/fkryan/gazelle/releases/download/v1.0.0/gazelle_dinov2_vitb14_hub.pt"
        ;;
      gazelle_dinov2_vitl14)
        GAZELLE_CKPT_URL="https://github.com/fkryan/gazelle/releases/download/v1.0.0/gazelle_dinov2_vitl14.pt"
        ;;
      gazelle_dinov2_vitb14_inout)
        GAZELLE_CKPT_URL="https://github.com/fkryan/gazelle/releases/download/v1.0.0/gazelle_dinov2_vitb14_inout.pt"
        ;;
      gazelle_dinov2_vitl14_inout)
        GAZELLE_CKPT_URL="https://github.com/fkryan/gazelle/releases/download/v1.0.0/gazelle_dinov2_vitl14_inout.pt"
        ;;
      *)
        echo "  [WARN] Unknown GAZELLE_MODEL_NAME=${GAZELLE_MODEL_NAME}; skip ckpt auto-download."
        ;;
    esac
    if [[ -n "${GAZELLE_CKPT_URL}" ]]; then
      echo "  - Downloading Gazelle ckpt -> ${GAZELLE_CKPT_PATH}"
      if ! $PYTHON - <<PY
import os, urllib.request
url = "${GAZELLE_CKPT_URL}"
dst = "${GAZELLE_CKPT_PATH}"
os.makedirs(os.path.dirname(dst), exist_ok=True)
urllib.request.urlretrieve(url, dst)
print("downloaded:", dst)
PY
      then
        echo "  [WARN] Gazelle ckpt auto-download failed. You can upload it manually: ${GAZELLE_CKPT_PATH}"
      fi
    fi
  fi
  cd "${WORK_DIR}"
else
  echo "8) Skipping Gazelle install (ENABLE_GAZELLE=${ENABLE_GAZELLE})"
fi

QUALITY_PYTHONPATH="${TP_DIR}/scalelsd:${TP_DIR}/gazelle:${PYTHONPATH:-}"
QUALITY_RUNTIME_PYTHONPATH="${WORK_DIR}/src/extract_features:${QUALITY_PYTHONPATH}"

# -----------------------------
# Verify
# -----------------------------
echo "=============================================="
echo "Verifying imports..."
PYTHONPATH="${QUALITY_PYTHONPATH}" $PYTHON - <<'PY'
import numpy as np
import torch
print("numpy:", np.__version__)
print("torch:", torch.__version__, "cuda:", torch.version.cuda)

import ray
import ultralytics
import open_clip
import pandas, pyarrow
import cv2
print("Base libs OK")

import mmengine
import mmcv
print("MMEngine/MMCV OK:", mmengine.__version__, getattr(mmcv, "__version__", "unknown"))

import mmdet
import mmpose
print("MMDet/MMPose OK:", mmdet.__version__, mmpose.__version__)

try:
    from scalelsd.ssl.misc.train_utils import load_scalelsd_model  # noqa: F401
    from scalelsd.ssl.models.detector import ScaleLSD  # noqa: F401
    print("ScaleLSD import(path used in C5): OK")
except Exception as e:
    print(f"ScaleLSD import(path used in C5): WARN ({e})")

try:
    import pythonjsonlogger  # noqa: F401
    print("python-json-logger import(path required by ScaleLSD): OK")
except Exception as e:
    print(f"python-json-logger import(path required by ScaleLSD): WARN ({e})")

try:
    from gazelle.model import get_gazelle_model  # noqa: F401
    print("Gazelle import(path used in C6): OK")
except Exception as e:
    print(f"Gazelle import(path used in C6): WARN ({e})")

try:
    import paddle  # noqa: F401
    from paddleocr import TextDetection  # noqa: F401
    print("PaddleOCR import(path used in C4): OK")
except Exception as e:
    print(f"PaddleOCR import(path used in C4): WARN ({e})")

print("All Good")
PY

if [[ "${STRICT_QUALITY_IMPORTS}" == "1" ]]; then
  echo "Verifying quality_first paths in strict mode..."
  if ! PYTHONPATH="${WORK_DIR}/src/extract_features:${QUALITY_PYTHONPATH}" $PYTHON - <<'PY'
from c3_pose import _patch_transformers_generation_compat
_patch_transformers_generation_compat()
import mmdet.models  # noqa: F401
print("strict check: C3 transformers/mmpretrain compatibility OK")
PY
  then
    echo "ERROR: strict C3 compatibility check failed."
    echo "  - Symptom usually appears as: TypeError: NoneType takes no arguments"
    echo "  - Path: mmpretrain.models.multimodal.blip.language_model (PreTrainedModel import chain)"
    echo "  - Verify current python/pip target and reinstall in same interpreter."
    exit 1
  fi
  if [[ "${ENABLE_SCALELSD}" == "1" ]]; then
    if ! PYTHONPATH="${QUALITY_PYTHONPATH}" $PYTHON - <<'PY'
from scalelsd.ssl.misc.train_utils import load_scalelsd_model  # noqa: F401
from scalelsd.ssl.models.detector import ScaleLSD  # noqa: F401
import pythonjsonlogger  # noqa: F401
print("strict check: C5 ScaleLSD path OK")
PY
    then
      echo "ERROR: strict C5 ScaleLSD import check failed."
      exit 1
    fi
    if ! PYTHONPATH="${QUALITY_RUNTIME_PYTHONPATH}" $PYTHON - <<'PY'
from PIL import Image
import numpy as np
from c5_geom import GeoFeatureExtractor

img = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
ext = GeoFeatureExtractor(priority="quality_first", c5_backend="auto")
out = ext.process_image(img)
print("strict runtime: C5 backend_runtime=", out.get("backend_runtime"), "method=", (out.get("horizon_roll") or {}).get("method"))
if out.get("backend_runtime") != "scalelsd":
    raise SystemExit(1)
PY
    then
      echo "ERROR: strict C5 runtime check failed (ScaleLSD not active)."
      exit 1
    fi
  fi
  if [[ "${ENABLE_GAZELLE}" == "1" ]]; then
    if ! PYTHONPATH="${QUALITY_PYTHONPATH}" $PYTHON - <<'PY'
from gazelle.model import get_gazelle_model  # noqa: F401
print("strict check: C6 Gazelle path OK")
PY
    then
      echo "ERROR: strict C6 Gazelle import check failed."
      exit 1
    fi
    if ! PYTHONPATH="${QUALITY_RUNTIME_PYTHONPATH}" \
      C6_GAZELLE_REPO="${TP_DIR}/gazelle" \
      C6_GAZELLE_CKPT="${GAZELLE_CKPT_PATH:-}" \
      C6_TORCH_HUB_DIR="${WORK_DIR}/.cache/torch/hub" \
      $PYTHON - <<'PY'
from PIL import Image
import numpy as np
from c6_gaze import GazeFeatureExtractor

img = Image.fromarray(np.zeros((448, 448, 3), dtype=np.uint8))
ext = GazeFeatureExtractor(priority="quality_first", c6_backend="auto")
pose = [{"bbox": [120, 120, 280, 360], "headpose_gaze": {"yaw_proxy": 0.1, "pitch_proxy": 0.0, "roll_deg": 0.0, "gaze_dir": "right", "conf": 0.6, "source": "pose_kp_proxy"}}]
out = ext.process_image(img, pose_items=pose, tags=["person"])
print("strict runtime: C6 backend_runtime=", ext.backend_runtime, "method=", out.get("method"))
if ext.backend_runtime != "gazelle" or out.get("method") != "gazelle_gaze_lle":
    raise SystemExit(1)
PY
    then
      echo "ERROR: strict C6 runtime check failed (Gazelle quality_first not active)."
      echo "HINT(offline server): upload these assets from a working machine:"
      echo "  1) ${TP_DIR}/gazelle  (repo source)"
      echo "  2) ${WORK_DIR}/.cache/torch/hub/facebookresearch_dinov2_main"
      echo "  3) ${WORK_DIR}/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth"
      echo "  4) ${GAZELLE_CKPT_PATH:-${WORK_DIR}/weights/gazelle/gazelle_dinov2_vitb14_inout.pt}"
      exit 1
    fi
  fi
  if [[ "${ENABLE_C4_OCR}" == "1" ]]; then
    if ! $PYTHON - <<'PY'
import paddle
from paddleocr import TextDetection  # noqa: F401
print("strict check: C4 paddle/paddleocr OK", paddle.__version__)
PY
    then
      echo "ERROR: strict C4 paddle/paddleocr import check failed."
      exit 1
    fi
  fi
fi
echo "=============================================="
echo "Installation complete!"
echo "NOTE:"
echo " - Constraints active: ${CONSTRAINTS_FILE} (numpy<2.0.0 enforced)"
echo " - mmdet/mmpose installed NON-editable (pip 26 PEP660 issue avoided)"
echo " - SAM2 CUDA build default OFF (SAM2_BUILD_CUDA=${SAM2_BUILD_CUDA})"
echo " - Qwen3 enforcement: ENABLE_QWEN3_VL=${ENABLE_QWEN3_VL}, QWEN3_TRY_PYPI_LATEST=${QWEN3_TRY_PYPI_LATEST}, QWEN3_ALLOW_GITHUB_FALLBACK=${QWEN3_ALLOW_GITHUB_FALLBACK}"
echo " - Qwen3 runtime-only mode: QWEN3_RUNTIME_ONLY=${QWEN3_RUNTIME_ONLY}"
echo " - C4 OCR install: ENABLE_C4_OCR=${ENABLE_C4_OCR}, REQUIRE_PADDLE_CUDA_FOR_C4=${REQUIRE_PADDLE_CUDA_FOR_C4}"
echo " - strict quality import checks: STRICT_QUALITY_IMPORTS=${STRICT_QUALITY_IMPORTS}"
echo "=============================================="
