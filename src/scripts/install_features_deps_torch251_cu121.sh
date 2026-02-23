#!/usr/bin/env bash
# Stable torch/cuda plan installer for SSTK Feature Extraction deps
# - Torch: 2.5.1 + cu121 (stable wheel set)
# - OpenMMLab: mmcv-lite==2.1.0 (no CUDA ops, but avoids mmcv/mmdet version deadlocks)
# - mmdet/mmpose: source editable install (pinned tags) + filtered runtime req install
#
# ==============================================================================
# USAGE
# ==============================================================================
#
# [권장] 새 가상환경 만들어서 설치 및 테스트:
#   python3 -m venv /home/jyju25/Venvs/py310_feat_test
#   source /home/jyju25/Venvs/py310_feat_test/bin/activate
#   bash src/scripts/install_features_deps_torch251_cu121.sh
#
# [기존 venv 활용]:
#   source /home/jyju25/Venvs/py310_gcf/bin/activate
#   bash src/scripts/install_features_deps_torch251_cu121.sh
#
# PyTorch/CUDA 버전 확인:
#   python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda)"
#
# ------------------------------------------------------------------------------
# [서버 환경] mmdetection / mmpose 빌드 에러 시 --no-build-isolation 방식으로 직접 설치:
# (mim install 또는 setup.py 빌드에서 pkg_resources / mmcv pin 충돌이 발생하는 경우 사용)
#
#   PROJ_ROOT="/group-volume/jaden.ju/Sources/ImageCropping"
#
#   cd "${PROJ_ROOT}/third_party/mmdetection"
#   python -m pip install -v . --no-deps --no-build-isolation
#
#   cd "${PROJ_ROOT}/third_party/mmpose"
#   python -m pip install -v . --no-deps --no-build-isolation
#
# ==============================================================================

set -euo pipefail

# -----------------------------
# Versions (pin for stability)
# -----------------------------
TORCH_VER="2.5.1"
TORCHVISION_VER="0.20.1"
TORCHAUDIO_VER="2.5.1"
TORCH_CUDA="cu121"

MMENGINE_VER=""          # empty = let mim pick compatible
MMCV_LITE_VER="2.1.0"

MMDET_TAG="v3.3.0"
MMPOSE_TAG="v1.3.2"

PADDLE_GPU_VER="3.3.0"
PADDLE_CUDA_INDEX="https://www.paddlepaddle.org.cn/packages/stable/cu126/"  # CUDA 12.6 runtime in wheel

# numpy pin: pycocotools/xtcocotools 바이너리 호환성을 위해 필요하지만,
# opencv-python-headless 4.11+ 는 numpy>=2 를 요구하므로 전역 강제 핀은 제거.
# -> 필요 시 아래 변수를 사용해 pycocotools 빌드 직전에만 한시적으로 적용.
NUMPY_COMPAT_PKG="numpy<2.0.0"  # 이 변수는 mmdet/mmpose source build 시에만 참조

# -----------------------------
PYTHON="${PYTHON:-python}"
PIP="$PYTHON -m pip"

echo "=============================================="
echo "SSTK Feature Extraction Dependencies Installer"
echo "STABLE TORCH/CUDA PLAN: torch=${TORCH_VER}+${TORCH_CUDA}"
echo "=============================================="

# -----------------------------
# Sanity checks
# -----------------------------
$PYTHON - <<'PY'
import sys
major, minor = sys.version_info[:2]
print("Python:", sys.version.replace("\n"," "))
if (major, minor) < (3, 10):
    raise SystemExit("ERROR: Python>=3.10 required (SAM2 requires >=3.10).")
PY

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "WARNING: VIRTUAL_ENV not set. Running inside a venv is strongly recommended."
fi

# -----------------------------
# Helper: 이미 명시한 버전이 설치되어 있으면 스킵
# Usage: pip_install_if_needed "package==X.Y.Z" [extra_pip_args...]
# -----------------------------
pip_install_if_needed () {
  local spec="$1"
  shift || true
  local pkg="${spec%%==*}"
  local ver="${spec#*==}"
  if [[ "$spec" == *"=="* ]]; then
    local installed
    installed=$($PYTHON -m pip show "$pkg" 2>/dev/null | grep "^Version:" | awk '{print $2}' || echo "")
    if [[ "$installed" == "$ver" ]]; then
      echo "  [SKIP] ${pkg}==${ver} already installed."
      return 0
    fi
  fi
  $PIP install "$spec" "$@"
}

# -----------------------------
# 0) Tooling
# -----------------------------
$PIP install -U pip setuptools wheel

# -----------------------------
# 1) Torch/CUDA (버전 일치 시 스킵)
# -----------------------------
echo "1) Installing PyTorch ${TORCH_VER} (${TORCH_CUDA}) + matching torchvision/torchaudio..."

TORCH_INSTALLED=$($PYTHON -m pip show torch 2>/dev/null | grep "^Version:" | awk '{print $2}' || echo "")
if [[ "$TORCH_INSTALLED" == "${TORCH_VER}+${TORCH_CUDA}" || "$TORCH_INSTALLED" == "${TORCH_VER}" ]]; then
  echo "  [SKIP] torch==${TORCH_VER}+${TORCH_CUDA} already installed."
else
  $PIP install --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" \
    "torch==${TORCH_VER}" "torchvision==${TORCHVISION_VER}" "torchaudio==${TORCHAUDIO_VER}"
fi

$PYTHON -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda)"

# -----------------------------
# 2) Base ML libs
# -----------------------------
echo "2) Installing base ML libraries..."
$PIP install ray[default] "ultralytics>=8.0.0" sentence-transformers timm webdataset
$PIP install open_clip_torch pyarrow pandas
$PIP install "opencv-python-headless>=4.6.0"
# NOTE: numpy<2 전역 핀 제거 — opencv 4.11+ 가 numpy>=2 를 요구하므로 충돌 발생.
# pycocotools(mmdet runtime dep) 는 source build 시 자체적으로 호환 numpy로 빌드됨.

# -----------------------------
# 3) OpenMMLab (mmcv-lite plan)
# -----------------------------
echo "3) Installing OpenMMLab core (MMEngine + MMCV-Lite)..."
# --no-deps: openmim의 의존성 체인(opendatalab→openxlab)이
# filelock~=3.14 / setuptools~=60.2 를 강제 다운그레이드하는 것을 방지
$PIP install -U openmim --no-deps
# openmim 런타임에 필요한 최소 deps만 명시적으로 설치
$PIP install -U click rich requests tabulate pyyaml

# MMEngine
MMENGINE_INSTALLED=$($PYTHON -m pip show mmengine 2>/dev/null | grep "^Version:" | awk '{print $2}' || echo "")
if [[ -z "${MMENGINE_VER}" ]]; then
  if [[ -z "${MMENGINE_INSTALLED}" ]]; then
    mim install mmengine
  else
    echo "  [SKIP] mmengine==${MMENGINE_INSTALLED} already installed."
  fi
else
  if [[ "${MMENGINE_INSTALLED}" == "${MMENGINE_VER}" ]]; then
    echo "  [SKIP] mmengine==${MMENGINE_VER} already installed."
  else
    mim install "mmengine==${MMENGINE_VER}"
  fi
fi

# MMCV-Lite (no CUDA ops) — avoids mmcv/mmdet version incompatibilities and source builds
MMCV_INSTALLED=$($PYTHON -m pip show mmcv-lite 2>/dev/null | grep "^Version:" | awk '{print $2}' || echo "")
if [[ "${MMCV_INSTALLED}" == "${MMCV_LITE_VER}" ]]; then
  echo "  [SKIP] mmcv-lite==${MMCV_LITE_VER} already installed."
else
  mim install "mmcv-lite==${MMCV_LITE_VER}"
fi

# Helper: install requirements but drop mmcv/mmengine pins to avoid pulling full mmcv
install_filtered_requirements () {
  local req_file="$1"
  if [[ -f "${req_file}" ]]; then
    echo "  - Installing filtered requirements from ${req_file}"
    tmp_req="$(mktemp)"
    # Exclude:
    #   mmcv/mmengine - avoid pulling full mmcv and overriding our lite version
    #   chumpy        - ancient package that does `import pip` inside setup.py,
    #                   broken on pip>=21. Not needed for pose inference at runtime.
    grep -vE '^\s*(mmcv|mmcv-lite|mmengine|chumpy)\b' "${req_file}" | sed '/^\s*#/d;/^\s*$/d' > "${tmp_req}"
    if [[ -s "${tmp_req}" ]]; then
      $PIP install -r "${tmp_req}"
    fi
    rm -f "${tmp_req}"
  fi
}

WORK_DIR="$(pwd)"
TP_DIR="${WORK_DIR}/third_party"
mkdir -p "${TP_DIR}"
cd "${TP_DIR}"

# -----------------------------
# 3a) MMDetection (source)
# 빌드 에러 시 USAGE 주석의 --no-build-isolation 방식으로 대체 설치 가능
# -----------------------------
echo "3a) Installing MMDetection (${MMDET_TAG}) from source..."
MMDET_INSTALLED=$($PYTHON -m pip show mmdet 2>/dev/null | grep "^Version:" | awk '{print $2}' || echo "")
if [[ -n "${MMDET_INSTALLED}" ]]; then
  echo "  [SKIP] mmdet==${MMDET_INSTALLED} already installed."
else
  if [[ ! -d "mmdetection" ]]; then
    git clone --depth 1 --branch "${MMDET_TAG}" https://github.com/open-mmlab/mmdetection.git
  fi
  cd mmdetection
  install_filtered_requirements "requirements/runtime.txt"
  # editable(-e) 모드는 setup.py 기반 프로젝트에서 PEP 660 훅 미지원 에러 발생.
  # -> 비-editable 방식(일반 설치)으로 변경
  $PIP install -v . --no-deps --no-build-isolation
  cd "${TP_DIR}"
fi

# -----------------------------
# 3b) MMPose (source)
# 빌드 에러 시 USAGE 주석의 --no-build-isolation 방식으로 대체 설치 가능
# -----------------------------
echo "3b) Installing MMPose (${MMPOSE_TAG}) from source..."
MMPOSE_INSTALLED=$($PYTHON -m pip show mmpose 2>/dev/null | grep "^Version:" | awk '{print $2}' || echo "")
if [[ -n "${MMPOSE_INSTALLED}" ]]; then
  echo "  [SKIP] mmpose==${MMPOSE_INSTALLED} already installed."
else
  if [[ ! -d "mmpose" ]]; then
    git clone --depth 1 --branch "${MMPOSE_TAG}" https://github.com/open-mmlab/mmpose.git
  fi
  cd mmpose
  install_filtered_requirements "requirements/runtime.txt"
  # editable(-e) 모드는 setup.py 기반 프로젝트에서 PEP 660 훅 미지원 에러 발생.
  # -> 비-editable 방식(일반 설치)으로 변경
  $PIP install -v . --no-deps --no-build-isolation
  cd "${TP_DIR}"
fi

cd "${WORK_DIR}"

# -----------------------------
# 4) PaddleOCR
# -----------------------------
#echo "4) Installing PaddlePaddle-GPU + PaddleOCR..."
#PADDLE_INSTALLED=$($PYTHON -m pip show paddlepaddle-gpu 2>/dev/null | grep "^Version:" | awk '{print $2}' || echo "")
#if [[ "${PADDLE_INSTALLED}" == "${PADDLE_GPU_VER}" ]]; then
#  echo "  [SKIP] paddlepaddle-gpu==${PADDLE_GPU_VER} already installed."
#else
#  $PIP install --no-cache-dir "paddlepaddle-gpu==${PADDLE_GPU_VER}" -i "${PADDLE_CUDA_INDEX}"
#fi
#$PIP install "paddleocr>=2.0.1" imgaug "PyMuPDF<1.21.0"

# -----------------------------
# 5) EfficientViT
# -----------------------------
echo "5) Installing EfficientViT..."
mkdir -p "${TP_DIR}"
cd "${TP_DIR}"

if [[ ! -d "efficientvit" ]]; then
  git clone --depth 1 https://github.com/mit-han-lab/efficientvit.git
  cd efficientvit
  $PIP install "opt-einsum>=3.3.0"
  $PIP install -e .
  cd ..
else
  echo "  [SKIP] efficientvit already exists in third_party."
fi

# -----------------------------
# 6) SAM2
# -----------------------------
echo "6) Installing SAM2..."
if [[ ! -d "sam2" ]]; then
  git clone --depth 1 https://github.com/facebookresearch/sam2.git
  cd sam2
  $PIP install ninja
  $PIP install -e .
  cd ..
else
  echo "  [SKIP] sam2 already exists in third_party."
fi

cd "${WORK_DIR}"

# -----------------------------
# Verify
# -----------------------------
echo "=============================================="
echo "Verifying imports..."
$PYTHON - <<'PY'
import torch
print("Torch OK:", torch.__version__, "CUDA:", torch.version.cuda)

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

from paddleocr import PaddleOCR
print("PaddleOCR OK")

print("All Good")
PY
echo "=============================================="
echo "Installation complete!"
echo "NOTE: mmcv-lite has no CUDA ops. If a model complains about missing ops (e.g., nms_cuda/DCN),"
echo "      you must switch to full mmcv + compatible torch/cuXXX combo."
echo "=============================================="