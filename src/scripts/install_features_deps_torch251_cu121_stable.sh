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
#   - PaddleOCR: C4 OCR 비필수이므로 기본 비활성화 (주석 해제로 설치 가능)
#
# ==============================================================================
# USAGE
#   # [권장] 새 venv 생성 후 실행:
#   python3 -m venv /home/jyju25/Venvs/py310_feat_test
#   source /home/jyju25/Venvs/py310_feat_test/bin/activate
#   bash src/scripts/install_features_deps_torch251_cu121_stable.sh
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
PADDLE_CUDA_INDEX="https://www.paddlepaddle.org.cn/packages/stable/cu126/"  # CUDA 12.6 runtime wheel

# Constraints (global)
NUMPY_CONSTRAINT="numpy<2.0.0"
TRANSFORMERS_CONSTRAINT="transformers<4.45.0"

# -----------------------------
PYTHON="${PYTHON:-python}"
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
# pip install args: venv 없어도 --user 미적용 (permission error 발생 시는 root로 실행하거나 venv를 생성하세요)
PIP_INSTALL_ARGS=()
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "WARNING: VIRTUAL_ENV not set. Proceeding without --user."
  echo "  If permission errors occur, create a venv first:"
  echo "    python3 -m venv ~/venvs/py310_feat && source ~/venvs/py310_feat/bin/activate"
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

# -----------------------------
# 0) Tooling (do NOT use openmim)
# -----------------------------
echo "0) Upgrading pip/setuptools/wheel..."
pip_install -U pip setuptools wheel

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

# -----------------------------
# 2) Base ML libs
# -----------------------------
echo "2) Installing base ML libraries..."
_VER() { $PYTHON -m pip show "$1" 2>/dev/null | awk '/^Version:/{print $2}'; }

pip_install -U "${NUMPY_CONSTRAINT}"

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

# sentence-transformers, timm, webdataset, pyarrow, pandas, opencv
pip_install sentence-transformers timm webdataset pyarrow pandas
pip_install "opencv-python-headless>=4.6.0"
pip_install -U "${NUMPY_CONSTRAINT}"  # re-enforce after opencv

# -----------------------------
# 3) OpenMMLab core (NO openmim)
# -----------------------------
echo "3) Installing OpenMMLab core via pip (mmengine + mmcv CUDA wheel + mmpretrain)..."
# mmcv-lite/mmcv-full/mmengine 이전 설치 모두 클린업
pip_uninstall mmcv mmcv-full mmcv-lite mmengine mmpretrain mmdet mmpose || true

# [CRITICAL] mmengine 설치 전 setuptools 복구:
# openxlab==0.1.3이 setuptools~=60.2.0으로 다운그레이드한 경우
# pkg_resources 미등록 에러가 발생할 수 있으므로 먼저 복구
echo "  [Pre-step] Restoring setuptools >= 68 for pkg_resources availability..."
$PIP install -U "setuptools>=68" wheel 2>/dev/null || true

pip_install "mmengine==${MMENGINE_VER}"
pip_install "mmpretrain>=1.0.0"

# mmcv CUDA 빌드 휠 설치 ————————————————————————————————————————
# mmpose inference 에 필요한 _ext CUDA ops 포함 버전
MMCV_INSTALLED="$($PYTHON -m pip show mmcv 2>/dev/null | awk '/^Version:/{print $2}' || true)"
if [[ "${MMCV_INSTALLED}" == "${MMCV_VER}" ]]; then
  echo "  [SKIP] mmcv==${MMCV_VER} already installed."
else
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
# 4) PaddleOCR
# -----------------------------
#echo "4) Installing PaddlePaddle-GPU + PaddleOCR..."
#PADDLE_INSTALLED="$($PYTHON -m pip show paddlepaddle-gpu 2>/dev/null | awk '/^Version:/{print $2}' || true)"
#if [[ "${PADDLE_INSTALLED}" == "${PADDLE_GPU_VER}" ]]; then
#  echo "  [SKIP] paddlepaddle-gpu already installed: ${PADDLE_INSTALLED}"
#else
#  pip_install --no-cache-dir -i "${PADDLE_CUDA_INDEX}" "paddlepaddle-gpu==${PADDLE_GPU_VER}"
#fi
#pip_install "paddleocr>=2.0.1" imgaug "PyMuPDF<1.21.0"

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
# Verify
# -----------------------------
echo "=============================================="
echo "Verifying imports..."
$PYTHON - <<'PY'
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

#from paddleocr import PaddleOCR
#print("PaddleOCR OK")

print("All Good")
PY
echo "=============================================="
echo "Installation complete!"
echo "NOTE:"
echo " - Constraints active: ${CONSTRAINTS_FILE} (numpy<2.0.0 enforced)"
echo " - mmdet/mmpose installed NON-editable (pip 26 PEP660 issue avoided)"
echo " - SAM2 CUDA build default OFF (SAM2_BUILD_CUDA=${SAM2_BUILD_CUDA})"
echo "=============================================="