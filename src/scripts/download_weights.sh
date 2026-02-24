#!/bin/bash
# ==============================================================================
# download_weights.sh  —  C2 ~ C3 모델 가중치 자동 다운로드
# ==============================================================================
# USAGE
#   bash src/scripts/download_weights.sh [--weights_dir /path/to/weights]
#
# 기본 weights_dir: PROJECT_ROOT/weights/
#
# 다운로드 목록:
#   C2 (SAM 2.1)         sam2.1_hiera_large.pt  (~900 MB)
#   C3 Detector          rtmdet_nano_person.pth  (~4.8 MB)
#   C3 Pose Estimator    vitpose-b-coco.pth      (~330 MB)
#
# NOTE: C3 config 파일은 third_party/ 소스에서 자동 참조합니다.
#       설치 스크립트를 먼저 실행해야 third_party/mmdetection, third_party/mmpose 가 존재합니다:
#         bash src/scripts/install_features_deps_torch251_cu121_stable.sh
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WEIGHTS_DIR="${PROJECT_ROOT}/weights"

# 옵션 파싱
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --weights_dir) WEIGHTS_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "${WEIGHTS_DIR}"
echo "======================================================="
echo "  Weight Download Script"
echo "  weights_dir: ${WEIGHTS_DIR}"
echo "======================================================="

# HF token 설정 (HuggingFace fallback 사용 시 필요)
export HUGGINGFACEHUB_API_TOKEN="${HUGGINGFACEHUB_API_TOKEN:-}"
export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACEHUB_API_TOKEN}}"

# ── 다운로드 헬퍼 ─────────────────────────────────────────────────────────────
download_file() {
    local dest="$1"
    shift
    local urls=("$@")

    if [[ -f "${dest}" ]]; then
        local sz_mb
        sz_mb=$(du -m "${dest}" | awk '{print $1}')
        echo "  [SKIP] $(basename "${dest}") already exists (${sz_mb} MB)"
        return 0
    fi

    echo "  Downloading $(basename "${dest}") ..."
    local tmp="${dest}.tmp"
    for url in "${urls[@]}"; do
        echo "    URL: ${url}"
        if curl -fsSL --retry 3 --retry-delay 5 \
               -H "Authorization: Bearer ${HF_TOKEN}" \
               -o "${tmp}" "${url}"; then
            mv "${tmp}" "${dest}"
            local sz_mb
            sz_mb=$(du -m "${dest}" | awk '{print $1}')
            echo "  ✓ Saved: ${dest} (${sz_mb} MB)"
            return 0
        else
            echo "  ✗ Failed: ${url}"
            rm -f "${tmp}"
        fi
    done

    echo "  [ERROR] All URLs failed for $(basename "${dest}")."
    echo "  Please download manually and place at: ${dest}"
    return 1
}

ERRORS=0

# ── C2: SAM 2.1 Large ─────────────────────────────────────────────────────────
echo ""
echo "[C2] SAM 2.1 Large (~900 MB)"
download_file \
    "${WEIGHTS_DIR}/sam2.1_hiera_large.pt" \
    "https://huggingface.co/facebook/sam2.1-hiera-large/resolve/main/sam2.1_hiera_large.pt" \
    || ERRORS=$((ERRORS+1))

# ── C3: RTMDet-nano (person detector) ────────────────────────────────────────
echo ""
echo "[C3] RTMDet-nano person detector (~4.8 MB)"
download_file \
    "${WEIGHTS_DIR}/rtmdet_nano_person.pth" \
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth" \
    || ERRORS=$((ERRORS+1))

# ── C3: ViTPose-Base (pose estimator) ────────────────────────────────────────
echo ""
echo "[C3] ViTPose-Base pose estimator (~330 MB)"
download_file \
    "${WEIGHTS_DIR}/vitpose-b-coco.pth" \
    "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192-216eae50_20230314.pth" \
    "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/torch/COCO/vitpose-b-coco.pth" \
    || ERRORS=$((ERRORS+1))

# ── C3 config 경로 확인 ───────────────────────────────────────────────────────
echo ""
echo "[C3] Verifying config files from third_party/..."
DET_CFG="${PROJECT_ROOT}/third_party/mmpose/demo/mmdetection_cfg/rtmdet_nano_320-8xb32_coco-person.py"
POSE_CFG="${PROJECT_ROOT}/third_party/mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192.py"

for cfg_path in "${DET_CFG}" "${POSE_CFG}"; do
    if [[ -f "${cfg_path}" ]]; then
        echo "  ✓ ${cfg_path}"
    else
        echo "  ✗ Config not found: ${cfg_path}"
        echo "    → Run install script first:"
        echo "      bash src/scripts/install_features_deps_torch251_cu121_stable.sh"
        ERRORS=$((ERRORS+1))
    fi
done

# ── 요약 ─────────────────────────────────────────────────────────────────────
echo ""
echo "======================================================="
echo "  Download Summary"
echo "======================================================="
ls -lh "${WEIGHTS_DIR}/" 2>/dev/null || true
echo ""

if [[ "${ERRORS}" -eq 0 ]]; then
    echo "  ✓ All weights ready."
else
    echo "  ✗ ${ERRORS} item(s) failed. Fix errors above before running the pipeline."
    exit 1
fi
