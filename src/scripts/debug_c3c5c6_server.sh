#!/usr/bin/env bash
# ============================================================================
# debug_c3c5c6_server.sh
# Server-side healthcheck/debug runner for precompute modules.
#
# What it does:
#  1) Environment/import/runtime smoke checks (C1/C2/C3/C5/C6)
#  2) Optional analysis of existing shard logs/jsonl outputs
#  3) Optional fresh smoke extraction run (default: C1+C2+C3+C5+C6)
#  4) PASS/FAIL summary with explicit reasons
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${SRC_DIR}/.." && pwd)"

INPUT_PARQUET=""
BUCKET="sstk_100"
IMAGE_DIR=""
TAR_DIR="/sstk/20230916/sstk_100"
SERVER_MODE=1
VENV_PATH="/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/activate"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_ID="0"
NUM_SAMPLES=100
RUN_SMOKE=1
SMOKE_COMPONENTS="c1 c2 c3 c5 c6"
SHARDS_DIR=""
OUT_DIR=""
OFFLINE_ROOT=""
BATCH_SIZE=8

usage() {
  cat <<'EOF'
Usage:
  bash src/scripts/debug_c3c5c6_server.sh \
    --input_parquet data/SSTK/Test_100/filtered_sstk_100.parquet \
    --bucket sstk_100 \
    --image_dir data/SSTK/Test_100/images \
    --tar_dir /sstk/20230916/sstk_100 \
    --server_mode 1 \
    --gpu_id 0 \
    --num_samples 5 \
    --components "c1 c2 c3 c5 c6" \
    --shards_dir data/SSTK/Test_100/artifacts/precompute/feats_c2c3c5_v2_strict_raw.jsonl.shards.32983

Examples:
  # 1) 서버(온라인): 기존 shard 분석 + 신규 smoke 추출
  bash src/scripts/debug_c3c5c6_server.sh \
    --input_parquet data/SSTK/Test_100/filtered_sstk_100.parquet \
    --bucket sstk_100 \
    --image_dir data/SSTK/Test_100/images \
    --tar_dir /sstk/20230916/sstk_100 \
    --server_mode 1 \
    --gpu_id 1 \
    --num_samples 5 \
    --shards_dir data/SSTK/Test_100/artifacts/precompute/feats_c2c3c5_v2_strict_raw.jsonl.shards.32983

  # 2) 서버(오프라인): 오프라인 자산 루트 고정 후 smoke 추출
  bash src/scripts/debug_c3c5c6_server.sh \
    --input_parquet data/SSTK/Test_100/filtered_sstk_100.parquet \
    --bucket sstk_100 \
    --image_dir data/SSTK/Test_100/images \
    --tar_dir /sstk/20230916/sstk_100 \
    --server_mode 1 \
    --gpu_id 0 \
    --num_samples 5 \
    --offline_root /group-volume/jaden.ju/Sources/ImageCropping

  # 3) 로그/산출물 분석만 수행 (smoke 미실행)
  bash src/scripts/debug_c3c5c6_server.sh \
    --run_smoke 0 \
    --shards_dir data/SSTK/Test_100/artifacts/precompute/feats_c2c3c5_v2_strict_raw.jsonl.shards.32983

Options:
  --input_parquet PATH   Input parquet for smoke run
  --bucket NAME          Bucket name (default: sstk_100)
  --image_dir PATH       Optional local image dir
  --tar_dir PATH         TAR root path (default: /sstk/20230916/sstk_100)
  --server_mode 0|1      1=server(no venv), 0=local(venv source)
  --venv_path PATH       venv activate path (used when server_mode=0)
  --python_bin BIN       Python binary (default: python3)
  --gpu_id ID            GPU id for smoke run (default: 0)
  --num_samples N        Subset size for smoke run (default: 100)
  --batch_size N         Batch size for smoke run (default: 8)
  --run_smoke 0|1        Run fresh extraction smoke test (default: 1)
  --components "..."     Components for smoke run (default: "c1 c2 c3 c5 c6")
  --shards_dir PATH      Existing shards dir to analyze
  --out_dir PATH         Output dir (default: artifacts/debug/c1c2c3c5c6_<timestamp>)
  --offline_root PATH    Root containing .cache/torch/hub and third_party/gazelle

Output files:
  env_check.txt
  env_check.json
  shard_log_issues.txt (when --shards_dir)
  shard_summary.json     (when --shards_dir)
  smoke_subset.parquet   (when --run_smoke=1)
  smoke_extract.log      (when --run_smoke=1)
  smoke_output.jsonl     (when --run_smoke=1)
  smoke_summary.json     (when --run_smoke=1)
  healthcheck_result.txt
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input_parquet) INPUT_PARQUET="$2"; shift 2 ;;
    --bucket) BUCKET="$2"; shift 2 ;;
    --image_dir) IMAGE_DIR="$2"; shift 2 ;;
    --tar_dir) TAR_DIR="$2"; shift 2 ;;
    --server_mode) SERVER_MODE="$2"; shift 2 ;;
    --venv_path) VENV_PATH="$2"; shift 2 ;;
    --python_bin) PYTHON_BIN="$2"; shift 2 ;;
    --gpu_id) GPU_ID="$2"; shift 2 ;;
    --num_samples) NUM_SAMPLES="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --run_smoke) RUN_SMOKE="$2"; shift 2 ;;
    --components) SMOKE_COMPONENTS="$2"; shift 2 ;;
    --shards_dir) SHARDS_DIR="$2"; shift 2 ;;
    --out_dir) OUT_DIR="$2"; shift 2 ;;
    --offline_root) OFFLINE_ROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "${OUT_DIR}" ]]; then
  OUT_DIR="${PROJECT_ROOT}/artifacts/debug/c1c2c3c5c6_$(date +%y%m%d_%H%M%S)"
fi
mkdir -p "${OUT_DIR}"
export OUT_DIR
export PROJECT_ROOT

SMOKE_COMPONENTS="${SMOKE_COMPONENTS//,/ }"
SMOKE_COMPONENTS="$(echo "${SMOKE_COMPONENTS}" | xargs)"
if [[ -z "${SMOKE_COMPONENTS}" ]]; then
  echo "[fatal] --components cannot be empty" >&2
  exit 2
fi
export SMOKE_COMPONENTS

if [[ "${SERVER_MODE}" != "1" ]]; then
  if [[ -f "${VENV_PATH}" ]]; then
    # shellcheck disable=SC1090
    source "${VENV_PATH}"
  fi
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[fatal] python not found: ${PYTHON_BIN}" >&2
  exit 2
fi

export PYTHONPATH="${SRC_DIR}/extract_features:${PROJECT_ROOT}/third_party/efficientvit:${PROJECT_ROOT}/third_party/sam2:${PROJECT_ROOT}/third_party/scalelsd:${PROJECT_ROOT}/third_party/gazelle:${PYTHONPATH:-}"

# Quality-first defaults
export C5_SCALELSD_CKPT="${C5_SCALELSD_CKPT:-${PROJECT_ROOT}/weights/scalelsd/scalelsd-vitbase-v1-train-sa1b.pt}"
export C6_GAZELLE_REPO="${C6_GAZELLE_REPO:-${PROJECT_ROOT}/third_party/gazelle}"
export C6_TORCH_HUB_DIR="${C6_TORCH_HUB_DIR:-${PROJECT_ROOT}/.cache/torch/hub}"
if [[ -z "${C6_GAZELLE_CKPT:-}" ]]; then
  if [[ -f "${PROJECT_ROOT}/weights/gazelle/gazelle_dinov2_vitb14_inout.pt" ]]; then
    export C6_GAZELLE_CKPT="${PROJECT_ROOT}/weights/gazelle/gazelle_dinov2_vitb14_inout.pt"
  else
    export C6_GAZELLE_CKPT="${C6_TORCH_HUB_DIR}/checkpoints/gazelle_dinov2_vitb14_inout.pt"
  fi
fi
export C6_GAZELLE_USE_TORCHHUB="${C6_GAZELLE_USE_TORCHHUB:-1}"

if [[ -n "${OFFLINE_ROOT}" ]]; then
  export C6_TORCH_HUB_DIR="${OFFLINE_ROOT}/.cache/torch/hub"
  export C6_GAZELLE_REPO="${OFFLINE_ROOT}/third_party/gazelle"
  export C6_GAZELLE_CKPT="${OFFLINE_ROOT}/.cache/torch/hub/checkpoints/gazelle_dinov2_vitb14_inout.pt"
fi

mkdir -p "${C6_TORCH_HUB_DIR}" || true

echo "[debug] out_dir=${OUT_DIR}"
echo "[debug] python_bin=${PYTHON_BIN}"
echo "[debug] C5_SCALELSD_CKPT=${C5_SCALELSD_CKPT}"
echo "[debug] C6_GAZELLE_REPO=${C6_GAZELLE_REPO}"
echo "[debug] C6_GAZELLE_CKPT=${C6_GAZELLE_CKPT}"
echo "[debug] C6_TORCH_HUB_DIR=${C6_TORCH_HUB_DIR}"
echo "[debug] smoke_components=${SMOKE_COMPONENTS}"

# ----------------------------------------------------------------------------
# 1) Environment checks + in-process runtime checks
# ----------------------------------------------------------------------------
"${PYTHON_BIN}" - <<'PY' > "${OUT_DIR}/env_check.txt" 2>&1
import json
import os
import sys

rep = {
    "python_executable": sys.executable,
    "python_version": sys.version.replace("\n", " "),
}

# versions / imports
mods = [
    "torch",
    "transformers",
    "mmcv",
    "mmdet",
    "mmpose",
    "mmpretrain",
    "pythonjsonlogger",
]
for m in mods:
    try:
        mod = __import__(m)
        rep[f"import_{m}"] = "ok"
        rep[f"version_{m}"] = getattr(mod, "__version__", "unknown")
    except Exception as e:
        rep[f"import_{m}"] = f"fail: {e}"

try:
    import torch
    rep["torch_cuda_available"] = bool(torch.cuda.is_available())
    rep["torch_cuda_device_count"] = int(torch.cuda.device_count())
except Exception:
    pass

# C3 compatibility probe
try:
    from c3_pose import _patch_transformers_generation_compat
    _patch_transformers_generation_compat()
    import mmdet.models  # noqa: F401
    rep["c3_compat_import"] = "ok"
except Exception as e:
    rep["c3_compat_import"] = f"fail: {e}"

# C1 runtime probe
try:
    from PIL import Image
    import numpy as np
    from c1_clip import ClipFeatureExtractor

    img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    ext = ClipFeatureExtractor(priority="quality_first")
    fi = ext.encode_images([img])
    ft = ext.encode_texts(["person"])
    rep["c1_probe_status"] = "ok"
    rep["c1_probe_img_dim"] = int(fi.shape[-1]) if getattr(fi, "size", 0) else 0
    rep["c1_probe_txt_dim"] = int(ft.shape[-1]) if getattr(ft, "size", 0) else 0
except Exception as e:
    rep["c1_probe_status"] = f"fail: {e}"

# C2 runtime probe
try:
    from PIL import Image
    import numpy as np
    from c2_seg import SegFeatureExtractor

    img = Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
    ext = SegFeatureExtractor(priority="quality_first", weights_dir=os.path.join(os.environ["PROJECT_ROOT"], "weights"))
    c2_out = ext.process_image(img, return_det=True)
    seg = []
    det = []
    if isinstance(c2_out, tuple) and len(c2_out) == 2:
        seg, det = c2_out
    elif isinstance(c2_out, list):
        seg = c2_out
    rep["c2_probe_status"] = "ok"
    rep["c2_probe_seg_count"] = len(seg) if isinstance(seg, list) else 0
    rep["c2_probe_det_count"] = len(det) if isinstance(det, list) else 0
except Exception as e:
    rep["c2_probe_status"] = f"fail: {e}"

# C5 runtime probe
try:
    from PIL import Image
    import numpy as np
    from c5_geom import GeoFeatureExtractor
    img = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
    ext = GeoFeatureExtractor(priority="quality_first", c5_backend="auto")
    out = ext.process_image(img)
    rep["c5_probe_backend_runtime"] = out.get("backend_runtime")
    rep["c5_probe_method"] = (out.get("horizon_roll") or {}).get("method")
except Exception as e:
    rep["c5_probe_backend_runtime"] = f"fail: {e}"

# C6 runtime probe
try:
    from PIL import Image
    import numpy as np
    from c6_gaze import GazeFeatureExtractor
    img = Image.fromarray(np.zeros((448, 448, 3), dtype=np.uint8))
    pose = [{
        "bbox": [120, 120, 280, 360],
        "headpose_gaze": {
            "yaw_proxy": 0.1,
            "pitch_proxy": 0.0,
            "roll_deg": 0.0,
            "gaze_dir": "right",
            "conf": 0.6,
            "source": "pose_kp_proxy",
        },
    }]
    ext = GazeFeatureExtractor(priority="quality_first", c6_backend="auto")
    out = ext.process_image(img, pose_items=pose, tags=["person"])
    rep["c6_probe_backend_runtime"] = ext.backend_runtime
    rep["c6_probe_method"] = out.get("method") if isinstance(out, dict) else None
except Exception as e:
    rep["c6_probe_backend_runtime"] = f"fail: {e}"

print("__JSON_START__")
print(json.dumps(rep, ensure_ascii=False, indent=2))
PY

"${PYTHON_BIN}" - <<'PY'
import json
import os
p = os.path.join(os.environ["OUT_DIR"], "env_check.txt")
with open(p, "r", encoding="utf-8") as f:
    txt = f.read()
print("[env_check]\n" + txt)
# Also store JSON-friendly object when possible.
try:
    marker = "__JSON_START__"
    idx = txt.rfind(marker)
    payload = txt[idx + len(marker):].strip() if idx >= 0 else txt.strip()
    obj = json.loads(payload)
    with open(os.path.join(os.environ["OUT_DIR"], "env_check.json"), "w", encoding="utf-8") as g:
        json.dump(obj, g, ensure_ascii=False, indent=2)
except Exception:
    pass
PY

# ----------------------------------------------------------------------------
# 2) Analyze existing shard outputs (optional)
# ----------------------------------------------------------------------------
if [[ -n "${SHARDS_DIR}" && -d "${SHARDS_DIR}" ]]; then
  export SHARDS_DIR
  echo "[debug] analyzing existing shards: ${SHARDS_DIR}"

  grep -nH -E "Traceback|NoneType takes no arguments|Failed to load Pose models|ScaleLSD unavailable|Gazelle unavailable|C2 Seg.*failed|C2 Seg.*unavailable|Warning: sam2 import failed|fallback" \
    "${SHARDS_DIR}"/part_*.log > "${OUT_DIR}/shard_log_issues.txt" || true

  "${PYTHON_BIN}" - <<'PY'
import glob
import json
import os

base = os.environ["SHARDS_DIR"]
out = os.path.join(os.environ["OUT_DIR"], "shard_summary.json")
summary = {
    "parts": {},
    "total": {
        "rows": 0,
        "c1_present_rows": 0,
        "c1_img_dim_hist": {},
        "c1_txt_dim_hist": {},
        "c2_seg_present_rows": 0,
        "c2_seg_nonempty_rows": 0,
        "c2_det_nonempty_rows": 0,
        "c3_nonempty": 0,
        "c5_backend_runtime": {},
        "c6_backend_runtime": {},
        "c6_method": {},
    },
}

def inc(d, k):
    d[k] = d.get(k, 0) + 1

for fp in sorted(glob.glob(os.path.join(base, "part_*.jsonl"))):
    part = os.path.basename(fp)
    ps = {
        "rows": 0,
        "c1_present_rows": 0,
        "c1_img_dim_hist": {},
        "c1_txt_dim_hist": {},
        "c2_seg_present_rows": 0,
        "c2_seg_nonempty_rows": 0,
        "c2_det_nonempty_rows": 0,
        "c3_nonempty": 0,
        "c5_backend_runtime": {},
        "c6_backend_runtime": {},
        "c6_method": {},
    }
    with open(fp, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            ps["rows"] += 1
            summary["total"]["rows"] += 1
            o = json.loads(ln)

            c1_img = o.get("c1_img_embed")
            c1_txt = o.get("c1_txt_embed")
            if isinstance(c1_img, list) and isinstance(c1_txt, list):
                ps["c1_present_rows"] += 1
                summary["total"]["c1_present_rows"] += 1
                inc(ps["c1_img_dim_hist"], str(len(c1_img)))
                inc(ps["c1_txt_dim_hist"], str(len(c1_txt)))
                inc(summary["total"]["c1_img_dim_hist"], str(len(c1_img)))
                inc(summary["total"]["c1_txt_dim_hist"], str(len(c1_txt)))

            c2_seg = o.get("c2_seg")
            if isinstance(c2_seg, list):
                ps["c2_seg_present_rows"] += 1
                summary["total"]["c2_seg_present_rows"] += 1
                if len(c2_seg) > 0:
                    ps["c2_seg_nonempty_rows"] += 1
                    summary["total"]["c2_seg_nonempty_rows"] += 1
            c2_det = o.get("c2_det")
            if isinstance(c2_det, list) and len(c2_det) > 0:
                ps["c2_det_nonempty_rows"] += 1
                summary["total"]["c2_det_nonempty_rows"] += 1

            c3 = o.get("c3_pose")
            if isinstance(c3, list) and len(c3) > 0:
                ps["c3_nonempty"] += 1
                summary["total"]["c3_nonempty"] += 1

            g = o.get("c5_geom") or {}
            z = o.get("c6_gaze") or {}

            r5 = g.get("backend_runtime", "")
            r6 = z.get("backend_runtime", "")
            m6 = z.get("method", "")

            inc(ps["c5_backend_runtime"], r5)
            inc(ps["c6_backend_runtime"], r6)
            inc(ps["c6_method"], m6)
            inc(summary["total"]["c5_backend_runtime"], r5)
            inc(summary["total"]["c6_backend_runtime"], r6)
            inc(summary["total"]["c6_method"], m6)

    summary["parts"][part] = ps

with open(out, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
fi

# ----------------------------------------------------------------------------
# 3) Fresh smoke extraction run (optional)
# ----------------------------------------------------------------------------
SMOKE_JSONL="${OUT_DIR}/smoke_output.jsonl"
SMOKE_LOG="${OUT_DIR}/smoke_extract.log"
SUBSET_PARQUET="${OUT_DIR}/smoke_subset.parquet"

if [[ "${RUN_SMOKE}" == "1" ]]; then
  if [[ -z "${INPUT_PARQUET}" || ! -f "${INPUT_PARQUET}" ]]; then
    echo "[fatal] --run_smoke=1 requires valid --input_parquet" >&2
    exit 2
  fi

  echo "[debug] building subset parquet: ${SUBSET_PARQUET}"
  INPUT_PARQUET="${INPUT_PARQUET}" SUBSET_PARQUET="${SUBSET_PARQUET}" NUM_SAMPLES="${NUM_SAMPLES}" "${PYTHON_BIN}" - <<'PY'
import os
import pandas as pd
inp = os.environ["INPUT_PARQUET"]
out = os.environ["SUBSET_PARQUET"]
n = int(os.environ["NUM_SAMPLES"])
df = pd.read_parquet(inp)
if n > 0:
    df = df.head(n).copy()
df.to_parquet(out, index=False)
print(f"subset_rows={len(df)} -> {out}")
PY

  echo "[debug] running smoke extraction (${SMOKE_COMPONENTS})..."
  read -r -a COMP_ARR <<< "${SMOKE_COMPONENTS}"
  if [[ "${#COMP_ARR[@]}" -eq 0 ]]; then
    echo "[fatal] parsed empty components from --components='${SMOKE_COMPONENTS}'" >&2
    exit 2
  fi
  CMD=(bash "${SCRIPT_DIR}/run_extract_component.sh"
    "${SUBSET_PARQUET}" "${BUCKET}" "${SMOKE_JSONL}"
    --component
  )
  CMD+=("${COMP_ARR[@]}")
  CMD+=(
    --priority quality_first
    --server_mode "${SERVER_MODE}"
    --tar_dir "${TAR_DIR}"
    --batch_size "${BATCH_SIZE}"
    --mode single
  )
  if [[ -n "${IMAGE_DIR}" ]]; then
    CMD+=(--image_dir "${IMAGE_DIR}")
  fi

  CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHON_BIN="${PYTHON_BIN}" "${CMD[@]}" 2>&1 | tee "${SMOKE_LOG}"

  if [[ ! -f "${SMOKE_JSONL}" ]]; then
    echo "[fatal] smoke output missing: ${SMOKE_JSONL}" >&2
    exit 2
  fi

  SMOKE_JSONL="${SMOKE_JSONL}" SMOKE_LOG="${SMOKE_LOG}" OUT_DIR="${OUT_DIR}" "${PYTHON_BIN}" - <<'PY'
import json
import os
import re

jpath = os.environ["SMOKE_JSONL"]
lpath = os.environ["SMOKE_LOG"]
out = os.path.join(os.environ["OUT_DIR"], "smoke_summary.json")

summary = {
    "rows": 0,
    "c1_present_rows": 0,
    "c1_img_dim_hist": {},
    "c1_txt_dim_hist": {},
    "c2_seg_present_rows": 0,
    "c2_seg_nonempty_rows": 0,
    "c2_det_nonempty_rows": 0,
    "c3_nonempty": 0,
    "c5_backend_runtime": {},
    "c5_method": {},
    "c6_backend_runtime": {},
    "c6_method": {},
    "log_flags": {
        "c1_error": False,
        "c2_error": False,
        "failed_pose": False,
        "none_type_error": False,
        "c5_fallback": False,
        "c6_fallback": False,
    },
}

def inc(d, k):
    d[k] = d.get(k, 0) + 1

with open(jpath, "r", encoding="utf-8") as f:
    for ln in f:
        if not ln.strip():
            continue
        summary["rows"] += 1
        o = json.loads(ln)
        c1_img = o.get("c1_img_embed")
        c1_txt = o.get("c1_txt_embed")
        if isinstance(c1_img, list) and isinstance(c1_txt, list):
            summary["c1_present_rows"] += 1
            inc(summary["c1_img_dim_hist"], str(len(c1_img)))
            inc(summary["c1_txt_dim_hist"], str(len(c1_txt)))
        c2_seg = o.get("c2_seg")
        if isinstance(c2_seg, list):
            summary["c2_seg_present_rows"] += 1
            if len(c2_seg) > 0:
                summary["c2_seg_nonempty_rows"] += 1
        c2_det = o.get("c2_det")
        if isinstance(c2_det, list) and len(c2_det) > 0:
            summary["c2_det_nonempty_rows"] += 1
        c3 = o.get("c3_pose")
        if isinstance(c3, list) and len(c3) > 0:
            summary["c3_nonempty"] += 1
        g = o.get("c5_geom") or {}
        z = o.get("c6_gaze") or {}
        inc(summary["c5_backend_runtime"], g.get("backend_runtime", ""))
        inc(summary["c5_method"], (g.get("horizon_roll") or {}).get("method", ""))
        inc(summary["c6_backend_runtime"], z.get("backend_runtime", ""))
        inc(summary["c6_method"], z.get("method", ""))

if os.path.isfile(lpath):
    txt = open(lpath, "r", encoding="utf-8", errors="ignore").read()
    summary["log_flags"]["c1_error"] = ("C1 setup failed" in txt) or ("open_clip import failed" in txt)
    summary["log_flags"]["c2_error"] = (
        "Warning: sam2 import failed" in txt
        or "[C2 Seg] SAM2 unavailable." in txt
        or "[C2 Seg] SAM2 init failed" in txt
        or "[C2 Seg] YOLO init failed" in txt
        or "[C2 Seg] EfficientViT-SAM unavailable." in txt
        or "[C2 Seg] EfficientViT-SAM init failed" in txt
    )
    summary["log_flags"]["failed_pose"] = ("Failed to load Pose models" in txt)
    summary["log_flags"]["none_type_error"] = ("NoneType takes no arguments" in txt)
    summary["log_flags"]["c5_fallback"] = ("ScaleLSD unavailable -> fallback" in txt)
    summary["log_flags"]["c6_fallback"] = ("Gazelle unavailable -> fallback" in txt)

with open(out, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
fi

# ----------------------------------------------------------------------------
# 4) Final pass/fail decision
# ----------------------------------------------------------------------------
RESULT_TXT="${OUT_DIR}/healthcheck_result.txt"
PASS=1
{
  echo "C1/C2/C3/C5/C6 Healthcheck"
  echo "out_dir=${OUT_DIR}"
  echo

  if [[ -f "${OUT_DIR}/env_check.json" ]]; then
    echo "[env_check.json]"
    cat "${OUT_DIR}/env_check.json"
    echo
  fi

  if [[ -f "${OUT_DIR}/smoke_summary.json" ]]; then
    echo "[smoke_summary.json]"
    cat "${OUT_DIR}/smoke_summary.json"
    echo

    if ! "${PYTHON_BIN}" - <<'PY'
import json, os, sys
p = os.path.join(os.environ["OUT_DIR"], "smoke_summary.json")
s = json.load(open(p, "r", encoding="utf-8"))
ok = True
if s.get("rows", 0) <= 0:
    print("FAIL: no rows in smoke output")
    ok = False
flags = s.get("log_flags", {})
components = os.environ.get("SMOKE_COMPONENTS", "")
def has_comp(name: str) -> bool:
    tok = f" {components} "
    return f" {name} " in tok
if has_comp("c1"):
    if s.get("c1_present_rows", 0) <= 0:
        print("FAIL: C1 embeddings not written")
        ok = False
    if flags.get("c1_error"):
        print("FAIL: C1 error detected in smoke log")
        ok = False
if has_comp("c2"):
    if s.get("c2_seg_present_rows", 0) <= 0:
        print("FAIL: C2 segmentation outputs not written")
        ok = False
    if flags.get("c2_error"):
        print("FAIL: C2 backend degraded/unavailable by log")
        ok = False
if has_comp("c3") and (flags.get("failed_pose") or flags.get("none_type_error")):
    print("FAIL: C3 pose init failed (Failed to load Pose models / NoneType error)")
    ok = False
if has_comp("c5") and s.get("c5_backend_runtime", {}).get("scalelsd", 0) <= 0:
    print("FAIL: C5 scalelsd backend not active")
    ok = False
if has_comp("c6") and s.get("c6_backend_runtime", {}).get("gazelle", 0) <= 0:
    print("FAIL: C6 gazelle backend not active")
    ok = False
sys.exit(0 if ok else 1)
PY
    then
      PASS=0
    fi
  fi

  if [[ -f "${OUT_DIR}/shard_log_issues.txt" ]]; then
    echo "[shard_log_issues.txt]"
    sed -n '1,120p' "${OUT_DIR}/shard_log_issues.txt"
    echo
  fi

  if [[ "${PASS}" == "1" ]]; then
    echo "RESULT=PASS"
  else
    echo "RESULT=FAIL"
  fi
} > "${RESULT_TXT}"

cat "${RESULT_TXT}"

if [[ "${PASS}" != "1" ]]; then
  exit 3
fi

exit 0
