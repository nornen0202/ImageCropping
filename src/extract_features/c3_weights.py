"""
c3_weights.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
C3 Pose Estimation 에 필요한 가중치 파일 레지스트리 및 자동 다운로드.

사용 모델 구성:
  - Person Detector  : RTMDet-nano  (mmdetection, COCO person-specific)
  - Pose Estimator   : ViTPose-Base (mmpose, COCO 17 keypoints)

두 모델 모두 OpenMMLab 공식 weights download server 에서 제공.
Download URL 이 방화벽에 막혀 있을 경우 HuggingFace fallback URL 을 시도.
"""
from __future__ import annotations

import os
import shutil
import urllib.request
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# PROJECT 경로 계산
# worker_core / c3_weights 는 src/extract_features/ 에 위치함.
# ──────────────────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))          # src/extract_features/
_PROJECT_ROOT = os.path.normpath(os.path.join(_HERE, "../.."))     # PROJECT_ROOT/
_TP_DIR      = os.path.join(_PROJECT_ROOT, "third_party")          # third_party/

# ──────────────────────────────────────────────────────────────────────────────
# 가중치 레지스트리
# (config: third_party 소스 내 상대 경로 / weight: 로컬 파일명 + 다운로드 URL)
# ──────────────────────────────────────────────────────────────────────────────
C3_REGISTRY = {
    "det": {
        # RTMDet-nano : person-specific, small & fast
        "weight_filenames": [
            "rtmdet_nano_person.pth",
            "rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth"
        ],
        "urls": [
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
            "rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth",
        ],
        # config 은 mmpose 소스(demo/mmdetection_cfg)에서 참조
        "config_relpath": os.path.join(
            "mmpose", "demo", "mmdetection_cfg",
            "rtmdet_nano_320-8xb32_coco-person.py"
        ),
    },
    "pose": {
        # ViTPose-Base : COCO 17-keypoint, top-down
        "weight_filenames": [
            "vitpose-b-coco.pth",
            "td-hm_ViTPose-base_8xb64-210e_coco-256x192-216eae50_20230314.pth"
        ],
        "urls": [
            "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/"
            "topdown_heatmap/coco/"
            "td-hm_ViTPose-base_8xb64-210e_coco-256x192-216eae50_20230314.pth",
            "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/"
            "torch/COCO/vitpose-b-coco.pth",
        ],
        # config 은 mmpose 소스에서 참조 (git clone 필요)
        "config_relpath": os.path.join(
            "mmpose", "configs", "body_2d_keypoint",
            "topdown_heatmap", "coco",
            "td-hm_ViTPose-base_8xb64-210e_coco-256x192.py"
        ),
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# 단일 파일 다운로드 헬퍼
# ──────────────────────────────────────────────────────────────────────────────
def _download_file(urls: list[str], dest: str) -> bool:
    """
    urls 목록을 순서대로 시도하여 첫 번째 성공한 것을 dest 에 저장.
    성공: True / 전부 실패: False
    """
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    for url in urls:
        tmp = dest + ".tmp"
        print(f"  → Downloading: {url}")
        try:
            # huggingface_hub 가 설치된 경우 HF token 지원
            if "huggingface.co" in url:
                try:
                    from huggingface_hub import hf_hub_download  # noqa
                    pass  # use urllib below with auth header if needed
                except ImportError:
                    pass
                hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN", "") or \
                           os.environ.get("HF_TOKEN", "")
                req = urllib.request.Request(url)
                if hf_token:
                    req.add_header("Authorization", f"Bearer {hf_token}")
                with urllib.request.urlopen(req, timeout=120) as r, \
                        open(tmp, "wb") as f:
                    shutil.copyfileobj(r, f)
            else:
                urllib.request.urlretrieve(url, tmp)

            shutil.move(tmp, dest)
            print(f"  ✓ Saved: {dest}")
            return True
        except Exception as e:
            print(f"  ✗ Failed ({url}): {e}")
            if os.path.exists(tmp):
                os.remove(tmp)

    return False


# ──────────────────────────────────────────────────────────────────────────────
# 공개 API
# ──────────────────────────────────────────────────────────────────────────────
def resolve_c3_paths(weights_dir: str) -> Optional[dict]:
    """
    C3 에 필요한 config / weight 경로를 확정하고 필요하면 자동 다운로드.

    Returns:
        dict with keys: det_cfg, det_w, pose_cfg, pose_w
        or None if setup could not be completed.
    """
    os.makedirs(weights_dir, exist_ok=True)

    paths = {}
    all_ok = True

    for role, meta in C3_REGISTRY.items():
        w_path = None
        w_name_used = None
        # 여러 허용된 파일명 중 디렉터리에 존재하는 파일이 있는지 확인
        for w_name in meta["weight_filenames"]:
            tmp_path = os.path.join(weights_dir, w_name)
            if os.path.exists(tmp_path):
                w_path = tmp_path
                w_name_used = w_name
                break
        
        # ── 가중치 파일 다운로드 ──────────────────────────────────────────────
        if w_path is None:
            # 여러 파일명 중 첫번째를 타겟으로 함
            w_name_used = meta["weight_filenames"][0]
            w_path = os.path.join(weights_dir, w_name_used)
            print(f"[C3 Weights] '{w_name_used}' not found. Attempting download...")
            ok = _download_file(meta["urls"], w_path)
            if not ok:
                print(f"[C3 Weights] ERROR: Could not download {w_name_used}.")
                print(f"  Please manually place the file at: {w_path}")
                print(f"  Download from one of:")
                for u in meta["urls"]:
                    print(f"    {u}")
                all_ok = False
        else:
            sz_mb = os.path.getsize(w_path) / 1e6
            print(f"[C3 Weights] '{w_name_used}' found ({sz_mb:.1f} MB). OK.")

        # ── config 파일 위치 확인 ────────────────────────────────────────────
        cfg_path = os.path.join(_TP_DIR, meta["config_relpath"])
        if not os.path.exists(cfg_path):
            # third_party 소스가 없으면 config 를 weights_dir 에도 찾아봄
            alt_cfg = os.path.join(weights_dir, os.path.basename(meta["config_relpath"]))
            if os.path.exists(alt_cfg):
                cfg_path = alt_cfg
            else:
                print(f"[C3 Weights] Config not found: {cfg_path}")
                print(f"  Install required: bash src/scripts/install_features_deps_torch251_cu121_stable.sh")
                print(f"  (this clones mmdetection/mmpose into third_party/)")
                all_ok = False

        paths[role] = {"cfg": cfg_path, "w": w_path}

    if not all_ok:
        return None

    return {
        "det_cfg":  paths["det"]["cfg"],
        "det_w":    paths["det"]["w"],
        "pose_cfg": paths["pose"]["cfg"],
        "pose_w":   paths["pose"]["w"],
    }


def print_c3_status(weights_dir: str) -> None:
    """C3 가중치 상태를 화면에 출력 (디버깅용)."""
    print("=" * 60)
    print("C3 Pose Weights Status")
    print(f"  weights_dir : {weights_dir}")
    print(f"  third_party : {_TP_DIR}")
    for role, meta in C3_REGISTRY.items():
        w_names = meta["weight_filenames"]
        w = os.path.join(weights_dir, w_names[0])
        # Found any config?
        found_w = any(os.path.exists(os.path.join(weights_dir, nm)) for nm in w_names)
        c = os.path.join(_TP_DIR, meta["config_relpath"])
        print(f"\n  [{role.upper()}]")
        print(f"    weight : {'✓' if found_w else '✗'} {w} (or alternative name)")
        print(f"    config : {'✓' if os.path.exists(c) else '✗'} {c}")
    print("=" * 60)
