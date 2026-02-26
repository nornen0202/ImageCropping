#!/usr/bin/env python
"""
Setup helper for public cropping teachers (GAIC/CACNet/CGS).

What this script does:
1) Clone repos under third_party/public_cropping_teachers
2) Apply minimal compatibility patches for PyTorch 2.x + Python 3.10
3) Build CGS CUDA extensions (roi_align_api / rod_align_api)
4) Download pretrained weights via gdown (best-effort)

If Google Drive download fails, the script prints exact manual links and
expected destination paths.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys
import zipfile


REPO_SPECS = {
    "gaic": {
        "url": "https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping-Pytorch",
        "dir_name": "gaic",
    },
    "cacnet": {
        "url": "https://github.com/bo-zhang-cs/CACNet-Pytorch",
        "dir_name": "cacnet",
    },
    "cgs": {
        "url": "https://github.com/bo-zhang-cs/CGS-Pytorch",
        "dir_name": "cgs",
    },
}


WEIGHT_SPECS = {
    "gaic": {
        "gdrive_id": "1OvLT_ul17zCK4ljAi4myGAgA50PmLy3Y",
        "dest_rel": "gaic/gaic_conference_pretrained.pth",
        "is_zip": False,
        "manual_url": "https://drive.google.com/file/d/1OvLT_ul17zCK4ljAi4myGAgA50PmLy3Y/view?usp=sharing",
    },
    "cacnet": {
        "gdrive_id": "19LUhHK1viHu9TYqzk2te2orqzKMA_ZRQ",
        "dest_rel": "cacnet/best-FLMS_iou.pth",
        "is_zip": False,
        "manual_url": "https://drive.google.com/file/d/19LUhHK1viHu9TYqzk2te2orqzKMA_ZRQ/view?usp=sharing",
    },
    "cgs": {
        "gdrive_id": "1CmMBcQdOc22Qnyle0xJlbO6BC8tdViWN",
        "dest_rel": "cgs/cgs_pretrained.zip",
        "is_zip": True,
        "manual_url": "https://drive.google.com/file/d/1CmMBcQdOc22Qnyle0xJlbO6BC8tdViWN/view?usp=sharing",
    },
}


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")


def clone_if_missing(root: Path, name: str) -> Path:
    spec = REPO_SPECS[name]
    repo_dir = root / spec["dir_name"]
    if (repo_dir / ".git").exists():
        print(f"[setup] repo exists: {repo_dir}")
        return repo_dir
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"[setup] cloning {name} -> {repo_dir}")
    run_cmd(["git", "clone", "--depth", "1", spec["url"], str(repo_dir)])
    return repo_dir


def _replace_once(text: str, old: str, new: str) -> str:
    if old in text:
        return text.replace(old, new, 1)
    return text


def patch_gaic_repo(gaic_dir: Path) -> None:
    """
    GAIC repo has several Python2-era issues under modern Python:
    - inconsistent tab/space in ShuffleNetV2.py
    - unconditional ShuffleNet import (breaks even when mobilenet is used)
    - fc_layers references undefined `dropout`
    """
    print(f"[setup] patching gaic repo for python3/torch2 compatibility: {gaic_dir}")

    for rel in ("ShuffleNetV2.py", "mobilenetv2.py", "croppingModel.py"):
        p = gaic_dir / rel
        if not p.exists():
            continue
        s = p.read_text(encoding="utf-8", errors="ignore")
        s = s.expandtabs(4)
        if rel == "ShuffleNetV2.py":
            # Known bad-indentation lines in the official GAIC repo.
            s = s.replace(
                "\n    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)\n",
                "\n        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)\n",
            )
            s = s.replace(
                "\n    self.globalpool = nn.Sequential(nn.AvgPool2d(int(input_size/32)))              \n",
                "\n        self.globalpool = nn.Sequential(nn.AvgPool2d(int(input_size/32)))\n",
            )
            s = s.replace("\n    # building classifier\n", "\n        # building classifier\n")
            s = s.replace(
                "\n    self.classifier = nn.Sequential(nn.Linear(self.stage_out_channels[-1], n_class))\n",
                "\n        self.classifier = nn.Sequential(nn.Linear(self.stage_out_channels[-1], n_class))\n",
            )
        p.write_text(s, encoding="utf-8")

    cm = gaic_dir / "croppingModel.py"
    if cm.exists():
        s = cm.read_text(encoding="utf-8", errors="ignore")
        canonical_shuffle_import = (
            "try:\n"
            "    from ShuffleNetV2 import shufflenetv2\n"
            "except Exception:\n"
            "    shufflenetv2 = None\n"
        )
        # Normalize duplicated/broken try-import blocks into one canonical block.
        s = re.sub(
            r"(?ms)try:\s*(?:try:\s*)?from ShuffleNetV2 import shufflenetv2\s*"
            r"except Exception:\s*shufflenetv2 = None(?:\s*except Exception:\s*shufflenetv2 = None)?\s*",
            canonical_shuffle_import,
            s,
            count=1,
        )
        if "from ShuffleNetV2 import shufflenetv2" in s and canonical_shuffle_import not in s:
            s = _replace_once(s, "from ShuffleNetV2 import shufflenetv2\n", canonical_shuffle_import)
        s = s.replace(
            "layers = nn.Sequential(conv1, conv2, dropout, conv3)",
            "layers = nn.Sequential(conv1, conv2, conv3)",
        )
        cm.write_text(s, encoding="utf-8")


def patch_cgs_repo(cgs_dir: Path) -> None:
    print(f"[setup] patching cgs cuda extensions for torch2: {cgs_dir}")

    roi_cuda = cgs_dir / "roi_align" / "src" / "roi_align_cuda.cpp"
    rod_cuda = cgs_dir / "rod_align" / "src" / "rod_align_cuda.cpp"
    rod_kernel_h = cgs_dir / "rod_align" / "src" / "rod_align_kernel.h"
    rod_kernel_cu = cgs_dir / "rod_align" / "src" / "rod_align_kernel.cu"

    for p in (roi_cuda, rod_cuda, rod_kernel_h):
        if not p.exists():
            continue
        s = p.read_text(encoding="utf-8", errors="ignore")
        s = s.replace("#include <THC/THC.h>\n", "")
        p.write_text(s, encoding="utf-8")

    if roi_cuda.exists():
        s = roi_cuda.read_text(encoding="utf-8", errors="ignore")
        if "#include <ATen/cuda/CUDAContext.h>" not in s:
            s = s.replace(
                "#include <torch/extension.h>\n",
                "#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n",
            )
        roi_cuda.write_text(s, encoding="utf-8")

    if rod_cuda.exists():
        s = rod_cuda.read_text(encoding="utf-8", errors="ignore")
        if "#include <torch/extension.h>" not in s:
            s = s.replace(
                "#include <math.h>\n",
                "#include <math.h>\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n",
            )
        elif "#include <ATen/cuda/CUDAContext.h>" not in s:
            s = s.replace(
                "#include <torch/extension.h>\n",
                "#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n",
            )
        rod_cuda.write_text(s, encoding="utf-8")

    if rod_kernel_cu.exists():
        s = rod_kernel_cu.read_text(encoding="utf-8", errors="ignore")
        if "#include <stdio.h>" not in s:
            s = s.replace("#include <float.h>\n", "#include <stdio.h>\n#include <float.h>\n")
        rod_kernel_cu.write_text(s, encoding="utf-8")


def build_cgs_extensions(cgs_dir: Path) -> None:
    print(f"[setup] building cgs roi_align extension: {cgs_dir / 'roi_align'}")
    run_cmd([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=cgs_dir / "roi_align")
    print(f"[setup] building cgs rod_align extension: {cgs_dir / 'rod_align'}")
    run_cmd([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=cgs_dir / "rod_align")


def maybe_import_gdown():
    try:
        import gdown  # type: ignore
    except Exception:
        return None
    return gdown


def looks_like_matlab_mat(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            head = f.read(64)
        return head.startswith(b"MATLAB 5.0 MAT-file")
    except Exception:
        return False


def download_weights(weights_root: Path) -> None:
    gdown = maybe_import_gdown()
    if gdown is None:
        print("[setup][warn] gdown is not installed. skipping automatic weight download.")
        print_manual_download_guide(weights_root)
        return

    for name, spec in WEIGHT_SPECS.items():
        dst = weights_root / spec["dest_rel"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.stat().st_size > 0:
            print(f"[setup] weight exists: {dst}")
            continue
        url = f"https://drive.google.com/uc?id={spec['gdrive_id']}"
        print(f"[setup] downloading {name} weight -> {dst}")
        try:
            gdown.download(url=url, output=str(dst), quiet=False, fuzzy=True)
        except Exception as e:
            print(f"[setup][warn] failed to download {name}: {e}")
            continue
        if name == "gaic" and dst.exists() and looks_like_matlab_mat(dst):
            print(
                "[setup][warn] downloaded GAIC file appears to be MATLAB .mat format, "
                "not a PyTorch state_dict checkpoint."
            )
            print(
                "[setup][warn] if GAIC init fails, please manually place a compatible "
                "PyTorch .pth checkpoint and pass --gaic_weight to inference script."
            )

    # Unzip CGS model package if downloaded.
    cgs_zip = weights_root / WEIGHT_SPECS["cgs"]["dest_rel"]
    cgs_dst = weights_root / "cgs" / "pretrained_model"
    if cgs_zip.exists() and cgs_zip.stat().st_size > 0 and not cgs_dst.exists():
        cgs_dst.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(cgs_zip, "r") as zf:
                zf.extractall(cgs_dst)
            print(f"[setup] extracted cgs zip -> {cgs_dst}")
        except Exception as e:
            print(f"[setup][warn] failed to extract cgs zip: {e}")


def print_manual_download_guide(weights_root: Path) -> None:
    print("\n[setup] manual weight download guide")
    for name, spec in WEIGHT_SPECS.items():
        dst = weights_root / spec["dest_rel"]
        print(f"- {name}:")
        print(f"  url : {spec['manual_url']}")
        print(f"  save: {dst}")
    print("")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Setup GAIC/CACNet/CGS public teacher repos and weights")
    p.add_argument(
        "--teacher_root_dir",
        default="third_party/public_cropping_teachers",
        help="where to clone teacher repos",
    )
    p.add_argument(
        "--weights_dir",
        default="weights/public_cropping_teachers",
        help="where to store pretrained weights",
    )
    p.add_argument("--clone", type=int, default=1)
    p.add_argument("--patch", type=int, default=1)
    p.add_argument("--build_ext", type=int, default=1)
    p.add_argument("--download_weights", type=int, default=1)
    p.add_argument("--print_manual_guide", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    teacher_root = Path(args.teacher_root_dir)
    weights_root = Path(args.weights_dir)

    repo_dirs: dict[str, Path] = {}
    if bool(int(args.clone)):
        for name in ("gaic", "cacnet", "cgs"):
            repo_dirs[name] = clone_if_missing(teacher_root, name)
    else:
        repo_dirs = {k: teacher_root / REPO_SPECS[k]["dir_name"] for k in REPO_SPECS.keys()}

    if bool(int(args.patch)):
        patch_gaic_repo(repo_dirs["gaic"])
        patch_cgs_repo(repo_dirs["cgs"])

    if bool(int(args.build_ext)):
        try:
            build_cgs_extensions(repo_dirs["cgs"])
        except Exception as e:
            print(f"[setup][warn] extension build failed: {e}")
            print("[setup][warn] GAIC/CGS inference may fail unless extensions are built manually.")

    if bool(int(args.download_weights)):
        download_weights(weights_root)

    if bool(int(args.print_manual_guide)):
        print_manual_download_guide(weights_root)

    print("[setup] done")


if __name__ == "__main__":
    main()
