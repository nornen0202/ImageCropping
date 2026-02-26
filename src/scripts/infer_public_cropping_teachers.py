#!/usr/bin/env python
"""
Run public cropping teachers (GAIC/CACNet/CGS) and dump raw inference outputs.

Output schema (one line per image x teacher):
{
  "image_id": "...",
  "teacher_id": "gaic|cacnet|cgs",
  "image_size": [W, H],
  "num_proposals": 4,
  "proposals": [
    {
      "bbox_xyxy": [x1,y1,x2,y2],
      "bbox_norm_xyxy": [x1n,y1n,x2n,y2n],
      "score": 0.123
    }
  ],
  "error": null
}
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import glob

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from tqdm import tqdm


IMAGE_NET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_NET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def normalize_box_xyxy(box_xyxy: Sequence[float], w: int, h: int) -> List[float]:
    ww = max(1.0, float(w))
    hh = max(1.0, float(h))
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    x1 = clamp(x1 / ww, 0.0, 1.0)
    y1 = clamp(y1 / hh, 0.0, 1.0)
    x2 = clamp(x2 / ww, 0.0, 1.0)
    y2 = clamp(y2 / hh, 0.0, 1.0)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6)]


def clip_box_xyxy(box_xyxy: Sequence[float], w: int, h: int) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    x1 = clamp(x1, 0.0, float(w))
    y1 = clamp(y1, 0.0, float(h))
    x2 = clamp(x2, 0.0, float(w))
    y2 = clamp(y2, 0.0, float(h))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def remove_module_prefix(state: Dict[str, Any]) -> Dict[str, Any]:
    if not state:
        return state
    keys = list(state.keys())
    if not all(isinstance(k, str) for k in keys):
        return state
    if all(k.startswith("module.") for k in keys):
        return {k[7:]: v for k, v in state.items()}
    return state


def unwrap_state_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return remove_module_prefix(obj["state_dict"])
    if isinstance(obj, dict):
        return remove_module_prefix(obj)
    raise TypeError(f"unsupported state dict object type: {type(obj)}")


def looks_like_matlab_mat(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            head = f.read(64)
        return head.startswith(b"MATLAB 5.0 MAT-file")
    except Exception:
        return False


def generate_gaic_grid_bboxes(resized_h: int, resized_w: int) -> List[List[float]]:
    """
    GAIC repo's default candidate generator.

    Returns boxes in resized-image coordinates with ordering:
    [y1, x1, y2, x2]
    """
    bins = 12.0
    step_h = resized_h / bins
    step_w = resized_w / bins
    out: List[List[float]] = []
    for x1 in range(0, 4):
        for y1 in range(0, 4):
            for x2 in range(8, 12):
                for y2 in range(8, 12):
                    area_ok = (x2 - x1) * (y2 - y1) > 0.4999 * bins * bins
                    ratio = (y2 - y1) * step_w / ((x2 - x1) * step_h + 1e-12)
                    ratio_ok = (ratio > 0.5) and (ratio < 2.0)
                    if area_ok and ratio_ok:
                        out.append(
                            [
                                float(step_h * (0.5 + x1)),
                                float(step_w * (0.5 + y1)),
                                float(step_h * (0.5 + x2)),
                                float(step_w * (0.5 + y2)),
                            ]
                        )
    return out


def preprocess_for_grid_teacher(image_rgb: np.ndarray, image_size: int = 256) -> Dict[str, Any]:
    """
    Shared preprocessing for GAIC/CGS style models.
    """
    h0, w0 = image_rgb.shape[:2]
    scale = float(image_size) / float(min(h0, w0))
    h = int(round(h0 * scale / 32.0) * 32)
    w = int(round(w0 * scale / 32.0) * 32)
    h = max(32, h)
    w = max(32, w)

    resized = cv2.resize(image_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    # GAIC repo divides by 256.0; we preserve this for fidelity.
    x = resized.astype(np.float32) / 256.0
    x = (x - IMAGE_NET_MEAN) / IMAGE_NET_STD
    x = np.transpose(x, (2, 0, 1))
    tensor = torch.from_numpy(x).unsqueeze(0).float()

    cand_yxyx = generate_gaic_grid_bboxes(resized_h=h, resized_w=w)
    sx = float(w0) / float(w)
    sy = float(h0) / float(h)
    source_xyxy: List[List[float]] = []
    rois_xyxy: List[List[float]] = []
    for y1, x1, y2, x2 in cand_yxyx:
        # model ROI on resized image (x1,y1,x2,y2)
        rois_xyxy.append([x1, y1, x2, y2])
        # map back to original image
        source_xyxy.append([x1 * sx, y1 * sy, x2 * sx, y2 * sy])

    return {
        "tensor": tensor,
        "resized_size": (w, h),
        "rois_xyxy": rois_xyxy,
        "source_boxes_xyxy": source_xyxy,
    }


@contextmanager
def prepend_sys_path(paths: Sequence[Path]):
    old = list(sys.path)
    for p in reversed(paths):
        sys.path.insert(0, str(p))
    try:
        yield
    finally:
        sys.path[:] = old


def import_module_from_file(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to build spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def ensure_cuda_for_teacher(teacher_id: str, device: torch.device) -> None:
    if teacher_id in {"gaic", "cgs"} and device.type != "cuda":
        raise RuntimeError(f"{teacher_id} requires CUDA (ROI/RoDAlign extension is CUDA-only)")


def score_sort_indices(scores: np.ndarray, topk: int) -> List[int]:
    if scores.size == 0:
        return []
    ids = np.argsort(-scores)
    k = max(1, min(int(topk), int(ids.shape[0])))
    return [int(i) for i in ids[:k]]


class TarImageLoader:
    def __init__(self, mapping: Dict[str, Dict[str, Any]], tar_dir: Path):
        self.mapping = mapping
        self.tar_dir = tar_dir
        self._current_tar_path: Optional[Path] = None
        self._current_tar: Optional[tarfile.TarFile] = None
        self._members_by_name: Dict[str, tarfile.TarInfo] = {}

    def close(self) -> None:
        if self._current_tar is not None:
            try:
                self._current_tar.close()
            except Exception:
                pass
        self._current_tar = None
        self._current_tar_path = None
        self._members_by_name = {}

    def _resolve_tar_path(self, tar_name: str, bucket: str) -> Optional[Path]:
        cands = [
            self.tar_dir / tar_name,
            self.tar_dir / str(bucket) / tar_name if bucket else None,
        ]
        for p in cands:
            if p is not None and p.exists():
                return p
        return None

    def _open_tar(self, tar_path: Path) -> None:
        if self._current_tar_path == tar_path and self._current_tar is not None:
            return
        self.close()
        tf = tarfile.open(str(tar_path), "r")
        members = {}
        for m in tf.getmembers():
            if not m.isfile():
                continue
            base = os.path.basename(m.name)
            if not base:
                continue
            members[base] = m
        self._current_tar = tf
        self._current_tar_path = tar_path
        self._members_by_name = members

    def load_rgb(self, image_id: str) -> Optional[Image.Image]:
        info = self.mapping.get(str(image_id))
        if not info:
            return None
        tar_name = str(info.get("tar_name", ""))
        bucket = str(info.get("bucket", "")) if "bucket" in info else ""
        if not tar_name:
            return None
        tar_path = self._resolve_tar_path(tar_name=tar_name, bucket=bucket)
        if tar_path is None:
            return None
        self._open_tar(tar_path)
        if self._current_tar is None:
            return None

        for ext in ("jpg", "jpeg", "png", "webp"):
            fname = f"{image_id}.{ext}"
            m = self._members_by_name.get(fname)
            if m is None:
                continue
            fobj = self._current_tar.extractfile(m)
            if fobj is None:
                continue
            data = fobj.read()
            try:
                return Image.open(io.BytesIO(data)).convert("RGB")
            except Exception:
                return None
        return None


def load_tar_mapping(parquet_path: Path) -> Dict[str, Dict[str, Any]]:
    df = pd.read_parquet(parquet_path)
    if "image_id" not in df.columns or "tar_name" not in df.columns:
        raise ValueError(f"parquet missing image_id/tar_name columns: {parquet_path}")
    if "bucket" in df.columns:
        out = df.set_index("image_id")[["tar_name", "bucket"]].to_dict("index")
    else:
        out = df.set_index("image_id")[["tar_name"]].to_dict("index")
    return {str(k): v for k, v in out.items()}


class GAICTeacher:
    teacher_id = "gaic"

    def __init__(
        self,
        *,
        gaic_repo: Path,
        cgs_repo: Path,
        weight_path: Path,
        device: torch.device,
        topk: int,
        gaic_model: str = "auto",
    ):
        ensure_cuda_for_teacher(self.teacher_id, device)
        self.device = device
        self.topk = max(1, int(topk))
        self.weight_path = weight_path
        if not self.weight_path.exists():
            raise FileNotFoundError(f"gaic weight not found: {self.weight_path}")
        if looks_like_matlab_mat(self.weight_path):
            raise RuntimeError(
                "gaic weight appears to be MATLAB .mat format (not PyTorch checkpoint). "
                "Please provide a compatible GAIC .pth via --gaic_weight."
            )

        with prepend_sys_path(
            [
                cgs_repo / "rod_align",
                cgs_repo / "roi_align",
                cgs_repo,
                gaic_repo,
            ]
        ):
            # Force using cgs-compatible roi/rod packages before GAIC module import.
            import roi_align  # noqa: F401
            import rod_align  # noqa: F401
            gaic_mod = import_module_from_file(
                "gaic_cropping_model_runtime",
                gaic_repo / "croppingModel.py",
            )

        self._gaic_mod = gaic_mod
        raw_state = torch.load(str(self.weight_path), map_location="cpu")
        state = unwrap_state_dict(raw_state)
        model_choice = str(gaic_model).strip().lower()
        if model_choice == "auto":
            key_text = " ".join(state.keys())
            name_text = self.weight_path.name.lower()
            if ("banch" in key_text) or ("shuffle" in name_text):
                model_choice = "shufflenetv2"
            else:
                model_choice = "mobilenetv2"
        if model_choice not in {"mobilenetv2", "shufflenetv2", "vgg16", "resnet50"}:
            raise ValueError(f"unsupported --gaic_model: {gaic_model}")
        print(f"[gaic] backbone={model_choice} weight={self.weight_path.name}")
        self.model = gaic_mod.build_crop_model(
            scale="multi",
            alignsize=9,
            reddim=8,
            loadweight=False,
            model=model_choice,
            downsample=4,
        )
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            print(f"[gaic][warn] missing keys: {len(missing)}")
        if unexpected:
            print(f"[gaic][warn] unexpected keys: {len(unexpected)}")
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, image: Image.Image) -> List[Dict[str, Any]]:
        image_rgb = np.asarray(image.convert("RGB"))
        h0, w0 = image_rgb.shape[:2]
        pre = preprocess_for_grid_teacher(image_rgb=image_rgb, image_size=256)

        x = pre["tensor"].to(self.device, non_blocking=True)
        rois_xyxy = pre["rois_xyxy"]
        source_boxes = pre["source_boxes_xyxy"]
        if not rois_xyxy:
            return []
        rois = [[0.0, b[0], b[1], b[2], b[3]] for b in rois_xyxy]
        roi_t = torch.tensor(rois, dtype=torch.float32, device=self.device)

        out = self.model(x, roi_t)
        scores = out.detach().float().reshape(-1).cpu().numpy()
        ids = score_sort_indices(scores, self.topk)

        results: List[Dict[str, Any]] = []
        for idx in ids:
            box = clip_box_xyxy(source_boxes[idx], w=w0, h=h0)
            results.append(
                {
                    "bbox_xyxy": [round(v, 3) for v in box],
                    "bbox_norm_xyxy": normalize_box_xyxy(box, w=w0, h=h0),
                    "score": float(scores[idx]),
                }
            )
        return results


class CGSTeacher:
    teacher_id = "cgs"

    def __init__(
        self,
        *,
        cgs_repo: Path,
        extractor_weight: Path,
        gnn_weight: Path,
        device: torch.device,
        topk: int,
    ):
        ensure_cuda_for_teacher(self.teacher_id, device)
        self.device = device
        self.topk = max(1, int(topk))
        if not extractor_weight.exists():
            raise FileNotFoundError(f"cgs extractor weight not found: {extractor_weight}")
        if not gnn_weight.exists():
            raise FileNotFoundError(f"cgs gnn weight not found: {gnn_weight}")

        with prepend_sys_path([cgs_repo / "rod_align", cgs_repo / "roi_align", cgs_repo]):
            cgs_mod = import_module_from_file(
                "cgs_cropping_model_runtime",
                cgs_repo / "croppingModel.py",
            )

        self.extractor = cgs_mod.RegionFeatureExtractor(loadweight=False)
        self.gnn = cgs_mod.CroppingGraph()

        ext_state = unwrap_state_dict(torch.load(str(extractor_weight), map_location="cpu"))
        gnn_state = unwrap_state_dict(torch.load(str(gnn_weight), map_location="cpu"))
        self.extractor.load_state_dict(ext_state, strict=False)
        self.gnn.load_state_dict(gnn_state, strict=False)
        self.extractor.to(self.device).eval()
        self.gnn.to(self.device).eval()

    @torch.no_grad()
    def predict(self, image: Image.Image) -> List[Dict[str, Any]]:
        image_rgb = np.asarray(image.convert("RGB"))
        h0, w0 = image_rgb.shape[:2]
        pre = preprocess_for_grid_teacher(image_rgb=image_rgb, image_size=256)
        x = pre["tensor"].to(self.device, non_blocking=True)
        rois_xyxy = pre["rois_xyxy"]
        source_boxes = pre["source_boxes_xyxy"]
        if not rois_xyxy:
            return []
        roi_t = torch.tensor([rois_xyxy], dtype=torch.float32, device=self.device)

        feat = self.extractor(x, roi_t)
        _adj, score = self.gnn(feat)
        scores = score.detach().float().reshape(-1).cpu().numpy()
        ids = score_sort_indices(scores, self.topk)

        results: List[Dict[str, Any]] = []
        for idx in ids:
            box = clip_box_xyxy(source_boxes[idx], w=w0, h=h0)
            results.append(
                {
                    "bbox_xyxy": [round(v, 3) for v in box],
                    "bbox_norm_xyxy": normalize_box_xyxy(box, w=w0, h=h0),
                    "score": float(scores[idx]),
                }
            )
        return results


class CACNetTeacher:
    teacher_id = "cacnet"

    def __init__(
        self,
        *,
        cacnet_repo: Path,
        weight_path: Path,
        device: torch.device,
        topk: int,
    ):
        self.device = device
        self.topk = max(1, int(topk))
        if not weight_path.exists():
            raise FileNotFoundError(f"cacnet weight not found: {weight_path}")

        with prepend_sys_path([cacnet_repo]):
            cacnet_mod = import_module_from_file(
                "cacnet_model_runtime",
                cacnet_repo / "CACNet.py",
            )
            cfg_mod = import_module_from_file(
                "cacnet_cfg_runtime",
                cacnet_repo / "config_cropping.py",
            )
        self._cfg = cfg_mod.cfg
        self.model = cacnet_mod.CACNet(loadweights=False)
        state = unwrap_state_dict(torch.load(str(weight_path), map_location="cpu"))
        self.model.load_state_dict(state, strict=False)
        self.model.to(self.device).eval()

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        im_w, im_h = image.size
        keep_ar = bool(getattr(self._cfg, "keep_aspect_ratio", False))
        if keep_ar:
            scale = float(self._cfg.image_size[0]) / float(min(im_h, im_w))
            h = int(round(im_h * scale / 32.0) * 32)
            w = int(round(im_w * scale / 32.0) * 32)
        else:
            w = int(self._cfg.image_size[0])
            h = int(self._cfg.image_size[1])

        # Pillow compatibility
        try:
            resample = Image.Resampling.LANCZOS
        except Exception:
            resample = Image.BILINEAR
        resized = image.resize((w, h), resample)
        arr = np.asarray(resized.convert("RGB")).astype(np.float32) / 255.0
        arr = (arr - IMAGE_NET_MEAN) / IMAGE_NET_STD
        arr = np.transpose(arr, (2, 0, 1))
        x = torch.from_numpy(arr).unsqueeze(0).float()
        return x

    @torch.no_grad()
    def predict(self, image: Image.Image) -> List[Dict[str, Any]]:
        im_w, im_h = image.size
        x = self._preprocess(image).to(self.device, non_blocking=True)
        _logits, _kcm, crop = self.model(x, only_classify=False)
        # One crop prediction from official demo.
        box = crop.detach().float().cpu().numpy().reshape(-1, 4)[0]
        x1, y1, x2, y2 = box.tolist()
        # Convert from resized coordinates to original image coordinates.
        rw = float(x.shape[-1])
        rh = float(x.shape[-2])
        x1 = x1 / rw * im_w
        x2 = x2 / rw * im_w
        y1 = y1 / rh * im_h
        y2 = y2 / rh * im_h
        box_xyxy = clip_box_xyxy([x1, y1, x2, y2], w=im_w, h=im_h)
        return [
            {
                "bbox_xyxy": [round(v, 3) for v in box_xyxy],
                "bbox_norm_xyxy": normalize_box_xyxy(box_xyxy, w=im_w, h=im_h),
                "score": 1.0,
            }
        ]


def resolve_weight_path(path_arg: str, fallback_patterns: Sequence[str]) -> Optional[Path]:
    if path_arg and Path(path_arg).exists():
        return Path(path_arg)
    for pat in fallback_patterns:
        found = sorted(Path(x) for x in glob.glob(pat, recursive=True))
        if found:
            return found[0]
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run GAIC/CACNet/CGS teacher inference and dump raw outputs")
    p.add_argument("--input_parquet", required=True)
    p.add_argument("--tar_dir", required=True)
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--teachers", nargs="+", default=["gaic", "cacnet", "cgs"])
    p.add_argument("--teacher_root_dir", default="third_party/public_cropping_teachers")
    p.add_argument("--weights_dir", default="weights/public_cropping_teachers")
    p.add_argument("--max_images", type=int, default=0, help="0 means all")
    p.add_argument("--device", default="auto", help="auto|cuda|cpu")
    p.add_argument("--skip_on_oom", type=int, default=1)
    p.add_argument("--run_setup", type=int, default=1, help="run setup_public_cropping_teachers.py first")
    p.add_argument("--setup_download_weights", type=int, default=1)

    p.add_argument("--gaic_weight", default="")
    p.add_argument(
        "--gaic_model",
        default="auto",
        help="auto|mobilenetv2|shufflenetv2|vgg16|resnet50",
    )
    p.add_argument("--cacnet_weight", default="")
    p.add_argument("--cgs_extractor_weight", default="")
    p.add_argument("--cgs_gnn_weight", default="")

    p.add_argument("--gaic_topk", type=int, default=4)
    p.add_argument("--cgs_topk", type=int, default=4)
    p.add_argument("--cacnet_topk", type=int, default=1)
    return p.parse_args()


def pick_device(device_arg: str) -> torch.device:
    s = str(device_arg).strip().lower()
    if s == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if s == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def maybe_run_setup(args: argparse.Namespace) -> None:
    if not bool(int(args.run_setup)):
        return
    setup_script = Path("src/scripts/setup_public_cropping_teachers.py")
    cmd = [
        sys.executable,
        str(setup_script),
        "--teacher_root_dir",
        str(args.teacher_root_dir),
        "--weights_dir",
        str(args.weights_dir),
        "--clone",
        "1",
        "--patch",
        "1",
        "--build_ext",
        "1",
        "--download_weights",
        str(int(args.setup_download_weights)),
        "--print_manual_guide",
        "1",
    ]
    print("[infer] running setup script")
    proc = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"setup script failed with exit code {proc.returncode}")


def build_teacher_adapters(args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    root = Path(args.teacher_root_dir)
    weights_root = Path(args.weights_dir)
    adapters: Dict[str, Any] = {}

    teacher_set = {str(t).strip().lower() for t in args.teachers}
    teacher_set = {t for t in teacher_set if t}
    if not teacher_set:
        return adapters

    gaic_repo = root / "gaic"
    cacnet_repo = root / "cacnet"
    cgs_repo = root / "cgs"

    if "gaic" in teacher_set:
        gaic_w = resolve_weight_path(
            args.gaic_weight,
            fallback_patterns=(
                f"{weights_root}/gaic/*.pth",
                f"{weights_root}/gaic/**/*.pth",
            ),
        )
        if gaic_w is None:
            print("[infer][warn] gaic weight not found. skipping gaic")
        else:
            try:
                adapters["gaic"] = GAICTeacher(
                    gaic_repo=gaic_repo,
                    cgs_repo=cgs_repo,
                    weight_path=gaic_w,
                    device=device,
                    topk=args.gaic_topk,
                    gaic_model=args.gaic_model,
                )
            except Exception as e:
                print(f"[infer][warn] failed to init gaic teacher: {e}")

    if "cacnet" in teacher_set:
        cac_w = resolve_weight_path(
            args.cacnet_weight,
            fallback_patterns=(f"{weights_root}/cacnet/*.pth", f"{weights_root}/cacnet/**/*.pth"),
        )
        if cac_w is None:
            print("[infer][warn] cacnet weight not found. skipping cacnet")
        else:
            try:
                adapters["cacnet"] = CACNetTeacher(
                    cacnet_repo=cacnet_repo,
                    weight_path=cac_w,
                    device=device,
                    topk=args.cacnet_topk,
                )
            except Exception as e:
                print(f"[infer][warn] failed to init cacnet teacher: {e}")

    if "cgs" in teacher_set:
        ext_w = resolve_weight_path(
            args.cgs_extractor_weight,
            fallback_patterns=(
                f"{weights_root}/cgs/**/*extractor*.pth",
                f"{weights_root}/cgs/**/*Extractor*.pth",
            ),
        )
        gnn_w = resolve_weight_path(
            args.cgs_gnn_weight,
            fallback_patterns=(
                f"{weights_root}/cgs/**/*gnn*.pth",
                f"{weights_root}/cgs/**/*GNN*.pth",
            ),
        )
        if ext_w is None or gnn_w is None:
            print("[infer][warn] cgs extractor/gnn weight not found. skipping cgs")
        else:
            try:
                adapters["cgs"] = CGSTeacher(
                    cgs_repo=cgs_repo,
                    extractor_weight=ext_w,
                    gnn_weight=gnn_w,
                    device=device,
                    topk=args.cgs_topk,
                )
            except Exception as e:
                print(f"[infer][warn] failed to init cgs teacher: {e}")

    return adapters


def is_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return ("out of memory" in msg) or ("cuda error: out of memory" in msg)


def main() -> None:
    args = parse_args()
    maybe_run_setup(args)

    device = pick_device(args.device)
    print(f"[infer] device={device}")

    input_parquet = Path(args.input_parquet)
    tar_dir = Path(args.tar_dir)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mapping = load_tar_mapping(input_parquet)
    loader = TarImageLoader(mapping=mapping, tar_dir=tar_dir)

    df = pd.read_parquet(input_parquet)
    if "image_id" not in df.columns:
        raise ValueError("input parquet missing image_id column")
    if args.max_images > 0:
        df = df.head(int(args.max_images))

    adapters = build_teacher_adapters(args=args, device=device)
    if not adapters:
        raise RuntimeError("no teacher adapters are ready (check setup/weights)")
    print(f"[infer] active teachers: {sorted(adapters.keys())}")

    with output_path.open("w", encoding="utf-8") as out_f:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="teacher-infer"):
            image_id = str(row["image_id"])
            image = loader.load_rgb(image_id)
            if image is None:
                for tid in sorted(adapters.keys()):
                    rec = {
                        "image_id": image_id,
                        "teacher_id": tid,
                        "image_size": None,
                        "num_proposals": 0,
                        "proposals": [],
                        "error": "image_load_failed",
                    }
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            w, h = image.size
            for tid, teacher in adapters.items():
                rec = {
                    "image_id": image_id,
                    "teacher_id": tid,
                    "image_size": [int(w), int(h)],
                    "num_proposals": 0,
                    "proposals": [],
                    "error": None,
                }
                try:
                    props = teacher.predict(image)
                    rec["proposals"] = props
                    rec["num_proposals"] = int(len(props))
                except Exception as e:
                    if bool(int(args.skip_on_oom)) and is_oom_error(e):
                        rec["error"] = f"oom_skipped:{type(e).__name__}"
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    else:
                        rec["error"] = f"{type(e).__name__}:{e}"
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    loader.close()
    print(f"[infer] done -> {output_path}")


if __name__ == "__main__":
    main()
