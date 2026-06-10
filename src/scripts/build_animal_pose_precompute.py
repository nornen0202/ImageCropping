#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from routing.routing_v2_simple import CAT_CLASS_ID, DOG_CLASS_ID, dogcat_evidence, iter_c2_instances, safe_float, safe_int  # noqa: E402


DEFAULT_PHASE_ROOT = Path("data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529")
DEFAULT_FEATURE_JSONL = DEFAULT_PHASE_ROOT / "artifacts/precompute/feats_c2c3c5_v2_strict_enriched_routed_final.jsonl"
DEFAULT_CHECKPOINT = Path("weights/mmpose/rtmpose-m_simcc-ap10k_pt-aic-coco_210e-256x256-7a041aa1_20230206.pth")
DEFAULT_CHECKPOINT_URLS = ",".join(
    [
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-ap10k_pt-aic-coco_210e-256x256-7a041aa1_20230206.pth",
        "https://huggingface.co/waveydaveygravy/animal_openpose/resolve/d82ce5b822d4168bf6cbdbb85d6417338ec8d71e/rtmpose-m_simcc-ap10k_pt-aic-coco_210e-256x256-7a041aa1_20230206.pth",
    ]
)
DEFAULT_CHECKPOINT_SHA256 = "896e3665d849ef7eb9b6ec0995955796cc9810f024fa0aa0bdc18acb0d68bf52"


def _default_config_path() -> Path:
    try:
        import mmpose

        return (
            Path(mmpose.__file__).resolve().parent
            / ".mim/configs/animal_2d_keypoint/rtmpose/ap10k/rtmpose-m_8xb64-210e_ap10k-256x256.py"
        )
    except Exception:
        return Path("configs/animal_2d_keypoint/rtmpose/ap10k/rtmpose-m_8xb64-210e_ap10k-256x256.py")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build dog/cat animal pose precompute for routing_v2_simple pet mode.")
    parser.add_argument("--feature_jsonl", type=Path, default=DEFAULT_FEATURE_JSONL)
    parser.add_argument("--image_root", type=Path, default=DEFAULT_PHASE_ROOT / "images")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=_default_config_path())
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--checkpoint_url", default=DEFAULT_CHECKPOINT_URLS, help="comma-separated URL fallback list")
    parser.add_argument("--checkpoint_sha256", default=DEFAULT_CHECKPOINT_SHA256)
    parser.add_argument("--download_checkpoint", action="store_true")
    parser.add_argument("--download_only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_images", type=int, default=300)
    parser.add_argument("--det_nms_iou", type=float, default=0.65)
    parser.add_argument("--det_nms_containment", type=float, default=0.85)
    parser.add_argument("--allow_proxy_fallback", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _image_path(image_root: Path, image_id: str) -> Path:
    for suffix in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = image_root / f"{image_id}{suffix}"
        if candidate.exists():
            return candidate
    return image_root / f"{image_id}.jpg"


def _dogcat_instances(row: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for idx, inst in enumerate(iter_c2_instances(row)):
        class_id = safe_int(inst.get("class_id"), -1)
        if class_id not in {CAT_CLASS_ID, DOG_CLASS_ID}:
            continue
        box = inst.get("box")
        if not isinstance(box, list) or len(box) < 4:
            continue
        out.append({"source_index": idx, "class_id": class_id, "species": "dog" if class_id == DOG_CLASS_ID else "cat", "bbox_xyxy": [float(v) for v in box[:4]], "score": safe_float(inst.get("score"), 0.0), "area_ratio": safe_float(inst.get("area_ratio"), 0.0)})
    return out


def _box_area(box: list[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def _box_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = _box_area(a) + _box_area(b) - inter
    return float(inter / union) if union > 0.0 else 0.0


def _box_containment(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    smaller = min(_box_area(a), _box_area(b))
    return float(inter / smaller) if smaller > 0.0 else 0.0


def _nms_dogcat_instances(
    instances: list[dict[str, Any]],
    *,
    iou_threshold: float,
    containment_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(
        instances,
        key=lambda inst: (
            -safe_float(inst.get("score"), 0.0),
            -safe_float(inst.get("area_ratio"), 0.0),
            safe_int(inst.get("source_index"), 0),
        ),
    )
    kept: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    for inst in ordered:
        duplicate_of: dict[str, Any] | None = None
        duplicate_reason = ""
        for keep in kept:
            iou = _box_iou(inst["bbox_xyxy"], keep["bbox_xyxy"])
            containment = _box_containment(inst["bbox_xyxy"], keep["bbox_xyxy"])
            if iou >= float(iou_threshold):
                duplicate_of = keep
                duplicate_reason = f"iou={iou:.3f}"
                break
            if containment >= float(containment_threshold):
                duplicate_of = keep
                duplicate_reason = f"containment={containment:.3f}"
                break
        if duplicate_of is None:
            kept.append(inst)
            continue
        dropped = dict(inst)
        dropped["suppressed_by_source_index"] = duplicate_of.get("source_index")
        dropped["suppress_reason"] = duplicate_reason
        suppressed.append(dropped)
    kept.sort(key=lambda inst: safe_int(inst.get("source_index"), 0))
    return kept, suppressed


def _proxy_pose(image_id: str, inst: dict[str, Any]) -> dict[str, Any]:
    x1, y1, x2, y2 = inst["bbox_xyxy"]
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    head = [x1 + 0.18 * w, y1 + 0.03 * h, x1 + 0.82 * w, y1 + 0.42 * h]
    eye_y = y1 + 0.18 * h
    nose_y = y1 + 0.28 * h
    left_eye = [x1 + 0.38 * w, eye_y, 0.35]
    right_eye = [x1 + 0.62 * w, eye_y, 0.35]
    nose = [x1 + 0.50 * w, nose_y, 0.40]
    conf = max(0.20, min(0.72, 0.35 + 0.35 * float(inst.get("score", 0.0))))
    return {
        "image_id": image_id,
        "instance_id": f"dogcat_{inst['source_index']}",
        "species_hint": inst["species"],
        "species_confidence": round(float(inst.get("score", 0.0)), 6),
        "bbox_xyxy": inst["bbox_xyxy"],
        "pose_model": "bbox_proxy_v1",
        "pose_available": False,
        "pose_confidence": round(conf, 6),
        "keypoints": {"left_eye": left_eye, "right_eye": right_eye, "nose": nose},
        "head_box_xyxy": [round(float(v), 3) for v in head],
        "body_box_xyxy": [round(float(v), 3) for v in inst["bbox_xyxy"]],
        "head_direction": "unknown",
        "motion_direction": "unknown",
        "visible_groups": {"head": round(conf, 6), "torso": round(conf * 0.75, 6), "front_paws": 0.0, "hind_paws": 0.0, "tail": 0.0},
        "notes": ["proxy_only_no_ap10k_checkpoint_or_inference_failed"],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_checkpoint_from_url(path: Path, url: str, *, status_path: Path | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    request = urllib.request.Request(url, headers={"User-Agent": "ImageCropping-routing-v2/1.0"})
    downloaded = 0
    last_status = 0.0
    with urllib.request.urlopen(request, timeout=30) as response, tmp.open("wb") as handle:
        total = int(response.headers.get("Content-Length") or 0)
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            now = time.time()
            if status_path is not None and (now - last_status >= 3.0 or downloaded == total):
                _write_json(
                    status_path,
                    {
                        "state": "running",
                        "phase": "download_checkpoint",
                        "checkpoint": str(path),
                        "tmp_path": str(tmp),
                        "downloaded_bytes": downloaded,
                        "total_bytes": total,
                        "downloaded_mb": round(downloaded / 1024.0 / 1024.0, 3),
                        "total_mb": round(total / 1024.0 / 1024.0, 3) if total else None,
                    },
                )
                last_status = now
    if tmp.stat().st_size <= 0:
        raise RuntimeError(f"downloaded checkpoint is empty: {tmp}")
    tmp.replace(path)


def _download_checkpoint(path: Path, urls_text: str, *, expected_sha256: str = "", status_path: Path | None = None) -> None:
    urls = [chunk.strip() for chunk in str(urls_text or "").split(",") if chunk.strip()]
    if not urls:
        raise ValueError("no checkpoint URL provided")
    errors: list[str] = []
    for idx, url in enumerate(urls, start=1):
        try:
            if status_path is not None:
                _write_json(status_path, {"state": "running", "phase": "download_checkpoint", "checkpoint": str(path), "url_index": idx, "url": url})
            _download_checkpoint_from_url(path, url, status_path=status_path)
            if expected_sha256:
                actual = _sha256(path)
                if actual.lower() != str(expected_sha256).strip().lower():
                    path.unlink(missing_ok=True)
                    raise RuntimeError(f"checkpoint sha256 mismatch: actual={actual} expected={expected_sha256}")
            return
        except Exception as exc:
            errors.append(f"{url}: {repr(exc)}")
            if status_path is not None:
                _write_json(
                    status_path,
                    {
                        "state": "running",
                        "phase": "download_checkpoint_retry",
                        "checkpoint": str(path),
                        "failed_url_index": idx,
                        "failed_url": url,
                        "error": repr(exc),
                    },
                )
    raise RuntimeError("all checkpoint downloads failed: " + " | ".join(errors))


def _init_mmpose_model(config: Path, checkpoint: Path, device: str) -> Any:
    _patch_transformers_for_mmpretrain()
    from mmpose.apis import init_model

    return init_model(str(config), str(checkpoint), device=device)


def _patch_transformers_for_mmpretrain() -> None:
    """Keep MMPretrain 1.2 importable with newer Transformers.

    MMPretrain's BLIP module imports a few helpers from
    transformers.modeling_utils. Transformers 4.57 moved them to
    transformers.pytorch_utils; the broad except in MMPretrain then nulls
    PreTrainedModel and breaks registry import even for non-BLIP backbones.
    """
    try:
        import transformers.modeling_utils as modeling_utils
        import transformers.pytorch_utils as pytorch_utils
    except Exception:
        return
    for name in ("apply_chunking_to_forward", "find_pruneable_heads_and_indices", "prune_linear_layer"):
        if not hasattr(modeling_utils, name) and hasattr(pytorch_utils, name):
            setattr(modeling_utils, name, getattr(pytorch_utils, name))


def _mmpose_instance(model: Any, image_path: Path, inst: dict[str, Any], image_id: str) -> dict[str, Any]:
    from mmpose.apis import inference_topdown

    bbox = np.asarray([inst["bbox_xyxy"]], dtype=np.float32)
    results = inference_topdown(model, str(image_path), bboxes=bbox, bbox_format="xyxy")
    if not results:
        return _proxy_pose(image_id, inst)
    sample = results[0]
    pred = getattr(sample, "pred_instances", None)
    keypoints = getattr(pred, "keypoints", None)
    scores = getattr(pred, "keypoint_scores", None)
    if keypoints is None:
        return _proxy_pose(image_id, inst)
    kps = np.asarray(keypoints)[0]
    ksc = np.asarray(scores)[0] if scores is not None else np.ones((kps.shape[0],), dtype=np.float32)
    names = [
        "left_eye",
        "right_eye",
        "nose",
        "neck",
        "root_of_tail",
        "left_shoulder",
        "left_elbow",
        "left_front_paw",
        "right_shoulder",
        "right_elbow",
        "right_front_paw",
        "left_hip",
        "left_knee",
        "left_back_paw",
        "right_hip",
        "right_knee",
        "right_back_paw",
    ]
    kp_dict = {}
    for idx, name in enumerate(names[: kps.shape[0]]):
        kp_dict[name] = [round(float(kps[idx, 0]), 3), round(float(kps[idx, 1]), 3), round(float(ksc[idx]), 6)]
    head_points = [kp_dict[name] for name in ("left_eye", "right_eye", "nose") if name in kp_dict and kp_dict[name][2] >= 0.10]
    if head_points:
        xs = [p[0] for p in head_points]
        ys = [p[1] for p in head_points]
        pad = 0.18 * max(1.0, inst["bbox_xyxy"][2] - inst["bbox_xyxy"][0])
        head_box = [min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad]
    else:
        head_box = _proxy_pose(image_id, inst)["head_box_xyxy"]
    pose_conf = float(np.mean(ksc)) if ksc.size else 0.0
    return {
        "image_id": image_id,
        "instance_id": f"dogcat_{inst['source_index']}",
        "species_hint": inst["species"],
        "species_confidence": round(float(inst.get("score", 0.0)), 6),
        "bbox_xyxy": inst["bbox_xyxy"],
        "pose_model": "mmpose_rtmpose_ap10k",
        "pose_available": True,
        "pose_confidence": round(pose_conf, 6),
        "keypoints": kp_dict,
        "head_box_xyxy": [round(float(v), 3) for v in head_box],
        "body_box_xyxy": [round(float(v), 3) for v in inst["bbox_xyxy"]],
        "head_direction": "unknown",
        "motion_direction": "unknown",
        "visible_groups": {
            "head": round(float(np.mean([kp_dict[k][2] for k in ("left_eye", "right_eye", "nose") if k in kp_dict])) if any(k in kp_dict for k in ("left_eye", "right_eye", "nose")) else 0.0, 6),
            "torso": round(float(np.mean([kp_dict[k][2] for k in ("left_shoulder", "right_shoulder", "left_hip", "right_hip") if k in kp_dict])) if any(k in kp_dict for k in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")) else 0.0, 6),
            "front_paws": round(float(np.mean([kp_dict[k][2] for k in ("left_front_paw", "right_front_paw") if k in kp_dict])) if any(k in kp_dict for k in ("left_front_paw", "right_front_paw")) else 0.0, 6),
            "hind_paws": round(float(np.mean([kp_dict[k][2] for k in ("left_back_paw", "right_back_paw") if k in kp_dict])) if any(k in kp_dict for k in ("left_back_paw", "right_back_paw")) else 0.0, 6),
            "tail": round(float(kp_dict.get("root_of_tail", [0, 0, 0])[2]), 6),
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "status.json"
    start = time.time()
    _write_json(status_path, {"state": "running", "phase": "prepare"})

    if args.download_checkpoint and not args.checkpoint.exists():
        _write_json(status_path, {"state": "running", "phase": "download_checkpoint", "checkpoint": str(args.checkpoint)})
        _download_checkpoint(
            args.checkpoint,
            args.checkpoint_url,
            expected_sha256=str(args.checkpoint_sha256 or ""),
            status_path=status_path,
        )
    if bool(args.download_only):
        summary = {
            "state": "completed",
            "phase": "download_only",
            "checkpoint": str(args.checkpoint),
            "checkpoint_exists": args.checkpoint.exists(),
            "checkpoint_size_bytes": args.checkpoint.stat().st_size if args.checkpoint.exists() else 0,
            "checkpoint_sha256": _sha256(args.checkpoint) if args.checkpoint.exists() else "",
            "elapsed_sec": round(time.time() - start, 3),
        }
        _write_json(args.output_dir / "summary.json", summary)
        _write_json(status_path, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    model = None
    model_error = ""
    if args.checkpoint.exists() and args.config.exists():
        try:
            _write_json(status_path, {"state": "running", "phase": "init_mmpose", "checkpoint": str(args.checkpoint)})
            model = _init_mmpose_model(args.config, args.checkpoint, args.device)
        except Exception as exc:  # pragma: no cover - environment dependent
            model_error = repr(exc)
            if not args.allow_proxy_fallback:
                raise

    rows_written = 0
    images_seen = 0
    raw_instances_seen = 0
    nms_suppressed_instances = 0
    pose_counter: Counter[str] = Counter()
    out_jsonl = args.output_dir / "animal_pose.jsonl"
    with args.feature_jsonl.open("r", encoding="utf-8") as src, out_jsonl.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            image_id = str(row.get("image_id") or "")
            if not dogcat_evidence(row)["dogcat_present"]:
                continue
            raw_instances = _dogcat_instances(row)
            instances, suppressed_instances = _nms_dogcat_instances(
                raw_instances,
                iou_threshold=float(args.det_nms_iou),
                containment_threshold=float(args.det_nms_containment),
            )
            if not instances:
                continue
            raw_instances_seen += len(raw_instances)
            nms_suppressed_instances += len(suppressed_instances)
            images_seen += 1
            image_path = _image_path(args.image_root, image_id)
            for inst in instances:
                try:
                    pose = _mmpose_instance(model, image_path, inst, image_id) if model is not None and image_path.exists() else _proxy_pose(image_id, inst)
                except Exception as exc:  # pragma: no cover - model/runtime dependent
                    pose = _proxy_pose(image_id, inst)
                    pose["notes"].append(f"mmpose_error:{repr(exc)[:200]}")
                dst.write(json.dumps(pose, ensure_ascii=False, sort_keys=True) + "\n")
                rows_written += 1
                pose_counter[str(pose.get("pose_model"))] += 1
                pose_counter[f"species:{pose.get('species_hint')}"] += 1
                pose_counter[f"pose_available:{bool(pose.get('pose_available'))}"] += 1
            if images_seen % 25 == 0:
                _write_json(
                    status_path,
                    {
                        "state": "running",
                        "phase": "precompute",
                        "images_seen": images_seen,
                        "raw_instances_seen": raw_instances_seen,
                        "nms_suppressed_instances": nms_suppressed_instances,
                        "rows_written": rows_written,
                        "pose_counter": dict(pose_counter),
                    },
                )
            if int(args.max_images) > 0 and images_seen >= int(args.max_images):
                break

    summary = {
        "state": "completed",
        "phase": "completed",
        "feature_jsonl": str(args.feature_jsonl),
        "image_root": str(args.image_root),
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "checkpoint_exists": args.checkpoint.exists(),
        "checkpoint_sha256": _sha256(args.checkpoint) if args.checkpoint.exists() else "",
        "model_error": model_error,
        "images_seen": images_seen,
        "raw_instances_seen": raw_instances_seen,
        "nms_suppressed_instances": nms_suppressed_instances,
        "det_nms_iou": float(args.det_nms_iou),
        "det_nms_containment": float(args.det_nms_containment),
        "rows_written": rows_written,
        "pose_counter": dict(pose_counter),
        "animal_pose_jsonl": str(out_jsonl),
        "elapsed_sec": round(time.time() - start, 3),
    }
    _write_json(args.output_dir / "summary.json", summary)
    _write_json(status_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
