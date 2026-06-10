"""
Visualization utility for Teacher Scorer outputs.

Draws baseline + Top-K selected boxes per (image, target_ar) and stores an
overview JSON for quick sanity checks.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import tarfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image


TOPK_COLORS_BGR: List[Tuple[int, int, int]] = [
    (0, 0, 255),
    (0, 200, 0),
    (0, 165, 255),
    (255, 0, 255),
    (255, 255, 0),
]
BASELINE_COLOR = (255, 80, 30)
SUBJECT_COLOR = (255, 255, 0)
IMAGE_EXTS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize teacher scorer results")
    p.add_argument("--teacher_scores_jsonl", required=True)
    p.add_argument("--features_jsonl", default="", help="optional routed/merged features jsonl for subject mask overlay")
    p.add_argument("--parquet", required=True, help="filtered parquet with tar_name/bucket")
    p.add_argument("--tar_dir", required=True)
    p.add_argument("--image_dir", default="", help="optional local curated image dir (<image_id>.<ext>)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--target_ar", default="all", help="e.g. FREE,1:1 or all")
    p.add_argument("--decision_filter", default="all", help="all|keep_full|minimal_crop|crop")
    p.add_argument("--num_samples", type=int, default=100)
    p.add_argument(
        "--image_ids",
        nargs="*",
        default=[],
        help="explicit image ids (space/comma separated). if set, tasks are filtered by these ids",
    )
    p.add_argument("--image_ids_file", default="", help="text file with image_id entries (1 per line)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--draw_subject_box", type=int, default=1)
    p.add_argument("--draw_subject_mask", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0, help="CPU render workers (0=all cores, 1=single).")
    return p.parse_args()


def resolve_num_workers(requested_workers: int, num_items: int) -> int:
    if num_items <= 1:
        return 1
    req = int(requested_workers)
    if req == 1:
        return 1
    if req <= 0:
        req = max(1, int(os.cpu_count() or 1))
    return max(1, min(req, num_items))


def parse_image_ids_arg(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    for raw in values:
        if raw is None:
            continue
        toks = [tok.strip() for tok in str(raw).split(",")]
        for tok in toks:
            if tok:
                out.append(tok)
    return out


def parse_image_ids_file(path: str) -> List[str]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"image_ids_file not found: {path}")
    out: List[str] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.extend([tok.strip() for tok in s.split(",") if tok.strip()])
    return out


def unique_keep_order(ids: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in ids:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def find_image_in_dir(image_dir: str, image_id: str) -> Optional[Path]:
    if not image_dir:
        return None
    root = Path(image_dir)
    for ext in IMAGE_EXTS:
        p = root / f"{image_id}{ext}"
        if p.exists():
            return p
    return None


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def to_int_pt(x: float, y: float) -> Tuple[int, int]:
    return int(round(float(x))), int(round(float(y)))


def denorm_box(box_norm: Sequence[float], w: int, h: int) -> Tuple[int, int, int, int]:
    x1n, y1n, x2n, y2n = [float(v) for v in box_norm]
    x1 = int(round(clamp(x1n, 0.0, 1.0) * w))
    y1 = int(round(clamp(y1n, 0.0, 1.0) * h))
    x2 = int(round(clamp(x2n, 0.0, 1.0) * w))
    y2 = int(round(clamp(y2n, 0.0, 1.0) * h))
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def decode_rle(rle_string: str, shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    if not rle_string:
        return np.zeros((h, w), dtype=bool)
    try:
        runs = np.asarray([int(x) for x in str(rle_string).split()], dtype=np.int64)
    except Exception:
        return np.zeros((h, w), dtype=bool)
    if runs.size == 0 or (runs.size % 2) != 0:
        return np.zeros((h, w), dtype=bool)
    starts = runs[0::2] - 1
    lengths = runs[1::2]
    ends = starts + lengths
    flat = np.zeros(h * w, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        lo_i = int(max(0, min(lo, flat.size)))
        hi_i = int(max(0, min(hi, flat.size)))
        if hi_i > lo_i:
            flat[lo_i:hi_i] = 1
    return flat.reshape((h, w)).astype(bool)


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 1e-8:
        return 0.0
    return float(inter / denom)


def resolve_tar_path(tar_dir: str, bucket: str, tar_name: str) -> Optional[str]:
    if not tar_name:
        return None
    cand = [
        os.path.join(tar_dir, tar_name),
        os.path.join(tar_dir, str(bucket), tar_name) if bucket else None,
    ]
    for p in cand:
        if p and os.path.exists(p):
            return p
    return None


def draw_text_block(img: np.ndarray, lines: Sequence[str], x: int = 8, y: int = 20) -> None:
    if not lines:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.47
    th = 1
    line_h = 18
    max_w = 0
    for s in lines:
        (tw, _), _ = cv2.getTextSize(s, font, scale, th)
        max_w = max(max_w, tw)

    h_block = line_h * len(lines) + 8
    cv2.rectangle(img, (x - 6, y - 16), (x + max_w + 8, y - 16 + h_block), (0, 0, 0), -1)
    for i, s in enumerate(lines):
        cv2.putText(img, s, (x, y + i * line_h), font, scale, (230, 230, 230), th, cv2.LINE_AA)


def load_c2_map(features_jsonl: Path, wanted_ids: Sequence[str]) -> Dict[str, List[Dict[str, Any]]]:
    wanted = set(str(x) for x in wanted_ids)
    out: Dict[str, List[Dict[str, Any]]] = {}
    if not features_jsonl.exists():
        return out
    with features_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            image_id = str(rec.get("image_id", ""))
            if image_id not in wanted:
                continue
            c2 = rec.get("c2_seg", [])
            out[image_id] = c2 if isinstance(c2, list) else []
    return out


def build_subject_mask(
    c2_list: Sequence[Dict[str, Any]],
    subject_box_norm: Optional[Sequence[float]],
    w: int,
    h: int,
) -> Optional[np.ndarray]:
    if not isinstance(subject_box_norm, (list, tuple)) or len(subject_box_norm) != 4:
        return None
    subject_box = [
        clamp(float(subject_box_norm[0]), 0.0, 1.0) * float(w),
        clamp(float(subject_box_norm[1]), 0.0, 1.0) * float(h),
        clamp(float(subject_box_norm[2]), 0.0, 1.0) * float(w),
        clamp(float(subject_box_norm[3]), 0.0, 1.0) * float(h),
    ]
    selected: List[Tuple[float, np.ndarray]] = []
    for inst in c2_list:
        if not isinstance(inst, dict) or bool(inst.get("bg_like", False)):
            continue
        box = inst.get("box")
        mask_rle = inst.get("mask_rle")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        if not isinstance(mask_rle, str) or not mask_rle.strip():
            continue
        box_xyxy = [float(v) for v in box]
        cx = 0.5 * (box_xyxy[0] + box_xyxy[2])
        cy = 0.5 * (box_xyxy[1] + box_xyxy[3])
        center_inside = subject_box[0] <= cx <= subject_box[2] and subject_box[1] <= cy <= subject_box[3]
        if iou_xyxy(box_xyxy, subject_box) < 0.05 and not center_inside:
            continue
        mask = decode_rle(mask_rle, (h, w))
        if not np.any(mask):
            continue
        score = float(inst.get("importance_score", inst.get("score", 0.0)))
        selected.append((score, mask))
    if not selected:
        return None
    selected.sort(key=lambda item: item[0], reverse=True)
    union = np.zeros((h, w), dtype=bool)
    for _, mask in selected[:3]:
        union |= mask
    return union if np.any(union) else None


def draw_subject_mask_overlay(image_bgr: np.ndarray, subject_mask: np.ndarray) -> np.ndarray:
    out = image_bgr.copy()
    if subject_mask is None or not np.any(subject_mask):
        return out
    overlay = np.zeros_like(out)
    overlay[subject_mask] = SUBJECT_COLOR
    blended = cv2.addWeighted(out, 0.72, overlay, 0.28, 0.0)
    out[subject_mask] = blended[subject_mask]
    contours, _ = cv2.findContours(subject_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(out, contours, -1, SUBJECT_COLOR, 2, cv2.LINE_AA)
    return out


def draw_one(
    raw_bgr: np.ndarray,
    task: Dict[str, Any],
    draw_subject_box: bool,
    subject_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    out = raw_bgr.copy()
    h, w = out.shape[:2]

    image_id = task["image_id"]
    ar = task["target_ar"]
    decision = task["decision_type"]
    delta = float(task.get("delta_improve", 0.0))
    tau = float(task.get("tau_improve", 0.0))

    if subject_mask is not None:
        out = draw_subject_mask_overlay(out, subject_mask)

    if draw_subject_box and task.get("subject_box") is not None:
        x1, y1, x2, y2 = denorm_box(task["subject_box"], w, h)
        cv2.rectangle(out, (x1, y1), (x2, y2), SUBJECT_COLOR, 2)
        cv2.putText(out, "subject", (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, SUBJECT_COLOR, 1, cv2.LINE_AA)

    baseline = task.get("baseline")
    if isinstance(baseline, dict):
        b = baseline.get("bbox_norm_xyxy")
        if isinstance(b, (list, tuple)) and len(b) == 4:
            x1, y1, x2, y2 = denorm_box(b, w, h)
            cv2.rectangle(out, (x1, y1), (x2, y2), BASELINE_COLOR, 2)
            cv2.putText(
                out,
                f"baseline ({baseline.get('source','')})",
                (x1, min(h - 6, y2 + 14)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                BASELINE_COLOR,
                1,
                cv2.LINE_AA,
            )

    topk = task.get("topk", [])
    for i, c in enumerate(topk):
        b = c.get("bbox_norm_xyxy")
        if not isinstance(b, (list, tuple)) or len(b) != 4:
            continue
        x1, y1, x2, y2 = denorm_box(b, w, h)
        color = TOPK_COLORS_BGR[i % len(TOPK_COLORS_BGR)]
        thickness = 3 if i == 0 else 2
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        rank = int(c.get("rank", i + 1))
        score = float(c.get("scores", {}).get("final", 0.0))
        cid = str(c.get("candidate_id", ""))
        label = f"#{rank} s={score:.3f} {cid}"
        cv2.putText(out, label, (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 1, cv2.LINE_AA)

    top1 = topk[0] if topk else {}
    ccheck = top1.get("composition_checks", {}) if isinstance(top1, dict) else {}

    lines = [
        f"{image_id} | AR={ar} | decision={decision}",
        f"delta={delta:.4f} tau={tau:.4f}",
    ]
    if isinstance(top1, dict):
        lines.append(f"top1={top1.get('candidate_id','')} final={float(top1.get('scores',{}).get('final',0.0)):.4f}")
        comps = top1.get("scores", {}).get("components", {}) if isinstance(top1.get("scores"), dict) else {}
        a_raw = comps.get("aesthetic_raw")
        a_norm = comps.get("aesthetic_norm")
        cos_it = comps.get("cosine_img_text")
        src = comps.get("expensive_source")
        lines.append(
            f"A_raw={a_raw if a_raw is not None else 'n/a'} "
            f"A_norm={float(a_norm):.3f} cos={float(cos_it):+.3f} src={src}"
            if a_norm is not None and cos_it is not None
            else "A_raw=n/a A_norm=n/a cos=n/a"
        )

    if isinstance(ccheck, dict):
        head_ok = ccheck.get("headroom", {}).get("pass") if isinstance(ccheck.get("headroom"), dict) else None
        look_ok = ccheck.get("lookroom", {}).get("pass") if isinstance(ccheck.get("lookroom"), dict) else None
        hor_ok = ccheck.get("horizon", {}).get("pass") if isinstance(ccheck.get("horizon"), dict) else None
        ctx_ok = ccheck.get("context", {}).get("pass") if isinstance(ccheck.get("context"), dict) else None
        lines.append(f"headroom={head_ok} lookroom={look_ok} horizon={hor_ok} context={ctx_ok}")

    draw_text_block(out, lines)
    return out


def load_tasks(
    score_jsonl: Path,
    target_ar: str,
    decision_filter: str,
) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    with score_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            image_id = str(rec.get("image_id", ""))
            if not image_id:
                continue

            subject_box = None
            sp = rec.get("subject_prior")
            if isinstance(sp, dict):
                sb = sp.get("bbox_norm_xyxy")
                if isinstance(sb, (list, tuple)) and len(sb) == 4:
                    subject_box = [float(v) for v in sb]

            t = rec.get("teacher_scorer", {}) if isinstance(rec.get("teacher_scorer"), dict) else {}
            by_ar = t.get("results_by_ar", {}) if isinstance(t.get("results_by_ar"), dict) else {}
            for ar, ar_res in by_ar.items():
                if target_ar != "all" and str(ar) != str(target_ar):
                    continue
                decision = str(ar_res.get("decision", {}).get("decision_type", ""))
                if decision_filter != "all" and decision != decision_filter:
                    continue

                topk = ar_res.get("selected_topk", []) if isinstance(ar_res.get("selected_topk"), list) else []
                baseline = ar_res.get("baseline_candidate", {}) if isinstance(ar_res.get("baseline_candidate"), dict) else {}
                decision_info = ar_res.get("decision", {}) if isinstance(ar_res.get("decision"), dict) else {}

                tasks.append(
                    {
                        "image_id": image_id,
                        "target_ar": str(ar),
                        "decision_type": decision,
                        "delta_improve": float(decision_info.get("delta_improve", 0.0)),
                        "tau_improve": float(decision_info.get("tau_improve", 0.0)),
                        "topk": topk,
                        "baseline": baseline,
                        "subject_box": subject_box,
                    }
                )
    return tasks


def load_images_from_tars(
    ids: Sequence[str],
    mapping: Dict[str, Dict[str, Any]],
    tar_dir: str,
    image_dir: str = "",
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}

    pending: List[str] = []
    for image_id in ids:
        p = find_image_in_dir(image_dir, image_id)
        if p is not None:
            raw = cv2.imread(str(p))
            if raw is not None:
                out[image_id] = raw
                continue
        pending.append(image_id)

    tar_to_ids: Dict[str, List[str]] = defaultdict(list)
    for image_id in pending:
        info = mapping.get(image_id)
        if not info:
            continue
        tar_name = str(info.get("tar_name", ""))
        bucket = str(info.get("bucket", "")) if "bucket" in info else ""
        tar_path = resolve_tar_path(tar_dir, bucket, tar_name)
        if tar_path:
            tar_to_ids[tar_path].append(image_id)

    for tar_path, wanted_ids in tar_to_ids.items():
        wanted = set(wanted_ids)
        try:
            with tarfile.open(tar_path, "r|") as tf:
                for member in tf:
                    name = os.path.basename(member.name)
                    base, ext = os.path.splitext(name)
                    if ext.lower() not in IMAGE_EXTS:
                        continue
                    if base not in wanted:
                        continue
                    fobj = tf.extractfile(member)
                    if fobj is None:
                        wanted.remove(base)
                        if not wanted:
                            break
                        continue

                    arr = np.asarray(Image.open(io.BytesIO(fobj.read())).convert("RGB"))
                    out[base] = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                    wanted.remove(base)
                    if not wanted:
                        break
        except Exception as e:
            print(f"[warn] failed reading tar={tar_path}: {e}")

    return out


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    score_jsonl = Path(args.teacher_scores_jsonl)
    parquet_path = Path(args.parquet)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "by_ar").mkdir(parents=True, exist_ok=True)

    if not score_jsonl.exists():
        raise FileNotFoundError(f"teacher_scores_jsonl not found: {score_jsonl}")
    if not parquet_path.exists():
        raise FileNotFoundError(f"parquet not found: {parquet_path}")

    tasks = load_tasks(score_jsonl, target_ar=args.target_ar, decision_filter=args.decision_filter)
    if not tasks:
        raise RuntimeError("No tasks selected. Check --target_ar / --decision_filter")

    explicit_ids = unique_keep_order(parse_image_ids_arg(args.image_ids) + parse_image_ids_file(args.image_ids_file))
    if explicit_ids:
        allowed = set(explicit_ids)
        tasks = [t for t in tasks if str(t.get("image_id", "")) in allowed]
        if not tasks:
            raise RuntimeError("No tasks selected after --image_ids filter")

    if args.num_samples > 0 and len(tasks) > args.num_samples:
        tasks = random.sample(tasks, args.num_samples)

    df = pd.read_parquet(parquet_path)
    if "bucket" in df.columns:
        mapping = df.set_index("image_id")[["tar_name", "bucket"]].to_dict("index")
    else:
        mapping = df.set_index("image_id")[["tar_name"]].to_dict("index")

    image_ids = sorted(set(t["image_id"] for t in tasks))
    img_map = load_images_from_tars(ids=image_ids, mapping=mapping, tar_dir=args.tar_dir, image_dir=args.image_dir)
    c2_map: Dict[str, List[Dict[str, Any]]] = {}
    if int(args.draw_subject_mask) != 0 and args.features_jsonl:
        c2_map = load_c2_map(Path(args.features_jsonl), image_ids)
    subject_mask_cache: Dict[str, Optional[np.ndarray]] = {}
    for image_id in image_ids:
        subject_mask_cache[image_id] = None
        raw = img_map.get(image_id)
        if raw is None:
            continue
        if int(args.draw_subject_mask) != 0 and image_id in c2_map:
            subject_mask_cache[image_id] = build_subject_mask(
                c2_list=c2_map.get(image_id, []),
                subject_box_norm=next((t.get("subject_box") for t in tasks if str(t.get("image_id", "")) == image_id), None),
                w=raw.shape[1],
                h=raw.shape[0],
            )

    missing_images = 0
    render_workers = resolve_num_workers(int(args.num_workers), len(tasks))

    def render_task(task: Dict[str, Any]) -> Tuple[str, str]:
        image_id = task["image_id"]
        raw = img_map.get(image_id)
        if raw is None:
            raise FileNotFoundError(image_id)
        ar = task["target_ar"]
        ar_tok = ar.replace(":", "x")
        subdir = out_dir / "by_ar" / ar_tok
        subdir.mkdir(parents=True, exist_ok=True)
        vis = draw_one(
            raw_bgr=raw,
            task=task,
            draw_subject_box=int(args.draw_subject_box) != 0,
            subject_mask=subject_mask_cache.get(image_id),
        )
        out_path = subdir / f"{image_id}.jpg"
        cv2.imwrite(str(out_path), vis)
        return ar, str(task["decision_type"])

    rendered = 0
    by_ar_count: Dict[str, int] = defaultdict(int)
    by_decision_count: Dict[str, int] = defaultdict(int)
    if render_workers <= 1:
        for task in tasks:
            try:
                ar, decision_type = render_task(task)
            except FileNotFoundError:
                missing_images += 1
                continue
            rendered += 1
            by_ar_count[ar] += 1
            by_decision_count[decision_type] += 1
    else:
        with ThreadPoolExecutor(max_workers=render_workers) as executor:
            future_map = {executor.submit(render_task, task): task for task in tasks}
            for future in as_completed(future_map):
                task = future_map[future]
                try:
                    ar, decision_type = future.result()
                except FileNotFoundError:
                    missing_images += 1
                    continue
                rendered += 1
                by_ar_count[ar] += 1
                by_decision_count[decision_type] += 1

    overview = {
        "inputs": {
            "teacher_scores_jsonl": str(score_jsonl),
            "features_jsonl": str(args.features_jsonl),
            "parquet": str(parquet_path),
            "tar_dir": str(args.tar_dir),
            "image_dir": str(args.image_dir),
        },
        "filters": {
            "target_ar": args.target_ar,
            "decision_filter": args.decision_filter,
            "num_samples": args.num_samples,
            "explicit_image_ids": explicit_ids,
            "num_workers": render_workers,
        },
        "summary": {
            "tasks_selected": len(tasks),
            "rendered": rendered,
            "missing_images": missing_images,
            "by_ar": dict(sorted(by_ar_count.items())),
            "by_decision": dict(sorted(by_decision_count.items())),
        },
        "output_dir": str(out_dir),
    }

    with (out_dir / "viz_overview.json").open("w", encoding="utf-8") as f:
        json.dump(overview, f, ensure_ascii=False, indent=2)

    print(f"[done] rendered={rendered} out_dir={out_dir}")
    print(f"[done] overview={out_dir / 'viz_overview.json'}")


if __name__ == "__main__":
    main()
