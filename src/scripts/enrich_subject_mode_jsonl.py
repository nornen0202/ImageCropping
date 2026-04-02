#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from tqdm import tqdm

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from routing.subject_mode_router import enrich_c2_topn, normalize_tags, route_subject_mode


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enrich precompute jsonl with C2 Top-N + subject-mode routing.")
    p.add_argument("--input_feats_jsonl", required=True)
    p.add_argument("--input_filtered_parquet", required=True)
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--caption_jsonl", default="")
    p.add_argument("--teacher_proposals_jsonl", default="")
    p.add_argument("--c2_top_n", type=int, default=5)
    p.add_argument("--c2_union_top_m", type=int, default=3)
    p.add_argument("--allow_det_proxy", type=int, default=1)
    p.add_argument("--progress", type=int, default=1)
    return p.parse_args()


def _load_meta_map(parquet_path: Path) -> Dict[str, Dict[str, Any]]:
    df = pd.read_parquet(parquet_path)
    needed = {"image_id", "width", "height"}
    if not needed.issubset(set(df.columns)):
        miss = sorted(needed - set(df.columns))
        raise ValueError(f"input parquet missing columns: {miss}")

    out: Dict[str, Dict[str, Any]] = {}
    for row in df.itertuples(index=False):
        image_id = str(getattr(row, "image_id"))
        out[image_id] = {
            "width": int(getattr(row, "width")),
            "height": int(getattr(row, "height")),
            "tags": getattr(row, "tags", None),
            "caption": getattr(row, "caption", ""),
            "super_cat": str(getattr(row, "super_cat", "")),
        }
    return out


def _load_caption_map(caption_jsonl: Path) -> Dict[str, Dict[str, Any]]:
    if not caption_jsonl.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    with caption_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            image_id = str(rec.get("image_id", "")).strip()
            if image_id:
                out[image_id] = rec
    return out


def _load_teacher_proposals_map(teacher_jsonl: Path) -> Dict[str, Dict[str, Any]]:
    if not teacher_jsonl.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    with teacher_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            image_id = str(rec.get("image_id", "")).strip()
            if image_id:
                block = rec.get("teacher_proposals", {})
                out[image_id] = block if isinstance(block, dict) else {}
    return out


def _first_int(rec: Dict[str, Any], keys: list[str], default: int = 0) -> int:
    for k in keys:
        if k in rec and rec.get(k) is not None:
            try:
                return int(float(rec.get(k)))
            except Exception:
                continue
    return int(default)


def _first_bool(rec: Dict[str, Any], keys: list[str], default: bool = False) -> bool:
    for k in keys:
        if k not in rec:
            continue
        v = rec.get(k)
        if isinstance(v, bool):
            return bool(v)
        s = str(v).strip().lower()
        if s in {"1", "true", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "no", "n", "off"}:
            return False
    return bool(default)


def _compute_blank_ratio_from_union(union_box: Any, width: int, height: int) -> float:
    if not isinstance(union_box, (list, tuple)) or len(union_box) != 4:
        return 0.0
    try:
        x1, y1, x2, y2 = [float(v) for v in union_box]
    except Exception:
        return 0.0
    w = max(1.0, float(width))
    h = max(1.0, float(height))
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ratio = max(0.0, min(1.0, area / (w * h)))
    return float(max(0.0, min(1.0, 1.0 - ratio)))


def _compute_blank_ratio_with_saliency(rec: Dict[str, Any], width: int, height: int) -> float:
    c7 = rec.get("c7_saliency", {}) if isinstance(rec.get("c7_saliency"), dict) else {}
    if bool(c7.get("available", False)):
        return float(
            max(
                0.0,
                min(
                    1.0,
                    c7.get(
                        "blank_ratio_est",
                        1.0 - float(c7.get("foreground_area_ratio", 0.0) or 0.0),
                    ),
                ),
            )
        )
    return _compute_blank_ratio_from_union(rec.get("c2_union_box_xyxy"), width, height)


def main() -> None:
    args = parse_args()
    input_jsonl = Path(args.input_feats_jsonl)
    input_parquet = Path(args.input_filtered_parquet)
    output_jsonl = Path(args.output_jsonl)
    caption_jsonl = Path(args.caption_jsonl).resolve() if str(args.caption_jsonl).strip() else None
    teacher_proposals_jsonl = (
        Path(args.teacher_proposals_jsonl).resolve() if str(args.teacher_proposals_jsonl).strip() else None
    )

    if not input_jsonl.exists():
        raise FileNotFoundError(f"input_feats_jsonl not found: {input_jsonl}")
    if not input_parquet.exists():
        raise FileNotFoundError(f"input_filtered_parquet not found: {input_parquet}")

    meta_map = _load_meta_map(input_parquet)
    caption_map = _load_caption_map(caption_jsonl) if caption_jsonl is not None else {}
    teacher_map = _load_teacher_proposals_map(teacher_proposals_jsonl) if teacher_proposals_jsonl is not None else {}
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    routed = 0
    missing_meta = 0
    mode_counter: Counter[str] = Counter()
    policy_counter: Counter[str] = Counter()

    with input_jsonl.open("r", encoding="utf-8") as fin, output_jsonl.open("w", encoding="utf-8") as fout:
        iterable = fin
        if int(args.progress) != 0:
            iterable = tqdm(fin, desc="subject-mode-enrich")
        for line in iterable:
            line = line.strip()
            if not line:
                continue
            total += 1
            rec = json.loads(line)
            image_id = str(rec.get("image_id", ""))
            if not image_id:
                continue

            meta = meta_map.get(image_id)
            if meta is None:
                missing_meta += 1
                width = int(rec.get("width", 1) or 1)
                height = int(rec.get("height", 1) or 1)
                tags = rec.get("tags", [])
                caption = str(rec.get("caption", "") or "")
                super_cat = ""
            else:
                width = int(meta["width"])
                height = int(meta["height"])
                tags = meta.get("tags", [])
                caption = str(meta.get("caption", "") or "")
                super_cat = str(meta.get("super_cat", ""))
            caption_rec = caption_map.get(image_id, {})
            if str(caption_rec.get("caption", "") or "").strip():
                caption = str(caption_rec.get("caption", "") or "")

            c2_payload = enrich_c2_topn(
                c2_seg=rec.get("c2_seg", []),
                c2_det=rec.get("c2_det", []),
                c3_pose=rec.get("c3_pose", []),
                width=width,
                height=height,
                top_n=max(1, int(args.c2_top_n)),
                union_top_m=max(1, int(args.c2_union_top_m)),
                allow_det_proxy=bool(int(args.allow_det_proxy)),
            )
            rec["c2_seg"] = c2_payload["instances"]
            rec["c2_primary_idx"] = c2_payload["c2_primary_idx"]
            rec["c2_union_box_xyxy"] = c2_payload["c2_union_box_xyxy"]
            rec["c2_topn"] = c2_payload["c2_topn"]
            rec["c2_stats"] = c2_payload["c2_stats"]

            tags_norm = normalize_tags(tags)
            ocr_text_boxes = _first_int(
                rec,
                [
                    "ocr_text_boxes",
                    "ocr_num_boxes",
                    "ocr_box_count",
                    "c4_text_boxes",
                    "c4_ocr_boxes",
                ],
                default=0,
            )
            text_overlay_likely = _first_bool(
                rec,
                [
                    "text_overlay_likely",
                    "has_text_overlay",
                    "ocr_text_overlay_likely",
                ],
                default=False,
            )
            copy_space_flag = _first_bool(
                rec,
                [
                    "copy_space",
                    "has_copy_space",
                    "special_flag_copy_space",
                ],
                default=False,
            )
            blank_ratio_est = _compute_blank_ratio_with_saliency(rec, width, height)
            c5_geom = rec.get("c5_geom", {}) if isinstance(rec.get("c5_geom"), dict) else {}
            c4_meta = rec.get("c4_ocr_meta", {}) if isinstance(rec.get("c4_ocr_meta"), dict) else {}
            horizon_conf = c5_geom.get("horizon_conf", 0.0)
            symmetry_score = c5_geom.get("symmetry_score", 0.0)
            routing = route_subject_mode(
                tags_norm=tags_norm,
                super_cat=super_cat,
                c3_pose=rec.get("c3_pose", []),
                c2_instances=rec.get("c2_seg", []),
                c2_union_box_xyxy=rec.get("c2_union_box_xyxy"),
                c2_primary_idx=int(rec.get("c2_primary_idx", -1)),
                width=width,
                height=height,
                c7_saliency=rec.get("c7_saliency", {}),
                ocr_text_boxes_count=ocr_text_boxes,
                text_overlay_likely=text_overlay_likely,
                copy_space_flag=copy_space_flag,
                blank_ratio_est=blank_ratio_est,
                horizon_conf=horizon_conf,
                symmetry_score=symmetry_score,
                ocr_backend_method=c4_meta.get("method"),
                caption_text=caption,
                public_teacher_proposals=teacher_map.get(image_id, {}),
            )
            rec["routing"] = routing

            mode = str(routing.get("subject_mode", "unknown"))
            policy = str(routing.get("policy_id", "unknown"))
            mode_counter[mode] += 1
            policy_counter[policy] += 1
            routed += 1

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(
        f"[done] total={total} routed={routed} missing_meta={missing_meta} "
        f"output={output_jsonl}"
    )
    print(f"[done] subject_mode_counts={dict(mode_counter)}")
    print(f"[done] policy_counts={dict(policy_counter)}")


if __name__ == "__main__":
    main()
