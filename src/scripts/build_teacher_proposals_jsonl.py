#!/usr/bin/env python
"""
Build canonical teacher_proposals_jsonl from raw public-teacher inference outputs.

Input (one or more JSONL files): each line should include at least
  - image_id
  - teacher_id
  - proposals: list of {bbox_norm_xyxy, score}

Output JSONL line format:
{
  "image_id": "...",
  "teacher_proposals": {
    "gaic": {
      "free_form": [{"bbox_norm_xyxy":[...], "score": 0.1}, ...],
      "by_ar": {}
    },
    ...
  }
}
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Dict, List


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def normalize_box(box: Any) -> List[float] | None:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = clamp(x1, 0.0, 1.0)
    y1 = clamp(y1, 0.0, 1.0)
    x2 = clamp(x2, 0.0, 1.0)
    y2 = clamp(y2, 0.0, 1.0)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if (x2 - x1) <= 1e-8 or (y2 - y1) <= 1e-8:
        return None
    return [round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build canonical teacher_proposals_jsonl")
    p.add_argument("--input_jsonl", nargs="+", required=True, help="raw inference output jsonl(s)")
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--topk_per_teacher", type=int, default=3)
    p.add_argument("--drop_error_rows", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    topk = max(1, int(args.topk_per_teacher))
    drop_error_rows = bool(int(args.drop_error_rows))

    agg: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))

    for p in [Path(x) for x in args.input_jsonl]:
        if not p.exists():
            print(f"[build][warn] missing input: {p}")
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                image_id = str(d.get("image_id", "")).strip()
                teacher_id = str(d.get("teacher_id", "")).strip().lower()
                if not image_id or not teacher_id:
                    continue

                err = d.get("error")
                if drop_error_rows and err not in (None, "", "null"):
                    continue

                proposals = d.get("proposals", [])
                if not isinstance(proposals, list):
                    continue
                for item in proposals:
                    if not isinstance(item, dict):
                        continue
                    b = item.get("bbox_norm_xyxy")
                    if b is None and "bbox_xyxy" in item and d.get("image_size"):
                        # optional fallback: convert px->norm if needed
                        im_sz = d.get("image_size")
                        if isinstance(im_sz, (list, tuple)) and len(im_sz) == 2:
                            w = max(float(im_sz[0]), 1.0)
                            h = max(float(im_sz[1]), 1.0)
                            bpx = item.get("bbox_xyxy")
                            if isinstance(bpx, (list, tuple)) and len(bpx) == 4:
                                b = [float(bpx[0]) / w, float(bpx[1]) / h, float(bpx[2]) / w, float(bpx[3]) / h]
                    bn = normalize_box(b)
                    if bn is None:
                        continue
                    score = float(item.get("score", 0.0))
                    agg[image_id][teacher_id].append(
                        {
                            "bbox_norm_xyxy": bn,
                            "score": score,
                        }
                    )

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as out_f:
        for image_id in sorted(agg.keys()):
            per_teacher = agg[image_id]
            teacher_block: Dict[str, Any] = {}
            for teacher_id in sorted(per_teacher.keys()):
                items = per_teacher[teacher_id]
                items = sorted(items, key=lambda x: float(x.get("score", 0.0)), reverse=True)
                items = items[:topk]
                teacher_block[teacher_id] = {
                    "free_form": items,
                    "by_ar": {},
                }
            if not teacher_block:
                continue
            rec = {
                "image_id": image_id,
                "teacher_proposals": teacher_block,
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    print(f"[build] done. written={written} output={output_path}")


if __name__ == "__main__":
    main()

