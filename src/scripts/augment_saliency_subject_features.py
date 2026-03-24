#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable

from PIL import Image
from tqdm import tqdm

PROJECT_SRC = Path(__file__).resolve().parents[1]
EXTRACT_SRC = PROJECT_SRC / "extract_features"
for p in (PROJECT_SRC, EXTRACT_SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from c7_saliency import SaliencyFeatureExtractor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Augment an existing feature jsonl with c7_saliency.")
    p.add_argument("--input_jsonl", required=True)
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--image_dir", required=True)
    p.add_argument("--priority", choices=["high_efficiency", "quality_first"], default="quality_first")
    p.add_argument("--weights_dir", default="")
    p.add_argument("--device", default="auto")
    p.add_argument("--overwrite", type=int, default=0)
    p.add_argument("--max_images", type=int, default=0)
    p.add_argument("--progress", type=int, default=1)
    p.add_argument("--summary_json", default="")
    return p.parse_args()


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _index_images(image_dir: Path) -> Dict[str, Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    out: Dict[str, Path] = {}
    for path in image_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in exts:
            continue
        out[path.stem] = path
    return out


def main() -> None:
    args = parse_args()
    input_jsonl = Path(args.input_jsonl)
    output_jsonl = Path(args.output_jsonl)
    image_dir = Path(args.image_dir)
    summary_json = Path(args.summary_json) if str(args.summary_json).strip() else None

    if not input_jsonl.exists():
        raise FileNotFoundError(f"input_jsonl not found: {input_jsonl}")
    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir not found: {image_dir}")

    image_map = _index_images(image_dir)
    extractor = SaliencyFeatureExtractor(
        priority=str(args.priority),
        weights_dir=str(args.weights_dir),
        device=str(args.device),
    )

    backend_counter: Counter[str] = Counter()
    fallback_counter: Counter[str] = Counter()
    missing_images = 0
    reused = 0
    processed = 0
    started = time.time()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    try:
        iterable = read_jsonl(input_jsonl)
        if int(args.progress) != 0:
            iterable = tqdm(iterable, desc="augment-c7-saliency")
        with output_jsonl.open("w", encoding="utf-8") as fout:
            for idx, rec in enumerate(iterable):
                if int(args.max_images) > 0 and idx >= int(args.max_images):
                    break
                image_id = str(rec.get("image_id", ""))
                if not image_id:
                    continue
                if (not bool(int(args.overwrite))) and isinstance(rec.get("c7_saliency"), dict):
                    payload = rec.get("c7_saliency", {})
                    backend_counter[str(payload.get("backend", "existing"))] += 1
                    reused += 1
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    continue
                image_path = image_map.get(image_id)
                if image_path is None:
                    missing_images += 1
                    rec["c7_saliency"] = {
                        "requested_backend": "missing_image",
                        "backend": "missing_image",
                        "fallback_reason": "image_missing",
                        "available": False,
                    }
                else:
                    with Image.open(image_path) as image:
                        rec["c7_saliency"] = extractor.process_image(image.convert("RGB"))
                    payload = rec["c7_saliency"]
                    backend_counter[str(payload.get("backend", "unknown"))] += 1
                    fallback_counter[str(payload.get("fallback_reason", ""))] += 1
                    processed += 1
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    finally:
        extractor.close()

    summary = {
        "status": "ok",
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "image_dir": str(image_dir),
        "requested_priority": str(args.priority),
        "processed": int(processed),
        "reused": int(reused),
        "missing_images": int(missing_images),
        "backend_counts": dict(backend_counter),
        "fallback_counts": {k: v for k, v in dict(fallback_counter).items() if k},
        "elapsed_sec": round(time.time() - started, 3),
    }
    if summary_json is not None:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
