#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

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
    p.add_argument("--priority", choices=["high_efficiency", "quality_first", "opencv", "opencv_only"], default="quality_first")
    p.add_argument("--weights_dir", default="")
    p.add_argument("--device", default="auto")
    p.add_argument("--overwrite", type=int, default=0)
    p.add_argument("--max_images", type=int, default=0)
    p.add_argument("--progress", type=int, default=1)
    p.add_argument("--summary_json", default="")
    p.add_argument("--multi_gpu", type=int, default=0)
    p.add_argument("--gpu_ids", default="")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
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


def parse_gpu_ids(gpu_ids: str) -> List[str]:
    return [part.strip() for part in str(gpu_ids or "").split(",") if part.strip()]


def compute_shard_range(total: int, shard_index: int, num_shards: int) -> Tuple[int, int]:
    if num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got {num_shards}")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"invalid shard_index={shard_index} for num_shards={num_shards}")
    base = total // num_shards
    extra = total % num_shards
    start = shard_index * base + min(shard_index, extra)
    end = start + base + (1 if shard_index < extra else 0)
    return start, end


def count_jsonl_rows(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def merge_counter_dicts(counters: Sequence[Dict[str, int]]) -> Dict[str, int]:
    merged: Counter[str] = Counter()
    for item in counters:
        merged.update({str(k): int(v) for k, v in item.items()})
    return dict(merged)


def build_summary(
    *,
    args: argparse.Namespace,
    processed: int,
    reused: int,
    missing_images: int,
    backend_counter: Counter[str],
    fallback_counter: Counter[str],
    elapsed_sec: float,
    shard_range: Tuple[int, int],
    total_rows: int,
) -> Dict[str, object]:
    return {
        "status": "ok",
        "input_jsonl": str(Path(args.input_jsonl)),
        "output_jsonl": str(Path(args.output_jsonl)),
        "image_dir": str(Path(args.image_dir)),
        "requested_priority": str(args.priority),
        "processed": int(processed),
        "reused": int(reused),
        "missing_images": int(missing_images),
        "backend_counts": dict(backend_counter),
        "fallback_counts": {k: v for k, v in dict(fallback_counter).items() if k},
        "elapsed_sec": round(float(elapsed_sec), 3),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "shard_range": {"start": int(shard_range[0]), "end": int(shard_range[1])},
        "total_input_rows": int(total_rows),
        "multi_gpu_requested": bool(int(args.multi_gpu) != 0),
    }


def write_summary(summary_json: Path | None, summary: Dict[str, object]) -> None:
    if summary_json is None:
        return
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def process_one_shard(args: argparse.Namespace) -> Dict[str, object]:
    input_jsonl = Path(args.input_jsonl)
    output_jsonl = Path(args.output_jsonl)
    image_dir = Path(args.image_dir)
    summary_json = Path(args.summary_json) if str(args.summary_json).strip() else None

    if not input_jsonl.exists():
        raise FileNotFoundError(f"input_jsonl not found: {input_jsonl}")
    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir not found: {image_dir}")

    total_rows = count_jsonl_rows(input_jsonl)
    effective_total_rows = min(total_rows, int(args.max_images)) if int(args.max_images) > 0 else total_rows
    shard_range = compute_shard_range(effective_total_rows, int(args.shard_index), int(args.num_shards))
    start_idx, end_idx = shard_range

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
        local_total = max(0, end_idx - start_idx)
        if int(args.progress) != 0:
            iterable = tqdm(iterable, desc=f"augment-c7-saliency[{args.shard_index}/{args.num_shards}]", total=local_total if local_total > 0 else None)
        with output_jsonl.open("w", encoding="utf-8") as fout:
            for rec_idx, rec in enumerate(iterable):
                if rec_idx < start_idx:
                    continue
                if rec_idx >= end_idx:
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

    summary = build_summary(
        args=args,
        processed=processed,
        reused=reused,
        missing_images=missing_images,
        backend_counter=backend_counter,
        fallback_counter=fallback_counter,
        elapsed_sec=time.time() - started,
        shard_range=shard_range,
        total_rows=effective_total_rows,
    )
    write_summary(summary_json, summary)
    return summary


def launch_multi_gpu(args: argparse.Namespace) -> Dict[str, object]:
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    if not gpu_ids:
        raise ValueError("--multi_gpu 1 requires --gpu_ids")
    workers = int(args.num_workers) if int(args.num_workers) > 0 else len(gpu_ids)
    if workers <= 1:
        one_args = argparse.Namespace(**vars(args))
        one_args.multi_gpu = 0
        one_args.num_shards = 1
        one_args.shard_index = 0
        return process_one_shard(one_args)

    output_jsonl = Path(args.output_jsonl)
    summary_json = Path(args.summary_json) if str(args.summary_json).strip() else None
    shard_dir = output_jsonl.parent / f"{output_jsonl.name}.shards.{os.getpid()}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    print(f"[multi] c7 saliency shard mode enabled: gpu_ids={','.join(gpu_ids)} workers={workers}")

    procs: List[Tuple[subprocess.Popen[str], Path]] = []
    shard_outputs: List[Path] = []
    shard_summaries: List[Path] = []
    shard_logs: List[Path] = []

    for i in range(workers):
        gpu_id = gpu_ids[i % len(gpu_ids)]
        shard_out = shard_dir / f"c7_saliency_shard_{i}.jsonl"
        shard_sum = shard_dir / f"c7_saliency_summary_shard_{i}.json"
        shard_log = shard_dir / f"c7_saliency_shard_{i}.log"
        shard_outputs.append(shard_out)
        shard_summaries.append(shard_sum)
        shard_logs.append(shard_log)

        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--input_jsonl",
            str(Path(args.input_jsonl)),
            "--output_jsonl",
            str(shard_out),
            "--image_dir",
            str(Path(args.image_dir)),
            "--priority",
            str(args.priority),
            "--device",
            "cuda:0",
            "--overwrite",
            str(int(args.overwrite)),
            "--max_images",
            str(int(args.max_images)),
            "--progress",
            "1" if i == 0 and int(args.progress) != 0 else "0",
            "--summary_json",
            str(shard_sum),
            "--multi_gpu",
            "0",
            "--num_shards",
            str(workers),
            "--shard_index",
            str(i),
        ]
        if str(args.weights_dir).strip():
            cmd.extend(["--weights_dir", str(args.weights_dir)])
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        print(f"[multi] launch shard={i}/{workers} gpu={gpu_id} -> {shard_out}")
        with shard_log.open("w", encoding="utf-8") as log_handle:
            proc = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT, text=True, env=env)
        procs.append((proc, shard_log))

    failed = False
    for proc, shard_log in procs:
        ret = proc.wait()
        if ret != 0:
            failed = True
            print(f"[error] c7 saliency shard failed: log={shard_log}", file=sys.stderr)
    if failed:
        for _, shard_log in procs:
            if shard_log.exists():
                print(f"----- tail: {shard_log} -----", file=sys.stderr)
                try:
                    tail = shard_log.read_text(encoding="utf-8").splitlines()[-80:]
                    for line in tail:
                        print(line, file=sys.stderr)
                except Exception:
                    pass
        raise RuntimeError(f"one or more c7 saliency shards failed. logs under: {shard_dir}")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as fout:
        for shard_out in shard_outputs:
            if shard_out.exists():
                with shard_out.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        fout.write(line)
    print(f"[multi] merged c7 saliency jsonl -> {output_jsonl}")

    shard_payloads: List[Dict[str, object]] = []
    for shard_sum in shard_summaries:
        if shard_sum.exists():
            shard_payloads.append(json.loads(shard_sum.read_text(encoding="utf-8")))

    summary = {
        "status": "ok",
        "input_jsonl": str(Path(args.input_jsonl)),
        "output_jsonl": str(output_jsonl),
        "image_dir": str(Path(args.image_dir)),
        "requested_priority": str(args.priority),
        "processed": int(sum(int(payload.get("processed", 0)) for payload in shard_payloads)),
        "reused": int(sum(int(payload.get("reused", 0)) for payload in shard_payloads)),
        "missing_images": int(sum(int(payload.get("missing_images", 0)) for payload in shard_payloads)),
        "backend_counts": merge_counter_dicts([payload.get("backend_counts", {}) for payload in shard_payloads]),
        "fallback_counts": merge_counter_dicts([payload.get("fallback_counts", {}) for payload in shard_payloads]),
        "elapsed_sec": round(sum(float(payload.get("elapsed_sec", 0.0)) for payload in shard_payloads), 3),
        "multi_gpu": {
            "enabled": True,
            "gpu_ids": gpu_ids,
            "workers": workers,
            "shard_dir": str(shard_dir),
        },
    }
    write_summary(summary_json, summary)
    return summary


def main() -> None:
    args = parse_args()
    enable_multi = int(args.multi_gpu) != 0
    if enable_multi:
        summary = launch_multi_gpu(args)
    else:
        summary = process_one_shard(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
