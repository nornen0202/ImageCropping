"""
extract_features_pipeline.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
멀티-GPU / Ray 기반 특징 추출 파이프라인.
GPU가 2개 이상이거나 대용량 데이터를 처리할 때 권장.

Usage 예시:
  # 전체 컴포넌트, 모든 GPU 사용
  python3 src/extract_features_pipeline.py \\
      --input_parquet data/SSTK/10K/filtered_sstk_100.parquet \\
      --bucket sstk_100 \\
      --tar_dir /sstk/20230916 \\
      --output_jsonl data/SSTK/10K/features_sstk_100.jsonl \\
      --component all

  # C2+C3만, 4-GPU, quality_first
  python3 src/extract_features_pipeline.py ... \\
      --component c2 c3 --num_workers 4 --priority quality_first
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd
import ray
from tqdm import tqdm

from worker_core import (
    FeatureWorker,
    _get_tags,
    iter_local_images,
    iter_tar_images,
    resolve_tar_path,
    resolve_weights_dir,
)


# ── Ray Remote 래퍼 (1 GPU / worker) ─────────────────────────────────────────
@ray.remote(num_gpus=1)
class RayFeatureWorker(FeatureWorker):
    """FeatureWorker를 Ray Actor로 실행하는 래퍼. 초기화 로직은 부모 클래스와 동일."""
    pass


# ── 인자 파싱 ─────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Feature Extraction Pipeline — Multi-GPU / Ray mode"
    )
    parser.add_argument("--input_parquet", required=True)
    parser.add_argument("--tar_dir",       required=True)
    parser.add_argument("--bucket",        default="sstk_100")
    parser.add_argument("--output_jsonl",  required=True)
    parser.add_argument(
        "--component",
        nargs="+",
        choices=["c1", "c2", "c3", "c4", "c5", "c6", "all"],
        default=["all"],
        help="Which components to run. e.g. --component c2 c3 c5  or  --component all",
    )
    parser.add_argument(
        "--priority",
        choices=["high_efficiency", "quality_first"],
        default="high_efficiency",
    )
    parser.add_argument("--batch_size",   type=int, default=16)
    parser.add_argument("--num_workers",  type=int, default=None,
                        help="Number of Ray workers (= GPUs to use). Default: auto-detect.")
    parser.add_argument("--weights_dir",  default="")
    parser.add_argument("--c4_lang",      default="en")
    parser.add_argument("--image_dir",    default="", help="optional local curated image dir (<image_id>.<ext>)")
    return parser.parse_args()


def main():
    args = parse_args()

    comps   = set(args.component)
    run_all = "all" in comps
    run_c1  = run_all or "c1" in comps
    run_c2  = run_all or "c2" in comps
    run_c3  = run_all or "c3" in comps
    run_c4  = run_all or "c4" in comps
    run_c5  = run_all or "c5" in comps
    run_c6  = run_all or "c6" in comps

    weights_dir = resolve_weights_dir(args.weights_dir, __file__)

    # ── Ray 초기화 ────────────────────────────────────────────────────────────
    ray.init()
    avail_gpus = int(ray.available_resources().get("GPU", 0))
    print(f"[Ray] Available GPUs: {avail_gpus}")

    num_workers = args.num_workers or max(avail_gpus, 1)
    print(f"[Ray] Launching {num_workers} worker(s). Components: {comps}  Priority: {args.priority}")
    if str(args.image_dir).strip():
        print(f"[Ray] Image dir: {args.image_dir}")

    workers = [
        RayFeatureWorker.remote(
            run_c1=run_c1,
            run_c2=run_c2,
            run_c3=run_c3,
            run_c4=run_c4,
            run_c5=run_c5,
            run_c6=run_c6,
            weights_dir=weights_dir,
            c4_lang=args.c4_lang,
            priority=args.priority,
        )
        for _ in range(num_workers)
    ]

    # ── 메타데이터 로드 ───────────────────────────────────────────────────────
    print(f"[Ray] Loading metadata: {args.input_parquet}")
    df = pd.read_parquet(args.input_parquet)
    required = {"image_id", "tar_name"}
    if not required.issubset(df.columns):
        raise ValueError(f"Input parquet missing columns: {required - set(df.columns)}")
    print(f"[Ray] Total images: {len(df)}")

    # ── 처리 ─────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)

    futures      = []
    worker_cycle = 0
    MAX_IN_FLIGHT = num_workers * 4

    with open(args.output_jsonl, "w") as out_f:

        def _flush(ready_futures):
            nonlocal futures
            for f in ready_futures:
                for r in ray.get(f):
                    out_f.write(json.dumps(r, ensure_ascii=False) + "\n")

        grouped = df.groupby("tar_name")
        for tar_name, group in tqdm(grouped, desc="Tars"):
            id_to_row = {str(r["image_id"]): r for _, r in group.iterrows()}
            wanted_ids = list(id_to_row.keys())
            id_to_img = {}
            if str(args.image_dir).strip():
                id_to_img = dict(iter_local_images(str(args.image_dir), wanted_ids))

            missing_ids = [iid for iid in wanted_ids if iid not in id_to_img]
            if missing_ids:
                tar_path = resolve_tar_path(args.tar_dir, args.bucket, tar_name)
                if tar_path is None:
                    if not id_to_img:
                        print(f"  [SKIP] tar not found: {tar_name}")
                        continue
                else:
                    id_to_img.update(dict(iter_tar_images(tar_path, missing_ids)))

            batch_all = []
            for img_id, img in id_to_img.items():
                row  = id_to_row.get(img_id, {})
                tags = _get_tags(row)
                batch_all.append((img_id, img, tags))

            for i in range(0, len(batch_all), args.batch_size):
                chunk  = batch_all[i : i + args.batch_size]
                worker = workers[worker_cycle % num_workers]
                futures.append(worker.process_batch.remote(chunk))
                worker_cycle += 1

                while len(futures) >= MAX_IN_FLIGHT:
                    ready, futures = ray.wait(futures, num_returns=1)
                    _flush(ready)

        # 잔여 futures 처리
        while futures:
            ready, futures = ray.wait(futures, num_returns=1)
            _flush(ready)

    print(f"[Ray] Feature Extraction completed → {args.output_jsonl}")
    ray.shutdown()


if __name__ == "__main__":
    main()
