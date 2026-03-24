"""
extract_features_single.py
━━━━━━━━━━━━━━━━━━━━━━━━━━
단일 GPU, Ray 없이 동작하는 특징 추출 파이프라인.

해당 모드를 선택하는 기준
- GPU 1장이거나
- Ray 오버헤드(Actor 초기화 수십 초, IPC 직렬화 비용)가
  실제 처리 속도보다 커서 불리한 경우

Usage 예시:
  # C1만 추출 (high_efficiency)
  python3 src/extract_features_single.py \\
      --input_parquet data/SSTK/10K_local/filtered_sstk_100.parquet \\
      --bucket sstk_100 \\
      --tar_dir /path/to/tars \\
      --output_jsonl data/SSTK/10K_local/feats_c1.jsonl \\
      --component c1

  # C2+C3만 추출 (quality_first)
  python3 src/extract_features_single.py ... --component c2 c3 --priority quality_first

  # 전체 (all)
  python3 src/extract_features_single.py ... --component all
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import pandas as pd
from tqdm import tqdm

# worker_core는 extract_features/ 하위에 있으므로 PYTHONPATH에 포함되어야 함.
# run_extract_component.sh 가 자동으로 설정해 준다.
from worker_core import (
    FeatureWorker,
    _get_c1_text,
    _get_tags,
    iter_local_images,
    iter_tar_images,
    resolve_tar_path,
    resolve_weights_dir,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Feature Extraction — Single-GPU, No-Ray mode"
    )
    parser.add_argument("--input_parquet", required=True)
    parser.add_argument("--tar_dir",       required=True)
    parser.add_argument("--bucket",        default="sstk_100")
    parser.add_argument("--output_jsonl",  required=True)
    parser.add_argument(
        "--component",
        nargs="+",
        choices=["c1", "c2", "c3", "c4", "c5", "c6", "c7", "all"],
        default=["all"],
        help="Which components to run. e.g. --component c1 c2 c5  or  --component all",
    )
    parser.add_argument(
        "--priority",
        choices=["high_efficiency", "quality_first"],
        default="high_efficiency",
    )
    parser.add_argument("--batch_size",   type=int, default=16)
    parser.add_argument("--weights_dir",  default="", help="Directory containing C3 weights")
    parser.add_argument("--c4_lang",      default="en")
    parser.add_argument("--image_dir",    default="", help="optional local curated image dir (<image_id>.<ext>)")
    parser.add_argument("--num_shards",   type=int, default=1, help="Dataset shard count for no-ray multi-gpu")
    parser.add_argument("--shard_index",  type=int, default=0, help="Current shard index [0, num_shards)")
    return parser.parse_args()


def stable_shard_of(image_id: str, num_shards: int) -> int:
    h = hashlib.sha1(str(image_id).encode("utf-8")).hexdigest()
    return int(h[:8], 16) % max(1, int(num_shards))


def main():
    args = parse_args()

    # ── component 해석 ─────────────────────────────────────────────────
    comps = set(args.component)
    run_all = "all" in comps
    run_c1 = run_all or "c1" in comps
    run_c2 = run_all or "c2" in comps
    run_c3 = run_all or "c3" in comps
    run_c4 = run_all or "c4" in comps
    run_c5 = run_all or "c5" in comps
    run_c6 = run_all or "c6" in comps
    run_c7 = run_all or "c7" in comps

    weights_dir = resolve_weights_dir(args.weights_dir, __file__)
    print(f"[Single-GPU] Components: {comps}  Priority: {args.priority}")
    print(f"[Single-GPU] Weights dir: {weights_dir}")
    if str(args.image_dir).strip():
        print(f"[Single-GPU] Image dir: {args.image_dir}")

    # ── 모델 초기화 ────────────────────────────────────────────────────
    worker = FeatureWorker(
        run_c1=run_c1,
        run_c2=run_c2,
        run_c3=run_c3,
        run_c4=run_c4,
        run_c5=run_c5,
        run_c6=run_c6,
        run_c7=run_c7,
        weights_dir=weights_dir,
        c4_lang=args.c4_lang,
        priority=args.priority,
    )

    # ── 메타데이터 로드 ────────────────────────────────────────────────
    print(f"[Single-GPU] Loading metadata: {args.input_parquet}")
    df = pd.read_parquet(args.input_parquet)
    required = {"image_id", "tar_name"}
    if not required.issubset(df.columns):
        raise ValueError(f"Input parquet missing columns: {required - set(df.columns)}")
    print(f"[Single-GPU] Total images: {len(df)}")

    if int(args.num_shards) < 1:
        raise ValueError(f"--num_shards must be >= 1 (got {args.num_shards})")
    if int(args.shard_index) < 0 or int(args.shard_index) >= int(args.num_shards):
        raise ValueError(
            f"--shard_index must satisfy 0 <= shard_index < num_shards "
            f"(got shard_index={args.shard_index}, num_shards={args.num_shards})"
        )

    if int(args.num_shards) > 1:
        all_rows = len(df)
        sid = int(args.shard_index)
        nsh = int(args.num_shards)
        mask = df["image_id"].astype(str).map(lambda x: stable_shard_of(x, nsh) == sid)
        df = df[mask].copy()
        print(
            f"[Single-GPU] Shard mode: shard_index={sid}/{nsh} "
            f"rows={len(df)} (from total={all_rows})"
        )

    # ── 처리 ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)

    written = 0
    with open(args.output_jsonl, "w") as out_f:
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

            # (image_id, PIL.Image, tags, c1_text) 목록 구성
            batch_all = []
            for img_id, img in id_to_img.items():
                row  = id_to_row.get(img_id, {})
                tags = _get_tags(row)
                c1_text = _get_c1_text(row)
                batch_all.append((img_id, img, tags, c1_text))

            # 배치 단위로 처리
            for i in range(0, len(batch_all), args.batch_size):
                chunk = batch_all[i : i + args.batch_size]
                results = worker.process_batch(chunk)
                for r in results:
                    out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    written += 1

    print(f"[Single-GPU] Done. Written {written} records → {args.output_jsonl}")


if __name__ == "__main__":
    main()
