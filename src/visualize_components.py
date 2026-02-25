'''
## Local
python3 src/visualize_components.py \
    --parquet data/SSTK/10K_local/filtered_sstk_100.parquet \
    --c2_jsonl data/SSTK/10K_local/feats_c2.jsonl \
    --c3_jsonl data/SSTK/10K_local/feats_c3.jsonl \
    --tar_dir /media/jyju25/T7_4TB_JY/Projects_26/Dataset/SSTK/20230916/sstk_100 \
    --out_dir data/SSTK/10K_local/visualizations \
    --num_samples 50

## Space
python3 src/visualize_components.py \
    --parquet data/SSTK/10K_local/filtered_sstk_100.parquet \
    --c2_jsonl data/SSTK/10K_local/feats_c2.jsonl \
    --c3_jsonl data/SSTK/10K_local/feats_c3.jsonl \
    --tar_dir /sstk/20230916/sstk_100 \
    --out_dir data/SSTK/10K_local/visualizations \
    --num_samples 50
'''

import os
import json
import argparse
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import tarfile
import io
from collections import defaultdict

def decode_rle(rle_string, shape):
    runs = np.array([int(x) for x in rle_string.split()])
    if len(runs) % 2 != 0:
        return np.zeros(shape, dtype=bool)
    starts = runs[::2] - 1
    lengths = runs[1::2]
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).astype(bool)

def resolve_tar_path(tar_dir: str, bucket: str, tar_name: str):
    import math
    if pd.isna(bucket): bucket = ""
    paths = [
        os.path.join(tar_dir, tar_name),
        os.path.join(tar_dir, str(bucket), tar_name) if bucket else None,
    ]
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

def process_visualizations(basename, raw_cv2, c2_feats, c3_feats, args):
    h, w = raw_cv2.shape[:2]
    # C2
    c2 = c2_feats.get(basename, [])
    if c2:
        vis_c2 = raw_cv2.copy()
        overlay = np.zeros_like(vis_c2)
        for obj in c2:
            box = obj['box']
            score = obj['score']
            if isinstance(score, list): score = score[0]
            score = float(score)
            mask_rle = obj['mask_rle']
            mask = decode_rle(mask_rle, (h, w))
            
            color = np.random.randint(0, 255, (3,), dtype=np.uint8).tolist()
            overlay[mask] = color
            
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(vis_c2, p1, p2, color, 2)
            cv2.putText(vis_c2, f"Score:{score:.2f}", (p1[0], max(0, p1[1] - 5)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        mask_bool = np.any(overlay != 0, axis=-1)
        if np.any(mask_bool):
            vis_c2[mask_bool] = cv2.addWeighted(vis_c2, 0.5, overlay, 0.5, 0)[mask_bool]
        cv2.imwrite(os.path.join(args.out_dir, "c2_seg", f"{basename}.jpg"), vis_c2)
        
    # C3
    c3 = c3_feats.get(basename, [])
    if c3:
        vis_c3 = raw_cv2.copy()
        for obj in c3:
            box = obj['bbox']
            score = obj.get('score', 0.0)
            if isinstance(score, list): score = score[0]
            score = float(score)
            kpts = obj.get('keypoints', [])
            
            color = (0, 255, 0)
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(vis_c3, p1, p2, color, 2)
            cv2.putText(vis_c3, f"{score:.2f}", (p1[0], max(0, p1[1] - 5)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            if len(kpts) > 0:
                for kpt in kpts:
                    x, y, s = kpt
                    if s > 0.05:
                        cv2.circle(vis_c3, (int(x), int(y)), 3, (0, 0, 255), -1)
        cv2.imwrite(os.path.join(args.out_dir, "c3_pose", f"{basename}.jpg"), vis_c3)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=str, default="data/SSTK/10K_local/filtered_sstk_100.parquet")
    parser.add_argument("--c1_jsonl", type=str, default="data/SSTK/10K_local/feats_c1.jsonl")
    parser.add_argument("--c2_jsonl", type=str, default="data/SSTK/10K_local/feats_c2.jsonl")
    parser.add_argument("--c3_jsonl", type=str, default="data/SSTK/10K_local/feats_c3.jsonl")
    parser.add_argument("--tar_dir", type=str, default="/media/jyju25/T7_4TB_JY/Projects_26/Dataset/SSTK/20230916/sstk_100")
    parser.add_argument("--out_dir", type=str, default="data/SSTK/10K_local/visualizations")
    parser.add_argument("--num_samples", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out_dir, "original"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "c2_seg"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "c3_pose"), exist_ok=True)

    print("Loading parquet...")
    df = pd.read_parquet(args.parquet)
    if 'bucket' in df.columns:
        mapping = df.set_index('image_id')[['tar_name', 'bucket']].to_dict('index')
    else:
        mapping = df.set_index('image_id')[['tar_name']].to_dict('index')

    print("Loading jsonl features...")
    c2_feats = {}
    if os.path.exists(args.c2_jsonl):
        with open(args.c2_jsonl, 'r') as f:
            for line in f:
                d = json.loads(line)
                c2_feats[d['image_id']] = d.get('c2_seg', [])

    c3_feats = {}
    if os.path.exists(args.c3_jsonl):
        with open(args.c3_jsonl, 'r') as f:
            for line in f:
                d = json.loads(line)
                c3_feats[d['image_id']] = d.get('c3_pose', [])

    print("Selecting samples...")
    sample_ids = []
    all_iids = set(c2_feats.keys()).union(c3_feats.keys())
    for iid in all_iids:
        if len(c2_feats.get(iid, [])) > 0 or len(c3_feats.get(iid, [])) > 0:
            sample_ids.append(iid)
            if len(sample_ids) >= args.num_samples:
                break
    
    print(f"Processing {len(sample_ids)} samples...")

    tar_to_ids = defaultdict(list)
    for iid in sample_ids:
        orig_path = os.path.join(args.out_dir, "original", f"{iid}.jpg")
        
        # If original image already exists locally, process C2/C3 without tar extraction.
        if os.path.exists(orig_path):
            raw_cv2 = cv2.imread(orig_path)
            if raw_cv2 is not None:
                process_visualizations(iid, raw_cv2, c2_feats, c3_feats, args)
        else:
            if iid not in mapping: continue
            info = mapping[iid]
            tar_name = info['tar_name']
            bucket = info.get('bucket', '')
            tar_path = resolve_tar_path(args.tar_dir, bucket, tar_name)
            if tar_path:
                tar_to_ids[tar_path].append(iid)

    for tar_path, iids in tar_to_ids.items():
        print(f"Opening {tar_path} for {len(iids)} images...")
        id_set = set(iids)
        
        try:
            with tarfile.open(tar_path, "r|") as tf:
                for member in tf:
                    name = os.path.basename(member.name)
                    basename, ext = os.path.splitext(name)
                    if ext.lower() not in ['.jpg', '.jpeg', '.png', '.webp']:
                        continue
                    if basename in id_set:
                        f = tf.extractfile(member)
                        if f:
                            img = Image.open(io.BytesIO(f.read())).convert("RGB")
                            raw_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                            
                            # Original
                            orig_path = os.path.join(args.out_dir, "original", f"{basename}.jpg")
                            cv2.imwrite(orig_path, raw_cv2)
                            
                            # C2, C3 Processing
                            process_visualizations(basename, raw_cv2, c2_feats, c3_feats, args)
                                
                        id_set.remove(basename)
                        if not id_set:
                            break
        except Exception as e:
            print(f"Error reading tar {tar_path}: {e}")

    print("Visualization complete.")

if __name__ == "__main__":
    main()
