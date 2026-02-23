import os
import argparse
import pandas as pd
import ray
import tarfile
import io
import json
from PIL import Image
from tqdm import tqdm

from c1_clip import ClipFeatureExtractor
from c2_seg import SegFeatureExtractor
from c3_pose import PoseFeatureExtractor
from c4_ocr import OcrFeatureExtractor

@ray.remote(num_gpus=1)
class FeatureExtractionWorker:
    def __init__(self, run_c1=True, run_c2=True, run_c3=True, run_c4=True, c3_ckpt_dir="", c4_lang="en", priority="high_efficiency"):
        self.run_c1 = run_c1
        self.run_c2 = run_c2
        self.run_c3 = run_c3
        self.run_c4 = run_c4
        self.priority = priority
        
        # Load models
        print(f"Initializing Worker models with priority='{self.priority}'...")
        self.c1 = ClipFeatureExtractor(priority=self.priority) if run_c1 else None
        self.c2 = SegFeatureExtractor(priority=self.priority) if run_c2 else None
        
        # Paths for det/pose configs & weights (Dummy/Defaults suitable for testing, modify for actual OpenMMLab paths)
        if run_c3:
            det_cfg = os.path.join(c3_ckpt_dir, "scrfd_10g_bnkps.py")
            det_w = os.path.join(c3_ckpt_dir, "scrfd_10g_bnkps.pth")
            pose_cfg = os.path.join(c3_ckpt_dir, "td-hm_vitpose-base-simple_8xb64-210e_coco-256x192.py")
            pose_w = os.path.join(c3_ckpt_dir, "vitpose-b-multi-coco.pth")
            
            # Simple fallback check to dodge errors if files missing for now (testing)
            if os.path.exists(det_cfg) and os.path.exists(pose_w):
                self.c3 = PoseFeatureExtractor(det_cfg, det_w, pose_cfg, pose_w, priority=self.priority)
            else:
                print(f"Pose ckpts not found at {c3_ckpt_dir}. Skipping C3 loading.")
                self.c3 = None
        else:
            self.c3 = None
            
        self.c4 = OcrFeatureExtractor(lang=c4_lang, priority=self.priority) if run_c4 else None

    def process_batch(self, batch_data):
        """
        batch_data: list of tuples (image_id, PIL.Image, tags_list)
        Returns list of feature dicts mapping 1:1 to batch_data
        """
        results = []
        
        # C1 Batched 
        if self.run_c1 and self.c1:
            images = [item[1] for item in batch_data]
            c1_img_features = self.c1.encode_images(images) # N x Dim
            c1_txt_features = self.c1.encode_texts([", ".join(item[2]) for item in batch_data])
            
        for i, (img_id, img, tags) in enumerate(batch_data):
            res_dict = {'image_id': img_id}
            
            # C1 Merge
            if self.run_c1 and self.c1:
                res_dict['c1_img_embed'] = c1_img_features[i].tolist() # Convert FP16 array back to list for JSON/Parquet safety
                res_dict['c1_txt_embed'] = c1_txt_features[i].tolist()
                
            # C2 Seg
            if self.run_c2 and self.c2:
                res_dict['c2_seg'] = self.c2.process_image(img)
                
            # C3 Pose (Conditional internally)
            if self.run_c3 and self.c3:
                res_dict['c3_pose'] = self.c3.process_image(img, tags=tags)
                
            # C4 OCR (Conditional internally)
            if self.run_c4 and self.c4:
                res_dict['c4_ocr'] = self.c4.process_image(img, tags=tags)
                
            results.append(res_dict)
            
        return results

def main():
    parser = argparse.ArgumentParser(description="Feature Extraction Pipeline (Ray Manager)")
    parser.add_argument("--input_parquet", required=True, help="Input metadata parquet file (e.g. filtered_sstk_100.parquet)")
    parser.add_argument("--tar_dir", type=str, default="/media/jyju25/T7_4TB_JY/Projects_26/Dataset/SSTK/tars", help="Base directory for image tarballs")
    parser.add_argument("--bucket", type=str, default="sstk_100", help="Bucket name")
    parser.add_argument("--output_jsonl", required=True, help="Output features file")
    
    # Feature selection
    parser.add_argument("--skip_c1", action='store_true')
    parser.add_argument("--skip_c2", action='store_true')
    parser.add_argument("--skip_c3", action='store_true')
    parser.add_argument("--skip_c4", action='store_true')
    parser.add_argument("--priority", type=str, choices=["high_efficiency", "quality_first"], default="high_efficiency", help="Feature extraction priority mode")
    
    # Batch / Workers
    parser.add_argument("--batch_size", type=int, default=16, help="Worker batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of Ray GPU Workers")
    
    args = parser.parse_args()
    
    ray.init()
    print(f"Started Ray. Available GPUs: {ray.available_resources().get('GPU', 0)}")
    
    # Instantiate Worker Pool
    workers = [
        FeatureExtractionWorker.remote(
            run_c1=not args.skip_c1,
            run_c2=not args.skip_c2,
            run_c3=not args.skip_c3,
            run_c4=not args.skip_c4,
            c3_ckpt_dir=os.path.join(os.path.dirname(__file__), "weights"),
            priority=args.priority
        ) for _ in range(args.num_workers)
    ]
    
    print(f"Loading Metadata from {args.input_parquet}...")
    df = pd.read_parquet(args.input_parquet)
    
    if 'tags' not in df.columns or 'tar_name' not in df.columns or 'image_id' not in df.columns:
        raise ValueError("Input parquet missing required columns (tags, tar_name, image_id).")
        
    print(f"Total entries to process: {len(df)}")
    
    # Group by tar_name to optimize I/O
    grouped = df.groupby('tar_name')
    
    out_f = open(args.output_jsonl, 'w')
    
    futures = []
    worker_cycle = 0
    
    for tar_name, group in tqdm(grouped, desc="Processing Tars"):
        tar_path = os.path.join(args.tar_dir, args.bucket, tar_name)
        if not os.path.exists(tar_path):
            print(f"Missing tar: {tar_path}")
            continue
            
        # Extract images from this Tar
        images_batch = []
        try:
            with tarfile.open(tar_path, 'r') as tf:
                members = tf.getmembers()
                member_dict = {m.name.split('/')[-1]: m for m in members}
                
                for _, row in group.iterrows():
                    img_id = row['image_id']
                    jpg_name = f"{img_id}.jpg"
                    if jpg_name in member_dict:
                        f = tf.extractfile(member_dict[jpg_name])
                        if f:
                            img = Image.open(io.BytesIO(f.read())).convert("RGB")
                            images_batch.append((img_id, img, list(row['tags'])))
        except Exception as e:
            print(f"Error reading {tar_path}: {e}")
            continue
            
        # Dispatch in chunks to ray workers
        for i in range(0, len(images_batch), args.batch_size):
            chunk = images_batch[i:i + args.batch_size]
            worker_id = workers[worker_cycle % args.num_workers]
            future = worker_id.process_batch.remote(chunk)
            futures.append(future)
            worker_cycle += 1
            
            # Limit in-flight futures
            MAX_IN_FLIGHT = args.num_workers * 4
            while len(futures) >= MAX_IN_FLIGHT:
                ready, futures = ray.wait(futures, num_returns=1)
                for res in ray.get(ready):
                    for r_dict in res:
                        # Write immediately
                        out_f.write(json.dumps(r_dict) + "\n")
    
    # Wait for completion of remaining
    while len(futures) > 0:
        ready, futures = ray.wait(futures, num_returns=1)
        for res in ray.get(ready):
            for r_dict in res:
                out_f.write(json.dumps(r_dict) + "\n")
                
    out_f.close()
    print("Feature Extraction completed.")
    ray.shutdown()

if __name__ == "__main__":
    main()
