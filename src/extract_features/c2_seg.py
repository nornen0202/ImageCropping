import os
import cv2
import torch
import numpy as np
import math
from PIL import Image

# YOLOv8
from ultralytics import YOLO

# EfficientViT-SAM (High Efficiency Mode)
try:
    try:
        from efficientvit.sam_model_zoo import create_sam_model
    except ImportError:
        from efficientvit.sam_model_zoo import create_efficientvit_sam_model as create_sam_model
    from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
except ImportError as e:
    create_sam_model = None
    EfficientViTSamPredictor = None
    print(f"Warning: efficientvit is not installed properly ({e}). C2 Seg (High Efficiency) will not work.")

# SAM 2.1 (Quality First Mode)
try:
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
except ImportError as e:
    build_sam2 = None
    SAM2AutomaticMaskGenerator = None
    print(f"Warning: sam2 is not installed properly ({e}). C2 Seg (Quality First) will not work.")

def encode_rle(mask):
    """
    Encode binary mask to RLE (Run Length Encoding) string
    mask: numpy array bool or 0/1
    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

class SegFeatureExtractor:
    """
    YOLOv8 + EfficientViT-SAM L2 (고효율) 또는 SAM 2.1 (품질 최우선) 기반 Segmentation 추출기
    - High Efficiency: YOLOv8로 주요 객체의 BBox 추출 -> 이를 prompt로 전달하여 마스크 획득
    - Quality First: SAM 2.1 Automatic Mask Generation -> 면적 및 중앙 집중도 기반 메인 피사체 선택
    """
    def __init__(self, yolo_model="yolov8n.pt", sam_model="l2", device=None,
                 priority="high_efficiency", weights_dir=""):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.priority = priority
        self.yolo = None
        self.predictor = None
        self.mask_generator = None

        # weights_dir 미지정 시 PROJECT_ROOT/weights → src/scripts/weights 순으로 탐색
        if not weights_dir:
            _here = os.path.dirname(os.path.abspath(__file__))
            for _cand in [
                os.path.normpath(os.path.join(_here, "../../weights")),
                os.path.normpath(os.path.join(_here, "../scripts/weights")),
            ]:
                if os.path.isdir(_cand):
                    weights_dir = _cand
                    break
            else:
                weights_dir = os.path.normpath(os.path.join(_here, "../../weights"))
        self._weights_dir = weights_dir

        print(f"[C2 Seg] Initializing Segmentation Models on {self.device} (Mode: {self.priority})...")
        print(f"[C2 Seg] Weights dir: {self._weights_dir}")

        if self.priority == "high_efficiency":
            # 1. Load YOLOv8
            self.yolo = YOLO(yolo_model)
            self.yolo.to(self.device)
            # 2. Load EfficientViT-SAM
            if create_sam_model is None:
                print("[C2 Seg] EfficientViT-SAM not available (import failed). Disabled.")
            else:
                try:
                    self.sam = create_sam_model(sam_model, True).to(self.device).eval()
                    self.predictor = EfficientViTSamPredictor(self.sam)
                    print("[C2 Seg] EfficientViT-SAM loaded OK.")
                except Exception as e:
                    print(f"[C2 Seg] Failed to load EfficientViT-SAM model: {e}")

        elif self.priority == "quality_first":
            if build_sam2 is None:
                print("[C2 Seg] SAM 2.1 not available (import failed). Disabled.")
            else:
                try:
                    cfg_name  = "configs/sam2.1/sam2.1_hiera_l.yaml"
                    ckpt_path = os.path.join(self._weights_dir, "sam2.1_hiera_large.pt")
                    if not os.path.exists(ckpt_path):
                        print(f"[C2 Seg] SAM 2.1 weights not found: {ckpt_path}")
                        print("[C2 Seg] Attempting auto-download from HuggingFace (facebook/sam2.1)...")
                        self._download_sam2_weights(ckpt_path)
                    self.sam2_model = build_sam2(cfg_name, ckpt_path, device=self.device,
                                                  apply_postprocessing=False)
                    self.mask_generator = SAM2AutomaticMaskGenerator(self.sam2_model)
                    print("[C2 Seg] SAM 2.1 loaded OK.")
                except Exception as e:
                    print(f"[C2 Seg] Failed to load SAM2.1 model: {e}")

    @staticmethod
    def _download_sam2_weights(dest_path: str):
        """HuggingFace Hub에서 SAM 2.1 large 가중치를 자동 다운로드."""
        import urllib.request
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        url = ("https://huggingface.co/facebook/sam2.1-hiera-large"
               "/resolve/main/sam2.1_hiera_large.pt")
        print(f"[C2 Seg] Downloading {url} → {dest_path}")
        try:
            urllib.request.urlretrieve(url, dest_path)
            print("[C2 Seg] Download complete.")
        except Exception as e:
            print(f"[C2 Seg] Auto-download failed: {e}. "
                  f"Please manually place sam2.1_hiera_large.pt in {os.path.dirname(dest_path)}")

    def process_image(self, image: Image.Image, conf_threshold=0.25):
        """
        image: PIL.Image
        반환: list of dicts { 'box': [x1,y1,x2,y2], 'class_id': int, 'score': float, 'mask_rle': str, 'area': int }
        """
        results_out = []
        
        # Convert PIL to BGR/RGB
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        h, w = img_rgb.shape[:2]
        img_area = h * w
        
        if self.priority == "high_efficiency":
            if self.yolo is None or self.predictor is None:
                return results_out
                
            yolo_res = self.yolo(img_bgr, conf=conf_threshold, verbose=False)[0]
            boxes = yolo_res.boxes.xyxy.cpu().numpy()
            scores = yolo_res.boxes.conf.cpu().numpy()
            class_ids = yolo_res.boxes.cls.cpu().numpy().astype(int)
            
            if len(boxes) == 0:
                return results_out
                
            self.predictor.set_image(img_rgb)
            for i, box in enumerate(boxes):
                mask, _, _ = self.predictor.predict(
                    point_coords=None, point_labels=None, box=box[None, :], multimask_output=False
                )
                mask_2d = mask[0]
                mask_rle = encode_rle(mask_2d)
                results_out.append({
                    'box': [float(x) for x in box], 'class_id': int(class_ids[i]),
                    'score': float(scores[i]), 'mask_rle': mask_rle, 'area': int(mask_2d.sum())
                })
            self.predictor.reset_image()
            
        elif self.priority == "quality_first":
            if self.mask_generator is None:
                return results_out
                
            # Generate all masks (SAM 2.1)
            masks = self.mask_generator.generate(img_rgb)
            if not masks:
                return results_out
                
            best_mask_info = None
            best_score = -float('inf')
            img_center = (w / 2.0, h / 2.0)
            max_dist = math.sqrt(img_center[0]**2 + img_center[1]**2)
            
            # 1. Heuristic Filtering & Scoring
            for m in masks:
                area = m['area']
                area_ratio = area / float(img_area)
                
                # Filter noise (<1%) and background (>90%)
                if area_ratio < 0.01 or area_ratio > 0.90:
                    continue
                    
                bbox = m['bbox'] # [x, y, w, h]
                cx = bbox[0] + bbox[2] / 2.0
                cy = bbox[1] + bbox[3] / 2.0
                dist_to_center = math.sqrt((cx - img_center[0])**2 + (cy - img_center[1])**2)
                dist_ratio = dist_to_center / max_dist
                
                score = area_ratio + 1.0 * (1.0 - dist_ratio)
                if score > best_score:
                    best_score = score
                    best_mask_info = m
                    
            if best_mask_info:
                bx, by, bw, bh = best_mask_info['bbox']
                box_xyxy = [bx, by, bx + bw, by + bh]
                
                mask_2d = best_mask_info['segmentation']
                mask_rle = encode_rle(mask_2d)
                
                results_out.append({
                    'box': [float(x) for x in box_xyxy],
                    'class_id': -1, # SAM doesn't provide class
                    'score': float(best_score),
                    'mask_rle': mask_rle,
                    'area': int(best_mask_info['area'])
                })
                
        return results_out
