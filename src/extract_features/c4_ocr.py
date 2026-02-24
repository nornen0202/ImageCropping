import os
import cv2
import numpy as np
from PIL import Image

try:
    from paddleocr import PaddleOCR
except ImportError:
    print("Warning: paddleocr is not installed properly. C4 OCR will not work.")

class OcrFeatureExtractor:
    """
    PaddleOCR 기반 제한적 텍스트 박스 탐지기 (Det-only)
    - 텍스트가 있을 가능성이 높은 이미지에 대해서만 bounding box와 confidence 출력
    """
    def __init__(self, use_gpu=True, lang="en", priority="high_efficiency"):
        self.use_gpu = use_gpu
        self.priority = priority
        
        # Quality First 모드일 경우 이미지 리사이징 제한을 대폭 늘려 작은 글자 인식률을 극도로 높임
        limit_side_len = 2560 if priority == "quality_first" else 960
        
        print(f"[C4 OCR] Initializing PaddleOCR (lang={lang}, use_gpu={self.use_gpu}, limit_side_len={limit_side_len}, Mode={self.priority})...")
        
        try:
            # We use det=True, rec=False, cls=False to save time. We only need text location.
            self.ocr = PaddleOCR(use_angle_cls=False, lang=lang, use_gpu=self.use_gpu, det=True, rec=False, cls=False, 
                                 det_limit_side_len=limit_side_len, show_log=False)
        except Exception as e:
            print(f"Failed to load PaddleOCR model: {e}")
            self.ocr = None

    def process_image(self, image: Image.Image, run_anyway=False, tags: list = None):
        """
        image: PIL.Image
        run_anyway: True면 무조건 실행, False면 tags 기반 OCR 필터링
        tags: meta데이터 태그 리스트
        반환: list of dicts { 'box': [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], 'score': float }
        """
        results_out = []
        if self.ocr is None:
            return results_out
            
        # 메타데이터 Gating (tags에 텍스트/문서/간판 관련 키워드가 없으면 bypass)
        TEXT_WORDS = {
            'document', 'paper', 'invoice', 'receipt', 'form', 'contract', 'text', 
            'typography', 'handwriting', 'banner', 'label', 'poster', 'writing', 
            'sign', 'letter', 'number', 'logo', 'brand', 'word'
        }
        
        if not run_anyway and tags is not None:
            if not any(t in TEXT_WORDS for t in tags):
                return results_out # Skip costly OCR detection
                
        # Convert PIL to BGR for OpenCV / PaddleOCR
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 3. Run OCR Detection (Det only)
        # result is a list of lists. each inner list corresponds to a detected text box:
        # [ [[x1,y1], [x2,y2], [x3,y3], [x4,y4]], score(which is not returned by rec=False standard, sometimes it does or defaults to 1.0) ]
        
        try:
            # det=True, rec=False returns nested lists. format: [[[box_points]]]
            # Usually paddleocr format varies slightly depending on v3/v4 detector.
            ocr_result = self.ocr.ocr(img_bgr, det=True, rec=False, cls=False)
            
            # when rec=False, ocr_result is typically [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], ...]
            # or [[[[x1,y1],[x2,y2],...]]] depending on batch inference config. First idx is usually batch.
            if ocr_result and len(ocr_result) > 0 and ocr_result[0] is not None:
                boxes = ocr_result[0] # assuming single image
                for box in boxes:
                    results_out.append({
                        'box': box, # list of 4 points -> 8 coordinates
                        'score': 1.0 # Detection only model doesn't output confidence score per box by default in PaddleOCR's wrapper
                    })
        except Exception as e:
            print(f"OCR Inference error: {e}")
            
        return results_out
