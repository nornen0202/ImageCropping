import os
import cv2
import numpy as np
from PIL import Image

try:
    from mmpose.apis import inference_topdown, init_model as init_pose_model
    from mmdet.apis import inference_detector, init_detector
    from mmpose.evaluation.functional import nms
    from mmpose.structures import merge_data_samples
    _MMPOSE_AVAILABLE = True
except ImportError as e:
    _MMPOSE_AVAILABLE = False
    print(f"Warning: mmpose or mmdet is not installed properly ({e}). C3 Pose will not work.")

class PoseFeatureExtractor:
    """
    ViTPose (Top-down) + SCRFD 기반 사람 관절 및 얼굴/전신 절단 탐지 파이프라인
    - (Option) mmdet/mmpose가 무거운 경우 YOLOv8-pose로 대체 가능하지만 여기서는 품질 1순위 스택 적용
    """
    def __init__(self, det_config, det_ckpt, pose_config, pose_ckpt, device=None, priority="high_efficiency"):
        self.device = device if device else "cuda:0"
        self.priority = priority
        print(f"[C3 Pose] Initializing Det({det_ckpt}) & ViTPose({pose_ckpt}) on {self.device} (Mode: {self.priority})...")
        
        if not _MMPOSE_AVAILABLE:
            print("mmpose/mmdet not available. C3 Pose disabled.")
            self.detector = None
            self.pose_estimator = None
        else:
            try:
                # 1. Object Detector (SCRFD or standard person detector like Faster R-CNN)
                self.detector = init_detector(det_config, det_ckpt, device=self.device)
                try:
                    from mmpose.utils import adapt_mmdet_pipeline
                    self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)
                except ImportError:
                    pass
                # 2. Pose Estimator (ViTPose)
                self.pose_estimator = init_pose_model(pose_config, pose_ckpt, device=self.device)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Failed to load Pose models: {e}")
                self.detector = None
                self.pose_estimator = None

    def process_image(self, image: Image.Image, run_anyway=False, tags: list = None):
        """
        image: PIL.Image
        run_anyway: True면 무조건 실행, False면 tags 기반 person 필터링
        tags: meta데이터 태그 리스트
        반환: list of dicts { 'bbox': [x1,y1,x2,y2], 'keypoints': [[x,y,score], ...], 'score': float }
        """
        results_out = []
        if self.detector is None or self.pose_estimator is None:
            return results_out
            
        # 메타데이터 Gating (tags에 사람 관련 키워드가 없으면 bypass)
        PERSON_WORDS = {'person', 'people', 'man', 'woman', 'boy', 'girl', 'face', 'portrait', 'selfie', 'couple', 'family', 'crowd', 'team'}
        if not run_anyway and tags is not None:
            if not any(t in PERSON_WORDS for t in tags):
                return results_out # Skip costly pose estimation
                
        # Convert PIL to BGR for MMCV/OpenCV
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 1. Run Detector (e.g. Find 'person' class bounding boxes)
        det_result = inference_detector(self.detector, img_bgr)
        pred_instances = det_result.pred_instances.cpu().numpy()
        
        # Filter class 0 (person usually in COCO), and apply score threshold
        person_indices = (pred_instances.labels == 0) & (pred_instances.scores > 0.3)
        bboxes = pred_instances.bboxes[person_indices]
        scores = pred_instances.scores[person_indices]
        
        if len(bboxes) == 0:
            return results_out
            
        # 2. Run Top-Down Pose Estimation
        # inference_topdown takes list of bboxes: [np.array([x1,y1,x2,y2]), ...]
        pose_results = inference_topdown(self.pose_estimator, img_bgr, bboxes, bbox_format='xyxy')
        
        for p_res in pose_results:
            p_inst = p_res.pred_instances
            kb = p_inst.keypoints[0] # (17, 2)
            ks = p_inst.keypoint_scores[0] # (17,)
            bb = p_inst.bboxes[0] # (4,)
            bs = p_inst.bbox_scores[0] # float
            
            # Combine into [x, y, score]
            kps_out = []
            for i in range(len(kb)):
                kps_out.append([float(kb[i][0]), float(kb[i][1]), float(ks[i])])
                
            results_out.append({
                'bbox': [float(x) for x in bb],
                'score': float(bs),
                'keypoints': kps_out
            })
            
        return results_out
