"""
worker_core.py
두 파이프라인(single-GPU / multi-GPU Ray)에서 공통으로 사용하는
모델 로딩 + 배치 처리 코어.
"""
from __future__ import annotations

import io
import os
import tarfile
from typing import List, Optional, Tuple

import pandas as pd
from PIL import Image


def _get_tags(row) -> List[str]:
    """parquet 컬럼에서 태그 목록을 안전하게 추출."""
    t = row.get("tags", "") if isinstance(row, dict) else row["tags"]
    if t is None:
        return []
    if isinstance(t, str):
        return [x.strip() for x in t.split("|") if x.strip()]
    try:
        return list(t)
    except Exception:
        return []


class FeatureWorker:
    """
    단일 GPU에서 동작하는 모델 컨테이너.
    Ray가 필요 없는 single-GPU 모드에서 직접 인스턴스화하여 사용.
    multi-GPU 모드에서는 @ray.remote 래퍼가 이 클래스를 감싼다.
    """

    def __init__(
        self,
        run_c1: bool = True,
        run_c2: bool = True,
        run_c3: bool = True,
        run_c4: bool = True,
        run_c5: bool = True,
        weights_dir: str = "",   # C2(SAM2.1 ckpt) + C3(SCRFD/ViTPose ckpt) 가중치 디렉토리
        c4_lang: str = "en",
        priority: str = "high_efficiency",
        device: Optional[str] = None,
    ):
        self.run_c1 = run_c1
        self.run_c2 = run_c2
        self.run_c3 = run_c3
        self.run_c4 = run_c4
        self.run_c5 = run_c5
        self.priority = priority

        print(f"[FeatureWorker] Initializing models with priority='{priority}' ...")

        # ---- C1 CLIP ----------------------------------------------------------------
        if run_c1:
            from c1_clip import ClipFeatureExtractor
            self.c1 = ClipFeatureExtractor(priority=priority)
        else:
            self.c1 = None

        # ---- C2 Segmentation --------------------------------------------------------
        if run_c2:
            from c2_seg import SegFeatureExtractor
            # SAM 2.1 가중치 경로를 명시적으로 전달하여 경로 오류 방지
            self.c2 = SegFeatureExtractor(priority=priority, weights_dir=weights_dir)
        else:
            self.c2 = None

        # ---- C3 Pose ----------------------------------------------------------------
        if run_c3:
            from c3_pose import PoseFeatureExtractor
            from c3_weights import resolve_c3_paths
            c3_paths = resolve_c3_paths(weights_dir)
            if c3_paths is not None:
                c3_verify_strict = os.environ.get("C3_PERSON_VERIFY_STRICT", "1").strip().lower() not in {
                    "0",
                    "false",
                    "no",
                }
                self.c3 = PoseFeatureExtractor(
                    c3_paths["det_cfg"], c3_paths["det_w"],
                    c3_paths["pose_cfg"], c3_paths["pose_w"],
                    priority=priority,
                    person_verify_strict=c3_verify_strict,
                )
            else:
                print(f"[FeatureWorker] C3 setup failed (see above). Skipping C3.")
                self.c3 = None
        else:
            self.c3 = None

        # ---- C4 OCR -----------------------------------------------------------------
        if run_c4:
            from c4_ocr import OcrFeatureExtractor
            self.c4 = OcrFeatureExtractor(lang=c4_lang, priority=priority)
        else:
            self.c4 = None

        # ---- C5 Geometry (horizon/roll + symmetry) -------------------------------
        if run_c5:
            from c5_geom import GeoFeatureExtractor
            self.c5 = GeoFeatureExtractor()
        else:
            self.c5 = None

    # ------------------------------------------------------------------
    def process_batch(
        self, batch_data: List[Tuple]
    ) -> List[dict]:
        """
        Args:
            batch_data: list of (image_id, PIL.Image, tags: List[str])
        Returns:
            list of result dicts (one per image)
        """
        results = []

        # C1 — batched encode (efficient)
        c1_img_feats = c1_txt_feats = None
        if self.c1:
            images = [x[1] for x in batch_data]
            texts  = [", ".join(x[2]) for x in batch_data]
            c1_img_feats = self.c1.encode_images(images)
            c1_txt_feats = self.c1.encode_texts(texts)

        for i, (img_id, img, tags) in enumerate(batch_data):
            res: dict = {"image_id": img_id}
            c2_seg = []
            c2_det = []

            if self.c1 and c1_img_feats is not None:
                res["c1_img_embed"] = c1_img_feats[i].tolist()
                res["c1_txt_embed"] = c1_txt_feats[i].tolist()

            if self.c2:
                c2_out = self.c2.process_image(img, return_det=True)
                if isinstance(c2_out, tuple) and len(c2_out) == 2:
                    c2_seg, c2_det = c2_out
                elif isinstance(c2_out, list):
                    c2_seg = c2_out
                if not isinstance(c2_seg, list):
                    c2_seg = []
                if not isinstance(c2_det, list):
                    c2_det = []
                res["c2_seg"] = c2_seg
                if c2_det:
                    res["c2_det"] = c2_det

            if self.c3:
                person_hint_boxes = []
                for d in c2_det:
                    if not isinstance(d, dict):
                        continue
                    try:
                        cid = int(d.get("class_id", -1))
                    except Exception:
                        cid = -1
                    if cid != 0:
                        continue
                    if float(d.get("score", 0.0)) < 0.20:
                        continue
                    box = d.get("box")
                    if isinstance(box, (list, tuple)) and len(box) == 4:
                        person_hint_boxes.append([float(v) for v in box])
                run_anyway = len(person_hint_boxes) > 0
                try:
                    res["c3_pose"] = self.c3.process_image(
                        img,
                        run_anyway=run_anyway,
                        tags=tags,
                        person_prior_boxes=person_hint_boxes if person_hint_boxes else None,
                    )
                except TypeError:
                    # Backward-compatible path if extractor does not expose person priors.
                    res["c3_pose"] = self.c3.process_image(img, run_anyway=run_anyway, tags=tags)

            if self.c4:
                res["c4_ocr"] = self.c4.process_image(img, tags=tags)

            if self.c5:
                res["c5_geom"] = self.c5.process_image(img)

            results.append(res)

        return results


# ---------------------------------------------------------------------------
# I/O helpers (공유)
# ---------------------------------------------------------------------------

def iter_tar_images(tar_path: str, image_ids) -> List[Tuple]:
    """
    tar 파일에서 요청된 image_id 목록의 PIL.Image를 읽어 반환.
    Returns list of (image_id, PIL.Image)
    """
    result = []
    id_set = {str(iid) for iid in image_ids}
    try:
        with tarfile.open(tar_path, "r|") as tf:
            for member in tf:
                basename = os.path.splitext(os.path.basename(member.name))[0]
                if basename in id_set:
                    f = tf.extractfile(member)
                    if f:
                        try:
                            img = Image.open(io.BytesIO(f.read())).convert("RGB")
                            result.append((basename, img))
                        except Exception:
                            pass
                    if len(result) == len(id_set):
                        break
    except Exception as e:
        print(f"  [iter_tar_images] Error reading {tar_path}: {e}")
    return result


def resolve_tar_path(tar_dir: str, bucket: str, tar_name: str) -> Optional[str]:
    """tar 경로를 flat / bucket-nested 두 패턴으로 탐색."""
    for candidate in [
        os.path.join(tar_dir, tar_name),
        os.path.join(tar_dir, bucket, tar_name),
    ]:
        if os.path.exists(candidate):
            return candidate
    return None


def resolve_weights_dir(weights_dir: str, script_file: str) -> str:
    """
    weights_dir 경로를 확정.
    - 인자가 주어진 경우: 존재 여부와 무관하게 항상 사용 (디렉터리가 없으면 자동 생성).
    - 인자가 빈 문자열인 경우에만 script_file 기준 'weights/' 하위 경로를 fallback으로 사용.
    주의: os.path.isdir() 체크를 하면 W/F 파일을 아직 복사하지 않은 상태에서
    인자가 무시되어 틀린 경로로 fallback되는 문제가 발생하므로 검사하지 않음.
    """
    if weights_dir:
        resolved = os.path.abspath(weights_dir)
    else:
        resolved = os.path.join(os.path.dirname(os.path.abspath(script_file)), "weights")
    os.makedirs(resolved, exist_ok=True)
    return resolved
