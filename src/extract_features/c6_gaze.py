from __future__ import annotations

import contextlib
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


POSITIVE_GAZE_TAGS = {
    "person",
    "people",
    "portrait",
    "face",
    "headshot",
    "selfie",
    "man",
    "woman",
    "boy",
    "girl",
    "family",
    "group",
    "couple",
}


@contextlib.contextmanager
def _file_lock(path: str):
    """
    Process-level lock for torch.hub initialization.
    Prevents concurrent workers from racing on ~/.cache/torch/hub/main.zip.
    """
    lock_path = str(path).strip()
    if not lock_path:
        yield
        return

    os.makedirs(os.path.dirname(os.path.abspath(lock_path)), exist_ok=True)
    fp = open(lock_path, "a+")
    try:
        try:
            import fcntl  # Linux/Unix

            fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
        except Exception:
            # Best effort: continue without lock on unsupported platforms.
            pass
        yield
    finally:
        try:
            import fcntl

            fcntl.flock(fp.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        fp.close()


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))


def _clamp01(v: float) -> float:
    return _clamp(float(v), 0.0, 1.0)


def _norm_tags(tags: Optional[Sequence[Any]]) -> List[str]:
    if tags is None:
        return []
    out: List[str] = []
    for t in tags:
        s = str(t).strip().lower()
        if s:
            out.append(s)
    return out


def _gaze_dir_from_dx(dx: float, thr: float = 0.05) -> str:
    if dx > thr:
        return "right"
    if dx < -thr:
        return "left"
    return "center"


def _bbox_center_xyxy_norm(box: Optional[Sequence[float]]) -> Tuple[float, float]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return 0.5, 0.5
    x1, y1, x2, y2 = [float(v) for v in box]
    return _clamp01(0.5 * (x1 + x2)), _clamp01(0.5 * (y1 + y2))


def _to_norm_xyxy(box: Sequence[float], w: int, h: int) -> Optional[List[float]]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    ww = max(1.0, float(w))
    hh = max(1.0, float(h))
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 /= ww
    x2 /= ww
    y1 /= hh
    y2 /= hh
    x1 = _clamp01(x1)
    x2 = _clamp01(x2)
    y1 = _clamp01(y1)
    y2 = _clamp01(y2)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if (x2 - x1) <= 1e-6 or (y2 - y1) <= 1e-6:
        return None
    return [round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6)]


def _head_box_norm_from_pose(pose_item: Dict[str, Any], image_w: int, image_h: int) -> Optional[List[float]]:
    if not isinstance(pose_item, dict):
        return None
    face = pose_item.get("face")
    if isinstance(face, dict):
        fb = face.get("bbox_norm")
        if isinstance(fb, (list, tuple)) and len(fb) == 4:
            out = [float(v) for v in fb]
            out = [_clamp01(v) for v in out]
            if out[2] > out[0] and out[3] > out[1]:
                return [round(v, 6) for v in out]
        fb_px = face.get("bbox")
        if isinstance(fb_px, (list, tuple)) and len(fb_px) == 4:
            out = _to_norm_xyxy([float(v) for v in fb_px], image_w, image_h)
            if out is not None:
                return out

    pb = pose_item.get("bbox")
    if isinstance(pb, (list, tuple)) and len(pb) == 4:
        return _to_norm_xyxy([float(v) for v in pb], image_w, image_h)
    return None


def _heatmap_peak_xy(heatmap: Any) -> Optional[Tuple[float, float, float]]:
    if heatmap is None:
        return None
    arr: np.ndarray
    if hasattr(heatmap, "detach"):
        arr = heatmap.detach().cpu().numpy().astype(np.float32)
    else:
        arr = np.asarray(heatmap, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return None
    h, w = arr.shape[:2]
    idx = int(np.argmax(arr))
    y = idx // w
    x = idx % w
    peak = float(arr[y, x])
    x_norm = (float(x) + 0.5) / max(1.0, float(w))
    y_norm = (float(y) + 0.5) / max(1.0, float(h))
    return _clamp01(x_norm), _clamp01(y_norm), _clamp01(peak)


class _GazelleBackend:
    """
    Optional Gaze-LLE backend via gazelle package.
    - local package path + checkpoint
    - or Torch Hub fallback
    """

    def __init__(
        self,
        model_name: str = "gazelle_dinov2_vitb14_inout",
        device: Optional[str] = None,
        ckpt_path: str = "",
        repo_dir: str = "",
        use_torchhub: bool = True,
    ) -> None:
        self.available = False
        self.model_name = str(model_name)
        self.device = str(device or os.environ.get("C6_GAZELLE_DEVICE", "cuda"))
        self.ckpt_path = str(ckpt_path or os.environ.get("C6_GAZELLE_CKPT", "")).strip()
        self.repo_dir = str(repo_dir or os.environ.get("C6_GAZELLE_REPO", "")).strip()
        if not self.repo_dir:
            here = os.path.dirname(os.path.abspath(__file__))
            candidates = [
                os.path.join(here, "..", "..", "third_party", "gazelle"),
                os.path.join(os.getcwd(), "third_party", "gazelle"),
            ]
            for cand in candidates:
                cand_abs = os.path.abspath(cand)
                if os.path.isdir(cand_abs):
                    self.repo_dir = cand_abs
                    break
        env_use_torchhub = os.environ.get("C6_GAZELLE_USE_TORCHHUB", "").strip().lower()
        if env_use_torchhub in {"0", "false", "no"}:
            use_torchhub = False
        elif env_use_torchhub in {"1", "true", "yes"}:
            use_torchhub = True
        self.use_torchhub = bool(use_torchhub)
        self.init_error: Optional[str] = None
        self._torch = None
        self._model = None
        self._transform = None
        self._hub_dir = ""

        try:
            import torch

            if self.device.startswith("cuda") and (not torch.cuda.is_available()):
                self.device = "cpu"
            self._torch = torch
        except Exception as e:
            self.init_error = f"torch import failed: {e}"
            return

        # Keep torch.hub cache deterministic and writable across workers.
        self._hub_dir = str(os.environ.get("C6_TORCH_HUB_DIR", "")).strip()
        if not self._hub_dir:
            home = os.path.expanduser("~")
            self._hub_dir = os.path.join(home, ".cache", "torch", "hub")

        if self.repo_dir and os.path.isdir(self.repo_dir) and self.repo_dir not in sys.path:
            sys.path.insert(0, self.repo_dir)

        try:
            self._init_model()
            if self._model is not None and self._transform is not None:
                self._model.eval()
                self._model.to(self.device)
                self.available = True
        except Exception as e:
            self.init_error = str(e)
            self.available = False

    @staticmethod
    def _is_torchhub_transient_error(exc: Exception) -> bool:
        s = str(exc)
        needles = [
            "main.zip",
            "No such file or directory",
            "Connection reset by peer",
            "timed out",
            "Temporary failure in name resolution",
        ]
        return any(n in s for n in needles)

    def _gazelle_ckpt_url(self) -> str:
        return {
            "gazelle_dinov2_vitb14": "https://github.com/fkryan/gazelle/releases/download/v1.0.0/gazelle_dinov2_vitb14_hub.pt",
            "gazelle_dinov2_vitl14": "https://github.com/fkryan/gazelle/releases/download/v1.0.0/gazelle_dinov2_vitl14.pt",
            "gazelle_dinov2_vitb14_inout": "https://github.com/fkryan/gazelle/releases/download/v1.0.0/gazelle_dinov2_vitb14_inout.pt",
            "gazelle_dinov2_vitl14_inout": "https://github.com/fkryan/gazelle/releases/download/v1.0.0/gazelle_dinov2_vitl14_inout.pt",
        }.get(self.model_name, "")

    def _resolve_local_ckpt_path(self) -> str:
        candidates: List[str] = []
        if self.ckpt_path:
            candidates.append(self.ckpt_path)
        if self._hub_dir:
            candidates.append(os.path.join(self._hub_dir, "checkpoints", f"{self.model_name}.pt"))
        for cand in candidates:
            cand_norm = str(cand).strip()
            if cand_norm and os.path.isfile(cand_norm):
                return cand_norm
        return ""

    def _init_model(self) -> None:
        assert self._torch is not None
        torch = self._torch

        if self._hub_dir:
            os.makedirs(self._hub_dir, exist_ok=True)
            torch.hub.set_dir(self._hub_dir)

        retries = int(os.environ.get("C6_GAZELLE_INIT_RETRIES", "3"))
        retries = max(1, retries)
        lock_path = str(os.environ.get("C6_GAZELLE_HUB_LOCK", "")).strip()
        if not lock_path:
            lock_path = os.path.join(self._hub_dir, ".gazelle_hub.lock")

        last_exc: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            try:
                with _file_lock(lock_path):
                    self._init_model_once()
                return
            except Exception as e:
                last_exc = e
                if attempt < retries and self._is_torchhub_transient_error(e):
                    time.sleep(min(2.0 * attempt, 5.0))
                    continue
                raise
        if last_exc is not None:
            raise last_exc

    def _init_model_once(self) -> None:
        assert self._torch is not None
        torch = self._torch
        validate_backup = getattr(torch.hub, "_validate_not_a_forked_repo", None)
        if callable(validate_backup):
            torch.hub._validate_not_a_forked_repo = lambda *args, **kwargs: None

        local_loaded = False
        local_exc: Optional[Exception] = None
        try:
            try:
                from gazelle.model import get_gazelle_model

                model, transform = get_gazelle_model(self.model_name)
                local_loaded = True
                ckpt_path = self._resolve_local_ckpt_path()
                if ckpt_path:
                    state = torch.load(ckpt_path, map_location="cpu")
                    if hasattr(model, "load_gazelle_state_dict"):
                        model.load_gazelle_state_dict(state)
                    else:
                        model.load_state_dict(state, strict=False)
                elif self.use_torchhub:
                    ckpt_url = self._gazelle_ckpt_url()
                    if not ckpt_url:
                        raise ValueError(f"unsupported model_name: {self.model_name}")
                    state = torch.hub.load_state_dict_from_url(ckpt_url, map_location="cpu")
                    if hasattr(model, "load_gazelle_state_dict"):
                        model.load_gazelle_state_dict(state)
                    else:
                        model.load_state_dict(state, strict=False)
                self._model = model
                self._transform = transform
            except Exception:
                local_loaded = False
                local_exc = sys.exc_info()[1]
        finally:
            if callable(validate_backup):
                torch.hub._validate_not_a_forked_repo = validate_backup

        if local_loaded:
            return

        if not self.use_torchhub:
            msg = "gazelle local package unavailable and torchhub disabled"
            if local_exc is not None:
                msg = f"{msg}; local_error={local_exc}"
            raise RuntimeError(msg)

        local_hub_exc: Optional[Exception] = None
        if self.repo_dir and os.path.isfile(os.path.join(self.repo_dir, "hubconf.py")):
            try:
                model, transform = torch.hub.load(
                    self.repo_dir,
                    self.model_name,
                    pretrained=True,
                    source="local",
                    trust_repo=True,
                    skip_validation=True,
                )
                self._model = model
                self._transform = transform
                return
            except Exception as e:
                local_hub_exc = e

        try:
            model, transform = torch.hub.load(
                "fkryan/gazelle",
                self.model_name,
                pretrained=True,
                trust_repo=True,
                skip_validation=True,
            )
        except Exception as e:
            details: List[str] = []
            if local_exc is not None:
                details.append(f"local_error={local_exc}")
            if local_hub_exc is not None:
                details.append(f"local_hub_error={local_hub_exc}")
            details.append(f"hub_error={e}")
            raise RuntimeError(f"gazelle init failed ({'; '.join(details)})")
        self._model = model
        self._transform = transform

    def infer(self, image: Image.Image, bboxes_norm: List[Optional[Tuple[float, float, float, float]]]) -> Optional[List[Dict[str, Any]]]:
        if not self.available or self._model is None or self._transform is None or self._torch is None:
            return None
        torch = self._torch
        if not isinstance(bboxes_norm, list):
            return None

        with torch.no_grad():
            inp = {
                "images": self._transform(image).unsqueeze(0).to(self.device),
                "bboxes": [bboxes_norm],
            }
            pred = self._model(inp)

        heat_all = pred.get("heatmap") if isinstance(pred, dict) else None
        inout_all = pred.get("inout") if isinstance(pred, dict) else None
        if not isinstance(heat_all, list) or len(heat_all) == 0:
            return None
        heat_person = heat_all[0]
        out_person: List[Dict[str, Any]] = []
        for i in range(len(bboxes_norm)):
            hm = heat_person[i] if i < len(heat_person) else None
            peak_xy = _heatmap_peak_xy(hm)
            if peak_xy is None:
                out_person.append(
                    {
                        "gaze_target_norm_xy": None,
                        "heatmap_peak": 0.0,
                        "inout_score": None,
                    }
                )
                continue
            gx, gy, peak = peak_xy
            inout_score = None
            if isinstance(inout_all, list) and len(inout_all) > 0 and inout_all[0] is not None:
                inout_i = inout_all[0][i] if i < len(inout_all[0]) else None
                if inout_i is not None:
                    if hasattr(inout_i, "detach"):
                        inout_score = float(inout_i.detach().cpu().item())
                    else:
                        inout_score = float(inout_i)
                    inout_score = _clamp01(inout_score)
            out_person.append(
                {
                    "gaze_target_norm_xy": [round(float(gx), 6), round(float(gy), 6)],
                    "heatmap_peak": round(float(peak), 6),
                    "inout_score": None if inout_score is None else round(float(inout_score), 6),
                }
            )
        return out_person


class GazeFeatureExtractor:
    """
    C6 Gaze/HeadPose extractor.

    Backend policy:
    - high_efficiency: proxy from C3 face/kp geometry
    - quality_first: Gaze-LLE(gazelle) if available, else proxy fallback
    """

    def __init__(
        self,
        priority: str = "high_efficiency",
        c6_backend: str = "auto",
        device: Optional[str] = None,
        gazelle_model_name: str = "gazelle_dinov2_vitb14_inout",
        gazelle_ckpt: str = "",
        gazelle_repo_dir: str = "",
        gazelle_use_torchhub: bool = True,
        inout_thr: float = 0.35,
    ) -> None:
        self.priority = str(priority)
        self.inout_thr = _clamp(float(inout_thr), 0.0, 1.0)

        env_backend = os.environ.get("C6_BACKEND", "").strip().lower()
        req = str(c6_backend or "auto").strip().lower()
        if env_backend in {"auto", "proxy", "gazelle"}:
            req = env_backend
        if req not in {"auto", "proxy", "gazelle"}:
            req = "auto"
        if req == "auto":
            req = "gazelle" if self.priority == "quality_first" else "proxy"

        self.requested_backend = req
        self.backend_runtime = "proxy"
        self.gazelle: Optional[_GazelleBackend] = None

        if self.requested_backend == "gazelle":
            self.gazelle = _GazelleBackend(
                model_name=str(os.environ.get("C6_GAZELLE_MODEL", gazelle_model_name)),
                device=device,
                ckpt_path=gazelle_ckpt,
                repo_dir=gazelle_repo_dir,
                use_torchhub=gazelle_use_torchhub,
            )
            if self.gazelle.available:
                self.backend_runtime = "gazelle"
            else:
                self.backend_runtime = "proxy_fallback"
                print(f"[C6 Gaze] Gazelle unavailable -> fallback to proxy. reason={self.gazelle.init_error}")

    @staticmethod
    def _should_run(tags: Optional[Sequence[Any]], pose_items: Sequence[Dict[str, Any]]) -> bool:
        if pose_items:
            return True
        tt = _norm_tags(tags)
        return any(t in POSITIVE_GAZE_TAGS for t in tt)

    def _proxy_people(self, pose_items: Sequence[Dict[str, Any]], image_w: int, image_h: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for idx, pose in enumerate(pose_items):
            if not isinstance(pose, dict):
                continue
            hp = pose.get("headpose_gaze", {}) if isinstance(pose.get("headpose_gaze"), dict) else {}
            yaw = float(hp.get("yaw_proxy", 0.0) or 0.0)
            pitch = float(hp.get("pitch_proxy", 0.0) or 0.0)
            roll = float(hp.get("roll_deg", 0.0) or 0.0)
            conf = _clamp01(float(hp.get("conf", 0.0) or 0.0))
            bbox_norm = _head_box_norm_from_pose(pose, image_w=image_w, image_h=image_h)
            cx, cy = _bbox_center_xyxy_norm(bbox_norm)
            gx = _clamp01(cx + 0.20 * yaw)
            gy = _clamp01(cy + 0.20 * pitch)
            gaze_dir = str(hp.get("gaze_dir", _gaze_dir_from_dx(gx - cx)))
            out.append(
                {
                    "person_index": int(idx),
                    "head_bbox_norm_xyxy": bbox_norm,
                    "gaze_target_norm_xy": [round(gx, 6), round(gy, 6)],
                    "yaw_proxy": round(yaw, 4),
                    "pitch_proxy": round(pitch, 4),
                    "roll_deg": round(roll, 3),
                    "gaze_dir": gaze_dir,
                    "inout_score": None,
                    "in_frame": None,
                    "heatmap_peak": None,
                    "conf": round(conf, 4),
                    "source": str(hp.get("source", "pose_kp_proxy")),
                }
            )
        return out

    def _gazelle_people(
        self,
        image: Image.Image,
        base_people: List[Dict[str, Any]],
    ) -> Optional[List[Dict[str, Any]]]:
        if self.gazelle is None or not self.gazelle.available:
            return None

        bboxes: List[Optional[Tuple[float, float, float, float]]] = []
        base_idx: List[int] = []
        for i, p in enumerate(base_people):
            hb = p.get("head_bbox_norm_xyxy")
            if isinstance(hb, (list, tuple)) and len(hb) == 4:
                bboxes.append((float(hb[0]), float(hb[1]), float(hb[2]), float(hb[3])))
                base_idx.append(i)
        if not bboxes:
            bboxes = [None]
            base_idx = [-1]

        pred_people = self.gazelle.infer(image=image, bboxes_norm=bboxes)
        if pred_people is None:
            return None

        out: List[Dict[str, Any]] = []
        for j, pred in enumerate(pred_people):
            person_ref = base_people[base_idx[j]] if (0 <= base_idx[j] < len(base_people)) else None
            if person_ref is not None:
                hb = person_ref.get("head_bbox_norm_xyxy")
                cx, cy = _bbox_center_xyxy_norm(hb)
                person_index = int(person_ref.get("person_index", base_idx[j]))
                roll = float(person_ref.get("roll_deg", 0.0) or 0.0)
            else:
                hb = None
                cx, cy = 0.5, 0.5
                person_index = int(j)
                roll = 0.0

            gaze_xy = pred.get("gaze_target_norm_xy")
            if isinstance(gaze_xy, (list, tuple)) and len(gaze_xy) == 2:
                gx = _clamp01(float(gaze_xy[0]))
                gy = _clamp01(float(gaze_xy[1]))
            else:
                gx, gy = cx, cy

            dx = gx - cx
            dy = gy - cy
            yaw = _clamp(dx * 2.0, -1.0, 1.0)
            pitch = _clamp(dy * 2.0, -1.0, 1.0)
            gaze_dir = _gaze_dir_from_dx(dx)

            inout_score = pred.get("inout_score")
            if inout_score is not None:
                inout_score = _clamp01(float(inout_score))
                in_frame = bool(inout_score >= self.inout_thr)
            else:
                in_frame = None

            heatmap_peak = _clamp01(float(pred.get("heatmap_peak", 0.0) or 0.0))
            conf = heatmap_peak if inout_score is None else _clamp01(0.5 * heatmap_peak + 0.5 * float(inout_score))

            out.append(
                {
                    "person_index": person_index,
                    "head_bbox_norm_xyxy": hb,
                    "gaze_target_norm_xy": [round(gx, 6), round(gy, 6)],
                    "yaw_proxy": round(yaw, 4),
                    "pitch_proxy": round(pitch, 4),
                    "roll_deg": round(roll, 3),
                    "gaze_dir": gaze_dir,
                    "inout_score": None if inout_score is None else round(float(inout_score), 4),
                    "in_frame": in_frame,
                    "heatmap_peak": round(float(heatmap_peak), 4),
                    "conf": round(float(conf), 4),
                    "source": "gaze_lle_gazelle",
                }
            )
        return out

    def process_image(
        self,
        image: Image.Image,
        pose_items: Optional[Sequence[Dict[str, Any]]] = None,
        tags: Optional[Sequence[Any]] = None,
    ) -> Dict[str, Any]:
        poses = list(pose_items or [])
        if not self._should_run(tags=tags, pose_items=poses):
            return {
                "people": [],
                "num_people": 0,
                "method": "skipped",
                "backend_runtime": self.backend_runtime,
                "requested_backend": self.requested_backend,
            }

        w, h = image.size
        proxy_people = self._proxy_people(poses, image_w=int(w), image_h=int(h))
        if self.backend_runtime == "gazelle":
            gazelle_people = self._gazelle_people(image=image, base_people=proxy_people)
            if gazelle_people is not None and len(gazelle_people) > 0:
                return {
                    "people": gazelle_people,
                    "num_people": len(gazelle_people),
                    "method": "gazelle_gaze_lle",
                    "backend_runtime": self.backend_runtime,
                    "requested_backend": self.requested_backend,
                }

        return {
            "people": proxy_people,
            "num_people": len(proxy_people),
            "method": "pose_proxy",
            "backend_runtime": self.backend_runtime,
            "requested_backend": self.requested_backend,
        }
