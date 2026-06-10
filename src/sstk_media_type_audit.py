from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont


DEFAULT_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")
POLICY_VERSION = "sstk_media_type_v5_20260529_camera_primary_no_monochrome"

HARD_NONPHOTO_KEYWORDS = {
    "vector": 1.0,
    "3d": 1.0,
    "3d render": 1.0,
    "render": 0.9,
    "rendering": 0.9,
    "cgi": 1.0,
    "cartoon": 1.0,
    "clipart": 1.0,
    "clip art": 1.0,
    "drawing": 0.9,
    "anime": 0.8,
    "digital art": 0.8,
}

SOFT_NONPHOTO_KEYWORDS = {
    "illustration": 0.8,
    "artwork": 0.5,
    "graphic": 0.5,
    "symbol": 0.4,
    "logo": 0.5,
    "icon": 0.6,
    "infographic": 0.7,
}

PACKSHOT_KEYWORDS = {
    "isolated",
    "white background",
    "isolated on white",
    "on white",
    "studio",
    "cut out",
    "cutout",
    "clipping path",
}

BACKGROUND_KEYWORDS = {
    "background",
    "texture",
    "pattern",
    "seamless",
    "wallpaper",
    "copy space",
    "copy-space",
    "copyspace",
    "abstract",
    "backdrop",
    "blank",
    "empty",
    "black background",
    "white background",
    "gray background",
    "grey background",
    "gray back",
    "grey back",
}

AI_GENERATED_KEYWORDS = {
    "ai generated",
    "generated ai",
    "generative ai",
    "artificial intelligence generated",
    "midjourney",
    "stable diffusion",
    "dall-e",
    "dalle",
}

DEFAULT_PROMPT_CLASSIFIER_MODEL = "google/siglip2-base-patch16-224"
DEFAULT_CLIP_CLASSIFIER_MODEL = "openai/clip-vit-base-patch32"

PROMPT_GROUPS = {
    "camera_photo": [
        "a natural photograph captured by a camera",
        "a real-world documentary photograph",
        "a normal camera photo with natural scene context",
    ],
    "studio_low_context_photo": [
        "a studio photograph on a plain backdrop",
        "a low-context photo with a single-color studio background",
    ],
    "packshot_or_isolated_photo": [
        "a product photograph isolated on a white background",
        "an object or animal cut out on a plain white background",
    ],
    "background_texture_copyspace": [
        "a plain background texture or empty copy space image",
        "an abstract backdrop or wallpaper background",
    ],
    "render_or_graphic": [
        "a 3D render or computer generated image",
        "a CGI rendering rather than a camera photograph",
    ],
    "illustration_or_render": [
        "a vector illustration, icon, clip art, or drawing",
        "a graphic design image rather than a camera photograph",
    ],
}

PROMPT_GROUP_TO_REVIEW = {
    "studio_low_context_photo": (
        "review_studio_low_context",
        "studio_low_context_photo",
        "prompt_studio_low_context",
    ),
    "packshot_or_isolated_photo": (
        "review_packshot_isolated",
        "packshot_or_isolated_photo",
        "prompt_packshot_or_isolated",
    ),
    "background_texture_copyspace": (
        "review_background_copyspace",
        "background_texture_copyspace",
        "prompt_background_or_copyspace",
    ),
    "render_or_graphic": (
        "review_graphic_ambiguous",
        "ambiguous_media_type",
        "prompt_render_or_graphic",
    ),
    "illustration_or_render": (
        "review_graphic_ambiguous",
        "ambiguous_media_type",
        "prompt_illustration_or_graphic",
    ),
}

PROMPT_RESULT_COLUMNS = [
    "prompt_classifier_model",
    "prompt_classifier_error",
    "prompt_classifier_applied",
    "prompt_top_group",
    "prompt_top_prompt",
    "prompt_top_score",
    "prompt_camera_photo_score",
    "prompt_nonphoto_score",
]
PROMPT_RESULT_COLUMNS.extend([f"prompt_score_{group}" for group in PROMPT_GROUPS])


def split_tags(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, float) and math.isnan(raw):
        return []
    if isinstance(raw, (list, tuple, set, np.ndarray)):
        values: Iterable[Any] = raw
    else:
        text = str(raw)
        sep = "|" if "|" in text else ","
        values = text.split(sep)
    out = []
    for value in values:
        item = str(value).strip().lower()
        if item:
            out.append(item)
    return out


def _match_keywords(text: str, weights_or_terms: dict[str, float] | set[str]) -> list[str]:
    if isinstance(weights_or_terms, dict):
        terms = weights_or_terms.keys()
    else:
        terms = weights_or_terms
    return sorted(term for term in terms if term in text)


def metadata_signals(row: dict[str, Any]) -> dict[str, Any]:
    tags = split_tags(row.get("tags"))
    tag_text = " | ".join(tags)
    hard_matches = _match_keywords(tag_text, HARD_NONPHOTO_KEYWORDS)
    soft_matches = _match_keywords(tag_text, SOFT_NONPHOTO_KEYWORDS)
    packshot_matches = _match_keywords(tag_text, PACKSHOT_KEYWORDS)
    background_matches = _match_keywords(tag_text, BACKGROUND_KEYWORDS)
    ai_matches = _match_keywords(tag_text, AI_GENERATED_KEYWORDS)

    hard_score = sum(HARD_NONPHOTO_KEYWORDS[t] for t in hard_matches)
    soft_score = sum(SOFT_NONPHOTO_KEYWORDS[t] for t in soft_matches)
    ai_score = float(len(ai_matches))
    nonphoto_score = min(1.0, 0.55 * hard_score + 0.35 * soft_score + 0.5 * ai_score)

    return {
        "tag_count": len(tags),
        "metadata_hard_nonphoto_terms": hard_matches,
        "metadata_soft_nonphoto_terms": soft_matches,
        "metadata_packshot_terms": packshot_matches,
        "metadata_background_terms": background_matches,
        "metadata_ai_terms": ai_matches,
        "metadata_nonphoto_score": float(nonphoto_score),
        "metadata_nonphoto_flag": bool(hard_matches or soft_matches or ai_matches),
        "metadata_packshot_flag": bool(packshot_matches),
        "metadata_background_flag": bool(background_matches),
        "metadata_ai_flag": bool(ai_matches),
    }


def resolve_image_path(image_dir: Path, image_id: str) -> Path | None:
    stem = Path(str(image_id)).stem
    for ext in DEFAULT_IMAGE_EXTS:
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def image_stats(path: Path | None, *, max_side: int = 256) -> dict[str, Any]:
    if path is None:
        return {
            "image_found": False,
            "image_path": "",
            "pixel_error": "missing_image",
        }
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            width, height = im.size
            im.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
            small = im.copy()
    except Exception as exc:
        return {
            "image_found": False,
            "image_path": str(path),
            "pixel_error": str(exc),
        }

    arr = np.asarray(small).astype(np.float32)
    total = int(arr.shape[0] * arr.shape[1])
    if total <= 0:
        return {
            "image_found": False,
            "image_path": str(path),
            "pixel_error": "empty_image",
        }

    gray_img = small.convert("L")
    gray = np.asarray(gray_img).astype(np.uint8)
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    prob = hist / max(1.0, float(hist.sum()))
    prob_nz = prob[prob > 0]
    entropy = float(-(prob_nz * np.log2(prob_nz)).sum())

    q = (arr // 16).astype(np.uint16)
    packed = ((q[:, :, 0] << 8) | (q[:, :, 1] << 4) | q[:, :, 2]).ravel()
    counts = np.bincount(packed, minlength=4096)
    nonzero_counts = counts[counts > 0]
    quant_unique_count = int(len(nonzero_counts))
    dominant_color_ratio = float(nonzero_counts.max() / max(1, total))

    edges = np.asarray(gray_img.filter(ImageFilter.FIND_EDGES)).astype(np.float32)
    edge_density = float((edges > 20).mean())
    white_ratio = float(((arr[:, :, 0] > 245) & (arr[:, :, 1] > 245) & (arr[:, :, 2] > 245)).mean())
    near_black_ratio = float(((arr[:, :, 0] < 12) & (arr[:, :, 1] < 12) & (arr[:, :, 2] < 12)).mean())
    max_rgb = np.maximum(arr.max(axis=2), 1.0)
    sat = (arr.max(axis=2) - arr.min(axis=2)) / max_rgb
    saturation_mean = float(sat.mean())
    saturation_std = float(sat.std())

    flat_background_score = 0.0
    if white_ratio >= 0.80:
        flat_background_score += 0.45
    elif white_ratio >= 0.60:
        flat_background_score += 0.25
    if dominant_color_ratio >= 0.55:
        flat_background_score += 0.30
    elif dominant_color_ratio >= 0.35:
        flat_background_score += 0.15
    if entropy <= 3.0:
        flat_background_score += 0.25
    elif entropy <= 4.5:
        flat_background_score += 0.12
    if edge_density <= 0.08:
        flat_background_score += 0.15
    flat_background_score = min(1.0, flat_background_score)

    graphic_texture_score = 0.0
    if entropy <= 4.5:
        graphic_texture_score += 0.20
    if dominant_color_ratio >= 0.25:
        graphic_texture_score += 0.20
    if quant_unique_count <= 160:
        graphic_texture_score += 0.20
    if edge_density <= 0.08:
        graphic_texture_score += 0.15
    if saturation_std <= 0.08:
        graphic_texture_score += 0.10
    graphic_texture_score = min(1.0, graphic_texture_score)

    monochrome_score = 0.0
    if saturation_mean <= 0.03:
        monochrome_score = 1.0
    elif saturation_mean <= 0.10 and saturation_std <= 0.12 and quant_unique_count <= 256:
        monochrome_score = 0.85
    elif saturation_mean <= 0.15 and saturation_std <= 0.15 and quant_unique_count <= 256:
        monochrome_score = 0.70

    return {
        "image_found": True,
        "image_path": str(path),
        "pixel_error": "",
        "actual_width": int(width),
        "actual_height": int(height),
        "entropy_luma": entropy,
        "quant_unique_count": quant_unique_count,
        "dominant_color_ratio": dominant_color_ratio,
        "white_ratio": white_ratio,
        "near_black_ratio": near_black_ratio,
        "edge_density": edge_density,
        "saturation_mean": saturation_mean,
        "saturation_std": saturation_std,
        "monochrome_score": float(monochrome_score),
        "monochrome_flag": bool(monochrome_score >= 0.70),
        "flat_background_score": float(flat_background_score),
        "graphic_texture_score": float(graphic_texture_score),
    }


def classify_media(meta: dict[str, Any], pix: dict[str, Any]) -> dict[str, Any]:
    metadata_nonphoto = float(meta.get("metadata_nonphoto_score", 0.0))
    flat_score = float(pix.get("flat_background_score", 0.0) or 0.0)
    graphic_score = float(pix.get("graphic_texture_score", 0.0) or 0.0)
    white_ratio = float(pix.get("white_ratio", 0.0) or 0.0)
    near_black_ratio = float(pix.get("near_black_ratio", 0.0) or 0.0)
    dominant_color_ratio = float(pix.get("dominant_color_ratio", 0.0) or 0.0)
    entropy = float(pix.get("entropy_luma", 8.0) or 8.0)
    edge_density = float(pix.get("edge_density", 1.0) or 1.0)
    saturation_mean = float(pix.get("saturation_mean", 1.0) or 0.0)
    saturation_std = float(pix.get("saturation_std", 1.0) or 0.0)
    quant_unique_count = int(pix.get("quant_unique_count", 4096) or 4096)
    monochrome_score = float(pix.get("monochrome_score", 0.0) or 0.0)
    if monochrome_score <= 0.0:
        if saturation_mean <= 0.03:
            monochrome_score = 1.0
        elif saturation_mean <= 0.10 and saturation_std <= 0.12 and quant_unique_count <= 256:
            monochrome_score = 0.85
        elif saturation_mean <= 0.15 and saturation_std <= 0.15 and quant_unique_count <= 256:
            monochrome_score = 0.70
    hard_terms = set(meta.get("metadata_hard_nonphoto_terms", []))
    soft_terms = set(meta.get("metadata_soft_nonphoto_terms", []))
    packshot = bool(meta.get("metadata_packshot_flag")) or white_ratio >= 0.50
    background = bool(meta.get("metadata_background_flag"))
    copyspace = any("copy" in str(term) for term in meta.get("metadata_background_terms", []))
    white_studio = white_ratio >= 0.45 or (
        white_ratio >= 0.40 and edge_density <= 0.20 and dominant_color_ratio >= 0.40
    )
    dark_studio = near_black_ratio >= 0.45 or (
        near_black_ratio >= 0.40 and edge_density <= 0.13 and dominant_color_ratio >= 0.42
    )
    color_dominant_low_texture = (
        dominant_color_ratio >= 0.58 and entropy <= 5.4 and edge_density <= 0.16
    ) or (
        dominant_color_ratio >= 0.50 and entropy <= 6.2 and edge_density <= 0.08
    )
    desaturated_low_context = (
        saturation_mean <= 0.10
        and edge_density <= 0.085
        and dominant_color_ratio >= 0.32
        and entropy <= 6.0
        and (flat_score >= 0.30 or graphic_score >= 0.35)
    )
    monochrome_photo = monochrome_score >= 0.70
    background_low_context = background and (
        white_ratio >= 0.25
        or near_black_ratio >= 0.25
        or dominant_color_ratio >= 0.42
        or entropy <= 5.4
        or edge_density <= 0.10
    )

    reason: list[str] = []
    decision = "keep_photo_primary"
    primary = "photo_primary"

    if meta.get("metadata_ai_flag"):
        decision = "reject_nonphoto"
        primary = "ai_generated_or_declared"
        reason.append("metadata_ai_terms")
    elif hard_terms:
        decision = "reject_nonphoto"
        primary = "render_or_graphic"
        reason.append("hard_nonphoto_terms")
    elif "illustration" in soft_terms and (flat_score >= 0.30 or graphic_score >= 0.25 or packshot):
        decision = "reject_nonphoto"
        primary = "illustration_or_render"
        reason.append("illustration_with_pixel_support")
    elif metadata_nonphoto >= 0.70 and (flat_score >= 0.30 or graphic_score >= 0.25):
        decision = "reject_nonphoto"
        primary = "graphic_nonphoto"
        reason.append("metadata_pixel_nonphoto_consensus")
    elif flat_score >= 0.80 and not packshot:
        decision = "reject_flat_background"
        primary = "flat_background"
        reason.append("pixel_flat_background")
    elif copyspace or background_low_context:
        decision = "review_background_copyspace"
        primary = "background_texture_copyspace"
        reason.append("background_or_copyspace_low_context")
    elif dark_studio:
        decision = "review_studio_low_context"
        primary = "studio_low_context_photo"
        reason.append("dark_or_uniform_low_context")
    elif desaturated_low_context or (color_dominant_low_texture and (background or flat_score >= 0.30 or graphic_score >= 0.35)):
        decision = "review_studio_low_context"
        primary = "studio_low_context_photo"
        reason.append("desaturated_or_uniform_low_context")
    elif packshot:
        decision = "review_packshot_isolated"
        primary = "packshot_or_isolated_photo"
        reason.append("packshot_or_white_background")
    elif white_studio:
        decision = "review_studio_low_context"
        primary = "studio_low_context_photo"
        reason.append("white_or_uniform_low_context")
    elif monochrome_photo:
        decision = "review_monochrome_desaturated"
        primary = "monochrome_or_desaturated_photo"
        reason.append("monochrome_or_desaturated_photo")
    elif soft_terms or metadata_nonphoto >= 0.35 or graphic_score >= 0.45:
        decision = "review_graphic_ambiguous"
        primary = "ambiguous_media_type"
        reason.append("ambiguous_nonphoto_signal")
    else:
        reason.append("no_nonphoto_signal")

    nonphoto_score = min(1.0, max(metadata_nonphoto, 0.6 * graphic_score + 0.4 * flat_score))
    photo_likeness_score = max(0.0, min(1.0, 1.0 - nonphoto_score))
    if decision.startswith("review_packshot"):
        photo_likeness_score = min(photo_likeness_score, 0.65)
    if decision.startswith("reject"):
        photo_likeness_score = min(photo_likeness_score, 0.25)

    return {
        "media_type_primary": primary,
        "media_decision": decision,
        "media_reject_reason": "|".join(reason),
        "nonphoto_score": float(nonphoto_score),
        "photo_likeness_score": float(photo_likeness_score),
        "curation_policy_version": POLICY_VERSION,
    }


def presampling_metadata_gate(row: dict[str, Any]) -> dict[str, Any]:
    meta = metadata_signals(row)
    hard_terms = set(meta.get("metadata_hard_nonphoto_terms", []))
    soft_terms = set(meta.get("metadata_soft_nonphoto_terms", []))
    reason: list[str] = []

    decision = "keep_photo_primary"
    primary = "photo_primary"

    if meta.get("metadata_ai_flag"):
        decision = "reject_nonphoto"
        primary = "ai_generated_or_declared"
        reason.append("metadata_ai_terms")
    elif hard_terms:
        decision = "reject_nonphoto"
        primary = "render_or_graphic"
        reason.append("hard_nonphoto_terms")
    elif meta.get("metadata_background_flag"):
        decision = "review_background_copyspace"
        primary = "background_texture_copyspace"
        reason.append("metadata_background_terms")
    elif meta.get("metadata_packshot_flag"):
        decision = "review_packshot_isolated"
        primary = "packshot_or_isolated_photo"
        reason.append("metadata_packshot_terms")
    elif soft_terms:
        decision = "review_graphic_ambiguous"
        primary = "ambiguous_media_type"
        reason.append("soft_nonphoto_terms")
    else:
        reason.append("no_metadata_nonphoto_signal")

    return {
        **meta,
        "presample_media_type_primary": primary,
        "presample_media_decision": decision,
        "presample_media_reject_reason": "|".join(reason),
    }


def _resolve_prompt_model(mode: str, model_id: str) -> str:
    requested = str(model_id or "").strip()
    if requested:
        return requested
    if str(mode).strip().lower() == "clip":
        return DEFAULT_CLIP_CLASSIFIER_MODEL
    return DEFAULT_PROMPT_CLASSIFIER_MODEL


def _prompt_labels() -> tuple[list[str], list[str]]:
    groups: list[str] = []
    prompts: list[str] = []
    for group, group_prompts in PROMPT_GROUPS.items():
        for prompt in group_prompts:
            groups.append(group)
            prompts.append(prompt)
    return groups, prompts


def _move_batch_to_device(batch: Any, device: str) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device)
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in dict(batch).items()
    }


class PromptMediaClassifier:
    def __init__(self, model_id: str, device: str = "auto") -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModel, AutoProcessor, AutoTokenizer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch = torch
        self.model_id = model_id
        self.device = device
        self.groups, self.prompts = _prompt_labels()
        self.processor = None
        self.image_processor = None
        self.tokenizer = None
        try:
            self.processor = AutoProcessor.from_pretrained(model_id)
        except Exception as exc:
            # Older transformers can recognize SigLIP2 as a SigLIP model but may
            # try to build the wrong combined processor. Loading the tokenizer
            # and image processor separately keeps the optional path usable.
            print(f"[prompt-classifier] AutoProcessor fallback for {model_id}: {exc}")
            self.image_processor = AutoImageProcessor.from_pretrained(model_id)
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()
        if self.processor is not None:
            text_inputs = self.processor(text=self.prompts, padding=True, return_tensors="pt")
        else:
            text_inputs = self.tokenizer(self.prompts, padding=True, return_tensors="pt")
        text_inputs = _move_batch_to_device(text_inputs, device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**text_inputs)
            self.text_features = torch.nn.functional.normalize(text_features, dim=-1)

    def classify_paths(self, paths: list[Path]) -> list[dict[str, Any]]:
        images = []
        valid_indices: list[int] = []
        results: list[dict[str, Any]] = []
        for path in paths:
            results.append(
                {
                    "prompt_classifier_model": self.model_id,
                    "prompt_classifier_error": "",
                    "prompt_classifier_applied": False,
                }
            )
            try:
                with Image.open(path) as im:
                    images.append(im.convert("RGB").copy())
                    valid_indices.append(len(results) - 1)
            except Exception as exc:
                results[-1]["prompt_classifier_error"] = str(exc)

        if not images:
            return results

        if self.processor is not None:
            inputs = self.processor(images=images, return_tensors="pt")
        else:
            inputs = self.image_processor(images=images, return_tensors="pt")
        inputs = _move_batch_to_device(inputs, self.device)
        with self.torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = self.torch.nn.functional.normalize(image_features, dim=-1)
            logits = image_features @ self.text_features.T
            logit_scale = getattr(self.model, "logit_scale", None)
            if logit_scale is not None:
                logits = logits * logit_scale.exp()
            logit_bias = getattr(self.model, "logit_bias", None)
            if logit_bias is not None:
                logits = logits + logit_bias
            probs = self.torch.softmax(logits, dim=-1).detach().cpu().numpy()

        for batch_idx, result_idx in enumerate(valid_indices):
            row_probs = probs[batch_idx]
            group_scores: dict[str, float] = {}
            for group in PROMPT_GROUPS:
                idxs = [idx for idx, g in enumerate(self.groups) if g == group]
                group_scores[group] = float(max(row_probs[idx] for idx in idxs))
            top_prompt_idx = int(row_probs.argmax())
            top_group = self.groups[top_prompt_idx]
            nonphoto_score = max(
                score for group, score in group_scores.items() if group != "camera_photo"
            )
            results[result_idx].update(
                {
                    "prompt_classifier_applied": True,
                    "prompt_top_group": top_group,
                    "prompt_top_prompt": self.prompts[top_prompt_idx],
                    "prompt_top_score": float(row_probs[top_prompt_idx]),
                    "prompt_camera_photo_score": float(group_scores.get("camera_photo", 0.0)),
                    "prompt_nonphoto_score": float(nonphoto_score),
                }
            )
            for group, score in group_scores.items():
                results[result_idx][f"prompt_score_{group}"] = float(score)
        return results


def run_prompt_classifier(
    df: pd.DataFrame,
    image_dir: Path,
    *,
    mode: str,
    model_id: str,
    device: str,
    batch_size: int,
    fail_on_error: bool,
    status_path: Path | None = None,
    status_base: dict[str, Any] | None = None,
    progress_interval: int = 50,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    resolved_model = _resolve_prompt_model(mode, model_id)
    status_base = dict(status_base or {})
    summary: dict[str, Any] = {
        "enabled": True,
        "mode": mode,
        "model_id": resolved_model,
        "device": device,
        "batch_size": int(batch_size),
        "load_error": "",
        "classified": 0,
        "errors": 0,
        "total_rows": int(len(df)),
    }
    write_status(
        status_path,
        {
            **status_base,
            "status": "running",
            "phase": "prompt_classifier_load",
            "prompt_classifier": summary,
        },
    )
    try:
        classifier = PromptMediaClassifier(resolved_model, device=device)
        summary["device"] = classifier.device
    except Exception as exc:
        summary["load_error"] = str(exc)
        write_status(
            status_path,
            {
                **status_base,
                "status": "failed" if fail_on_error else "running",
                "phase": "prompt_classifier_load_failed",
                "prompt_classifier": summary,
                "error": str(exc),
            },
        )
        if fail_on_error:
            raise
        print(f"[prompt-classifier] disabled after load failure: {exc}")
        return {}, summary

    results: dict[str, dict[str, Any]] = {}
    batch_paths: list[Path] = []
    batch_ids: list[str] = []
    batch_size = max(1, int(batch_size))
    progress_interval = max(1, int(progress_interval))
    rows = df[["image_id"]].to_dict(orient="records")
    last_status_done = 0
    try:
        from tqdm import tqdm

        row_iter = tqdm(rows, total=len(rows), desc="prompt-media-classifier")
    except Exception:
        row_iter = rows

    def maybe_write_prompt_status(force: bool = False) -> None:
        nonlocal last_status_done
        done = int(summary["classified"] + summary["errors"])
        if not force and done - last_status_done < progress_interval:
            return
        last_status_done = done
        write_status(
            status_path,
            {
                **status_base,
                "status": "running",
                "phase": "prompt_classifier",
                "prompt_classifier": summary,
                "prompt_classifier_done": done,
                "prompt_classifier_total": int(len(rows)),
            },
        )

    def flush() -> None:
        if not batch_paths:
            return
        batch_results = classifier.classify_paths(batch_paths)
        for image_id, result in zip(batch_ids, batch_results):
            results[image_id] = result
            if result.get("prompt_classifier_applied"):
                summary["classified"] += 1
            if result.get("prompt_classifier_error"):
                summary["errors"] += 1
        batch_paths.clear()
        batch_ids.clear()
        maybe_write_prompt_status()

    for row in row_iter:
        image_id = str(row["image_id"])
        path = resolve_image_path(image_dir, image_id)
        if path is None:
            results[image_id] = {
                "prompt_classifier_model": resolved_model,
                "prompt_classifier_applied": False,
                "prompt_classifier_error": "missing_image",
            }
            summary["errors"] += 1
            maybe_write_prompt_status()
            continue
        batch_paths.append(path)
        batch_ids.append(image_id)
        if len(batch_paths) >= batch_size:
            flush()
    flush()
    maybe_write_prompt_status(force=True)
    return results, summary


def reuse_prompt_classifier_columns(
    df: pd.DataFrame,
    *,
    mode: str,
    model_id: str,
    device: str,
    batch_size: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    resolved_model = _resolve_prompt_model(mode, model_id)
    available_cols = [col for col in PROMPT_RESULT_COLUMNS if col in df.columns]
    if "prompt_classifier_applied" not in available_cols:
        raise ValueError("input parquet does not contain prompt_classifier_applied")

    results: dict[str, dict[str, Any]] = {}
    classified = 0
    errors = 0
    model_values: list[str] = []
    for row in df[["image_id", *available_cols]].to_dict(orient="records"):
        image_id = str(row.get("image_id", ""))
        prompt: dict[str, Any] = {}
        for col in available_cols:
            value = row.get(col)
            if isinstance(value, float) and math.isnan(value):
                continue
            prompt[col] = _safe_value(value)
        if prompt.get("prompt_classifier_model"):
            model_values.append(str(prompt["prompt_classifier_model"]))
        if bool(prompt.get("prompt_classifier_applied")):
            classified += 1
        if str(prompt.get("prompt_classifier_error", "") or ""):
            errors += 1
        results[image_id] = prompt

    summary: dict[str, Any] = {
        "enabled": True,
        "mode": mode,
        "model_id": model_values[0] if model_values else resolved_model,
        "device": device,
        "batch_size": int(batch_size),
        "load_error": "",
        "classified": int(classified),
        "errors": int(errors),
        "total_rows": int(len(df)),
        "reused_columns": True,
    }
    return results, summary


def apply_prompt_classifier_result(
    audit: dict[str, Any],
    prompt: dict[str, Any] | None,
    *,
    min_score: float = 0.28,
    min_margin: float = 0.03,
) -> dict[str, Any]:
    out = dict(audit)
    out["heuristic_media_decision"] = audit.get("media_decision")
    out["heuristic_media_type_primary"] = audit.get("media_type_primary")
    out["heuristic_media_reject_reason"] = audit.get("media_reject_reason")
    out["prompt_classifier_adjusted"] = False
    if not prompt:
        return out

    out.update(prompt)
    if prompt.get("prompt_classifier_error") or not prompt.get("prompt_classifier_applied"):
        return out

    top_group = str(prompt.get("prompt_top_group", ""))
    if top_group not in PROMPT_GROUP_TO_REVIEW:
        return out
    if str(audit.get("media_decision")) != "keep_photo_primary":
        return out

    top_score = float(prompt.get("prompt_score_" + top_group, prompt.get("prompt_top_score", 0.0)) or 0.0)
    camera_score = float(prompt.get("prompt_camera_photo_score", 0.0) or 0.0)
    if top_score < float(min_score) or (top_score - camera_score) < float(min_margin):
        return out

    decision, primary, reason = PROMPT_GROUP_TO_REVIEW[top_group]
    out["media_decision"] = decision
    out["media_type_primary"] = primary
    old_reason = str(out.get("media_reject_reason", "") or "")
    out["media_reject_reason"] = f"{old_reason}|{reason}" if old_reason else reason
    out["prompt_classifier_adjusted"] = True
    prompt_nonphoto = float(prompt.get("prompt_nonphoto_score", 0.0) or 0.0)
    out["nonphoto_score"] = float(max(float(out.get("nonphoto_score", 0.0) or 0.0), prompt_nonphoto))
    out["photo_likeness_score"] = float(min(float(out.get("photo_likeness_score", 1.0) or 1.0), 1.0 - prompt_nonphoto))
    return out


def audit_row(row: dict[str, Any], image_dir: Path) -> dict[str, Any]:
    image_id = str(row.get("image_id", ""))
    meta = metadata_signals(row)
    pix = image_stats(resolve_image_path(image_dir, image_id))
    cls = classify_media(meta, pix)
    return {
        "image_id": image_id,
        **meta,
        **pix,
        **cls,
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")


def _safe_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    payload = {str(k): _safe_value(v) for k, v in payload.items()}
    payload["last_update_utc"] = _utc_now()
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_path, path)


def write_summary(path: Path, audited: pd.DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    decisions = audited["media_decision"].value_counts(dropna=False).to_dict()
    primary = audited["media_type_primary"].value_counts(dropna=False).to_dict()
    missing_images = int((~audited["image_found"].astype(bool)).sum()) if "image_found" in audited else 0
    explicit_nonphoto = int(audited["metadata_nonphoto_flag"].astype(bool).sum())
    summary: dict[str, Any] = {
        "policy_version": POLICY_VERSION,
        "input_parquet": str(args.input_parquet),
        "image_dir": str(args.image_dir),
        "output_dir": str(args.output_dir),
        "total_rows": int(len(audited)),
        "max_images": int(args.max_images),
        "missing_images": missing_images,
        "metadata_nonphoto_flag_count": explicit_nonphoto,
        "decision_counts": {str(k): int(v) for k, v in decisions.items()},
        "media_type_primary_counts": {str(k): int(v) for k, v in primary.items()},
    }
    if hasattr(args, "_prompt_classifier_summary"):
        summary["prompt_classifier"] = getattr(args, "_prompt_classifier_summary")
        if "prompt_classifier_adjusted" in audited.columns:
            summary["prompt_classifier"]["adjusted_rows"] = int(
                audited["prompt_classifier_adjusted"].fillna(False).astype(bool).sum()
            )
    if "super_cat" in audited.columns:
        by_cat = (
            audited.groupby(["super_cat", "media_decision"], dropna=False)
            .size()
            .reset_index(name="count")
        )
        summary["by_super_cat_decision"] = [
            {str(k): _safe_value(v) for k, v in row.items()}
            for row in by_cat.to_dict(orient="records")
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def write_markdown_report(path: Path, summary: dict[str, Any], audited: pd.DataFrame) -> None:
    lines = [
        "# SSTK Media Type Audit Report",
        "",
        f"- Policy version: `{summary['policy_version']}`",
        f"- Input parquet: `{summary['input_parquet']}`",
        f"- Image dir: `{summary['image_dir']}`",
        f"- Total rows audited: `{summary['total_rows']}`",
        f"- Missing images: `{summary['missing_images']}`",
        "",
        "## Decision Counts",
        "",
        "| Decision | Count |",
        "| --- | ---: |",
    ]
    for key, value in sorted(summary["decision_counts"].items()):
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "## Media Type Counts", "", "| Type | Count |", "| --- | ---: |"])
    for key, value in sorted(summary["media_type_primary_counts"].items()):
        lines.append(f"| `{key}` | {value} |")

    lines.extend(["", "## High Risk Examples", ""])
    cols = ["image_id", "media_decision", "media_type_primary", "media_reject_reason", "nonphoto_score"]
    examples = audited.sort_values("nonphoto_score", ascending=False).head(20)
    lines.extend(["| Image ID | Decision | Type | Reason | Nonphoto |", "| --- | --- | --- | --- | ---: |"])
    for _, row in examples.iterrows():
        lines.append(
            f"| `{row.get('image_id')}` | `{row.get('media_decision')}` | "
            f"`{row.get('media_type_primary')}` | `{row.get('media_reject_reason')}` | "
            f"{float(row.get('nonphoto_score', 0.0)):.3f} |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_split_parquets(audited: pd.DataFrame, out_dir: Path) -> dict[str, str]:
    split_paths = {
        "audited": out_dir / "filtered_sstk_media_audited.parquet",
        "photo_primary": out_dir / "filtered_sstk_photo_primary.parquet",
        "media_review": out_dir / "filtered_sstk_media_review.parquet",
        "packshot_review": out_dir / "filtered_sstk_packshot_review.parquet",
        "reject_nonphoto": out_dir / "filtered_sstk_reject_nonphoto.parquet",
    }
    audited.to_parquet(split_paths["audited"], index=False)
    audited[audited["media_decision"] == "keep_photo_primary"].to_parquet(split_paths["photo_primary"], index=False)
    audited[audited["media_decision"].astype(str).str.startswith("review_")].to_parquet(
        split_paths["media_review"], index=False
    )
    audited[audited["media_decision"] == "review_packshot_isolated"].to_parquet(
        split_paths["packshot_review"], index=False
    )
    audited[audited["media_decision"].astype(str).str.startswith("reject_")].to_parquet(
        split_paths["reject_nonphoto"], index=False
    )
    return {k: str(v) for k, v in split_paths.items()}


def load_mapped_super_cat(mapped_path: Path, image_ids: set[str]) -> pd.DataFrame:
    if not mapped_path.exists() or not image_ids:
        return pd.DataFrame(columns=["image_id", "super_cat"])
    try:
        import pyarrow.compute as pc
        import pyarrow.dataset as ds

        dataset = ds.dataset(str(mapped_path), format="parquet")
        filt = pc.field("image_id").isin(list(image_ids))
        table = dataset.to_table(columns=["image_id", "super_cat"], filter=filt)
        return table.to_pandas().drop_duplicates("image_id")
    except Exception:
        mapped = pd.read_parquet(mapped_path, columns=["image_id", "super_cat"])
        return mapped[mapped["image_id"].astype(str).isin(image_ids)].drop_duplicates("image_id")


def write_contact_sheets(audited: pd.DataFrame, image_dir: Path, out_dir: Path, *, max_per_sheet: int = 40) -> None:
    sheet_dir = out_dir / "contact_sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    decisions = [
        "reject_nonphoto",
        "reject_flat_background",
        "review_studio_low_context",
        "review_background_copyspace",
        "review_monochrome_desaturated",
        "review_graphic_ambiguous",
        "review_packshot_isolated",
        "keep_photo_primary",
    ]
    font = ImageFont.load_default()
    for decision in decisions:
        subset = audited[audited["media_decision"] == decision].copy()
        if subset.empty:
            continue
        subset = subset.sort_values("nonphoto_score", ascending=False).head(max_per_sheet)
        thumb_w, thumb_h = 160, 120
        label_h = 34
        cols = 5
        rows = int(math.ceil(len(subset) / cols))
        canvas = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), "white")
        draw = ImageDraw.Draw(canvas)
        for idx, (_, row) in enumerate(subset.iterrows()):
            x = (idx % cols) * thumb_w
            y = (idx // cols) * (thumb_h + label_h)
            p = resolve_image_path(image_dir, str(row["image_id"]))
            if p is not None:
                try:
                    with Image.open(p) as im:
                        im = im.convert("RGB")
                        im.thumbnail((thumb_w, thumb_h), Image.Resampling.BILINEAR)
                        px = x + (thumb_w - im.width) // 2
                        py = y + (thumb_h - im.height) // 2
                        canvas.paste(im, (px, py))
                except Exception:
                    pass
            label = f"{row['image_id']}\n{float(row.get('nonphoto_score', 0.0)):.2f}"
            draw.text((x + 3, y + thumb_h + 2), label[:54], fill="black", font=font)
        canvas.save(sheet_dir / f"{decision}.jpg", quality=90)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit SSTK curated pool media type quality.")
    p.add_argument("--input_parquet", required=True)
    p.add_argument("--image_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--mapped_cache_parquet", default="")
    p.add_argument("--max_images", type=int, default=0, help="0=all rows")
    p.add_argument("--image_ids", default="", help="comma-separated image ids to audit")
    p.add_argument("--write_contact_sheets", type=int, default=1)
    p.add_argument("--contact_sheet_max_per_decision", type=int, default=40)
    p.add_argument(
        "--prompt_classifier",
        default="none",
        choices=["none", "siglip2", "clip", "auto"],
        help="Optional prompt VLM classifier used only to demote keep_photo_primary rows to review.",
    )
    p.add_argument(
        "--prompt_classifier_model",
        default="",
        help="HF model id. Empty uses google/siglip2-base-patch16-224 for siglip2/auto, openai/clip-vit-base-patch32 for clip.",
    )
    p.add_argument("--prompt_classifier_device", default="auto", help="auto|cpu|cuda|cuda:0")
    p.add_argument("--prompt_classifier_batch_size", type=int, default=16)
    p.add_argument("--prompt_classifier_demote_min_score", type=float, default=0.28)
    p.add_argument("--prompt_classifier_demote_margin", type=float, default=0.03)
    p.add_argument(
        "--reuse_prompt_classifier_columns",
        type=int,
        default=0,
        help="If 1, reuse prompt_* columns from input_parquet instead of loading the prompt model.",
    )
    p.add_argument(
        "--prompt_classifier_fail_on_error",
        type=int,
        default=0,
        help="If 1, fail the audit when the optional prompt classifier cannot load.",
    )
    p.add_argument(
        "--status_json",
        default="",
        help="Optional durable progress JSON path. Default: <output_dir>/status.json.",
    )
    p.add_argument(
        "--progress_interval",
        type=int,
        default=50,
        help="Rows between durable status updates for long audit runs.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    input_parquet = Path(args.input_parquet)
    image_dir = Path(args.image_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = Path(args.status_json) if str(args.status_json).strip() else out_dir / "status.json"
    started = time.time()
    progress_interval = max(1, int(args.progress_interval))
    status_base: dict[str, Any] = {
        "pid": os.getpid(),
        "policy_version": POLICY_VERSION,
        "input_parquet": str(input_parquet),
        "image_dir": str(image_dir),
        "output_dir": str(out_dir),
        "status_json": str(status_path),
        "started_utc": _utc_now(),
    }
    write_status(status_path, {**status_base, "status": "running", "phase": "load_input"})

    try:
        df = pd.read_parquet(input_parquet)
        if args.image_ids.strip():
            allow = {x.strip() for x in args.image_ids.split(",") if x.strip()}
            df = df[df["image_id"].astype(str).isin(allow)].copy()
        if args.max_images > 0:
            df = df.head(int(args.max_images)).copy()

        if args.mapped_cache_parquet:
            mapped_path = Path(args.mapped_cache_parquet)
            mapped = load_mapped_super_cat(mapped_path, set(df["image_id"].astype(str).tolist()))
            if not mapped.empty:
                df = df.merge(mapped, on="image_id", how="left")

        status_base["total_rows"] = int(len(df))
        write_status(
            status_path,
            {
                **status_base,
                "status": "running",
                "phase": "input_loaded",
                "total_rows": int(len(df)),
            },
        )

        prompt_results: dict[str, dict[str, Any]] = {}
        if str(args.prompt_classifier).strip().lower() != "none":
            if int(args.reuse_prompt_classifier_columns):
                prompt_summary: dict[str, Any]
                prompt_results, prompt_summary = reuse_prompt_classifier_columns(
                    df,
                    mode=str(args.prompt_classifier).strip().lower(),
                    model_id=str(args.prompt_classifier_model),
                    device=str(args.prompt_classifier_device),
                    batch_size=int(args.prompt_classifier_batch_size),
                )
                write_status(
                    status_path,
                    {
                        **status_base,
                        "status": "running",
                        "phase": "prompt_classifier_reused",
                        "prompt_classifier": prompt_summary,
                    },
                )
            else:
                prompt_results, prompt_summary = run_prompt_classifier(
                    df,
                    image_dir,
                    mode=str(args.prompt_classifier).strip().lower(),
                    model_id=str(args.prompt_classifier_model),
                    device=str(args.prompt_classifier_device),
                    batch_size=int(args.prompt_classifier_batch_size),
                    fail_on_error=bool(int(args.prompt_classifier_fail_on_error)),
                    status_path=status_path,
                    status_base=status_base,
                    progress_interval=progress_interval,
                )
            args._prompt_classifier_summary = prompt_summary

        records: list[dict[str, Any]] = []
        try:
            from tqdm import tqdm

            row_iter = tqdm(df.iterrows(), total=len(df), desc="media-type-audit")
        except Exception:
            row_iter = df.iterrows()

        processed = 0
        last_status_processed = 0
        for _, row in row_iter:
            base = {str(k): _safe_value(v) for k, v in row.to_dict().items()}
            audit = audit_row(base, image_dir)
            if prompt_results:
                audit = apply_prompt_classifier_result(
                    audit,
                    prompt_results.get(str(base.get("image_id", ""))),
                    min_score=float(args.prompt_classifier_demote_min_score),
                    min_margin=float(args.prompt_classifier_demote_margin),
                )
            records.append({**base, **audit})
            processed += 1
            if processed - last_status_processed >= progress_interval:
                last_status_processed = processed
                write_status(
                    status_path,
                    {
                        **status_base,
                        "status": "running",
                        "phase": "media_type_audit",
                        "audit_processed": int(processed),
                        "audit_total": int(len(df)),
                    },
                )

        write_status(
            status_path,
            {
                **status_base,
                "status": "running",
                "phase": "write_outputs",
                "audit_processed": int(processed),
                "audit_total": int(len(df)),
            },
        )
        audited = pd.DataFrame(records)
        split_paths = write_split_parquets(audited, out_dir)
        write_jsonl(out_dir / "media_type_audit.jsonl", records)
        summary = write_summary(out_dir / "media_type_audit_summary.json", audited, args)
        summary["split_paths"] = split_paths
        summary_path = out_dir / "media_type_audit_summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        write_markdown_report(out_dir / "MEDIA_TYPE_AUDIT_REPORT.md", summary, audited)
        if int(args.write_contact_sheets):
            write_contact_sheets(
                audited,
                image_dir,
                out_dir,
                max_per_sheet=int(args.contact_sheet_max_per_decision),
            )

        write_status(
            status_path,
            {
                **status_base,
                "status": "completed",
                "phase": "complete",
                "elapsed_sec": round(time.time() - started, 3),
                "summary_json": str(summary_path),
                "report_md": str(out_dir / "MEDIA_TYPE_AUDIT_REPORT.md"),
                "split_paths": split_paths,
                "decision_counts": summary.get("decision_counts", {}),
                "prompt_classifier": summary.get("prompt_classifier", {}),
            },
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    except Exception as exc:
        write_status(
            status_path,
            {
                **status_base,
                "status": "failed",
                "phase": "failed",
                "elapsed_sec": round(time.time() - started, 3),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    main()
