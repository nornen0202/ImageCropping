#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


DEFAULT_PHASE_ROOT = Path("data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529")
DEFAULT_V16_TAG = "260609_v16_routing_v2_mode_catalog_nofood_petstrict_shotstrict_nms_v2"
DEFAULT_LABEL_JSONL = DEFAULT_PHASE_ROOT / "artifacts/training_labels" / DEFAULT_V16_TAG / "routing_v2_v16_full.jsonl"
DEFAULT_MULTIMODE_JSON = DEFAULT_PHASE_ROOT / "artifacts/training_labels_multimode" / DEFAULT_V16_TAG / "label_json/multimode_labels_full.json"
DEFAULT_OUTPUT_DIR = DEFAULT_PHASE_ROOT / "artifacts/training_labels_multimode" / DEFAULT_V16_TAG / "routing_v2_auto_triage_decision_pack_260609"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build reviewer decision pack for routing_v2 follow-up items.")
    parser.add_argument("--label_jsonl", type=Path, default=DEFAULT_LABEL_JSONL)
    parser.add_argument("--multimode_json", type=Path, default=DEFAULT_MULTIMODE_JSON)
    parser.add_argument("--image_root", type=Path, default=DEFAULT_PHASE_ROOT / "images")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_per_bucket", type=int, default=80)
    parser.add_argument("--contact_cols", type=int, default=5)
    parser.add_argument("--thumb_w", type=int, default=260)
    parser.add_argument("--thumb_h", type=int, default=190)
    return parser


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _source_id(image: dict[str, Any]) -> str:
    return str(image.get("source_image_id") or Path(str(image.get("file_name") or "")).stem)


def _ann_score(ann: dict[str, Any]) -> float:
    for key in ("score_mode", "score", "final_score"):
        if ann.get(key) is not None:
            return _safe_float(ann.get(key))
    attrs = ann.get("attributes") if isinstance(ann.get("attributes"), dict) else {}
    scores = attrs.get("checklist_scores") if isinstance(attrs.get("checklist_scores"), dict) else {}
    return _safe_float(scores.get("final_score"))


def _ann_target_ar(ann: dict[str, Any]) -> str:
    attrs = ann.get("attributes") if isinstance(ann.get("attributes"), dict) else {}
    return str(attrs.get("target_ar") or ann.get("target_ar") or "FREE")


def _ann_routing(ann: dict[str, Any]) -> dict[str, Any]:
    attrs = ann.get("attributes") if isinstance(ann.get("attributes"), dict) else {}
    routing = attrs.get("routing_v2_simple")
    return routing if isinstance(routing, dict) else {}


def _load_positive_votes(path: Path) -> dict[str, dict[str, Any]]:
    payload = _read_json(path)
    images = payload.get("images") if isinstance(payload.get("images"), list) else []
    anns = payload.get("annotations") if isinstance(payload.get("annotations"), list) else []
    id_to_source = {int(img.get("id")): _source_id(img) for img in images if img.get("id") is not None}
    out: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "positive_count": 0,
        "best_score": 0.0,
        "positive_mode_counts": Counter(),
        "positive_route_counts": Counter(),
        "positive_ar_mode_counts": Counter(),
    })
    for ann in anns:
        if not isinstance(ann, dict) or _safe_int(ann.get("gt_flag"), 0) != 1:
            continue
        source_id = id_to_source.get(_safe_int(ann.get("image_id"), -1))
        if not source_id:
            continue
        mode = str(ann.get("mode_name") or "unknown")
        routing = _ann_routing(ann)
        route = str(routing.get("route_family_v2") or "missing")
        target_ar = _ann_target_ar(ann)
        bucket = out[source_id]
        bucket["positive_count"] += 1
        bucket["best_score"] = max(float(bucket["best_score"]), _ann_score(ann))
        bucket["positive_mode_counts"][mode] += 1
        bucket["positive_route_counts"][route] += 1
        bucket["positive_ar_mode_counts"][f"{mode}|{target_ar}"] += 1
    serializable = {}
    for source_id, item in out.items():
        serializable[source_id] = {
            "positive_count": int(item["positive_count"]),
            "best_score": round(float(item["best_score"]), 6),
            "positive_mode_counts": dict(item["positive_mode_counts"]),
            "positive_route_counts": dict(item["positive_route_counts"]),
            "positive_ar_mode_counts": dict(item["positive_ar_mode_counts"]),
        }
    return serializable


def _resolve_image(image_root: Path, row: dict[str, Any]) -> Path:
    image_id = str(row.get("image_id") or "")
    image_path = Path(str(row.get("image_path") or ""))
    if image_path.exists():
        return image_path
    if image_path.name:
        candidate = image_root / image_path.name
        if candidate.exists():
            return candidate
    file_name = str(row.get("file_name") or "")
    if file_name:
        candidate = image_root / file_name
        if candidate.exists():
            return candidate
    for suffix in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = image_root / f"{image_id}{suffix}"
        if candidate.exists():
            return candidate
    return image_root / f"{image_id}.jpg"


def _flatten_row(row: dict[str, Any], votes: dict[str, Any], image_root: Path) -> dict[str, Any]:
    routing = row.get("routing_v2_simple") if isinstance(row.get("routing_v2_simple"), dict) else {}
    feats = row.get("probe_features") if isinstance(row.get("probe_features"), dict) else {}
    food = routing.get("food_evidence") if isinstance(routing.get("food_evidence"), dict) else {}
    dogcat = routing.get("dogcat_evidence") if isinstance(routing.get("dogcat_evidence"), dict) else {}
    person = routing.get("person_pose_signals") if isinstance(routing.get("person_pose_signals"), dict) else {}
    env = routing.get("environmental_portrait_evidence") if isinstance(routing.get("environmental_portrait_evidence"), dict) else {}
    image_id = str(row.get("image_id") or "")
    vote = votes.get(image_id, {})
    positive_routes = vote.get("positive_route_counts") if isinstance(vote.get("positive_route_counts"), dict) else {}
    positive_modes = vote.get("positive_mode_counts") if isinstance(vote.get("positive_mode_counts"), dict) else {}
    positive_ar_modes = vote.get("positive_ar_mode_counts") if isinstance(vote.get("positive_ar_mode_counts"), dict) else {}
    image_path = _resolve_image(image_root, row)
    food_signal = max(_safe_float(food.get("food_score_max")), _safe_float(food.get("tableware_score_max")))
    dogcat_signal = _safe_float(dogcat.get("dogcat_score_max"))
    person_signal = max(_safe_float(feats.get("num_person")), _safe_float(feats.get("num_person_c2")), 1.0 if person.get("pose_available") else 0.0)
    env_score = _safe_float(env.get("environmental_score"), _safe_float(feats.get("environmental_environmental_score")))
    out = {
        "image_id": image_id,
        "file_name": str(row.get("file_name") or image_path.name),
        "image_path": str(image_path),
        "image_route": str(row.get("image_route_name_no_placement") or ""),
        "image_route_id": _safe_int(row.get("image_route_id_no_placement"), -1),
        "route_family_v2": str(routing.get("route_family_v2") or ""),
        "person_shot_type": str(routing.get("person_shot_type") or "na"),
        "context_intent": str(routing.get("context_intent") or ""),
        "placement_aux": str(routing.get("placement_intent") or ""),
        "v16_mode_aux": str(row.get("v16_mode_name") or ""),
        "flat_route_class_aux": str(row.get("flat_route_class") or ""),
        "routing_confidence": round(_safe_float(routing.get("routing_confidence"), _safe_float(row.get("label_weight"))), 6),
        "teacher_reliability": str(routing.get("teacher_reliability") or ""),
        "legacy_subject_mode": str(routing.get("legacy_subject_mode") or ""),
        "positive_vote_route_family_share": round(_safe_float(routing.get("positive_vote_route_family_share")), 6),
        "positive_vote_placement_share_aux": round(_safe_float(routing.get("positive_vote_placement_share")), 6),
        "positive_count": _safe_int(vote.get("positive_count"), 0),
        "best_score": round(_safe_float(vote.get("best_score")), 6),
        "positive_route_counts_json": json.dumps(positive_routes, ensure_ascii=False, sort_keys=True),
        "positive_mode_counts_json": json.dumps(positive_modes, ensure_ascii=False, sort_keys=True),
        "positive_ar_mode_counts_json": json.dumps(positive_ar_modes, ensure_ascii=False, sort_keys=True),
        "positive_person_single_count": _safe_int(positive_routes.get("person_single"), 0),
        "positive_pet_dogcat_count": _safe_int(positive_routes.get("pet_dogcat"), 0),
        "positive_food_count": _safe_int(positive_routes.get("food"), 0),
        "positive_scene_count": _safe_int(positive_routes.get("scene"), 0),
        "food_count": _safe_int(food.get("food_count"), _safe_int(feats.get("food_count"), 0)),
        "tableware_count": _safe_int(food.get("tableware_count"), _safe_int(feats.get("tableware_count"), 0)),
        "food_score_max": round(_safe_float(food.get("food_score_max"), _safe_float(feats.get("food_score_max"))), 6),
        "tableware_score_max": round(_safe_float(food.get("tableware_score_max"), _safe_float(feats.get("tableware_score_max"))), 6),
        "food_area_sum": round(_safe_float(food.get("food_area_sum"), _safe_float(feats.get("food_area_sum"))), 6),
        "tableware_area_sum": round(_safe_float(food.get("tableware_area_sum"), _safe_float(feats.get("tableware_area_sum"))), 6),
        "dog_count": _safe_int(dogcat.get("dog_count"), _safe_int(feats.get("dog_count"), 0)),
        "cat_count": _safe_int(dogcat.get("cat_count"), _safe_int(feats.get("cat_count"), 0)),
        "dogcat_count": _safe_int(dogcat.get("dogcat_count"), _safe_int(feats.get("dogcat_count"), 0)),
        "animal_count": _safe_int(dogcat.get("animal_count"), _safe_int(feats.get("animal_count"), 0)),
        "dogcat_score_max": round(dogcat_signal, 6),
        "dogcat_area_sum": round(_safe_float(dogcat.get("dogcat_area_sum"), _safe_float(feats.get("dogcat_area_sum"))), 6),
        "animal_area_sum": round(_safe_float(dogcat.get("animal_area_sum"), _safe_float(feats.get("animal_area_sum"))), 6),
        "species_hint": str(dogcat.get("species") or ""),
        "pet_pose_head_available": routing.get("pet_pose_head_available"),
        "num_person": round(_safe_float(feats.get("num_person")), 6),
        "num_person_c2": round(_safe_float(feats.get("num_person_c2")), 6),
        "person_union_area_ratio": round(_safe_float(feats.get("person_union_area_ratio")), 6),
        "person_pose_available": bool(person.get("pose_available")),
        "face_available": bool(person.get("face_available")),
        "face_ratio": round(_safe_float(person.get("face_ratio"), _safe_float(feats.get("face_ratio"))), 6),
        "has_ankle": bool(person.get("has_ankle", _safe_float(feats.get("has_ankle")) > 0.0)),
        "ankle_conf_max": round(_safe_float(person.get("ankle_conf_max"), _safe_float(feats.get("ankle_conf_max"))), 6),
        "lower_body_reliable": bool(person.get("lower_body_reliable", _safe_float(feats.get("lower_body_reliable")) > 0.0)),
        "visible_group_count": _safe_int(person.get("visible_group_count"), _safe_int(feats.get("visible_group_count"), 0)),
        "environmental_score": round(env_score, 6),
        "environmental_person_area": round(_safe_float(env.get("person_area"), _safe_float(feats.get("environmental_person_area"))), 6),
        "environmental_scene_score": round(_safe_float(env.get("scene_score"), _safe_float(feats.get("environmental_scene_score"))), 6),
        "environmental_blank_ratio_est": round(_safe_float(env.get("blank_ratio_est"), _safe_float(feats.get("environmental_blank_ratio_est"))), 6),
        "food_signal": round(food_signal, 6),
        "dogcat_signal": round(dogcat_signal, 6),
        "person_signal": round(person_signal, 6),
        "reviewer_primary_route": "",
        "reviewer_pet_decision": "",
        "reviewer_shot_decision": "",
        "reviewer_person_pet_joint": "",
        "reviewer_confidence_ok": "",
        "reviewer_notes": "",
    }
    out["food_boundary_risk"] = round(
        (1.0 if out["image_route"] == "food" else 0.0)
        + 0.6 * min(1.0, out["person_signal"])
        + 0.5 * (1.0 if str(out["legacy_subject_mode"]).startswith("portrait") else 0.0)
        + 0.4 * (1.0 if out["food_count"] <= 0 and out["tableware_count"] > 0 else 0.0)
        + 0.4 * (1.0 if out["food_signal"] < 0.55 else 0.0),
        6,
    )
    out["pet_boundary_risk"] = round(
        (1.0 if out["image_route"] == "pet_dogcat" else 0.0)
        + 0.6 * (1.0 if out["dogcat_count"] <= 0 else 0.0)
        + 0.5 * (1.0 if out["animal_count"] > out["dogcat_count"] else 0.0)
        + 0.4 * (1.0 if out["dogcat_signal"] < 0.55 else 0.0)
        + 0.3 * (1.0 if out["pet_pose_head_available"] is not True else 0.0),
        6,
    )
    out["false_person_boundary_risk"] = round(
        (1.0 if str(out["route_family_v2"]).startswith("person") else 0.0)
        * (0.5 * (1.0 if out["person_signal"] <= 0.0 else 0.0) + 0.5 * max(out["food_signal"], out["dogcat_signal"]))
        + 0.4 * (1.0 if out["image_route"] in {"pet_dogcat"} and str(out["legacy_subject_mode"]).startswith("portrait") else 0.0)
        + 0.4 * (1.0 if out["food_signal"] >= 0.80 and str(out["legacy_subject_mode"]).startswith("portrait") else 0.0),
        6,
    )
    out["shot_boundary_risk"] = round(
        (1.0 if out["image_route"] == "person_single_full_body" else 0.0)
        + 0.8 * (1.0 if out["person_shot_type"] == "full_body" and not out["lower_body_reliable"] else 0.0)
        + 0.4 * (1.0 if out["person_shot_type"] == "full_body" and out["face_ratio"] >= 0.13 else 0.0),
        6,
    )
    out["environmental_boundary_risk"] = round(
        (0.8 if out["context_intent"] == "environmental" else 0.0)
        + 0.5 * (1.0 if out["environmental_score"] >= 0.65 else 0.0)
        + 0.4 * (1.0 if out["environmental_person_area"] <= 0.35 else 0.0)
        + 0.3 * (1.0 if out["routing_confidence"] < 0.65 else 0.0),
        6,
    )
    out["confidence_review_priority"] = round(
        (1.0 - out["routing_confidence"])
        + 0.5 * (1.0 - out["positive_vote_route_family_share"])
        + 0.3 * (1.0 - out["positive_vote_placement_share_aux"]),
        6,
    )
    auto_action = "keep"
    auto_reason = "strong_enough"
    needs_review = False
    auto_confidence = 0.90
    if out["image_route"] == "pet_dogcat" and (out["pet_pose_head_available"] is not True or out["dogcat_score_max"] < 0.55):
        auto_action = "review_pet_or_demote_scene_person"
        auto_reason = "weak_pet_pose_or_detector"
        needs_review = True
        auto_confidence = 0.55
    elif out["image_route"] == "person_single_full_body" and (not out["lower_body_reliable"] or out["face_ratio"] >= 0.13):
        auto_action = "review_full_body_vs_upper_half"
        auto_reason = "full_body_lower_body_evidence_weak"
        needs_review = True
        auto_confidence = 0.60
    elif out["image_route"].startswith("person_single") and out["food_signal"] >= 0.80 and out["person_union_area_ratio"] <= 0.32:
        auto_action = "review_person_vs_scene_object"
        auto_reason = "food_tableware_false_person_risk"
        needs_review = True
        auto_confidence = 0.60
    elif out["routing_confidence"] < 0.65 or out["positive_vote_route_family_share"] < 0.55:
        auto_action = "review_low_confidence"
        auto_reason = "low_confidence_or_vote_disagreement"
        needs_review = True
        auto_confidence = 0.65
    out["auto_suggested_action"] = auto_action
    out["auto_review_reason"] = auto_reason
    out["auto_confidence"] = round(auto_confidence, 6)
    out["needs_user_review"] = bool(needs_review)
    return out


def _top(rows: list[dict[str, Any]], key: str, limit: int) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (-_safe_float(row.get(key)), str(row.get("image_id") or "")))[: max(0, int(limit))]


def _bucket_rows(rows: list[dict[str, Any]], max_per_bucket: int) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    buckets["pet_hard_negative_review"] = _top(
        [
            r
            for r in rows
            if r["image_route"] == "pet_dogcat"
            or (r["animal_count"] > 0 and r["dogcat_count"] <= 0)
            or (r["dogcat_signal"] >= 0.45 and r["image_route"] != "pet_dogcat")
        ],
        "pet_boundary_risk",
        max_per_bucket,
    )
    buckets["pet_false_negative_candidates"] = _top(
        [r for r in rows if r["image_route"] != "pet_dogcat" and (r["dogcat_count"] > 0 or r["positive_pet_dogcat_count"] > 0)],
        "dogcat_signal",
        max_per_bucket,
    )
    buckets["co_primary_person_pet_review"] = _top(
        [r for r in rows if r["positive_person_single_count"] >= 2 and r["positive_pet_dogcat_count"] >= 2],
        "positive_count",
        max_per_bucket,
    )
    buckets["shot_full_body_boundary_review"] = _top(
        [
            r
            for r in rows
            if r["image_route"] == "person_single_full_body"
            or (r["route_family_v2"] == "person_single" and r["has_ankle"] and not r["lower_body_reliable"])
        ],
        "shot_boundary_risk",
        max_per_bucket,
    )
    buckets["environmental_context_review"] = _top(
        [
            r
            for r in rows
            if r["route_family_v2"] == "person_single"
            and (r["context_intent"] == "environmental" or r["environmental_score"] >= 0.65)
        ],
        "environmental_boundary_risk",
        max_per_bucket,
    )
    buckets["false_person_boundary_review"] = _top(
        [
            r
            for r in rows
            if str(r["route_family_v2"]).startswith("person")
            and (r["person_signal"] <= 0.0 or r["food_signal"] >= 0.35 or r["dogcat_signal"] >= 0.35)
        ]
        + [r for r in rows if r["image_route"] in {"pet_dogcat"} and str(r["legacy_subject_mode"]).startswith("portrait")],
        "false_person_boundary_risk",
        max_per_bucket,
    )
    buckets["low_confidence_calibration_review"] = _top(
        [r for r in rows if r["routing_confidence"] < 0.65 or r["positive_vote_route_family_share"] < 0.75],
        "confidence_review_priority",
        max_per_bucket,
    )
    buckets["auto_needs_user_review_top"] = _top(
        [r for r in rows if r.get("needs_user_review")],
        "confidence_review_priority",
        max_per_bucket,
    )
    return buckets


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


FONT_TITLE = _font(22, bold=True)
FONT = _font(14)
FONT_SMALL = _font(12)


def _draw_wrapped(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, *, width: int, font: ImageFont.ImageFont, fill: tuple[int, int, int]) -> int:
    words = str(text).split()
    lines: list[str] = []
    line = ""
    for word in words:
        trial = word if not line else f"{line} {word}"
        if draw.textbbox((0, 0), trial, font=font)[2] <= width:
            line = trial
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    x, y = xy
    for line in lines[:4]:
        draw.text((x, y), line, font=font, fill=fill)
        y += font.size + 3
    return y


def _contact_sheet(bucket: str, rows: list[dict[str, Any]], output_path: Path, *, cols: int, thumb_w: int, thumb_h: int) -> None:
    if not rows:
        return
    pad = 16
    title_h = 52
    card_h = thumb_h + 168
    rows_n = math.ceil(len(rows) / cols)
    sheet = Image.new("RGB", (cols * (thumb_w + pad) + pad, title_h + rows_n * (card_h + pad) + pad), (244, 244, 241))
    draw = ImageDraw.Draw(sheet)
    draw.rectangle((0, 0, sheet.width, title_h), fill=(35, 65, 88))
    draw.text((pad, 13), f"{bucket} | {len(rows)} samples", font=FONT_TITLE, fill=(255, 255, 255))
    for idx, row in enumerate(rows):
        col = idx % cols
        rr = idx // cols
        x = pad + col * (thumb_w + pad)
        y = title_h + pad + rr * (card_h + pad)
        draw.rounded_rectangle((x, y, x + thumb_w, y + card_h), radius=6, fill=(255, 255, 255), outline=(205, 205, 200), width=1)
        image_path = Path(str(row.get("image_path") or ""))
        try:
            im = Image.open(image_path).convert("RGB")
            scale = min(thumb_w / max(1, im.width), thumb_h / max(1, im.height))
            size = (max(1, int(round(im.width * scale))), max(1, int(round(im.height * scale))))
            thumb = im.resize(size, Image.Resampling.LANCZOS)
            tx = x + (thumb_w - thumb.width) // 2
            ty = y + 6 + (thumb_h - thumb.height) // 2
            sheet.paste(thumb, (tx, ty))
        except Exception:
            draw.rectangle((x + 6, y + 6, x + thumb_w - 6, y + thumb_h + 6), fill=(230, 230, 230))
            draw.text((x + 12, y + 24), "image load failed", font=FONT, fill=(80, 80, 80))
        text_y = y + thumb_h + 16
        title = f"{idx + 1}. {row.get('image_id')} | {row.get('image_route')} conf={_safe_float(row.get('routing_confidence')):.3f}"
        text_y = _draw_wrapped(draw, (x + 8, text_y), title, width=thumb_w - 16, font=FONT_SMALL, fill=(20, 20, 20))
        lines = [
            f"family={row.get('route_family_v2')} shot={row.get('person_shot_type')}",
            f"auto={row.get('auto_suggested_action')} review={row.get('needs_user_review')}",
            f"dogcat={row.get('dogcat_score_max')}/{row.get('dogcat_count')} animal={row.get('animal_count')}",
            f"full_body_rel={row.get('lower_body_reliable')} ankle={row.get('ankle_conf_max')} face={row.get('face_ratio')}",
        ]
        for line in lines:
            draw.text((x + 8, text_y), line, font=FONT_SMALL, fill=(65, 70, 76))
            text_y += 16
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def _confidence_bins(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bins = [(0.0, 0.35), (0.35, 0.50), (0.50, 0.65), (0.65, 0.80), (0.80, 1.01)]
    out = []
    for lo, hi in bins:
        subset = [r for r in rows if lo <= _safe_float(r.get("routing_confidence")) < hi]
        out.append(
            {
                "bin": f"[{lo:.2f},{hi:.2f})",
                "count": len(subset),
                "mean_confidence": round(sum(_safe_float(r.get("routing_confidence")) for r in subset) / max(1, len(subset)), 6),
                "mean_route_vote_share": round(sum(_safe_float(r.get("positive_vote_route_family_share")) for r in subset) / max(1, len(subset)), 6),
                "food_count": sum(1 for r in subset if r.get("image_route") == "food"),
                "pet_count": sum(1 for r in subset if r.get("image_route") == "pet_dogcat"),
                "environmental_context_count": sum(1 for r in subset if r.get("context_intent") == "environmental"),
            }
        )
    return out


def _decision_rows(output_dir: Path, bucket_counts: dict[str, int]) -> list[dict[str, Any]]:
    return [
        {
            "decision_id": "D1_pet_hard_negative_policy",
            "decision_needed": "dog/cat pet positive와 non-dog/cat animal/toy/sculpture/animal-print hard negative 경계를 결정",
            "artifact_to_review": str(output_dir / "pet_hard_negative_review.csv"),
            "contact_sheet": str(output_dir / "contact_sheets/pet_hard_negative_review.jpg"),
            "samples": bucket_counts.get("pet_hard_negative_review", 0),
            "review_columns_to_fill": "needs_user_review=true 행만 reviewer_primary_route, reviewer_pet_decision, reviewer_notes",
            "allowed_decisions": "keep_pet / hard_negative_scene / non_dogcat_object / unsure",
            "downstream_effect": "pet teacher precision, pet pose/head gate fallback, hard-negative mining 기준을 고정",
        },
        {
            "decision_id": "D2_person_pet_joint",
            "decision_needed": "사람과 pet이 공동 주피사체인 이미지를 single dominant route로 둘지 multi-label/person_pet_joint 실험을 열지 결정",
            "artifact_to_review": str(output_dir / "co_primary_person_pet_review.csv"),
            "contact_sheet": str(output_dir / "contact_sheets/co_primary_person_pet_review.jpg"),
            "samples": bucket_counts.get("co_primary_person_pet_review", 0),
            "review_columns_to_fill": "needs_user_review=true 행만 reviewer_primary_route, reviewer_person_pet_joint, reviewer_notes",
            "allowed_decisions": "single_dominant / allow_multilabel / add_person_pet_joint_ablation / unsure",
            "downstream_effect": "image route head label space와 route-conditioned crop scorer의 multi-intent 처리 방식 결정",
        },
        {
            "decision_id": "D3_full_body_precision_policy",
            "decision_needed": "full_body teacher를 현재처럼 ankle/face-ratio 기반 보수 정책으로 유지할지, 추가 visual/VLM teacher로 recall을 보강할지 결정",
            "artifact_to_review": str(output_dir / "shot_full_body_boundary_review.csv"),
            "contact_sheet": str(output_dir / "contact_sheets/shot_full_body_boundary_review.jpg"),
            "samples": bucket_counts.get("shot_full_body_boundary_review", 0),
            "review_columns_to_fill": "needs_user_review=true 행만 reviewer_primary_route, reviewer_shot_decision, reviewer_notes",
            "allowed_decisions": "keep_strict / add_vlm_audit / relax_threshold / unsure",
            "downstream_effect": "person_shot_type full_body precision/recall과 route head class balance 기준을 고정",
        },
        {
            "decision_id": "D4_confidence_calibration_seed",
            "decision_needed": "routing_confidence가 낮은 bucket에서 teacher hard label을 그대로 쓸지 low-weight/calibration split으로 돌릴지 결정",
            "artifact_to_review": str(output_dir / "low_confidence_calibration_review.csv"),
            "contact_sheet": str(output_dir / "contact_sheets/low_confidence_calibration_review.jpg"),
            "samples": bucket_counts.get("low_confidence_calibration_review", 0),
            "review_columns_to_fill": "needs_user_review=true 행만 reviewer_primary_route, reviewer_confidence_ok, reviewer_notes",
            "allowed_decisions": "hard_label_ok / low_weight_only / calibration_only / relabel_needed",
            "downstream_effect": "confidence-weighted loss, ECE/reliability calibration, train/eval split policy 결정",
        },
    ]


def _write_guide(path: Path, summary: dict[str, Any], decisions: list[dict[str, Any]]) -> None:
    lines = [
        "# Routing v2 Follow-up Decision Pack",
        "",
        "이 산출물은 routing_v2 후속 의사결정을 auto triage 우선으로 빠르게 검수하도록 만든 audit pack이다. image-level route는 positive crop vote로 override하지 않으며, positive vote 관련 column은 불일치 audit용 참고값이다.",
        "",
        "## 생성 요약",
        "",
        f"- total images: `{summary['total_images']}`",
        f"- output dir: `{summary['output_dir']}`",
        f"- max per bucket: `{summary['max_per_bucket']}`",
        "",
        "## 사용 방법",
        "",
        "1. `contact_sheets/auto_needs_user_review_top.jpg`를 먼저 훑는다.",
        "2. 같은 이름의 CSV에서 `needs_user_review=true` 행만 확인한다.",
        "3. 자동 권고가 틀린 행에만 `reviewer_*` column을 채운다.",
        "4. `decision_matrix_for_user.csv`의 `allowed_decisions` 중 하나를 선택한다.",
        "5. 확정 결정은 다음 label builder threshold 및 route head 학습 config에 반영한다.",
        "",
        "## 반드시 결정해야 하는 항목",
        "",
        "| decision_id | 검토할 산출물 | 결정 질문 | 허용 결정값 |",
        "|---|---|---|---|",
    ]
    for decision in decisions:
        lines.append(
            f"| `{decision['decision_id']}` | `{Path(decision['artifact_to_review']).name}` / `{Path(decision['contact_sheet']).name}` | "
            f"{decision['decision_needed']} | `{decision['allowed_decisions']}` |"
        )
    lines.extend(
        [
            "",
            "## Bucket 설명",
            "",
            "| bucket | 의미 |",
            "|---|---|",
            "| `pet_hard_negative_review` | pet positive와 non-dog/cat animal/object hard negative 경계 |",
            "| `pet_false_negative_candidates` | route는 pet이 아니지만 dog/cat 또는 pet positive evidence가 있는 후보 |",
            "| `co_primary_person_pet_review` | person_single positive와 pet_dogcat positive가 동시에 강한 공동 주피사체 후보 |",
            "| `shot_full_body_boundary_review` | full_body와 upper_half_body 경계, 특히 하반신 keypoint 신뢰도가 약한 후보 |",
            "| `environmental_context_review` | 제거된 environmental_portrait class가 context_intent=environmental로 잘 흡수됐는지 보는 후보 |",
            "| `false_person_boundary_review` | person으로 접힐 위험이 있는 tableware/pet/object 또는 person false positive 경계 |",
            "| `low_confidence_calibration_review` | routing_confidence가 낮거나 route vote agreement가 낮은 calibration seed |",
            "| `auto_needs_user_review_top` | 자동 규칙이 사람 검수를 요구한 상위 후보 |",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "status.json"
    _write_json(status_path, {"state": "running", "phase": "load"})
    votes = _load_positive_votes(args.multimode_json)
    rows = [_flatten_row(row, votes, args.image_root) for row in _iter_jsonl(args.label_jsonl)]
    _write_json(status_path, {"state": "running", "phase": "bucket", "rows": len(rows)})
    buckets = _bucket_rows(rows, int(args.max_per_bucket))
    contact_dir = args.output_dir / "contact_sheets"
    bucket_counts = {name: len(items) for name, items in buckets.items()}
    for name, items in buckets.items():
        _write_csv(args.output_dir / f"{name}.csv", items)
        _contact_sheet(name, items, contact_dir / f"{name}.jpg", cols=int(args.contact_cols), thumb_w=int(args.thumb_w), thumb_h=int(args.thumb_h))
    confidence = _confidence_bins(rows)
    _write_csv(args.output_dir / "confidence_bins_proxy.csv", confidence)
    decisions = _decision_rows(args.output_dir, bucket_counts)
    _write_csv(args.output_dir / "decision_matrix_for_user.csv", decisions)
    summary = {
        "state": "completed",
        "label_jsonl": str(args.label_jsonl),
        "multimode_json": str(args.multimode_json),
        "output_dir": str(args.output_dir),
        "total_images": len(rows),
        "max_per_bucket": int(args.max_per_bucket),
        "bucket_counts": bucket_counts,
        "image_route_counts": dict(sorted(Counter(str(r["image_route"]) for r in rows).items())),
        "route_family_counts": dict(sorted(Counter(str(r["route_family_v2"]) for r in rows).items())),
        "confidence_bins": confidence,
        "decisions": decisions,
    }
    _write_json(args.output_dir / "summary.json", summary)
    _write_guide(args.output_dir / "DECISION_REVIEW_GUIDE_KO.md", summary, decisions)
    _write_json(status_path, {"state": "completed", "summary": str(args.output_dir / "summary.json")})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
