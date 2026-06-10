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

from scripts.visualize_multimode_crop_review import (  # noqa: E402
    _ann_mode,
    _ann_score,
    _ann_target_ar,
    _select_display_annotations,
)


DEFAULT_PHASE_ROOT = Path("data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529")
DEFAULT_TAG = "260609_v16_routing_v2_mode_catalog_nofood_petstrict_shotstrict_nms_v2"
DEFAULT_MM_JSON = DEFAULT_PHASE_ROOT / "artifacts/training_labels_multimode" / DEFAULT_TAG / "label_json/multimode_labels_full.json"
DEFAULT_LABEL_JSONL = DEFAULT_PHASE_ROOT / "artifacts/training_labels" / DEFAULT_TAG / "routing_v2_v16_full.jsonl"
DEFAULT_FEATURE_JSONL = DEFAULT_PHASE_ROOT / "artifacts/precompute/feats_c2c3c5_v2_strict_enriched_routed_final.jsonl"
DEFAULT_PANEL_DIR = (
    DEFAULT_PHASE_ROOT
    / "artifacts/training_labels_multimode"
    / DEFAULT_TAG
    / "crop_review_routing_v2_v16_nofood_petstrict_shotstrict"
    / "panels_unified_220"
)
DEFAULT_OUTPUT_DIR = DEFAULT_PANEL_DIR / "audit_jy_260609"


USER_ISSUE_CROPS = {
    "bigstock_image_96641153": {4, 7},
    "bigstock_image_97607294": {3},
    "bigstock_image_166429337": {4},
    "bigstock_image_184240093": {1},
    "bigstock_image_185099542": {1},
    "bigstock_image_187705027": {6},
    "bigstock_image_215961286": {1},
}
USER_ISSUE_IMAGES = {
    "bigstock_image_187705027",
    "bigstock_image_199044151",
    "bigstock_image_173600372",
    "bigstock_image_180607570",
    "bigstock_image_97607294",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit routing_v2 crop-review panels and full multimode labels.")
    parser.add_argument("--multimode_json", type=Path, default=DEFAULT_MM_JSON)
    parser.add_argument("--label_jsonl", type=Path, default=DEFAULT_LABEL_JSONL)
    parser.add_argument("--feature_jsonl", type=Path, default=DEFAULT_FEATURE_JSONL)
    parser.add_argument("--panel_dir", type=Path, default=DEFAULT_PANEL_DIR)
    parser.add_argument("--image_root", type=Path, default=DEFAULT_PHASE_ROOT / "images")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_contact", type=int, default=40)
    return parser


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


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


def _load_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            image_id = str(row.get("image_id") or "").strip()
            if image_id:
                rows[image_id] = row
    return rows


def _load_features(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            image_id = str(row.get("image_id") or "").strip()
            if image_id:
                rows[image_id] = row
    return rows


def _source_id(image: dict[str, Any]) -> str:
    return str(image.get("source_image_id") or Path(str(image.get("file_name") or "")).stem)


def _ann_routing(ann: dict[str, Any]) -> dict[str, Any]:
    attrs = ann.get("attributes") if isinstance(ann.get("attributes"), dict) else {}
    routing = attrs.get("routing_v2_simple")
    return routing if isinstance(routing, dict) else {}


def _flat_route(ann: dict[str, Any]) -> str:
    attrs = ann.get("attributes") if isinstance(ann.get("attributes"), dict) else {}
    return str(attrs.get("flat_route_class") or "")


def _xywh_to_xyxy(box: Any) -> tuple[float, float, float, float]:
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return 0.0, 0.0, 0.0, 0.0
    x, y, w, h = [_safe_float(v) for v in box[:4]]
    return x, y, x + max(0.0, w), y + max(0.0, h)


def _xyxy_area(box: Any) -> float:
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return 0.0
    x1, y1, x2, y2 = [_safe_float(v) for v in box[:4]]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _inter_area(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _in_box(point: Any, crop: tuple[float, float, float, float], conf_thr: float = 0.15, margin: float = 3.0) -> bool:
    if not isinstance(point, (list, tuple)) or len(point) < 3:
        return False
    x = _safe_float(point[0])
    y = _safe_float(point[1])
    conf = _safe_float(point[2])
    return conf >= conf_thr and crop[0] - margin <= x <= crop[2] + margin and crop[1] - margin <= y <= crop[3] + margin


def _pose_box(pose: dict[str, Any]) -> tuple[float, float, float, float]:
    box = pose.get("bbox")
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return 0.0, 0.0, 0.0, 0.0
    return tuple(_safe_float(v) for v in box[:4])  # type: ignore[return-value]


def _best_pose_for_crop(feature_row: dict[str, Any] | None, crop: tuple[float, float, float, float]) -> dict[str, Any] | None:
    poses = feature_row.get("c3_pose") if isinstance(feature_row, dict) else None
    if not isinstance(poses, list):
        return None
    best = None
    best_key = (-1.0, -1.0, -1.0)
    for pose in poses:
        if not isinstance(pose, dict):
            continue
        pbox = _pose_box(pose)
        pose_area = max(1.0, _xyxy_area(pbox))
        overlap = _inter_area(crop, pbox) / pose_area
        kps = pose.get("keypoints") if isinstance(pose.get("keypoints"), list) else []
        inside = sum(_in_box(kp, crop) for kp in kps)
        key = (overlap, float(inside), _safe_float(pose.get("score")))
        if key > best_key:
            best = pose
            best_key = key
    return best


def _crop_pose_visibility(feature_row: dict[str, Any] | None, bbox_xywh: Any, image_w: int, image_h: int) -> dict[str, Any]:
    crop = _xywh_to_xyxy(bbox_xywh)
    crop_w = max(0.0, crop[2] - crop[0])
    crop_h = max(0.0, crop[3] - crop[1])
    crop_area = crop_w * crop_h
    pose = _best_pose_for_crop(feature_row, crop)
    if not isinstance(pose, dict):
        return {
            "crop_w": round(crop_w, 3),
            "crop_h": round(crop_h, 3),
            "crop_area_ratio": round(crop_area / max(1.0, float(image_w * image_h)), 6),
            "pose_available": False,
            "crop_shot_inferred": "unknown",
            "full_body_crop_ok": False,
            "face_only_like": False,
        }
    kps = pose.get("keypoints") if isinstance(pose.get("keypoints"), list) else []
    groups = {
        "eye": (1, 2),
        "shoulder": (5, 6),
        "hip": (11, 12),
        "knee": (13, 14),
        "ankle": (15, 16),
    }
    visible = {name: any(idx < len(kps) and _in_box(kps[idx], crop) for idx in indices) for name, indices in groups.items()}
    pbox = _pose_box(pose)
    pose_area = max(1.0, _xyxy_area(pbox))
    pose_overlap = _inter_area(crop, pbox) / pose_area
    face = pose.get("face") if isinstance(pose.get("face"), dict) else {}
    face_box = face.get("bbox")
    face_xyxy = tuple(_safe_float(v) for v in face_box[:4]) if isinstance(face_box, (list, tuple)) and len(face_box) >= 4 else None
    face_area = _xyxy_area(face_xyxy) if face_xyxy is not None else 0.0
    face_overlap = _inter_area(crop, face_xyxy) / max(1.0, face_area) if face_xyxy is not None else 0.0
    face_to_crop = face_area / max(1.0, crop_area)
    full_body_crop_ok = bool(visible["shoulder"] and visible["hip"] and visible["knee"] and visible["ankle"] and pose_overlap >= 0.45)
    face_only_like = bool(face_overlap >= 0.55 and not (visible["hip"] or visible["knee"] or visible["ankle"]) and pose_overlap <= 0.38)
    if full_body_crop_ok:
        crop_shot = "full_body"
    elif face_only_like or (face_overlap >= 0.55 and not visible["shoulder"]):
        crop_shot = "face_headshot"
    elif visible["shoulder"] or visible["hip"] or visible["knee"]:
        crop_shot = "upper_half_body"
    elif face_overlap > 0.0:
        crop_shot = "face_headshot"
    else:
        crop_shot = "unknown"
    return {
        "crop_w": round(crop_w, 3),
        "crop_h": round(crop_h, 3),
        "crop_area_ratio": round(crop_area / max(1.0, float(image_w * image_h)), 6),
        "pose_available": True,
        "pose_overlap_ratio": round(pose_overlap, 6),
        "face_overlap_ratio": round(face_overlap, 6),
        "face_area_to_crop": round(face_to_crop, 6),
        "crop_has_eye": visible["eye"],
        "crop_has_shoulder": visible["shoulder"],
        "crop_has_hip": visible["hip"],
        "crop_has_knee": visible["knee"],
        "crop_has_ankle": visible["ankle"],
        "crop_shot_inferred": crop_shot,
        "full_body_crop_ok": full_body_crop_ok,
        "face_only_like": face_only_like,
    }


def _probe(row: dict[str, Any], key: str) -> float:
    feats = row.get("probe_features") if isinstance(row.get("probe_features"), dict) else {}
    return _safe_float(feats.get(key))


def _panel_source_ids(panel_dir: Path) -> set[str]:
    manifest = panel_dir / "routing_v2_v16_nofood_petstrict_manifest.json"
    if manifest.exists():
        payload = _read_json(manifest)
        rows = payload.get("rows") if isinstance(payload, dict) else payload
        if isinstance(rows, list):
            return {str(row.get("source_image_id") or row.get("image_id") or "").strip() for row in rows if isinstance(row, dict)}
    panel_files = panel_dir.joinpath("panels").glob("review_*.jpg")
    return {path.stem.removeprefix("review_") for path in panel_files}


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _make_panel_contact_sheet(path: Path, rows: list[dict[str, Any]], panel_dir: Path, title: str, max_items: int) -> None:
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        source_id = str(row.get("source_image_id") or row.get("image_id") or "")
        if not source_id or source_id in seen:
            continue
        panel_path = panel_dir / "panels" / f"review_{source_id}.jpg"
        if not panel_path.exists():
            continue
        unique.append(row)
        seen.add(source_id)
        if len(unique) >= max_items:
            break
    if not unique:
        return
    cols = 3
    thumb_w = 360
    thumb_h = 260
    caption_h = 56
    header_h = 52
    rows_n = math.ceil(len(unique) / cols)
    canvas = Image.new("RGB", (cols * thumb_w, header_h + rows_n * (thumb_h + caption_h)), (246, 247, 249))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, header_h), fill=(40, 69, 98))
    draw.text((16, 14), title, font=_font(22, True), fill=(255, 255, 255))
    for idx, row in enumerate(unique):
        x = (idx % cols) * thumb_w
        y = header_h + (idx // cols) * (thumb_h + caption_h)
        src = panel_dir / "panels" / f"review_{row.get('source_image_id') or row.get('image_id')}.jpg"
        with Image.open(src) as im:
            im = im.convert("RGB")
            im.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            ox = x + (thumb_w - im.width) // 2
            oy = y + (thumb_h - im.height) // 2
            canvas.paste(im, (ox, oy))
        caption = str(row.get("caption") or row.get("issue_type") or row.get("image_route") or "")
        source_id = str(row.get("source_image_id") or row.get("image_id") or "")
        draw.text((x + 8, y + thumb_h + 4), source_id[:42], font=_font(14, True), fill=(20, 24, 31))
        draw.text((x + 8, y + thumb_h + 24), caption[:54], font=_font(13), fill=(70, 76, 86))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=92)


def _routing_label(routing: dict[str, Any]) -> str:
    return "/".join(
        [
            str(routing.get("route_family_v2") or "na"),
            str(routing.get("person_shot_type") or "na"),
            str(routing.get("placement_intent") or "na"),
            str(routing.get("context_intent") or "na"),
        ]
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = _read_json(args.multimode_json)
    images = payload.get("images") if isinstance(payload.get("images"), list) else []
    annotations = payload.get("annotations") if isinstance(payload.get("annotations"), list) else []
    image_by_id = {int(img.get("id")): img for img in images if img.get("id") is not None}
    source_by_image_id = {image_id: _source_id(img) for image_id, img in image_by_id.items()}
    anns_by_image_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in annotations:
        if isinstance(ann, dict) and ann.get("image_id") is not None:
            anns_by_image_id[int(ann.get("image_id"))].append(ann)
    label_rows = _load_jsonl_by_id(args.label_jsonl)
    features = _load_features(args.feature_jsonl)
    selected_panel_ids = _panel_source_ids(args.panel_dir)

    user_issue_rows: list[dict[str, Any]] = []
    crop_mismatch_full: list[dict[str, Any]] = []
    crop_mismatch_selected: list[dict[str, Any]] = []
    selected_ann_count = 0
    selected_full_body_count = 0
    full_ann_count = 0
    full_body_ann_count = 0

    for image_id, anns in anns_by_image_id.items():
        image = image_by_id.get(image_id, {})
        source_id = source_by_image_id.get(image_id, "")
        feature_row = features.get(source_id)
        image_w = _safe_int(image.get("width"), 0)
        image_h = _safe_int(image.get("height"), 0)
        selected = _select_display_annotations(anns, 8)
        selected_ids = {id(ann): idx for idx, ann in enumerate(selected, 1)}
        for ann in anns:
            if _safe_int(ann.get("gt_flag"), 0) != 1:
                continue
            full_ann_count += 1
            routing = _ann_routing(ann)
            is_full = routing.get("route_family_v2") == "person_single" and routing.get("person_shot_type") == "full_body"
            if is_full:
                full_body_ann_count += 1
            is_selected = source_id in selected_panel_ids and id(ann) in selected_ids
            if is_selected:
                selected_ann_count += 1
                if is_full:
                    selected_full_body_count += 1
            if not is_full:
                continue
            vis = _crop_pose_visibility(feature_row, ann.get("bbox"), image_w, image_h)
            mismatch = bool(vis.get("pose_available") and not vis.get("full_body_crop_ok"))
            if not mismatch:
                continue
            row = {
                "source_image_id": source_id,
                "image_route": str(label_rows.get(source_id, {}).get("image_route_name_no_placement") or ""),
                "panel_crop_index": selected_ids.get(id(ann), ""),
                "query_id": str(ann.get("query_id") or ""),
                "mode_name": _ann_mode(ann),
                "target_ar": _ann_target_ar(ann),
                "score": round(_ann_score(ann), 6),
                "bbox_xywh": json.dumps(ann.get("bbox"), ensure_ascii=False),
                "annotation_route": _routing_label(routing),
                "flat_route_class": _flat_route(ann),
                **vis,
                "issue_type": "crop_full_body_without_lower_body",
                "caption": f"#{selected_ids.get(id(ann), '-')} full_body->{vis.get('crop_shot_inferred')}",
            }
            crop_mismatch_full.append(row)
            if is_selected:
                crop_mismatch_selected.append(row)
            if source_id in USER_ISSUE_CROPS and selected_ids.get(id(ann)) in USER_ISSUE_CROPS[source_id]:
                user_issue_rows.append({**row, "user_issue_kind": "user_marked_crop"})

    image_group_suspicious: list[dict[str, Any]] = []
    image_face_should_fullbody: list[dict[str, Any]] = []
    image_route_counts: Counter[str] = Counter()
    for source_id, row in label_rows.items():
        routing = row.get("routing_v2_simple") if isinstance(row.get("routing_v2_simple"), dict) else {}
        image_route = str(row.get("image_route_name_no_placement") or "")
        image_route_counts[image_route] += 1
        person = routing.get("person_pose_signals") if isinstance(routing.get("person_pose_signals"), dict) else {}
        reasons = routing.get("reasons") if isinstance(routing.get("reasons"), list) else []
        num_person = _probe(row, "num_person")
        num_person_c2 = _probe(row, "num_person_c2")
        if image_route == "person_group" and max(num_person, num_person_c2) <= 1.05:
            image_group_suspicious.append(
                {
                    "source_image_id": source_id,
                    "image_route": image_route,
                    "legacy_subject_mode": str(routing.get("legacy_subject_mode") or ""),
                    "routing_confidence": _safe_float(routing.get("routing_confidence")),
                    "num_person": num_person,
                    "num_person_c2": num_person_c2,
                    "person_union_area_ratio": _probe(row, "person_union_area_ratio"),
                    "pose_available": bool(person.get("pose_available")),
                    "positive_vote_route_family_share": _safe_float(routing.get("positive_vote_route_family_share")),
                    "positive_vote_route_family_weight": _safe_float(routing.get("positive_vote_route_family_weight")),
                    "reasons": ";".join(str(x) for x in reasons),
                    "in_selected_panel": source_id in selected_panel_ids,
                    "issue_type": "image_group_with_single_person_evidence",
                    "caption": f"group but num_person={max(num_person, num_person_c2):.0f}",
                    "user_marked": source_id in USER_ISSUE_IMAGES,
                }
            )
        if image_route == "person_single_face_headshot":
            lower = bool(person.get("lower_body_reliable"))
            face_ratio = _safe_float(person.get("face_ratio"))
            ankle = _safe_float(person.get("ankle_conf_max"))
            if lower or (ankle >= 0.30 and face_ratio <= 0.13):
                image_face_should_fullbody.append(
                    {
                        "source_image_id": source_id,
                        "image_route": image_route,
                        "legacy_subject_mode": str(routing.get("legacy_subject_mode") or ""),
                        "routing_confidence": _safe_float(routing.get("routing_confidence")),
                        "face_ratio": face_ratio,
                        "ankle_conf_max": ankle,
                        "ankle_y_norm": _safe_float(person.get("ankle_y_norm")),
                        "pose_bbox_height_ratio": _safe_float(person.get("pose_bbox_height_ratio")),
                        "lower_body_reliable": lower,
                        "reasons": ";".join(str(x) for x in reasons),
                        "in_selected_panel": source_id in selected_panel_ids,
                        "issue_type": "image_face_headshot_with_full_body_pose",
                        "caption": f"face route but lower_body={int(lower)}",
                        "user_marked": source_id in USER_ISSUE_IMAGES,
                    }
                )

    # Include user-marked image-level rows even when the heuristic bucket does not catch them.
    for source_id in sorted(USER_ISSUE_IMAGES):
        row = label_rows.get(source_id)
        if not row:
            continue
        routing = row.get("routing_v2_simple") if isinstance(row.get("routing_v2_simple"), dict) else {}
        if any(r.get("source_image_id") == source_id for r in user_issue_rows):
            continue
        user_issue_rows.append(
            {
                "source_image_id": source_id,
                "image_route": str(row.get("image_route_name_no_placement") or ""),
                "annotation_route": "",
                "issue_type": "user_marked_image_route",
                "caption": str(row.get("image_route_name_no_placement") or ""),
                "reasons": ";".join(str(x) for x in routing.get("reasons", [])) if isinstance(routing.get("reasons"), list) else "",
            }
        )

    crop_mismatch_full.sort(key=lambda r: (str(r.get("source_image_id")), _safe_int(r.get("panel_crop_index"), 999), -_safe_float(r.get("score"))))
    crop_mismatch_selected.sort(key=lambda r: (str(r.get("source_image_id")), _safe_int(r.get("panel_crop_index"), 999), -_safe_float(r.get("score"))))
    image_group_suspicious.sort(key=lambda r: (not bool(r.get("in_selected_panel")), str(r.get("source_image_id"))))
    image_face_should_fullbody.sort(key=lambda r: (not bool(r.get("in_selected_panel")), str(r.get("source_image_id"))))

    _write_csv(args.output_dir / "user_marked_issue_reproduction.csv", user_issue_rows)
    _write_csv(args.output_dir / "crop_full_body_mismatch_selected_220.csv", crop_mismatch_selected)
    _write_csv(args.output_dir / "crop_full_body_mismatch_full_annotations.csv", crop_mismatch_full)
    _write_csv(args.output_dir / "image_group_single_person_suspicious.csv", image_group_suspicious)
    _write_csv(args.output_dir / "image_face_headshot_fullbody_pose_suspicious.csv", image_face_should_fullbody)

    _make_panel_contact_sheet(
        args.output_dir / "contact_sheets/crop_full_body_mismatch_selected_220.jpg",
        crop_mismatch_selected,
        args.panel_dir,
        "Selected panels: full_body crop without lower-body evidence",
        args.max_contact,
    )
    _make_panel_contact_sheet(
        args.output_dir / "contact_sheets/image_group_single_person_suspicious.jpg",
        image_group_suspicious,
        args.panel_dir,
        "Selected panels: image_route person_group but single-person evidence",
        args.max_contact,
    )
    _make_panel_contact_sheet(
        args.output_dir / "contact_sheets/image_face_headshot_fullbody_pose_suspicious.jpg",
        image_face_should_fullbody,
        args.panel_dir,
        "Selected panels: face_headshot route but full-body pose evidence",
        args.max_contact,
    )
    _make_panel_contact_sheet(
        args.output_dir / "contact_sheets/user_marked_issue_reproduction.jpg",
        user_issue_rows,
        args.panel_dir,
        "User-marked issue reproduction",
        args.max_contact,
    )

    selected_image_count = len(selected_panel_ids)
    summary = {
        "multimode_json": str(args.multimode_json),
        "label_jsonl": str(args.label_jsonl),
        "feature_jsonl": str(args.feature_jsonl),
        "panel_dir": str(args.panel_dir),
        "selected_panel_image_count": selected_image_count,
        "selected_panel_annotation_count": selected_ann_count,
        "selected_panel_full_body_annotation_count": selected_full_body_count,
        "selected_panel_full_body_mismatch_count": len(crop_mismatch_selected),
        "selected_panel_full_body_mismatch_rate": round(len(crop_mismatch_selected) / max(1, selected_full_body_count), 6),
        "full_positive_annotation_count": full_ann_count,
        "full_positive_full_body_annotation_count": full_body_ann_count,
        "full_positive_full_body_mismatch_count": len(crop_mismatch_full),
        "full_positive_full_body_mismatch_rate": round(len(crop_mismatch_full) / max(1, full_body_ann_count), 6),
        "image_route_counts": dict(sorted(image_route_counts.items())),
        "image_group_single_person_suspicious_count": len(image_group_suspicious),
        "image_group_single_person_suspicious_selected_count": sum(1 for r in image_group_suspicious if r.get("in_selected_panel")),
        "image_face_headshot_fullbody_pose_suspicious_count": len(image_face_should_fullbody),
        "image_face_headshot_fullbody_pose_suspicious_selected_count": sum(1 for r in image_face_should_fullbody if r.get("in_selected_panel")),
        "user_marked_issue_rows": len(user_issue_rows),
        "outputs": {
            "user_marked_issue_reproduction_csv": str(args.output_dir / "user_marked_issue_reproduction.csv"),
            "crop_full_body_mismatch_selected_csv": str(args.output_dir / "crop_full_body_mismatch_selected_220.csv"),
            "crop_full_body_mismatch_full_csv": str(args.output_dir / "crop_full_body_mismatch_full_annotations.csv"),
            "image_group_single_person_suspicious_csv": str(args.output_dir / "image_group_single_person_suspicious.csv"),
            "image_face_headshot_fullbody_pose_suspicious_csv": str(args.output_dir / "image_face_headshot_fullbody_pose_suspicious.csv"),
            "contact_sheets": str(args.output_dir / "contact_sheets"),
        },
    }
    _write_json(args.output_dir / "audit_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
