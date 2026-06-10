#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from routing.routing_v2_simple import (  # noqa: E402
    ANIMAL_POSE_CONF_MIN,
    ANIMAL_POSE_HEAD_CONF_MIN,
    ANIMAL_POSE_SPECIES_CONF_MIN,
    CONTEXT_INTENT,
    HIERARCHICAL_VOCABS,
    PERSON_SHOT_TYPE,
    PLACEMENT_INTENT,
    ROUTE_FAMILY_V2,
    SCHEMA_VERSION,
    derive_image_routing_v2_simple,
    derive_query_routing_v2_simple,
    flat_route_class_key,
    hierarchical_target_ids,
    probe_features_from_feature_row,
)


DEFAULT_PHASE_ROOT = Path("data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529")
DEFAULT_V14_RUN = DEFAULT_PHASE_ROOT / "artifacts/training_labels_multimode/260601_v14_photo_primary_fullsstk_phaseA_scene_subjectsafe_residual_patch_split9010"
DEFAULT_FEATURE_JSONL = DEFAULT_PHASE_ROOT / "artifacts/precompute/feats_c2c3c5_v2_strict_enriched_routed_final.jsonl"
DEFAULT_ANIMAL_POSE_JSONL = DEFAULT_PHASE_ROOT / "artifacts/precompute/animal_pose_260609_routing_v2_simple_gpu_full_ap10k_nms_v2/animal_pose.jsonl"
DEFAULT_RUN_TAG = "260609_routing_v2_simple_v8_nofood_petstrict_shotstrict_nms_v2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build routing_v2_simple labels from PhaseA v14 multimode artifacts.")
    parser.add_argument("--phase_root", type=Path, default=DEFAULT_PHASE_ROOT)
    parser.add_argument("--source_multimode_run_dir", type=Path, default=DEFAULT_V14_RUN)
    parser.add_argument("--feature_jsonl", type=Path, default=DEFAULT_FEATURE_JSONL)
    parser.add_argument("--animal_pose_jsonl", type=Path, default=DEFAULT_ANIMAL_POSE_JSONL)
    parser.add_argument("--run_tag", default=DEFAULT_RUN_TAG)
    parser.add_argument("--image_root", type=Path, default=None)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--high_conf_min", type=float, default=0.65)
    return parser


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _load_features(path: Path, *, max_images: int = 0) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            image_id = str(row.get("image_id") or "").strip()
            if not image_id:
                continue
            features[image_id] = row
            if max_images > 0 and len(features) >= max_images:
                break
    return features


def _load_animal_pose_head_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            image_id = str(row.get("image_id") or "").strip()
            if not image_id:
                continue
            groups = row.get("visible_groups") if isinstance(row.get("visible_groups"), dict) else {}
            head = groups.get("head")
            species_conf = float(row.get("species_confidence") or 0.0)
            pose_conf = float(row.get("pose_confidence") or 0.0)
            head_conf = float(head) if isinstance(head, (int, float)) else (1.0 if head is True else 0.0)
            species_hint = str(row.get("species_hint") or "").lower()
            if (
                species_hint in {"dog", "cat", "dogcat_mixed"}
                and species_conf >= ANIMAL_POSE_SPECIES_CONF_MIN
                and pose_conf >= ANIMAL_POSE_CONF_MIN
                and head_conf >= ANIMAL_POSE_HEAD_CONF_MIN
            ):
                ids.add(image_id)
    return ids


def _source_image_id(image: dict[str, Any]) -> str:
    return str(image.get("source_image_id") or Path(str(image.get("file_name") or "")).stem)


def _resolve_image_path(image_root: Path, image: dict[str, Any], source_id: str) -> Path:
    file_name = str(image.get("file_name") or f"{source_id}.jpg")
    direct = image_root / file_name
    if direct.exists():
        return direct
    for suffix in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = image_root / f"{source_id}{suffix}"
        if candidate.exists():
            return candidate
    return direct


def _score_value(ann: dict[str, Any]) -> float:
    for key in ("score_mode", "score", "final_score"):
        try:
            value = float(ann.get(key))
            return value
        except (TypeError, ValueError):
            continue
    attrs = ann.get("attributes") if isinstance(ann.get("attributes"), dict) else {}
    scores = attrs.get("checklist_scores") if isinstance(attrs.get("checklist_scores"), dict) else {}
    try:
        return float(scores.get("final_score"))
    except (TypeError, ValueError):
        return 0.0


def _vote_summary(anns: list[dict[str, Any]]) -> dict[str, Any]:
    weighted: dict[str, Counter[str]] = {
        "route_family_v2": Counter(),
        "person_shot_type": Counter(),
        "placement_intent": Counter(),
        "context_intent": Counter(),
    }
    mode_feasible = False
    best_score = 0.0
    for ann in anns:
        if int(ann.get("gt_flag") or 0) != 1:
            continue
        attrs = ann.get("attributes") if isinstance(ann.get("attributes"), dict) else {}
        routing = attrs.get("routing_v2_simple") if isinstance(attrs.get("routing_v2_simple"), dict) else {}
        if not routing:
            continue
        score = max(0.05, _score_value(ann))
        conf = float(routing.get("routing_confidence") or 0.0)
        weight = score * max(0.20, conf)
        best_score = max(best_score, score)
        mode_feasible = mode_feasible or bool(routing.get("mode_feasible", True))
        for key in weighted:
            value = routing.get(key)
            if value is not None:
                weighted[key][str(value)] += weight
    out: dict[str, Any] = {"mode_feasible": bool(mode_feasible), "vote_score": round(best_score, 6)}
    for key, counter in weighted.items():
        if counter:
            out[key] = counter.most_common(1)[0][0]
            out[f"{key}_votes"] = {k: round(float(v), 6) for k, v in counter.most_common()}
    return out


def _augment_coco_split(
    *,
    split: str,
    source_path: Path,
    output_path: Path,
    features: dict[str, dict[str, Any]],
    image_root: Path,
    flat_vocab: dict[str, int] | None,
    animal_pose_head_ids: set[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], Counter[str]]:
    payload = _read_json(source_path)
    images = payload.get("images") if isinstance(payload.get("images"), list) else []
    anns = payload.get("annotations") if isinstance(payload.get("annotations"), list) else []
    image_by_id = {int(image.get("id")): image for image in images if image.get("id") is not None}
    source_id_by_image_id = {image_id: _source_image_id(image) for image_id, image in image_by_id.items()}
    anns_by_image_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
    counter: Counter[str] = Counter()

    for ann in anns:
        image_id_int = int(ann.get("image_id"))
        source_id = source_id_by_image_id.get(image_id_int, "")
        feature_row = features.get(source_id)
        attrs = ann.setdefault("attributes", {})
        if not isinstance(attrs, dict):
            attrs = {}
            ann["attributes"] = attrs
        pet_pose_head_available = source_id in animal_pose_head_ids if animal_pose_head_ids else None
        routing = derive_query_routing_v2_simple(
            mode_name=str(ann.get("mode_name") or ""),
            feature_row=feature_row,
            attributes=attrs,
            positive_exists=bool(attrs.get("query_has_positive", True)),
            positive_score=_score_value(ann),
            no_positive_reason=str(attrs.get("negative_reason") or ""),
            pet_pose_head_available=pet_pose_head_available,
        )
        if routing.get("route_family_v2") == "pet_dogcat" and pet_pose_head_available is False:
            attrs["pet_pose_head_required"] = 1
            attrs["pet_pose_head_available"] = 0
            attrs["pet_pose_gate_suppressed_positive"] = int(int(ann.get("gt_flag") or 0) == 1)
            ann["gt_flag"] = 0
            ann["is_best"] = 0
        elif routing.get("route_family_v2") == "pet_dogcat":
            attrs["pet_pose_head_required"] = 1
            attrs["pet_pose_head_available"] = int(bool(pet_pose_head_available))
        flat_key = flat_route_class_key(routing)
        attrs["routing_v2_simple"] = routing
        attrs["flat_route_class"] = flat_key
        attrs["routing_v2_hierarchical_targets"] = hierarchical_target_ids(routing)
        if flat_vocab is not None and flat_key in flat_vocab:
            attrs["flat_route_class_id"] = int(flat_vocab[flat_key])
        anns_by_image_id[image_id_int].append(ann)
        counter[f"annotation_route_family:{routing['route_family_v2']}"] += 1
        counter[f"annotation_flat:{flat_key}"] += 1

    image_rows: list[dict[str, Any]] = []
    for image in images:
        source_id = _source_image_id(image)
        feature_row = features.get(source_id)
        vote = _vote_summary(anns_by_image_id.get(int(image.get("id")), []))
        pet_pose_head_available = source_id in animal_pose_head_ids if animal_pose_head_ids else None
        routing = derive_image_routing_v2_simple(feature_row, positive_vote_summary=vote, pet_pose_head_available=pet_pose_head_available)
        flat_key = flat_route_class_key(routing)
        targets = hierarchical_target_ids(routing)
        image["routing_v2_simple"] = routing
        image["flat_route_class"] = flat_key
        if flat_vocab is not None and flat_key in flat_vocab:
            image["flat_route_class_id"] = int(flat_vocab[flat_key])
        image["routing_v2_hierarchical_targets"] = targets
        image_path = _resolve_image_path(image_root, image, source_id)
        row = {
            "image_id": source_id,
            "file_name": str(image.get("file_name") or image_path.name),
            "image_path": str(image_path),
            "width": int(image.get("width") or 0),
            "height": int(image.get("height") or 0),
            "split": split,
            "routing_v2_simple": routing,
            "flat_route_class": flat_key,
            "flat_route_class_id": int(flat_vocab.get(flat_key, -1)) if flat_vocab is not None else -1,
            "hierarchical_targets": targets,
            "label_weight": float(routing.get("routing_confidence") or 0.0),
            "positive_vote_summary": vote,
            "probe_features": probe_features_from_feature_row(feature_row),
        }
        image_rows.append(row)
        counter[f"image_route_family:{routing['route_family_v2']}"] += 1
        counter[f"image_flat:{flat_key}"] += 1
        counter[f"image_reliability:{routing['teacher_reliability']}"] += 1

    _write_json(output_path, payload)
    return payload, image_rows, counter


def _collect_flat_vocab(
    *,
    source_label_json: Path,
    features: dict[str, dict[str, Any]],
    image_root: Path,
    animal_pose_head_ids: set[str],
) -> dict[str, int]:
    tmp_payload = _read_json(source_label_json)
    image_by_id = {int(image.get("id")): image for image in tmp_payload.get("images", []) if image.get("id") is not None}
    source_id_by_image_id = {image_id: _source_image_id(image) for image_id, image in image_by_id.items()}
    keys: set[str] = set()
    for ann in tmp_payload.get("annotations", []):
        source_id = source_id_by_image_id.get(int(ann.get("image_id")), "")
        attrs = ann.get("attributes") if isinstance(ann.get("attributes"), dict) else {}
        pet_pose_head_available = source_id in animal_pose_head_ids if animal_pose_head_ids else None
        routing = derive_query_routing_v2_simple(
            mode_name=str(ann.get("mode_name") or ""),
            feature_row=features.get(source_id),
            attributes=attrs,
            positive_score=_score_value(ann),
            pet_pose_head_available=pet_pose_head_available,
        )
        keys.add(flat_route_class_key(routing))
    # Include all image-level keys from the source full JSON as well.
    anns_by_image_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in tmp_payload.get("annotations", []):
        anns_by_image_id[int(ann.get("image_id"))].append(ann)
    for image_id, image in image_by_id.items():
        source_id = _source_image_id(image)
        enriched_anns = []
        for ann in anns_by_image_id.get(image_id, []):
            attrs = ann.get("attributes") if isinstance(ann.get("attributes"), dict) else {}
            pet_pose_head_available = source_id in animal_pose_head_ids if animal_pose_head_ids else None
            routing = derive_query_routing_v2_simple(
                mode_name=str(ann.get("mode_name") or ""),
                feature_row=features.get(source_id),
                attributes=attrs,
                positive_score=_score_value(ann),
                pet_pose_head_available=pet_pose_head_available,
            )
            ann = dict(ann)
            ann["attributes"] = {**attrs, "routing_v2_simple": routing}
            enriched_anns.append(ann)
        vote = _vote_summary(enriched_anns)
        pet_pose_head_available = source_id in animal_pose_head_ids if animal_pose_head_ids else None
        keys.add(flat_route_class_key(derive_image_routing_v2_simple(features.get(source_id), positive_vote_summary=vote, pet_pose_head_available=pet_pose_head_available)))
    return {key: idx for idx, key in enumerate(sorted(keys))}


def _augment_query_status(
    *,
    source_path: Path,
    output_path: Path,
    features: dict[str, dict[str, Any]],
    flat_vocab: dict[str, int],
    animal_pose_head_ids: set[str],
) -> tuple[int, Counter[str]]:
    total = 0
    counter: Counter[str] = Counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            feature_row = features.get(str(row.get("source_image_id") or ""))
            pet_pose_head_available = str(row.get("source_image_id") or "") in animal_pose_head_ids if animal_pose_head_ids else None
            attrs = row.get("attributes") if isinstance(row.get("attributes"), dict) else {}
            routing = derive_query_routing_v2_simple(
                mode_name=str(row.get("mode_name") or ""),
                feature_row=feature_row,
                attributes=attrs,
                positive_exists=bool(row.get("positive_exists", 0)),
                positive_score=row.get("positive_score") if row.get("positive_score") is not None else row.get("best_score"),
                no_positive_reason=str(row.get("no_positive_reason") or ""),
                pet_pose_head_available=pet_pose_head_available,
            )
            flat_key = flat_route_class_key(routing)
            row["routing_v2_simple"] = routing
            row["flat_route_class"] = flat_key
            row["flat_route_class_id"] = int(flat_vocab.get(flat_key, -1))
            row["routing_v2_hierarchical_targets"] = hierarchical_target_ids(routing)
            dst.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            total += 1
            counter[f"query_route_family:{routing['route_family_v2']}"] += 1
            counter[f"query_feasible:{routing['mode_feasible']}"] += 1
            if "support_grounding_miss" in str(row.get("no_positive_reason") or ""):
                counter[f"support_grounding_miss:{routing.get('person_shot_type') or 'na'}"] += 1
    return total, counter


def _write_summary_csv(path: Path, counter: Counter[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["key", "count"])
        for key, count in sorted(counter.items()):
            writer.writerow([key, int(count)])


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    start = time.time()
    phase_root = args.phase_root
    image_root = args.image_root or phase_root / "images"
    mm_out_dir = phase_root / "artifacts" / "training_labels_multimode" / args.run_tag
    label_out_dir = mm_out_dir / "label_json"
    tl_out_dir = phase_root / "artifacts" / "training_labels" / args.run_tag
    mm_out_dir.mkdir(parents=True, exist_ok=True)
    tl_out_dir.mkdir(parents=True, exist_ok=True)
    status_path = mm_out_dir / "status.json"
    _write_json(status_path, {"state": "running", "phase": "load_features", "run_tag": args.run_tag})

    features = _load_features(args.feature_jsonl, max_images=int(args.max_images))
    animal_pose_head_ids = _load_animal_pose_head_ids(args.animal_pose_jsonl)
    _write_json(
        status_path,
        {
            "state": "running",
            "phase": "collect_flat_vocab",
            "feature_rows": len(features),
            "animal_pose_head_ids": len(animal_pose_head_ids),
        },
    )
    flat_vocab = _collect_flat_vocab(
        source_label_json=args.source_multimode_run_dir / "label_json" / "multimode_labels_full.json",
        features=features,
        image_root=image_root,
        animal_pose_head_ids=animal_pose_head_ids,
    )
    vocabs = {
        "schema_version": SCHEMA_VERSION,
        "flat_route_class": flat_vocab,
        "hierarchical": HIERARCHICAL_VOCABS,
        "notes": {
            "route_family_v2": list(ROUTE_FAMILY_V2),
            "person_shot_type": list(PERSON_SHOT_TYPE),
            "placement_intent": list(PLACEMENT_INTENT),
            "context_intent": list(CONTEXT_INTENT),
        },
    }
    _write_json(mm_out_dir / "routing_v2_simple_vocabs.json", vocabs)
    _write_json(tl_out_dir / "routing_v2_simple_vocabs.json", vocabs)

    all_image_rows: dict[str, list[dict[str, Any]]] = {}
    total_counter: Counter[str] = Counter()
    split_summaries: dict[str, Any] = {}
    for split in ("full", "train", "val"):
        _write_json(status_path, {"state": "running", "phase": f"augment_{split}", "feature_rows": len(features)})
        source_path = args.source_multimode_run_dir / "label_json" / f"multimode_labels_{split}.json"
        output_path = label_out_dir / f"multimode_labels_{split}.json"
        payload, rows, counter = _augment_coco_split(
            split=split,
            source_path=source_path,
            output_path=output_path,
            features=features,
            image_root=image_root,
            flat_vocab=flat_vocab,
            animal_pose_head_ids=animal_pose_head_ids,
        )
        all_image_rows[split] = rows
        total_counter.update(counter)
        _write_jsonl(tl_out_dir / f"routing_v2_simple_{split}.jsonl", rows)
        split_summaries[split] = {
            "images": len(payload.get("images", [])),
            "annotations": len(payload.get("annotations", [])),
            "image_rows": len(rows),
            "counter": dict(counter),
        }
        target_ar_source = args.source_multimode_run_dir / "label_json" / f"multimode_labels_target_ar_only_{split}.json"
        if target_ar_source.exists():
            shutil.copy2(target_ar_source, label_out_dir / f"multimode_labels_target_ar_only_{split}.json")

    query_total, query_counter = _augment_query_status(
        source_path=args.source_multimode_run_dir / "mode_query_status.jsonl",
        output_path=mm_out_dir / "mode_query_status_routing_v2_simple.jsonl",
        features=features,
        flat_vocab=flat_vocab,
        animal_pose_head_ids=animal_pose_head_ids,
    )
    total_counter.update(query_counter)
    for sidecar_name in ("categories.json", "dataset_stats.json", "mode_stats.csv", "target_ar_stats.csv", "mode_target_ar_stats.csv", "validation_summary.json"):
        source = args.source_multimode_run_dir / sidecar_name
        if source.exists():
            shutil.copy2(source, mm_out_dir / sidecar_name)
    if (args.source_multimode_run_dir / "splits").exists():
        shutil.copytree(args.source_multimode_run_dir / "splits", mm_out_dir / "splits", dirs_exist_ok=True)

    full_rows = all_image_rows.get("full", [])
    high_conf_min = float(args.high_conf_min)
    confidence_values = [float(row["routing_v2_simple"].get("routing_confidence") or 0.0) for row in full_rows]
    summary = {
        "state": "completed",
        "phase": "completed",
        "run_tag": args.run_tag,
        "source_multimode_run_dir": str(args.source_multimode_run_dir),
        "feature_jsonl": str(args.feature_jsonl),
        "animal_pose_jsonl": str(args.animal_pose_jsonl),
        "image_root": str(image_root),
        "multimode_output_dir": str(mm_out_dir),
        "training_labels_output_dir": str(tl_out_dir),
        "feature_rows": len(features),
        "animal_pose_head_ids": len(animal_pose_head_ids),
        "flat_class_count": len(flat_vocab),
        "query_status_rows": query_total,
        "split_summaries": split_summaries,
        "routing_confidence": {
            "min": min(confidence_values) if confidence_values else None,
            "max": max(confidence_values) if confidence_values else None,
            "mean": sum(confidence_values) / len(confidence_values) if confidence_values else None,
            "high_conf_min": high_conf_min,
            "high_conf_rate": sum(1 for v in confidence_values if v >= high_conf_min) / len(confidence_values) if confidence_values else None,
        },
        "counter": dict(total_counter),
        "outputs": {
            "multimode_label_json": str(label_out_dir),
            "mode_query_status": str(mm_out_dir / "mode_query_status_routing_v2_simple.jsonl"),
            "training_labels_jsonl": str(tl_out_dir),
            "vocabs": str(tl_out_dir / "routing_v2_simple_vocabs.json"),
        },
        "elapsed_sec": round(time.time() - start, 3),
    }
    _write_json(mm_out_dir / "summary.json", summary)
    _write_json(tl_out_dir / "summary.json", summary)
    _write_summary_csv(mm_out_dir / "routing_v2_simple_counts.csv", total_counter)
    _write_json(status_path, summary)
    _write_json(tl_out_dir / "status.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
