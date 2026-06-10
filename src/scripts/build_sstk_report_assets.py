#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

try:
    from progress_utils import ProgressTracker, progress_log
except ModuleNotFoundError:
    from scripts.progress_utils import ProgressTracker, progress_log


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build SSTK report summary/evidence assets from rerun outputs.")
    p.add_argument("--run_tag", required=True)
    p.add_argument("--routed_feats_jsonl", required=True)
    p.add_argument("--teacher_scores_jsonl", required=True)
    p.add_argument("--teacher_qa_json", required=True)
    p.add_argument("--components_viz_dir", required=True)
    p.add_argument("--output_report_dir", required=True)
    p.add_argument("--examples_per_bucket", type=int, default=5)
    p.add_argument("--progress", type=int, default=1)
    p.add_argument("--progress_every", type=int, default=1000)
    p.add_argument("--progress_min_seconds", type=float, default=10.0)
    return p.parse_args()


def safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(float(v))
    except Exception:
        return int(default)


def load_teacher_decision_counts(path: Path) -> Counter:
    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            by_ar = rec.get("teacher_scorer", {}).get("results_by_ar", {})
            if not isinstance(by_ar, dict):
                continue
            for ar_res in by_ar.values():
                if not isinstance(ar_res, dict):
                    continue
                decision = ar_res.get("decision", {})
                if not isinstance(decision, dict):
                    continue
                counts[str(decision.get("decision_type", "unknown"))] += 1
    return counts


def decision_counts_from_qa(qa: Dict[str, Any]) -> Counter:
    counts = qa.get("global", {}).get("decision_counts", {})
    if not isinstance(counts, dict):
        return Counter()
    out: Counter[str] = Counter()
    for key, value in counts.items():
        try:
            out[str(key)] = int(float(value))
        except Exception:
            continue
    return out


def build_paths(image_id: str, routed_feats_jsonl: str, components_viz_dir: str) -> Dict[str, str]:
    return {
        "source_routed_feats_jsonl": routed_feats_jsonl,
        "original_image": f"{components_viz_dir}/original/{image_id}.jpg",
        "precompute_combined": f"{components_viz_dir}/combined_all/{image_id}.jpg",
        "precompute_c2": f"{components_viz_dir}/c2_seg/{image_id}.jpg",
        "precompute_c3": f"{components_viz_dir}/c3_pose/{image_id}.jpg",
        "precompute_c4": f"{components_viz_dir}/c4_ocr/{image_id}.jpg",
        "precompute_c5": f"{components_viz_dir}/c5_geom/{image_id}.jpg",
        "precompute_c6": f"{components_viz_dir}/c6_gaze/{image_id}.jpg",
    }


def subject_count_bucket(num_person: int) -> str:
    if num_person >= 2:
        return "ge2"
    if num_person == 1:
        return "eq1"
    return "eq0"


def sort_examples(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        rows,
        key=lambda r: (
            -float(r.get("subject_mode_conf", 0.0)),
            -int(r.get("subject_set", {}).get("num_effective_subjects", 0)),
            str(r.get("image_id", "")),
        ),
    )


def main() -> None:
    args = parse_args()
    progress_enabled = bool(int(args.progress))
    progress_every = max(1, int(args.progress_every))
    progress_min_seconds = max(0.0, float(args.progress_min_seconds))
    run_tag = str(args.run_tag)
    routed_feats_jsonl = Path(args.routed_feats_jsonl)
    teacher_scores_jsonl = Path(args.teacher_scores_jsonl)
    teacher_qa_json = Path(args.teacher_qa_json)
    output_report_dir = Path(args.output_report_dir)

    analytics_dir = output_report_dir / "assets" / "analytics"
    evidence_dir = output_report_dir / "assets" / "evidence"
    subject_count_dir = evidence_dir / "subject_count"
    subject_mode_dir = evidence_dir / "subject_mode"
    analytics_dir.mkdir(parents=True, exist_ok=True)
    subject_count_dir.mkdir(parents=True, exist_ok=True)
    subject_mode_dir.mkdir(parents=True, exist_ok=True)

    progress_log(
        f"build_sstk_report_assets: start | run_tag={run_tag} | routed_feats_jsonl={routed_feats_jsonl.name}",
        enabled=progress_enabled,
    )
    qa = json.loads(teacher_qa_json.read_text(encoding="utf-8"))
    decision_counts = decision_counts_from_qa(qa)
    if decision_counts:
        progress_log(
            "build_sstk_report_assets: using decision_counts from teacher QA; skipped teacher JSONL full scan",
            enabled=progress_enabled,
        )
    else:
        progress_log(
            "build_sstk_report_assets: teacher QA has no decision_counts; scanning teacher JSONL fallback",
            enabled=progress_enabled,
        )
        decision_counts = load_teacher_decision_counts(teacher_scores_jsonl)

    mode_counter: Counter[str] = Counter()
    person_bucket_counter: Counter[str] = Counter()
    inst_bucket_counter: Counter[str] = Counter()
    effective_bucket_counter: Counter[str] = Counter()
    subject_count_examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    subject_mode_examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    routed_feats_rel = str(routed_feats_jsonl).replace("\\", "/")
    components_viz_dir = str(Path(args.components_viz_dir)).replace("\\", "/")

    total = 0
    routed_tracker = ProgressTracker(
        "build_sstk_report_assets:routed_feats",
        unit="rows",
        every=progress_every,
        min_seconds=progress_min_seconds,
        enabled=progress_enabled,
    )
    with routed_feats_jsonl.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            total += 1
            rec = json.loads(line)
            image_id = str(rec.get("image_id", ""))
            routing = rec.get("routing", {}) if isinstance(rec.get("routing"), dict) else {}
            subject_set = routing.get("subject_set", {}) if isinstance(routing.get("subject_set"), dict) else {}
            router_signals = routing.get("router_signals", {}) if isinstance(routing.get("router_signals"), dict) else {}
            mode = str(routing.get("subject_mode", "other_ambiguous"))
            num_person = safe_int(subject_set.get("num_person", 0))
            num_c2_instances = safe_int(subject_set.get("num_c2_instances", subject_set.get("num_subject_inst", 0)))
            num_effective_subjects = safe_int(subject_set.get("num_effective_subjects", num_c2_instances))

            mode_counter[mode] += 1
            person_bucket_counter[subject_count_bucket(num_person)] += 1
            if num_c2_instances >= 2:
                inst_bucket_counter["ge2"] += 1
            elif num_c2_instances == 1:
                inst_bucket_counter["eq1"] += 1
            else:
                inst_bucket_counter["eq0"] += 1
            if num_effective_subjects >= 2:
                effective_bucket_counter["ge2"] += 1
            elif num_effective_subjects == 1:
                effective_bucket_counter["eq1"] += 1
            else:
                effective_bucket_counter["eq0"] += 1

            row = {
                "line_index": idx,
                "image_id": image_id,
                "subject_mode": mode,
                "subject_mode_conf": float(routing.get("subject_mode_conf", 0.0)),
                "subject_mode_reasons": routing.get("subject_mode_reasons", []),
                "subject_mode_flags": routing.get("flags", {}),
                "subject_mode_conflict": bool(routing.get("subject_mode_conflict", False)),
                "policy_id": str(routing.get("policy_id", "")),
                "router_rule_id": str(routing.get("router_rule_id", "")),
                "router_signals": router_signals,
                "subject_set": subject_set,
                "c2_seg_count": len(rec.get("c2_seg", [])) if isinstance(rec.get("c2_seg"), list) else 0,
                "c2_det_count": len(rec.get("c2_det", [])) if isinstance(rec.get("c2_det"), list) else 0,
                "c3_pose_count": len(rec.get("c3_pose", [])) if isinstance(rec.get("c3_pose"), list) else 0,
                "c4_num_boxes": safe_int(router_signals.get("ocr_text_boxes", 0)),
                "c4_meta": rec.get("c4_ocr_meta", {}),
                "c5_geom": rec.get("c5_geom", {}),
                "c6_gaze": rec.get("c6_gaze", {}),
                "paths": build_paths(image_id=image_id, routed_feats_jsonl=routed_feats_rel, components_viz_dir=components_viz_dir),
            }
            subject_count_examples[subject_count_bucket(num_person)].append(row)
            subject_mode_examples[mode].append(row)
            routed_tracker.update(total, extra=f"image_id={image_id} mode={mode}")
    routed_tracker.finish(total, extra=f"modes={len(mode_counter)}")

    limit = max(1, int(args.examples_per_bucket))
    subject_count_examples_sorted = {
        bucket: sort_examples(rows)[:limit]
        for bucket, rows in sorted(subject_count_examples.items())
    }
    subject_mode_examples_sorted = {
        mode: sort_examples(rows)[:limit]
        for mode, rows in sorted(subject_mode_examples.items())
    }

    for bucket, rows in subject_count_examples_sorted.items():
        for row in rows:
            out = subject_count_dir / f"{bucket}_{row['image_id']}.json"
            out.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    for mode, rows in subject_mode_examples_sorted.items():
        for row in rows:
            out = subject_mode_dir / f"{mode}_{row['image_id']}.json"
            out.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "run_tag": run_tag,
        "paths": {
            "routed_feats_jsonl": routed_feats_rel,
            "teacher_scores_jsonl": str(teacher_scores_jsonl).replace("\\", "/"),
            "teacher_qa_json": str(teacher_qa_json).replace("\\", "/"),
        },
        "counts": {
            "num_images": total,
            "num_top1_ar_results": int(sum(decision_counts.values())),
            "subject_mode_counts": dict(mode_counter),
            "subject_count_person": dict(person_bucket_counter),
            "subject_count_instances": dict(inst_bucket_counter),
            "subject_count_effective": dict(effective_bucket_counter),
            "decision_counts": dict(decision_counts),
        },
        "qa_snapshots": {
            "global_p0_release_gates": qa.get("global", {}).get("p0_release_gates", {}),
            "mode_qa_summary": qa.get("mode_qa_summary", {}),
        },
    }

    (analytics_dir / f"summary_{run_tag}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (evidence_dir / f"subject_count_examples_{run_tag}.json").write_text(
        json.dumps(subject_count_examples_sorted, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (evidence_dir / f"subject_mode_examples_{run_tag}.json").write_text(
        json.dumps(subject_mode_examples_sorted, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    progress_log(
        f"build_sstk_report_assets: finished | num_images={total} | subject_modes={len(mode_counter)}",
        enabled=progress_enabled,
    )

    print(f"[done] summary={analytics_dir / f'summary_{run_tag}.json'}")
    print(f"[done] subject_count_examples={evidence_dir / f'subject_count_examples_{run_tag}.json'}")
    print(f"[done] subject_mode_examples={evidence_dir / f'subject_mode_examples_{run_tag}.json'}")


if __name__ == "__main__":
    main()
