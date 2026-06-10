#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SSTK_UCTR_ROOT = "data/SSTK/Full_10000/artifacts/training_labels_uctr_stage3_product_ar_260422_hashsplit_80_10_10_260422"
SSTK_PUBLIC_ROOT = "data/SSTK/Full_10000/artifacts/training_labels_public_ensemble_best_product_ar_260422_hashsplit_80_10_10_260422"
PUBLIC_MANIFEST = "artifacts/unified_public_benchmark_20260420/stage_full/benchmark_task_manifest.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build extended MobileCropNet v4 shortlist manifest.")
    parser.add_argument("--base_manifest_json", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--route_root", type=Path, default=Path("artifacts/mobilecropnet_v4/route_gate_recover_20260424"))
    parser.add_argument("--subject_box_root", type=Path, default=Path("artifacts/mobilecropnet_v4/subject_box_head_uctr_20260423"))
    parser.add_argument("--subject_box_patch_root", type=Path, default=Path("artifacts/mobilecropnet_v4/subject_box_iou_patch_20260424_full"))
    parser.add_argument(
        "--release_gate_root",
        type=Path,
        action="append",
        default=[
            Path("artifacts/mobilecropnet_v4/release_gate_v1_20260424"),
            Path("artifacts/mobilecropnet_v4/release_gate_v2_20260424"),
        ],
    )
    return parser


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _has_run_artifacts(run_dir: Path, checkpoint: Path) -> bool:
    if checkpoint.exists():
        return True
    return any((run_dir / name).exists() for name in ("run_summary.json", "metrics.json", "config.json", "status.json"))


def _entry(
    *,
    track: str,
    variant: str,
    profile: str,
    run_name: str,
    run_dir: str,
    checkpoint: str,
    val_root: str,
    test_root: str,
    output_dir: str,
    method_id: str,
) -> dict[str, Any]:
    return {
        "track": track,
        "variant": variant,
        "profile": profile,
        "run_name": run_name,
        "run_dir": run_dir,
        "checkpoint": checkpoint,
        "val_jsonl": f"{val_root}/val/train_conditional_detr_batch.jsonl" if not val_root.endswith("/val") else f"{val_root}/train_conditional_detr_batch.jsonl",
        "test_jsonl": f"{test_root}/test/train_conditional_detr_batch.jsonl" if not test_root.endswith("/test") else f"{test_root}/train_conditional_detr_batch.jsonl",
        "public_task_manifest_jsonl": PUBLIC_MANIFEST,
        "output_dir": output_dir,
        "method_id": method_id,
    }


def _add_static_entries(entries: list[dict[str, Any]], args: argparse.Namespace) -> None:
    route_root = args.route_root
    route_specs = [
        {
            "track": "SSTK public routefix",
            "variant": "subjectprior_route_recover_v1",
            "profile": "balanced_288",
            "run_dir": route_root / "full_runs/sstk_public/mcn-sstk-public-routefix-balanced-288-20260424",
            "output_dir": route_root / "final_bundle/sstk_public_balanced_288_routefix",
            "method_id": "sstk_public__balanced_288__routefix_20260424",
            "labels": SSTK_PUBLIC_ROOT,
        },
        {
            "track": "SSTK UCTR routefix",
            "variant": "subjectprior_route_recover_v1",
            "profile": "balanced_288",
            "run_dir": route_root / "full_runs/sstk_uctr/mcn-sstk-uctr-routefix-balanced-288-20260424",
            "output_dir": route_root / "final_bundle/sstk_uctr_balanced_288_routefix",
            "method_id": "sstk_uctr_subjectprior__balanced_288__routefix_20260424",
            "labels": SSTK_UCTR_ROOT,
        },
    ]
    for spec in route_specs:
        run_dir = Path(spec["run_dir"])
        checkpoint = run_dir / "best.pt"
        if not run_dir.exists():
            continue
        entries.append(
            _entry(
                track=spec["track"],
                variant=spec["variant"],
                profile=spec["profile"],
                run_name=run_dir.name,
                run_dir=str(run_dir),
                checkpoint=str(checkpoint),
                val_root=spec["labels"],
                test_root=spec["labels"],
                output_dir=str(spec["output_dir"]),
                method_id=spec["method_id"],
            )
        )

    subj_root = args.subject_box_root / "sstk_uctr_subjectprior_v2_balanced_valid"
    subject_specs = [
        ("rank_320", "mcn-sstk-uctr-subjboxv2-rank-320-20260423-subjboxv2", "sstk_uctr_subjectbox_v2__rank_320", "sstk_uctr_subjectbox_v2_rank_320"),
        ("plus_384", "mcn-sstk-uctr-subjboxv2-plus-384-20260423-subjboxv2", "sstk_uctr_subjectbox_v2__plus_384", "sstk_uctr_subjectbox_v2_plus_384"),
    ]
    for profile, run_name, method_id, bundle_name in subject_specs:
        run_dir = subj_root / run_name
        checkpoint = run_dir / "best.pt"
        if not checkpoint.exists():
            continue
        entries.append(
            _entry(
                track="SSTK UCTR subject-box v2",
                variant="subject_box_prior_v2_balanced_valid",
                profile=profile,
                run_name=run_name,
                run_dir=str(run_dir),
                checkpoint=str(checkpoint),
                val_root=SSTK_UCTR_ROOT,
                test_root=SSTK_UCTR_ROOT,
                output_dir=str(args.subject_box_root / "final_bundle" / bundle_name),
                method_id=method_id,
            )
        )

    patch_root = args.subject_box_patch_root / "sstk_uctr_subjectprior"
    patch_specs = [
        ("rank_320", "mcn-sstk-uctr-subjboxpatch-r320-20260424", "sstk_uctr_subjectbox_patch__rank_320", "sstk_uctr_subjectbox_patch_rank_320"),
        ("plus_384", "mcn-sstk-uctr-subjboxpatch-p384-20260424", "sstk_uctr_subjectbox_patch__plus_384", "sstk_uctr_subjectbox_patch_plus_384"),
    ]
    for profile, run_name, method_id, bundle_name in patch_specs:
        run_dir = patch_root / run_name
        checkpoint = run_dir / "best.pt"
        if not _has_run_artifacts(run_dir, checkpoint):
            continue
        entries.append(
            _entry(
                track="SSTK UCTR subject-box patch",
                variant="subject_box_iou_patch",
                profile=profile,
                run_name=run_name,
                run_dir=str(run_dir),
                checkpoint=str(checkpoint),
                val_root=SSTK_UCTR_ROOT,
                test_root=SSTK_UCTR_ROOT,
                output_dir=str(args.subject_box_patch_root / "final_bundle" / bundle_name),
                method_id=method_id,
            )
        )


def _infer_release_variant(run_name: str) -> str:
    if "releasegate-v1" in run_name:
        return "release_gate_v1"
    if "rg-v2-ftsubj" in run_name:
        return "release_gate_v2_finetune_subjectpatch"
    if "rg-v2-valid" in run_name:
        return "release_gate_v2_validcal"
    if "rg-v2-npr" in run_name:
        return "release_gate_v2_nopriorroute"
    if "rg-v2" in run_name:
        return "release_gate_v2"
    for variant in ("release_gate_v2_nopriorroute", "release_gate_v2_validcal", "release_gate_v2", "release_gate_v1"):
        token = variant.replace("release_gate_", "rg-").replace("_", "-")
        if variant in run_name or token in run_name:
            return variant
    return "release_gate"


def _add_release_gate_entries(entries: list[dict[str, Any]], release_root: Path) -> None:
    full_root = release_root / "full_runs/sstk_uctr"
    for run_dir in sorted(full_root.glob("mcn-sstk-uctr-*")):
        checkpoint = run_dir / "best.pt"
        if not _has_run_artifacts(run_dir, checkpoint):
            continue
        config = _read_json(run_dir / "config.json") or {}
        profile = str(config.get("model_profile_effective") or config.get("model_profile") or "")
        if not profile:
            for token, candidate in (("b288", "balanced_288"), ("r320", "rank_320"), ("p384", "plus_384")):
                if token in run_dir.name:
                    profile = candidate
                    break
        if not profile:
            profile = "unknown"
        variant = _infer_release_variant(run_dir.name)
        method_id = f"sstk_uctr__{variant}__{profile}"
        entries.append(
            _entry(
                track=f"SSTK UCTR {variant}",
                variant=variant,
                profile=profile,
                run_name=run_dir.name,
                run_dir=str(run_dir),
                checkpoint=str(checkpoint),
                val_root=SSTK_UCTR_ROOT,
                test_root=SSTK_UCTR_ROOT,
                output_dir=str(release_root / "final_bundle" / run_dir.name),
                method_id=method_id,
            )
        )


def main() -> int:
    args = build_parser().parse_args()
    base = _read_json(args.base_manifest_json)
    if not isinstance(base, dict) or not isinstance(base.get("entries"), list):
        raise SystemExit(f"invalid base manifest: {args.base_manifest_json}")
    entries = [dict(row) for row in base["entries"] if isinstance(row, dict)]
    _add_static_entries(entries, args)
    for release_root in args.release_gate_root:
        _add_release_gate_entries(entries, release_root)
    seen: set[tuple[str, str, str]] = set()
    deduped = []
    for row in entries:
        key = (str(row.get("track")), str(row.get("profile")), str(row.get("run_name")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    payload = {"entry_count": len(deduped), "entries": deduped}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"entry_count": len(deduped), "output_json": str(args.output_json)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
