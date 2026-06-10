#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an interim track-level leaderboard for MobileCropNet v4 product experiments.")
    parser.add_argument("--matrix_status_json", type=Path, required=True)
    parser.add_argument("--sstk_public_summary_json", type=Path, default=None)
    parser.add_argument("--sstk_public_collection_json", type=Path, default=None)
    parser.add_argument("--sstk_t1_summary_json", type=Path, default=None)
    parser.add_argument("--sstk_t1_collection_json", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _f(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _best_by(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    if not rows:
        return {}
    return max(rows, key=lambda row: (_f(row.get(key)), str(row.get("profile", ""))))


def _matrix_track_rows(track_name: str, matrix_payload: dict[str, Any]) -> dict[str, Any]:
    tracks = matrix_payload.get("tracks", {})
    track = tracks.get(track_name, {})
    rows = list(track.get("profiles", []))
    completed = int(track.get("completed_profile_count") or 0)
    total = int(track.get("total_profile_count") or len(rows))
    return {
        "track": track_name,
        "variant": track.get("variant"),
        "rows": rows,
        "completed_profile_count": completed,
        "total_profile_count": total,
        "remaining_profiles": sorted(row.get("profile") for row in rows if not row.get("has_run_summary")),
    }


def _override_track_rows(
    track_name: str,
    variant: str,
    summary_payload: dict[str, Any],
    collection_payload: dict[str, Any],
) -> dict[str, Any]:
    rows = list(summary_payload.get("runs", []))
    selected_runs = list(collection_payload.get("selected_runs", []))
    selected_by_profile = {row.get("profile"): row for row in selected_runs}
    total = int(collection_payload.get("selected_profile_count") or len(selected_runs) or len(rows))
    completed = len(rows)
    remaining = sorted(
        profile
        for profile, row in selected_by_profile.items()
        if row.get("artifact_status") != "complete"
    )
    for row in rows:
        row.setdefault("track", track_name)
        row.setdefault("variant", variant)
    return {
        "track": track_name,
        "variant": variant,
        "rows": rows,
        "completed_profile_count": completed,
        "total_profile_count": total,
        "remaining_profiles": remaining,
    }


def _track_summary(track_payload: dict[str, Any]) -> dict[str, Any]:
    rows = list(track_payload.get("rows", []))
    selection_best = _best_by(rows, "best_selection_score")
    mos_best = _best_by(rows, "official_top1_mos")
    pcc_best = _best_by(rows, "official_pcc")
    srcc_best = _best_by(rows, "official_srcc")
    shortlist = []
    for row in [selection_best, mos_best, pcc_best, srcc_best]:
        profile = row.get("profile")
        if profile and profile not in shortlist:
            shortlist.append(str(profile))
    return {
        "track": track_payload.get("track"),
        "variant": track_payload.get("variant"),
        "completed_profile_count": track_payload.get("completed_profile_count"),
        "total_profile_count": track_payload.get("total_profile_count"),
        "remaining_profiles": track_payload.get("remaining_profiles", []),
        "selection_best": {
            "profile": selection_best.get("profile"),
            "score": _f(selection_best.get("best_selection_score")),
        },
        "top1_mos_best": {
            "profile": mos_best.get("profile"),
            "score": _f(mos_best.get("official_top1_mos")),
            "regret": _f(mos_best.get("official_regret")),
        },
        "pcc_best": {
            "profile": pcc_best.get("profile"),
            "score": _f(pcc_best.get("official_pcc")),
        },
        "srcc_best": {
            "profile": srcc_best.get("profile"),
            "score": _f(srcc_best.get("official_srcc")),
        },
        "shortlist_profiles": shortlist,
    }


def _write_markdown(path: Path, summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# MobileCropNet v4 Interim Track Leaderboard",
        "",
        "| track | variant | done | selection-best | top1 MOS-best | PCC-best | SRCC-best | shortlist | remaining |",
        "| --- | --- | ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summaries:
        done = f"{row['completed_profile_count']}/{row['total_profile_count']}"
        sel = row["selection_best"]
        mos = row["top1_mos_best"]
        pcc = row["pcc_best"]
        srcc = row["srcc_best"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["track"]),
                    str(row.get("variant") or ""),
                    done,
                    f"{sel.get('profile') or ''} ({sel.get('score', 0.0):.6f})",
                    f"{mos.get('profile') or ''} ({mos.get('score', 0.6):.6f})",
                    f"{pcc.get('profile') or ''} ({pcc.get('score', 0.0):.6f})",
                    f"{srcc.get('profile') or ''} ({srcc.get('score', 0.0):.6f})",
                    ", ".join(row.get("shortlist_profiles", [])),
                    ", ".join(row.get("remaining_profiles", [])),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    matrix_payload = _read_json(args.matrix_status_json)
    track_payloads = [
        _matrix_track_rows("GAIC T1", matrix_payload),
        _matrix_track_rows("GAIC UCTR", matrix_payload),
        _matrix_track_rows("GAIC public", matrix_payload),
        _matrix_track_rows("GAIC UCTR subjectprior", matrix_payload),
        _matrix_track_rows("SSTK UCTR subjectprior", matrix_payload),
    ]

    sstk_public_summary = _read_json(args.sstk_public_summary_json)
    sstk_public_collection = _read_json(args.sstk_public_collection_json)
    if sstk_public_summary:
        track_payloads.append(
            _override_track_rows(
                "SSTK public",
                "subjectprior_route_proposal_v1",
                sstk_public_summary,
                sstk_public_collection,
            )
        )

    sstk_t1_summary = _read_json(args.sstk_t1_summary_json)
    sstk_t1_collection = _read_json(args.sstk_t1_collection_json)
    if sstk_t1_summary:
        track_payloads.append(
            _override_track_rows(
                "SSTK T1",
                "baseline_current",
                sstk_t1_summary,
                sstk_t1_collection,
            )
        )

    summaries = [_track_summary(track_payload) for track_payload in track_payloads if track_payload.get("rows")]
    payload = {"track_count": len(summaries), "tracks": summaries}
    (args.output_dir / "interim_track_leaderboard.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_markdown(args.output_dir / "interim_track_leaderboard.md", summaries)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
