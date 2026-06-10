#!/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


METHOD_ROWS: list[dict[str, Any]] = [
    {
        "method": "sstk_teacher_t1_historical_gaic_plus_public_proxy",
        "family": "historical_teacher",
        "fcdb_iou_top1": 0.7204,
        "cpc_weighted_pairwise": 0.8374,
        "gnmc_iou_top1": 0.7329,
        "top1_mos": 3.5985,
        "srcc": 0.4979,
        "accw4_of_top10": 0.2856,
        "source": "Implement_Docs/UniversalCropTeacher_H_Implementation_Report_KO_2026-04-20.md:491",
    },
    {
        "method": "teacher_proxy_compact_utility_pool",
        "family": "teacher_proxy",
        "fcdb_iou_top1": 0.7204,
        "cpc_weighted_pairwise": 0.8374,
        "gnmc_iou_top1": 0.7329,
        "top1_mos": 3.6731,
        "srcc": 0.5022,
        "accw4_of_top10": 0.3140,
        "source": "Implement_Docs/UniversalCropTeacher_H_Implementation_Report_KO_2026-04-20.md:492",
    },
    {
        "method": "public_cropper_gaic",
        "family": "public_cropper",
        "fcdb_iou_top1": 0.7376,
        "cpc_weighted_pairwise": 0.8784,
        "gnmc_iou_top1": 0.7692,
        "top1_mos": 3.9853,
        "srcc": 0.8459,
        "accw4_of_top10": 0.6062,
        "source": "Implement_Docs/UniversalCropTeacher_H_Implementation_Report_KO_2026-04-20.md:493",
    },
    {
        "method": "public_cropper_cgs",
        "family": "public_cropper",
        "fcdb_iou_top1": 0.7334,
        "cpc_weighted_pairwise": 0.8851,
        "gnmc_iou_top1": 0.7702,
        "top1_mos": 3.9210,
        "srcc": 0.7998,
        "accw4_of_top10": 0.5521,
        "source": "Implement_Docs/UniversalCropTeacher_H_Implementation_Report_KO_2026-04-20.md:494",
    },
    {
        "method": "public_cropper_ensemble_best",
        "family": "public_cropper",
        "fcdb_iou_top1": 0.7351,
        "cpc_weighted_pairwise": 0.8918,
        "gnmc_iou_top1": 0.7772,
        "top1_mos": 3.9841,
        "srcc": 0.8524,
        "accw4_of_top10": 0.6240,
        "source": "Implement_Docs/UniversalCropTeacher_H_Implementation_Report_KO_2026-04-20.md:495",
    },
    {
        "method": "universal_crop_teacher_h_stage2_fullhonest",
        "family": "uctr_teacher",
        "fcdb_iou_top1": 0.9080,
        "cpc_weighted_pairwise": 0.9120,
        "gnmc_iou_top1": 0.9168,
        "top1_mos": 3.4324,
        "srcc": 0.3810,
        "accw4_of_top10": 0.2272,
        "source": "Implement_Docs/UniversalCropTeacher_H_Implementation_Report_KO_2026-04-20.md:498",
    },
    {
        "method": "universal_crop_teacher_h_stage3_gate128",
        "family": "uctr_teacher",
        "fcdb_iou_top1": 0.9085,
        "cpc_weighted_pairwise": 0.9135,
        "gnmc_iou_top1": 0.9173,
        "top1_mos": 3.8461,
        "srcc": 0.6294,
        "accw4_of_top10": 0.4438,
        "source": "Implement_Docs/UniversalCropTeacher_H_Implementation_Report_KO_2026-04-20.md:499",
    },
    {
        "method": "production_final_hybrid",
        "family": "hybrid_teacher",
        "fcdb_iou_top1": 0.9085,
        "cpc_weighted_pairwise": 0.9135,
        "gnmc_iou_top1": 0.9173,
        "top1_mos": 3.9869,
        "srcc": 0.8530,
        "accw4_of_top10": 0.6283,
        "source": "Implement_Docs/UniversalCropTeacher_H_Implementation_Report_KO_2026-04-20.md:730",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build equal-4 normalized crop benchmark leaderboard for public cropper / teacher methods."
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def _gaic_primary(row: dict[str, Any]) -> float:
    return float(
        0.50 * (float(row["top1_mos"]) / 5.0)
        + 0.35 * float(row["srcc"])
        + 0.15 * float(row["accw4_of_top10"])
    )


def _zscore(value: float, values: list[float]) -> float:
    sigma = float(pstdev(values))
    if sigma <= 1e-12:
        return 0.0
    return (float(value) - float(mean(values))) / sigma


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "-"
    return f"{number:.{digits}f}"


def build_rows() -> list[dict[str, Any]]:
    rows = [dict(item) for item in METHOD_ROWS]
    for row in rows:
        row["gaic_primary"] = _gaic_primary(row)

    metric_keys = ("fcdb_iou_top1", "cpc_weighted_pairwise", "gnmc_iou_top1", "gaic_primary")
    metric_values = {key: [float(row[key]) for row in rows] for key in metric_keys}
    for row in rows:
        for key in metric_keys:
            row[f"z_{key}"] = _zscore(float(row[key]), metric_values[key])
        row["equal4_raw_mean"] = float(sum(float(row[key]) for key in metric_keys) / float(len(metric_keys)))
        row["equal4_zscore"] = float(
            sum(float(row[f"z_{key}"]) for key in metric_keys) / float(len(metric_keys))
        )
        row["worst_dataset_z"] = float(min(float(row[f"z_{key}"]) for key in metric_keys))
    rows.sort(key=lambda item: (float(item["equal4_zscore"]), float(item["gaic_primary"])), reverse=True)
    for idx, row in enumerate(rows, start=1):
        row["rank_equal4_zscore"] = idx
    return rows


def build_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Equal-4 Teacher/Public Cropper Leaderboard",
        "",
        "- scope: `public cropper` and `data curation teacher` methods only",
        "- note: original `UNIFIED_PUBLIC_BENCHMARK_REPORT_KO.md` did not contain this equal-4 table; this is a post-hoc recomputation from already recorded benchmark metrics",
        "- normalization: per-dataset primary metric z-score across the compared method set",
        "- primaries:",
        "  - `FCDB`: `IoU@top1`",
        "  - `CPC`: `weighted pairwise accuracy`",
        "  - `GNMC`: `IoU@top1`",
        "  - `GAIC`: `0.50*(top1_mos/5) + 0.35*SRCC + 0.15*Accw4@10`",
        "- raw score: `equal4_raw_mean = mean(fcdb_primary, cpc_primary, gnmc_primary, gaic_primary)`",
        "- normalized score: `equal4_zscore = mean(z_fcdb, z_cpc, z_gnmc, z_gaic)`",
        "- diagnostic: `worst_dataset_z` is reported to expose methods that win on average while collapsing on one dataset",
        "",
        "| rank | method | family | equal4_raw_mean | equal4_zscore | worst_dataset_z | FCDB IoU | CPC weighted | GNMC IoU | GAIC primary | GAIC top1 MOS | GAIC SRCC | GAIC Accw4@10 |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {rank} | `{method}` | {family} | {raw} | {score} | {worst} | {fcdb} | {cpcw} | {gnmc} | {gaicp} | {top1} | {srcc} | {accw4} |".format(
                rank=row["rank_equal4_zscore"],
                method=row["method"],
                family=row["family"],
                raw=_fmt(row["equal4_raw_mean"], 6),
                score=_fmt(row["equal4_zscore"], 6),
                worst=_fmt(row["worst_dataset_z"], 6),
                fcdb=_fmt(row["fcdb_iou_top1"]),
                cpcw=_fmt(row["cpc_weighted_pairwise"]),
                gnmc=_fmt(row["gnmc_iou_top1"]),
                gaicp=_fmt(row["gaic_primary"]),
                top1=_fmt(row["top1_mos"]),
                srcc=_fmt(row["srcc"]),
                accw4=_fmt(row["accw4_of_top10"]),
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `equal4_raw_mean` is easy to read directly because all four primaries are on roughly `[0,1]` scale.",
            "- `equal4_zscore` is still useful because it compensates for the fact that the compared method set has different variance on each dataset axis.",
            "- `production_final_hybrid` is the strongest method on equal-4 average and also has the best worst-dataset z among the compared methods.",
            "- `UCTR stage3` is the strongest single learned teacher among non-hybrid rows.",
            "- `UCTR stage2` ranks high on equal-4 average because FCDB/CPC/GNMC are extremely strong, but its `worst_dataset_z` is very poor because GAIC is weak. This is exactly why worst-dataset gate should still be applied for final selection.",
            "- `public_cropper_ensemble_best` is the winner under the older `public_macro + gaic_component` scalar, but only mid-pack under equal-4 because FCDB/GNMC geometry is much weaker than `UCTR stage3` / `production_final_hybrid`.",
            "",
            "## Sources",
            "",
        ]
    )
    for row in rows:
        lines.append(f"- `{row['method']}`: `{row['source']}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    payload = {
        "method_count": len(rows),
        "metric_definition": {
            "fcdb_primary": "iou_top1",
            "cpc_primary": "weighted_pairwise_acc",
            "gnmc_primary": "iou_top1",
            "gaic_primary": "0.50*(top1_mos/5) + 0.35*srcc + 0.15*accw4_of_top10",
            "raw_score": "mean(fcdb_primary, cpc_primary, gnmc_primary, gaic_primary)",
            "normalization": "zscore across compared method set",
            "normalized_score": "mean(z_fcdb, z_cpc, z_gnmc, z_gaic)",
        },
        "rows": rows,
    }
    (args.output_dir / "equal4_teacher_leaderboard.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "equal4_teacher_leaderboard.md").write_text(
        build_markdown(rows),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
