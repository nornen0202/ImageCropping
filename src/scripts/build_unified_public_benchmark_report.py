#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


PUBLIC_METRICS = (
    "iou_top1",
    "gt_rank_at_1",
    "gt_rank_at_5",
    "pairwise_acc",
    "weighted_pairwise_acc",
    "coverage_at_09",
    "ar_constraint_violation_rate",
    "schema_fail_rate",
)

GAIC_METRICS = (
    "pcc",
    "srcc",
    "acc1_of_top5",
    "acc1_of_top10",
    "acc4_of_top5",
    "acc4_of_top10",
    "accw4_of_top5",
    "accw4_of_top10",
    "top1_mos",
    "top1_mos_regret",
    "top1_rank_percentile",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a unified FCDB/CPC/GNMC/GAIC benchmark report from method registry artifacts.")
    parser.add_argument("--registry_json", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--title", default="Unified FCDB/CPC/GNMC/GAIC Benchmark Report")
    return parser.parse_args()


def load_json(path: str | Path | None) -> dict[str, Any]:
    if path is None or str(path).strip() == "":
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def nested_get(row: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = row
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "-"
    return f"{number:.{digits}f}"


def method_public_summary(method: dict[str, Any]) -> dict[str, Any]:
    return load_json(method.get("public_summary_json"))


def method_gaic_summary(method: dict[str, Any]) -> dict[str, Any]:
    payload = load_json(method.get("gaic_metrics_json"))
    if not payload:
        return {}
    method_name = method.get("gaic_method_name") or method.get("method_id")
    runs = payload.get("runs")
    if isinstance(runs, list):
        run_name = method.get("gaic_run_name") or method.get("run_name") or method_name
        chosen = None
        for run in runs:
            if isinstance(run, dict) and str(run.get("run_name", "")) == str(run_name):
                chosen = run
                break
        if chosen is None and len(runs) == 1 and isinstance(runs[0], dict):
            chosen = runs[0]
        if isinstance(chosen, dict):
            return {
                "metrics": {
                    "pcc": chosen.get("official_pcc"),
                    "srcc": chosen.get("official_srcc"),
                    "acc1_of_top5": chosen.get("official_acc1_top5"),
                    "acc1_of_top10": chosen.get("official_acc1_top10"),
                    "acc4_of_top5": chosen.get("official_acc4_top5"),
                    "acc4_of_top10": chosen.get("official_acc4_top10"),
                    "accw4_of_top5": chosen.get("official_accw4_top5"),
                    "accw4_of_top10": chosen.get("official_accw4_top10"),
                    "top1_mos": chosen.get("official_top1_mos"),
                    "top1_mos_regret": chosen.get("official_regret"),
                    "top1_rank_percentile": chosen.get("official_top1_rank_percentile"),
                },
                "run": chosen,
            }
    methods = payload.get("methods")
    if isinstance(methods, dict):
        if method_name in methods:
            return methods[method_name]
        aliases = method.get("gaic_method_aliases") or []
        for alias in aliases:
            if alias in methods:
                return methods[alias]
        # If the JSON contains exactly one deployable non-oracle method, use it as a fallback.
        deployable = [name for name in methods if "oracle" not in name.lower()]
        if len(deployable) == 1:
            return methods[deployable[0]]
    # SSTK teacher benchmark_summary.json shape.
    if "Gc" in payload or "Ge" in payload:
        field = nested_get(payload, "config.score_profile.semantics.official_score_field", "crop_utility_raw")
        trend = nested_get(payload, f"Gc.trend_by_field.{field}", {})
        if not isinstance(trend, dict):
            trend = nested_get(payload, "Gc.trend_by_field.crop_utility_raw", {})
        prod = nested_get(payload, "Ge.prod_selection_crop_utility", {})
        return {
            "metrics": {
                "pcc": None,
                "srcc": trend.get("mean_spearman"),
                "acc1_of_top5": None,
                "acc1_of_top10": None,
                "acc4_of_top5": None,
                "acc4_of_top10": None,
                "accw4_of_top5": None,
                "accw4_of_top10": None,
                "top1_mos": None,
                "top1_mos_regret": None,
                "top1_rank_percentile": prod.get("matched_gt_mos_percentile_mean") if isinstance(prod, dict) else None,
                "teacher_mean_weighted_pair_acc": trend.get("mean_weighted_pair_acc"),
                "teacher_pooled_weighted_pair_acc": trend.get("pooled_weighted_pair_acc"),
                "teacher_hit_at_1": trend.get("mean_hit_at_1"),
                "teacher_ndcg_at_topq": trend.get("mean_ndcg_at_topq"),
            }
        }
    return payload


def metrics_dict(row: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(row, dict):
        return {}
    if isinstance(row.get("metrics"), dict):
        return row["metrics"]
    return row


def dataset_cell(public_summary: dict[str, Any], dataset: str, metric: str) -> Any:
    return nested_get(public_summary, f"datasets.{dataset}.{metric}")


def public_overall_cell(public_summary: dict[str, Any], metric: str) -> Any:
    return nested_get(public_summary, f"overall.{metric}")


def delta(a: Any, b: Any) -> Any:
    try:
        aval = float(a)
        bval = float(b)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(aval) or not math.isfinite(bval):
        return None
    return aval - bval


def build_report(registry: dict[str, Any]) -> str:
    methods = registry.get("methods") or []
    lines: list[str] = [
        f"# {registry.get('title') or 'Unified FCDB/CPC/GNMC/GAIC Benchmark Report'}",
        "",
        f"- generated_at: `{time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime())}`",
        f"- artifact_root: `{registry.get('artifact_root', '')}`",
        f"- registry_json: `{registry.get('registry_json', '')}`",
        "",
        "## Scope",
        "",
        "이 리포트는 FCDB/CPC/GNMC public benchmark와 GAIC v2 official benchmark를 하나의 method registry로 묶어 비교한다. Public benchmark는 candidate-window scoring protocol을 primary로 사용하며, GAIC는 기존 official annotation candidate ranking metric을 함께 표시한다.",
        "",
        "주의할 점은 FCDB/CPC/GNMC와 GAIC가 같은 의미의 지표를 제공하지 않는다는 것이다. FCDB/GNMC는 주로 crop geometry와 GT/editor crop 회수율을 보고, CPC는 pairwise preference ranking을 본다. GAIC는 MOS candidate ranking과 top-return quality를 본다.",
        "",
        "## Method Registry",
        "",
        "| method | family | role | public status | GAIC source |",
        "| --- | --- | --- | --- | --- |",
    ]
    public_by_method: dict[str, dict[str, Any]] = {}
    gaic_by_method: dict[str, dict[str, Any]] = {}
    for method in methods:
        mid = str(method.get("method_id", ""))
        public = method_public_summary(method)
        gaic = method_gaic_summary(method)
        public_by_method[mid] = public
        gaic_by_method[mid] = gaic
        public_status = "ok" if public else "missing"
        gaic_status = str(method.get("gaic_metrics_json", "")) if gaic else ""
        lines.append(
            f"| `{mid}` | {method.get('family', '')} | {method.get('role', '')} | {public_status} | `{gaic_status}` |"
        )

    lines.extend(
        [
            "",
            "## Public Overall",
            "",
            "| method | tasks | images | IoU@top1 | rank@1 | rank@5 | CPC pairwise | CPC weighted | cov@0.9 | AR viol | schema fail |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for method in methods:
        mid = str(method.get("method_id", ""))
        public = public_by_method.get(mid, {})
        overall = public.get("overall", {}) if isinstance(public, dict) else {}
        lines.append(
            "| {mid} | {tasks} | {images} | {iou} | {r1} | {r5} | {pair} | {wpair} | {cov} | {ar} | {schema} |".format(
                mid=f"`{mid}`",
                tasks=int(overall.get("n_tasks", 0) or 0),
                images=int(overall.get("n_images", 0) or 0),
                iou=fmt(overall.get("iou_top1")),
                r1=fmt(overall.get("gt_rank_at_1")),
                r5=fmt(overall.get("gt_rank_at_5")),
                pair=fmt(overall.get("pairwise_acc")),
                wpair=fmt(overall.get("weighted_pairwise_acc")),
                cov=fmt(overall.get("coverage_at_09")),
                ar=fmt(overall.get("ar_constraint_violation_rate")),
                schema=fmt(overall.get("schema_fail_rate")),
            )
        )

    for dataset in ("fcdb", "cpc", "gnmc"):
        lines.extend(
            [
                "",
                f"## Public Dataset: {dataset.upper()}",
                "",
                "| method | tasks | IoU@top1 | rank@1 | rank@5 | pairwise | weighted pairwise | cov@0.9 | AR viol |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for method in methods:
            mid = str(method.get("method_id", ""))
            row = nested_get(public_by_method.get(mid, {}), f"datasets.{dataset}", {})
            lines.append(
                "| {mid} | {tasks} | {iou} | {r1} | {r5} | {pair} | {wpair} | {cov} | {ar} |".format(
                    mid=f"`{mid}`",
                    tasks=int(row.get("n_tasks", 0) or 0) if isinstance(row, dict) else 0,
                    iou=fmt(row.get("iou_top1") if isinstance(row, dict) else None),
                    r1=fmt(row.get("gt_rank_at_1") if isinstance(row, dict) else None),
                    r5=fmt(row.get("gt_rank_at_5") if isinstance(row, dict) else None),
                    pair=fmt(row.get("pairwise_acc") if isinstance(row, dict) else None),
                    wpair=fmt(row.get("weighted_pairwise_acc") if isinstance(row, dict) else None),
                    cov=fmt(row.get("coverage_at_09") if isinstance(row, dict) else None),
                    ar=fmt(row.get("ar_constraint_violation_rate") if isinstance(row, dict) else None),
                )
            )

    lines.extend(
        [
            "",
            "## GAIC v2 Reference",
            "",
            "| method | PCC | SRCC | Acc1/5 | Acc1/10 | Acc4/5 | Acc4/10 | Accw4/5 | Accw4/10 | top1 MOS | regret | rank pct |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for method in methods:
        mid = str(method.get("method_id", ""))
        metrics = metrics_dict(gaic_by_method.get(mid, {}))
        lines.append(
            "| {mid} | {pcc} | {srcc} | {a15} | {a110} | {a45} | {a410} | {aw45} | {aw410} | {mos} | {regret} | {pct} |".format(
                mid=f"`{mid}`",
                pcc=fmt(metrics.get("pcc")),
                srcc=fmt(metrics.get("srcc")),
                a15=fmt(metrics.get("acc1_of_top5")),
                a110=fmt(metrics.get("acc1_of_top10")),
                a45=fmt(metrics.get("acc4_of_top5")),
                a410=fmt(metrics.get("acc4_of_top10")),
                aw45=fmt(metrics.get("accw4_of_top5")),
                aw410=fmt(metrics.get("accw4_of_top10")),
                mos=fmt(metrics.get("top1_mos")),
                regret=fmt(metrics.get("top1_mos_regret")),
                pct=fmt(metrics.get("top1_rank_percentile")),
            )
        )

    lines.extend(build_interpretation(methods, public_by_method, gaic_by_method))
    return "\n".join(lines) + "\n"


def build_interpretation(
    methods: list[dict[str, Any]],
    public_by_method: dict[str, dict[str, Any]],
    gaic_by_method: dict[str, dict[str, Any]],
) -> list[str]:
    by_id = {str(method.get("method_id", "")): method for method in methods}
    public_score_ids = [mid for mid, method in by_id.items() if "public_score" in str(method.get("role", "")).lower() or "pubscore" in mid]
    teacher_ids = [mid for mid, method in by_id.items() if "t6" in mid.lower() or "teacher" in str(method.get("role", "")).lower()]

    lines = [
        "",
        "## Interpretation",
        "",
        "GAIC public cropper score로 만든 학습 라벨의 일반화 여부는 GAIC test500의 top-return만으로 판단하면 안 된다. 핵심은 같은 student가 FCDB/CPC/GNMC에서 editor/expert crop과 pairwise preference를 얼마나 회수하는지다.",
        "",
    ]
    if public_score_ids and teacher_ids:
        lines.append("### GAIC Public-Score Student vs T1-T6/Teacher-Style Student")
        lines.append("")
        for pub_id in public_score_ids:
            pub_public = public_by_method.get(pub_id, {})
            pub_gaic = metrics_dict(gaic_by_method.get(pub_id, {}))
            lines.append(
                f"- `{pub_id}`: GAIC top1 MOS `{fmt(pub_gaic.get('top1_mos'))}`, regret `{fmt(pub_gaic.get('top1_mos_regret'))}`, public overall IoU `{fmt(public_overall_cell(pub_public, 'iou_top1'))}`, CPC weighted pairwise `{fmt(dataset_cell(pub_public, 'cpc', 'weighted_pairwise_acc'))}`."
            )
            for teacher_id in teacher_ids:
                teacher_public = public_by_method.get(teacher_id, {})
                teacher_gaic = metrics_dict(gaic_by_method.get(teacher_id, {}))
                d_public_iou = delta(public_overall_cell(pub_public, "iou_top1"), public_overall_cell(teacher_public, "iou_top1"))
                d_cpc_pair = delta(dataset_cell(pub_public, "cpc", "weighted_pairwise_acc"), dataset_cell(teacher_public, "cpc", "weighted_pairwise_acc"))
                d_gnmc_iou = delta(dataset_cell(pub_public, "gnmc", "iou_top1"), dataset_cell(teacher_public, "gnmc", "iou_top1"))
                d_gaic_mos = delta(pub_gaic.get("top1_mos"), teacher_gaic.get("top1_mos"))
                lines.append(
                    f"- 비교 `{teacher_id}`: GAIC top1 MOS `{fmt(teacher_gaic.get('top1_mos'))}`, regret `{fmt(teacher_gaic.get('top1_mos_regret'))}`, public overall IoU `{fmt(public_overall_cell(teacher_public, 'iou_top1'))}`, CPC weighted pairwise `{fmt(dataset_cell(teacher_public, 'cpc', 'weighted_pairwise_acc'))}`."
                )
                lines.append(
                    f"- delta `{pub_id}` - `{teacher_id}`: GAIC top1 MOS `{fmt(d_gaic_mos)}`, public overall IoU `{fmt(d_public_iou)}`, CPC weighted pairwise `{fmt(d_cpc_pair)}`, GNMC IoU `{fmt(d_gnmc_iou)}`."
                )
        lines.append("")
        lines.append(
            "해석 기준은 명확하다. public-score student가 GAIC에서는 높지만 FCDB/CPC/GNMC에서 T1-T6 계열보다 낮으면 GAIC score label은 GAIC candidate distribution에 맞춘 강한 teacher일 뿐 일반 crop preference로는 과적합 위험이 있다. 반대로 CPC weighted pairwise와 FCDB/GNMC IoU까지 같이 오르면 public cropper score label이 teacher utility보다 더 일반적인 aesthetic/crop prior를 제공한다고 볼 수 있다."
        )
    else:
        lines.append("public-score student와 T1-T6/Teacher-style student가 registry에 함께 없어서 일반화 비교를 자동 서술하지 못했다.")
    lines.extend(
        [
            "",
            "### Protocol Caveats",
            "",
            "- FCDB와 GNMC의 candidate windows는 synthetic/evaluation window가 포함된다. 따라서 수치는 deploy-time proposal recall이 아니라 scorer가 주어진 후보를 얼마나 잘 rank하는지에 가깝다.",
            "- CPC는 native candidate set과 pairwise preference가 있으므로 cross-dataset ranking 일반화를 보는 데 가장 강한 축이다.",
            "- CACNet처럼 단일 crop만 출력하는 method는 candidate-ranking primary metric과 동등 비교하면 안 된다. 이 경우 projection-only 또는 product-crop protocol로 분리해야 한다.",
            "- MobileCropNet v4의 target AR vocabulary에는 `2:1`이 없다. exporter가 `nearest` policy를 쓰면 `2:1`은 `16:9` conditioning으로 평가되며, GNMC 2:1 결과 해석 시 이 제약을 같이 봐야 한다.",
        ]
    )
    return lines


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    registry = load_json(args.registry_json.resolve())
    registry["title"] = args.title
    registry["registry_json"] = str(args.registry_json.resolve())
    report = build_report(registry)
    report_path = args.output_dir / "UNIFIED_PUBLIC_BENCHMARK_REPORT_KO.md"
    report_path.write_text(report, encoding="utf-8")
    (args.output_dir / "unified_public_benchmark_registry_expanded.json").write_text(
        json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"report_md": str(report_path.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
