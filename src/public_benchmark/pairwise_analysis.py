from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def extract_pairwise_failures(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in rows:
        for failure in row.get("pairwise_failures", []) or []:
            if isinstance(failure, dict):
                failures.append(failure)
    failures.sort(
        key=lambda item: (
            float(item.get("weight", 0.0)) * (1.0 - float(item.get("hit", 0.0))),
            float(item.get("weight", 0.0)),
        ),
        reverse=True,
    )
    return failures


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _flatten_numeric(payload: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if not isinstance(payload, dict):
        return out
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(_flatten_numeric(value, name))
        else:
            try:
                fval = float(value)
            except (TypeError, ValueError):
                continue
            if fval == fval:
                out[name] = fval
    return out


def _candidate_feature_map(candidate: dict[str, Any]) -> dict[str, float]:
    out = {}
    out.update(_flatten_numeric(candidate.get("scores", {}), "scores"))
    out.update(_flatten_numeric(candidate.get("macro_scores", {}), "macro_scores"))
    return out


def summarize_pairwise_failures(failures: list[dict[str, Any]]) -> dict[str, Any]:
    by_dataset = Counter(str(item.get("dataset", "")) for item in failures)
    by_source_pair = Counter()
    score_diffs: list[float] = []
    component_diffs: dict[str, list[float]] = defaultdict(list)
    for failure in failures:
        cand_a = failure.get("candidate_a", {}) if isinstance(failure.get("candidate_a"), dict) else {}
        cand_b = failure.get("candidate_b", {}) if isinstance(failure.get("candidate_b"), dict) else {}
        source_pair = f"{cand_a.get('source', '')}->{cand_b.get('source', '')}"
        if failure.get("preferred") == "b":
            source_pair = f"{cand_b.get('source', '')}->{cand_a.get('source', '')}"
        by_source_pair[source_pair] += 1
        try:
            score_diffs.append(float(failure.get("preferred_score_minus_other", 0.0)))
        except (TypeError, ValueError):
            pass
        preferred = cand_a if failure.get("preferred") == "a" else cand_b
        other = cand_b if failure.get("preferred") == "a" else cand_a
        pref_features = _candidate_feature_map(preferred)
        other_features = _candidate_feature_map(other)
        for key in sorted(set(pref_features).intersection(other_features)):
            component_diffs[key].append(pref_features[key] - other_features[key])

    component_rows = []
    for key, values in component_diffs.items():
        if not values:
            continue
        component_rows.append(
            {
                "feature": key,
                "count": len(values),
                "mean_preferred_minus_other": sum(values) / float(len(values)),
                "min_preferred_minus_other": min(values),
                "max_preferred_minus_other": max(values),
            }
        )
    component_rows.sort(key=lambda row: (float(row["mean_preferred_minus_other"]), -int(row["count"])))
    return {
        "failure_count": len(failures),
        "by_dataset": dict(by_dataset),
        "by_preferred_to_other_source": dict(by_source_pair.most_common(20)),
        "preferred_score_minus_other_mean": sum(score_diffs) / float(len(score_diffs)) if score_diffs else None,
        "component_deltas_worst": component_rows[:25],
    }


def build_pairwise_failure_report(summary: dict[str, Any], failures: list[dict[str, Any]]) -> str:
    lines = ["# Pairwise Failure Analysis", ""]
    lines.append(f"- failure_count_recorded: `{summary.get('failure_count', 0)}`")
    lines.append(f"- preferred_score_minus_other_mean: `{_fmt(summary.get('preferred_score_minus_other_mean'))}`")
    lines.append("")
    lines.append("## By Dataset")
    lines.append("")
    lines.append("| dataset | failures |")
    lines.append("|---|---|")
    for dataset, count in sorted((summary.get("by_dataset") or {}).items()):
        lines.append(f"| {dataset} | {count} |")
    lines.append("")
    lines.append("## Worst Mean Feature Deltas")
    lines.append("")
    lines.append("| feature | count | mean preferred-other |")
    lines.append("|---|---|---|")
    for row in summary.get("component_deltas_worst", [])[:15]:
        lines.append(
            f"| {row.get('feature')} | {row.get('count')} | {_fmt(row.get('mean_preferred_minus_other'))} |"
        )
    lines.append("")
    lines.append("## Top Failures")
    lines.append("")
    lines.append("| dataset | image | pair | preferred | predicted | weight | score_pref-other |")
    lines.append("|---|---|---|---|---|---|---|")
    for failure in failures[:25]:
        pair = f"{failure.get('candidate_id_a')} vs {failure.get('candidate_id_b')}"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(failure.get("dataset", "")),
                    str(failure.get("image_id", "")),
                    pair,
                    str(failure.get("preferred", "")),
                    str(failure.get("predicted", "")),
                    _fmt(failure.get("weight")),
                    _fmt(failure.get("preferred_score_minus_other")),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)
