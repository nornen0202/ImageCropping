from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


DEFAULT_GATE_CONFIG: dict[str, Any] = {
    "public_metrics": {
        "weighted_pairwise_acc": {"direction": "higher", "max_drop": 0.02},
        "gt_rank_at_5": {"direction": "higher", "max_drop": 0.05},
        "coverage_at_07_preinject": {"direction": "higher", "max_drop": 0.02},
        "iou_top1_preinject": {"direction": "higher", "max_drop": 0.02},
        "schema_fail_rate": {"direction": "lower", "max_increase": 0.0},
        "nan_score_rate": {"direction": "lower", "max_increase": 0.0},
        "ar_constraint_violation_rate": {"direction": "lower", "max_increase": 0.02},
    },
    "gaic_protocol_metrics": {
        "mean_spearman": {"direction": "higher", "max_drop": 0.01},
        "mean_weighted_pair_acc": {"direction": "higher", "max_drop": 0.01},
        "mean_topq_jaccard": {"direction": "higher", "max_drop": 0.01},
    },
    "gaic_prod_selection_metrics": {
        "matched_gt_iou_mean": {"direction": "higher", "max_drop": 0.01},
        "matched_gt_mos_percentile_mean": {"direction": "higher", "max_drop": 0.01},
        "gt_best_hit_at_07": {"direction": "higher", "max_drop": 0.01},
    },
}


def load_gate_config(path: Optional[Path]) -> dict[str, Any]:
    if path is None:
        return json.loads(json.dumps(DEFAULT_GATE_CONFIG))
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = json.loads(json.dumps(DEFAULT_GATE_CONFIG))
    for section, value in payload.items():
        if isinstance(value, dict) and isinstance(config.get(section), dict):
            config[section].update(value)
        else:
            config[section] = value
    return config


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:
        return None
    return out


def _check_metric(
    *,
    suite: str,
    item: str,
    metric: str,
    current: Any,
    baseline: Any,
    rule: dict[str, Any],
) -> dict[str, Any]:
    cur = _as_float(current)
    base = _as_float(baseline)
    row = {
        "suite": suite,
        "item": item,
        "metric": metric,
        "current": cur,
        "baseline": base,
        "status": "skip",
        "delta": None,
        "threshold": None,
        "direction": rule.get("direction", "higher"),
    }
    if cur is None or base is None:
        row["reason"] = "missing_metric"
        return row
    delta = cur - base
    row["delta"] = float(delta)
    direction = str(rule.get("direction", "higher"))
    if direction == "higher":
        max_drop = float(rule.get("max_drop", 0.0))
        row["threshold"] = -max_drop
        row["status"] = "fail" if delta < -max_drop else "pass"
    elif direction == "lower":
        max_increase = float(rule.get("max_increase", 0.0))
        row["threshold"] = max_increase
        row["status"] = "fail" if delta > max_increase else "pass"
    else:
        row["status"] = "skip"
        row["reason"] = f"unsupported_direction:{direction}"
    return row


def compare_summaries(
    *,
    current: dict[str, Any],
    baseline: dict[str, Any],
    gate_config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    config = gate_config or DEFAULT_GATE_CONFIG
    checks: list[dict[str, Any]] = []

    public_rules = config.get("public_metrics", {})
    for dataset, cur_row in sorted((current.get("datasets") or {}).items()):
        base_row = (baseline.get("datasets") or {}).get(dataset, {})
        for metric, rule in sorted(public_rules.items()):
            checks.append(
                _check_metric(
                    suite="public",
                    item=str(dataset),
                    metric=str(metric),
                    current=cur_row.get(metric) if isinstance(cur_row, dict) else None,
                    baseline=base_row.get(metric) if isinstance(base_row, dict) else None,
                    rule=rule if isinstance(rule, dict) else {},
                )
            )

    protocol_rules = config.get("gaic_protocol_metrics", {})
    for protocol in ("Gc", "Ge"):
        cur_row = (((current.get("gaic") or {}).get("protocols") or {}).get(protocol) or {})
        base_row = (((baseline.get("gaic") or {}).get("protocols") or {}).get(protocol) or {})
        for metric, rule in sorted(protocol_rules.items()):
            checks.append(
                _check_metric(
                    suite="gaic_protocol",
                    item=protocol,
                    metric=str(metric),
                    current=cur_row.get(metric) if isinstance(cur_row, dict) else None,
                    baseline=base_row.get(metric) if isinstance(base_row, dict) else None,
                    rule=rule if isinstance(rule, dict) else {},
                )
            )

    prod_rules = config.get("gaic_prod_selection_metrics", {})
    cur_prod = (current.get("gaic") or {}).get("prod_selection") or {}
    base_prod = (baseline.get("gaic") or {}).get("prod_selection") or {}
    for metric, rule in sorted(prod_rules.items()):
        checks.append(
            _check_metric(
                suite="gaic_prod_selection",
                item="Ge",
                metric=str(metric),
                current=cur_prod.get(metric) if isinstance(cur_prod, dict) else None,
                baseline=base_prod.get(metric) if isinstance(base_prod, dict) else None,
                rule=rule if isinstance(rule, dict) else {},
            )
        )

    failures = [row for row in checks if row.get("status") == "fail"]
    passes = [row for row in checks if row.get("status") == "pass"]
    skips = [row for row in checks if row.get("status") == "skip"]
    return {
        "status": "fail" if failures else "pass",
        "failure_count": len(failures),
        "pass_count": len(passes),
        "skip_count": len(skips),
        "checks": checks,
        "failures": failures,
    }


def build_regression_report(result: dict[str, Any]) -> str:
    lines = ["# Public Benchmark Regression Gate", ""]
    lines.append(f"- status: `{result.get('status', 'unknown')}`")
    lines.append(f"- failures: `{result.get('failure_count', 0)}`")
    lines.append(f"- passes: `{result.get('pass_count', 0)}`")
    lines.append(f"- skips: `{result.get('skip_count', 0)}`")
    lines.append("")
    lines.append("| suite | item | metric | baseline | current | delta | status |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in result.get("checks", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("suite", "")),
                    str(row.get("item", "")),
                    str(row.get("metric", "")),
                    _fmt(row.get("baseline")),
                    _fmt(row.get("current")),
                    _fmt(row.get("delta")),
                    str(row.get("status", "")),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)
