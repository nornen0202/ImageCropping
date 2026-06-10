#!/usr/bin/env python3
"""Safe research-agent bridge for the ImageCropping project.

This module gives external research agents a narrow, reproducible interface to
the existing SSTK/GAIC/VLM tooling.  It intentionally avoids free-form project
mutation: agents should call this bridge, inspect structured JSON outputs, and
then propose small patches through the normal review path.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import platform
import re
import shutil
import shlex
import subprocess
import sys
import time
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
PROJECT_PYTHON = Path(
    "/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/python"
)
EVOSCIENTIST_PY311 = Path(
    "/media/jyju25/Disk_JY/Projects_26/Venvs/EvoScientist_Py311/bin/python"
)
AI_SCIENTIST_V2_PY311 = Path(
    "/media/jyju25/Disk_JY/Projects_26/Venvs/AIScientistV2_Py311/bin/python"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "research_agent_runs"
DEFAULT_REMOTE_HOST = "jaden.ju@10.8.103.149"
DEFAULT_REMOTE_PROJECT_ROOT = "/group-volume/users/jaden.ju/Sources/ImageCropping"
DEFAULT_REMOTE_PYTHON = "/usr/local/bin/python3"
DEFAULT_REMOTE_MODEL_ID = (
    "/group-volume/users/jaden.ju/Sources/ImageCropping/"
    "weights/hf/Qwen3-VL-4B-Instruct"
)
DEFAULT_JUMP_HOST = "jump-n6"
DEFAULT_IDENTITY_FILE = "/home/jyju25/SPACE/key.pem"

SMOKE_TESTS = {
    "agent_smoke": [
        "tests/test_vlm_teacher_labeler.py",
        "tests/test_gaic_benchmark_eval.py",
        "tests/test_crop_score_priority_experiments.py",
    ],
    "vlm_schema": ["tests/test_vlm_teacher_labeler.py"],
    "benchmark_math": ["tests/test_gaic_benchmark_eval.py"],
    "score_priority": ["tests/test_crop_score_priority_experiments.py"],
    "training_labels": ["tests/test_finalscore_training_data.py"],
}


def utc_timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )


def run_command(
    argv: list[str],
    *,
    timeout: int = 120,
    cwd: Path = PROJECT_ROOT,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    started = time.time()
    try:
        proc = subprocess.run(
            argv,
            cwd=str(cwd),
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return {
            "argv": argv,
            "cwd": str(cwd),
            "returncode": proc.returncode,
            "elapsed_sec": round(time.time() - started, 3),
            "stdout_tail": proc.stdout[-8000:],
            "stderr_tail": proc.stderr[-8000:],
            "ok": proc.returncode == 0,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "argv": argv,
            "cwd": str(cwd),
            "returncode": None,
            "elapsed_sec": round(time.time() - started, 3),
            "stdout_tail": (exc.stdout or "")[-8000:]
            if isinstance(exc.stdout, str)
            else "",
            "stderr_tail": (exc.stderr or "")[-8000:]
            if isinstance(exc.stderr, str)
            else "",
            "ok": False,
            "error": f"timeout after {timeout}s",
        }
    except FileNotFoundError as exc:
        return {
            "argv": argv,
            "cwd": str(cwd),
            "returncode": None,
            "elapsed_sec": round(time.time() - started, 3),
            "stdout_tail": "",
            "stderr_tail": "",
            "ok": False,
            "error": f"not found: {exc.filename}",
        }


def find_executable(name: str, extra_paths: list[Path] | None = None) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    for path in extra_paths or []:
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    return None


def ccproxy_health(port: int) -> dict[str, Any]:
    url = f"http://127.0.0.1:{port}/health/live"
    started = time.time()
    try:
        with urlopen(url, timeout=2.0) as resp:
            body = resp.read(2000).decode("utf-8", errors="replace")
            return {
                "url": url,
                "ok": 200 <= int(resp.status) < 300,
                "status": int(resp.status),
                "elapsed_sec": round(time.time() - started, 3),
                "body_tail": body[-1000:],
            }
    except URLError as exc:
        return {
            "url": url,
            "ok": False,
            "status": None,
            "elapsed_sec": round(time.time() - started, 3),
            "error": str(exc),
        }


def build_diagnosis(args: argparse.Namespace) -> dict[str, Any]:
    codex_bin = find_executable("codex")
    ccproxy_bin = find_executable(
        "ccproxy",
        [
            Path(os.environ.get("CCPROXY_BIN", "")),
            EVOSCIENTIST_PY311.parent / "ccproxy",
            AI_SCIENTIST_V2_PY311.parent / "ccproxy",
        ],
    )

    diagnosis: dict[str, Any] = {
        "schema_version": "sstk_research_agent_doctor_v1",
        "created_at_utc": utc_timestamp(),
        "project_root": PROJECT_ROOT,
        "project_python": PROJECT_PYTHON,
        "evoscientist_python": EVOSCIENTIST_PY311,
        "ai_scientist_v2_python": AI_SCIENTIST_V2_PY311,
        "host": {
            "platform": platform.platform(),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
        },
        "checks": {},
    }

    checks = diagnosis["checks"]
    checks["project_python_exists"] = {
        "ok": PROJECT_PYTHON.exists() and os.access(PROJECT_PYTHON, os.X_OK),
        "path": str(PROJECT_PYTHON),
    }
    if checks["project_python_exists"]["ok"]:
        checks["project_python_version"] = run_command(
            [str(PROJECT_PYTHON), "--version"],
            timeout=20,
        )
    checks["evoscientist_python_exists"] = {
        "ok": EVOSCIENTIST_PY311.exists() and os.access(EVOSCIENTIST_PY311, os.X_OK),
        "path": str(EVOSCIENTIST_PY311),
    }
    if checks["evoscientist_python_exists"]["ok"]:
        checks["evoscientist_python_version"] = run_command(
            [str(EVOSCIENTIST_PY311), "--version"],
            timeout=20,
        )
    checks["ai_scientist_v2_python_exists"] = {
        "ok": AI_SCIENTIST_V2_PY311.exists()
        and os.access(AI_SCIENTIST_V2_PY311, os.X_OK),
        "path": str(AI_SCIENTIST_V2_PY311),
    }
    if checks["ai_scientist_v2_python_exists"]["ok"]:
        checks["ai_scientist_v2_python_version"] = run_command(
            [str(AI_SCIENTIST_V2_PY311), "--version"],
            timeout=20,
        )

    checks["codex_cli"] = {
        "ok": bool(codex_bin),
        "path": codex_bin or "",
    }
    if codex_bin:
        checks["codex_version"] = run_command([codex_bin, "--version"], timeout=20)
        checks["codex_login_status"] = run_command(
            [codex_bin, "login", "status"],
            timeout=20,
        )

    checks["ccproxy_cli"] = {
        "ok": bool(ccproxy_bin),
        "path": ccproxy_bin or "",
    }
    if ccproxy_bin:
        auth_status = run_command(
            [ccproxy_bin, "auth", "status", "codex"],
            timeout=20,
        )
        auth_text = (
            str(auth_status.get("stdout_tail", ""))
            + "\n"
            + str(auth_status.get("stderr_tail", ""))
        ).lower()
        if "not authenticated" in auth_text or "provider not found" in auth_text:
            auth_status["ok"] = False
        checks["ccproxy_codex_auth_status"] = auth_status
        checks["ccproxy_health"] = ccproxy_health(int(args.ccproxy_port))
    else:
        checks["ccproxy_install_hint"] = {
            "ok": False,
            "message": (
                "Install ccproxy in the external agent Python 3.11 environment "
                "with: <agent-python> -m pip install 'evoscientist[oauth]' "
                "or ccproxy-api, then run: ccproxy auth login codex"
            ),
        }

    repo_paths = {
        "evoscientist_repo": Path(args.evoscientist_repo).expanduser(),
        "ai_scientist_v2_repo": Path(args.ai_scientist_v2_repo).expanduser(),
    }
    for key, path in repo_paths.items():
        checks[key] = {"ok": path.exists(), "path": str(path)}
        if path.exists() and (path / ".git").exists():
            checks[f"{key}_commit"] = run_command(
                ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
                timeout=20,
                cwd=PROJECT_ROOT,
            )

    required_project_files = [
        "src/vlm_teacher_labeler.py",
        "src/scripts/run_gaic_benchmark_eval.py",
        "src/scripts/run_crop_score_priority_experiments.py",
        "src/scripts/build_finalscore_training_data.py",
        "src/scripts/run_gaic_to_teacher_e2e.sh",
        "src/scripts/run_phaseA_to_teacher_e2e.sh",
    ]
    checks["required_project_files"] = {
        "ok": all((PROJECT_ROOT / p).exists() for p in required_project_files),
        "files": {
            p: (PROJECT_ROOT / p).exists()
            for p in required_project_files
        },
    }

    if args.run_tests:
        checks["agent_smoke_tests"] = run_pytest_suite("agent_smoke", args.timeout)

    diagnosis["codex_auth_ready"] = bool(
        checks.get("codex_cli", {}).get("ok")
        and checks.get("codex_login_status", {}).get("ok")
    )
    diagnosis["ccproxy_cli_ready"] = bool(checks.get("ccproxy_cli", {}).get("ok"))
    diagnosis["ccproxy_codex_login_ready"] = bool(
        checks.get("ccproxy_codex_auth_status", {}).get("ok")
    )
    diagnosis["ccproxy_server_ready"] = bool(
        checks.get("ccproxy_health", {}).get("ok")
    )
    diagnosis["ccproxy_oauth_ready"] = bool(
        checks.get("ccproxy_cli", {}).get("ok")
        and checks.get("ccproxy_codex_auth_status", {}).get("ok")
        and checks.get("ccproxy_health", {}).get("ok")
    )
    diagnosis["ok"] = all(
        bool(v.get("ok", False))
        for k, v in checks.items()
        if k
        in {
            "project_python_exists",
            "codex_cli",
            "required_project_files",
        }
    )
    return diagnosis


def run_pytest_suite(suite: str, timeout: int) -> dict[str, Any]:
    if suite not in SMOKE_TESTS:
        raise ValueError(f"unknown suite: {suite}. choices={sorted(SMOKE_TESTS)}")
    argv = [str(PROJECT_PYTHON), "-m", "pytest", *SMOKE_TESTS[suite], "-q"]
    return run_command(argv, timeout=timeout)


def build_campaign_templates() -> dict[str, Any]:
    return {
        "schema_version": "sstk_research_campaign_templates_v1",
        "ok": True,
        "baseline": {
            "adopted_scorer_lane": "single_stage2",
            "rerun_policy": (
                "scorer-only changes rerun teacher->labels->benchmark; "
                "routing/guidance changes rerun routed/features->candidates->teacher->labels->benchmark"
            ),
        },
        "campaigns": [
            {
                "campaign_id": "score_decomposition_ablation",
                "claim_target": (
                    "A_gen + A_crop + A_comp decomposition improves crop utility "
                    "without increasing monotonic violations."
                ),
                "change_family": "scorer",
                "rerun_boundary": "teacher -> training_labels -> GAIC benchmark -> report",
                "entrypoints": [
                    "src/scripts/run_crop_score_priority_experiments.py",
                    "src/scripts/build_finalscore_training_data.py",
                    "src/scripts/run_gaic_benchmark_eval.py",
                ],
                "required_metrics": [
                    "Gc Spearman",
                    "Ge Spearman",
                    "IoU to GT MOS best",
                    "GT percentile",
                    "monotonic violations",
                    "why-tag consistency",
                ],
            },
            {
                "campaign_id": "c7_support_preservation_ablation",
                "claim_target": (
                    "C7 saliency-guided latent support preservation improves subject "
                    "coverage and support mass on person/object slices."
                ),
                "change_family": "guidance",
                "rerun_boundary": "C7 augment -> routing -> candidates -> teacher -> benchmark",
                "entrypoints": [
                    "src/scripts/augment_saliency_subject_features.py",
                    "src/scripts/run_gaic_subject_region_ab.py",
                    "src/scripts/run_gaic_benchmark_eval.py",
                ],
                "required_metrics": [
                    "support recall gap",
                    "subject mass in box",
                    "candidate recall upper bound",
                    "slice-level Gc/Ge",
                ],
            },
            {
                "campaign_id": "vlm_teacher_schema_and_fidelity",
                "claim_target": (
                    "VLM Teacher explanations are schema-valid, candidate-grounded, "
                    "and faithful to numeric teacher top-k decisions."
                ),
                "change_family": "vlm_teacher",
                "rerun_boundary": "teacher -> VLM Teacher -> schema audit -> explanation report",
                "entrypoints": [
                    "src/vlm_teacher_labeler.py",
                    "src/scripts/run_vlm_teacher_labeler.sh",
                    "src/scripts/visualize_vlm_teacher_labels.py",
                ],
                "required_metrics": [
                    "schema_ok rate",
                    "candidate-id consistency",
                    "fallback rate",
                    "placeholder leakage count",
                    "why-tag allowed-vocab rate",
                ],
            },
            {
                "campaign_id": "pose_aware_person_mode",
                "claim_target": (
                    "Eye/torso/support anchors reduce portrait joint cuts and "
                    "groundedness failures versus bbox-center placement."
                ),
                "change_family": "routing_guidance_candidate",
                "rerun_boundary": "routing/guidance -> candidates -> teacher -> portrait slice eval",
                "entrypoints": [
                    "src/portrait_composition.py",
                    "src/routing/subject_mode_router.py",
                    "src/scripts/build_subject_mode_stress_set.py",
                ],
                "required_metrics": [
                    "joint cut rate",
                    "face cut rate",
                    "lookroom violation rate",
                    "groundedness fail count",
                    "portrait slice rank metrics",
                ],
            },
        ],
    }


def build_paper_pack(run_dir: Path, output: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    json_reports = (
        sorted(p for p in run_dir.glob("*.json") if not p.name.startswith("paper_pack"))
        if run_dir.exists()
        else []
    )
    summaries: list[dict[str, Any]] = []
    for report_path in json_reports:
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as exc:
            summaries.append(
                {
                    "path": str(report_path),
                    "ok": False,
                    "error": f"failed to parse json: {exc}",
                }
            )
            continue
        schema_version = str(payload.get("schema_version", ""))
        payload_ok = bool(payload.get("ok", False))
        if schema_version == "vlm_teacher_summary_v1":
            counts = payload.get("counts", {}) if isinstance(payload.get("counts"), dict) else {}
            payload_ok = int(counts.get("task_written", 0) or 0) > 0 and int(counts.get("backend_init_fail", 0) or 0) == 0
        summaries.append(
            {
                "path": str(report_path),
                "ok": payload_ok,
                "schema_version": schema_version,
                "created_at_utc": payload.get("created_at_utc", ""),
                "checks": sorted(payload.get("checks", {}).keys())
                if isinstance(payload.get("checks"), dict)
                else [],
                "analysis_metrics": payload.get("analysis", {}).get("metrics", {})
                if isinstance(payload.get("analysis"), dict)
                else {},
                "analysis_caveats": payload.get("analysis", {}).get("caveats", [])
                if isinstance(payload.get("analysis"), dict)
                else [],
            }
        )

    lines = [
        "# Explainable Candidate-First Image Cropping with Policy-Aware Teacher Distillation",
        "",
        "## Draft Status",
        "",
        f"- evidence_pack_created_at_utc: `{utc_timestamp()}`",
        f"- source_run_dir: `{run_dir}`",
        f"- parsed_json_reports: `{len(json_reports)}`",
        "",
        "## Core Claims",
        "",
        "1. Candidate-first crop selection separates candidate recall, ranking, and policy errors.",
        "2. Decomposed teacher scoring (`A_gen + A_crop + A_comp`) gives stronger ablation handles than a single score.",
        "3. VLM Teacher outputs can be constrained to candidate-grounded explanations with deterministic fallback.",
        "4. Benchmark/CI gates prevent slice-level regressions from being hidden by aggregate metrics.",
        "",
        "## Evidence Links",
        "",
    ]
    if not summaries:
        lines.append("- No JSON evidence reports were found yet.")
    for item in summaries:
        lines.append(
            f"- `{item['path']}`: ok=`{item['ok']}`, "
            f"schema=`{item.get('schema_version', '')}`"
        )
        metrics = item.get("analysis_metrics", {})
        if isinstance(metrics, dict) and metrics:
            compact_keys = [
                "num_label_rows",
                "schema_ok_rate",
                "numeric_consistency_ok_rate",
                "candidate_id_consistency_rate",
                "fallback_rate",
                "placeholder_leakage_count",
                "why_tag_allowed_vocab_rate",
                "model_direct_pick_total",
                "model_direct_pick_rows",
            ]
            metric_bits = [
                f"{key}=`{metrics[key]}`"
                for key in compact_keys
                if key in metrics
            ]
            if metric_bits:
                lines.append(f"  - metrics: {', '.join(metric_bits)}")
            caveats = item.get("analysis_caveats", [])
            if caveats:
                lines.append(f"  - caveats: `{', '.join(str(x) for x in caveats)}`")

    lines.extend(
        [
            "",
            "## Required Experiments Before Submission",
            "",
            "- GAIC Gc/Ge synced full validation for adopted lane and each ablation.",
            "- VLM Teacher schema/fidelity audit on candidate-grounded labels.",
            "- Slice-level report for portrait, object, copyspace, text_document, and scene_general modes.",
            "- Paired bootstrap confidence intervals over image tasks.",
            "- Failure gallery with claim-specific counterexamples.",
            "",
            "## Disclosure",
            "",
            "Research agents were used for experiment orchestration, analysis packaging, and draft assistance. Human authors selected the final claims, verified code changes, and approved reported conclusions.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "schema_version": "sstk_paper_pack_v1",
        "created_at_utc": utc_timestamp(),
        "run_dir": run_dir,
        "output": output,
        "num_json_reports": len(json_reports),
        "ok": output.exists(),
    }


def read_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def resolve_vlm_teacher_input_paths(
    data_root: str,
    run_tag: str,
    teacher_scores_jsonl: str = "",
    image_dir: str = "",
) -> dict[str, Path]:
    data_root_path = (PROJECT_ROOT / data_root).resolve() if not Path(data_root).is_absolute() else Path(data_root)
    teacher_path = (
        Path(teacher_scores_jsonl)
        if str(teacher_scores_jsonl).strip()
        else data_root_path / "artifacts" / "teacher" / "scores" / f"teacher_scores_ar_{run_tag}.jsonl"
    )
    image_path = Path(image_dir) if str(image_dir).strip() else data_root_path / "images"
    if not teacher_path.is_absolute():
        teacher_path = PROJECT_ROOT / teacher_path
    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path
    return {
        "data_root": data_root_path,
        "teacher_scores_jsonl": teacher_path,
        "image_dir": image_path,
    }


def _relative_to_project(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)


def _count_placeholders(value: Any) -> int:
    from vlm_teacher_labeler import is_template_placeholder_text

    if isinstance(value, str):
        return 1 if is_template_placeholder_text(value) else 0
    if isinstance(value, dict):
        return sum(_count_placeholders(v) for v in value.values())
    if isinstance(value, list):
        return sum(_count_placeholders(v) for v in value)
    return 0


def _load_candidate_index(
    teacher_scores_jsonl: Path,
    needed_samples: set[tuple[str, str]],
    *,
    top_m: int,
    top_k: int,
    prompt_version: str,
) -> dict[tuple[str, str], dict[str, dict[str, Any]]]:
    from vlm_teacher_labeler import build_task

    by_image: dict[str, set[str]] = {}
    for image_id, target_ar in needed_samples:
        by_image.setdefault(image_id, set()).add(target_ar)

    out: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    if not teacher_scores_jsonl.exists():
        return out

    with teacher_scores_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            image_id = str(row.get("image_id", "")).strip()
            target_ars = by_image.get(image_id)
            if not target_ars:
                continue
            for target_ar in target_ars:
                task = build_task(
                    row,
                    target_ar,
                    top_m=max(1, int(top_m)),
                    top_k=max(1, int(top_k)),
                    prompt_version=str(prompt_version),
                )
                if task is None:
                    continue
                cand_map = {
                    str(c.get("candidate_id", "")): c
                    for c in task.get("candidates", [])
                    if isinstance(c, dict) and str(c.get("candidate_id", "")).strip()
                }
                out[(image_id, target_ar)] = cand_map
            if len(out) >= len(needed_samples):
                break
    return out


def analyze_vlm_teacher_outputs(
    *,
    labels_jsonl: Path,
    meta_jsonl: Path,
    summary_json: Path,
    teacher_scores_jsonl: Path,
    top_m: int,
    top_k: int,
    prompt_version: str,
) -> dict[str, Any]:
    from vlm_teacher_labeler import ALLOWED_WHY_TAGS

    label_rows = read_jsonl_rows(labels_jsonl)
    meta_rows_count = count_jsonl_rows(meta_jsonl)
    summary_payload = read_json_or_empty(summary_json)

    needed_samples = {
        (str(row.get("image_id", "")), str(row.get("target_ar", "")))
        for row in label_rows
        if str(row.get("image_id", "")).strip() and str(row.get("target_ar", "")).strip()
    }
    candidate_index = _load_candidate_index(
        teacher_scores_jsonl,
        needed_samples,
        top_m=top_m,
        top_k=top_k,
        prompt_version=prompt_version,
    )

    counts: Counter[str] = Counter()
    target_ar_counter: Counter[str] = Counter()
    backend_counter: Counter[str] = Counter()
    selected_len_counter: Counter[int] = Counter()
    model_direct_pick_total = 0
    model_direct_pick_rows = 0

    for row in label_rows:
        counts["rows"] += 1
        target_ar = str(row.get("target_ar", ""))
        image_id = str(row.get("image_id", ""))
        target_ar_counter[target_ar] += 1

        validator = row.get("validator", {}) if isinstance(row.get("validator"), dict) else {}
        if bool(validator.get("schema_ok", False)):
            counts["schema_ok"] += 1
        if bool(validator.get("numeric_consistency_ok", False)):
            counts["numeric_consistency_ok"] += 1

        teacher = row.get("teacher", {}) if isinstance(row.get("teacher"), dict) else {}
        backend = str(teacher.get("backend", "unknown"))
        backend_counter[backend] += 1
        notes = str(validator.get("notes", ""))
        if backend == "heuristic" or "fallback" in notes.lower():
            counts["fallback_like"] += 1

        placeholder_count = _count_placeholders(row)
        counts["placeholder_leakage"] += int(placeholder_count)

        selected = row.get("selected_topk", [])
        selected = selected if isinstance(selected, list) else []
        selected_len_counter[len(selected)] += 1
        if selected:
            counts["rows_with_selected_topk"] += 1

        cand_map = candidate_index.get((image_id, target_ar), {})
        if cand_map:
            counts["rows_with_candidate_index"] += 1
        for item in selected:
            if not isinstance(item, dict):
                continue
            counts["selected_items"] += 1
            cid = str(item.get("candidate_id", "")).strip()
            if cid and cid in cand_map:
                counts["candidate_id_consistent"] += 1
            for tag in item.get("why_tags", []) if isinstance(item.get("why_tags"), list) else []:
                counts["why_tags_total"] += 1
                if str(tag) in ALLOWED_WHY_TAGS:
                    counts["why_tags_allowed"] += 1

        explanations = row.get("explanations", {}) if isinstance(row.get("explanations"), dict) else {}
        long_text = str(explanations.get("long", ""))
        model_count_raw = validator.get("model_selected_count")
        if model_count_raw is not None:
            model_direct_pick_rows += 1
            try:
                model_direct_pick_total += int(model_count_raw)
            except Exception:
                pass
        else:
            match = re.search(r"모델 직접 선택=(\d+)개", long_text)
            if match:
                model_direct_pick_rows += 1
                model_direct_pick_total += int(match.group(1))

    rows = max(1, int(counts["rows"]))
    selected_items = max(1, int(counts["selected_items"]))
    why_tags_total = max(1, int(counts["why_tags_total"]))

    metrics = {
        "num_label_rows": int(counts["rows"]),
        "num_meta_rows": int(meta_rows_count),
        "num_unique_images": len({str(row.get("image_id", "")) for row in label_rows}),
        "schema_ok_rate": counts["schema_ok"] / rows,
        "numeric_consistency_ok_rate": counts["numeric_consistency_ok"] / rows,
        "candidate_id_consistency_rate": counts["candidate_id_consistent"] / selected_items,
        "rows_with_candidate_index_rate": counts["rows_with_candidate_index"] / rows,
        "fallback_rate": counts["fallback_like"] / rows,
        "placeholder_leakage_count": int(counts["placeholder_leakage"]),
        "why_tag_allowed_vocab_rate": counts["why_tags_allowed"] / why_tags_total,
        "rows_with_selected_topk": int(counts["rows_with_selected_topk"]),
        "selected_topk_size_histogram": {str(k): int(v) for k, v in sorted(selected_len_counter.items())},
        "target_ar_counts": dict(target_ar_counter),
        "teacher_backend_counts": dict(backend_counter),
        "model_direct_pick_rows": int(model_direct_pick_rows),
        "model_direct_pick_total": int(model_direct_pick_total),
    }

    ok = bool(
        counts["rows"] > 0
        and metrics["schema_ok_rate"] >= 0.99
        and metrics["numeric_consistency_ok_rate"] >= 0.99
        and metrics["candidate_id_consistency_rate"] >= 0.995
        and metrics["placeholder_leakage_count"] == 0
        and metrics["why_tag_allowed_vocab_rate"] >= 0.99
    )
    caveats = []
    if metrics["fallback_rate"] > 0.05:
        caveats.append("fallback_rate_above_5_percent")
    if metrics["model_direct_pick_rows"] and metrics["model_direct_pick_total"] == 0:
        caveats.append("vlm_generated_json_was_normalized_with_numeric_topk_fill")
    if metrics["rows_with_candidate_index_rate"] < 1.0:
        caveats.append("some_outputs_could_not_be_matched_to_teacher_score_candidates")

    return {
        "schema_version": "sstk_vlm_teacher_audit_analysis_v1",
        "created_at_utc": utc_timestamp(),
        "labels_jsonl": labels_jsonl,
        "meta_jsonl": meta_jsonl,
        "summary_json": summary_json,
        "teacher_scores_jsonl": teacher_scores_jsonl,
        "summary_payload": summary_payload,
        "metrics": metrics,
        "caveats": caveats,
        "ok": ok,
    }


def _ssh_base_args(args: argparse.Namespace) -> list[str]:
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=60",
        "-o",
        "ConnectionAttempts=1",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "StrictHostKeyChecking=no",
        "-J",
        str(args.jump_host),
        "-i",
        str(args.identity_file),
        str(args.remote_host),
    ]


def sync_remote_research_agent_code(args: argparse.Namespace) -> dict[str, Any]:
    files = [
        "src/vlm_teacher_labeler.py",
        "src/scripts/run_vlm_teacher_labeler.sh",
        "src/scripts/research_agent_bridge.py",
        "research_agents/remote_gpu/run_qwen3_vlm_teacher_probe.sh",
        "research_agents/remote_gpu/download_vlm_teacher_artifacts.sh",
    ]
    quoted_files = " ".join(shlex.quote(p) for p in files)
    ssh_args = " ".join(shlex.quote(x) for x in _ssh_base_args(args))
    remote_cmd = (
        f"cd {shlex.quote(str(args.remote_project_root))} && "
        "tar -xf - && "
        "chmod +x src/scripts/run_vlm_teacher_labeler.sh "
        "research_agents/remote_gpu/run_qwen3_vlm_teacher_probe.sh "
        "research_agents/remote_gpu/download_vlm_teacher_artifacts.sh"
    )
    cmd = f"tar -cf - {quoted_files} | {ssh_args} {shlex.quote(remote_cmd)}"
    return run_command(["bash", "-lc", cmd], timeout=int(args.sync_timeout))


def check_remote_vlm_inputs(
    args: argparse.Namespace,
    *,
    teacher_scores_rel: str,
    image_dir_rel: str,
) -> dict[str, Any]:
    remote_cmd = (
        "set -euo pipefail; "
        f"cd {shlex.quote(str(args.remote_project_root))}; "
        f"test -f {shlex.quote(teacher_scores_rel)}; "
        f"test -d {shlex.quote(image_dir_rel)}; "
        f"printf 'teacher_scores_lines='; wc -l < {shlex.quote(teacher_scores_rel)}; "
        f"printf 'image_files='; find {shlex.quote(image_dir_rel)} -maxdepth 1 "
        "\\( -type f -o -type l \\) | wc -l"
    )
    return run_command([*_ssh_base_args(args), remote_cmd], timeout=120)


def download_remote_debug_dir(
    args: argparse.Namespace,
    *,
    remote_debug_rel: str,
    local_run_dir: Path,
) -> dict[str, Any]:
    ssh_opts = [
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=60",
        "-o",
        "ConnectionAttempts=1",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        f"ProxyJump={args.jump_host}",
        "-i",
        str(args.identity_file),
    ]
    local_debug = local_run_dir / "debug"
    local_debug.mkdir(parents=True, exist_ok=True)
    remote_path = f"{args.remote_host}:{args.remote_project_root}/{remote_debug_rel.rstrip('/')}/"
    return run_command(
        ["scp", "-r", *ssh_opts, remote_path, str(local_debug)],
        timeout=600,
    )


def build_vlm_teacher_audit(args: argparse.Namespace) -> dict[str, Any]:
    paths = resolve_vlm_teacher_input_paths(
        args.data_root,
        args.run_tag,
        args.teacher_scores_jsonl,
        args.image_dir,
    )
    run_dir = Path(args.run_dir) if str(args.run_dir).strip() else DEFAULT_OUTPUT_ROOT / (
        f"vlm_teacher_audit_{args.run_tag}_{utc_timestamp()}"
    )
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    prompt_version = str(args.prompt_version)
    output_run_tag = str(args.output_run_tag).strip() or f"{args.run_tag}_qwen3_{utc_timestamp()}"
    labels_name = f"crop_label_v1_{output_run_tag}.jsonl"
    meta_name = f"meta_norm_v1_{output_run_tag}.jsonl"
    summary_name = f"vlm_teacher_summary_{output_run_tag}.json"

    local_labels = run_dir / labels_name
    local_meta = run_dir / meta_name
    local_summary = run_dir / summary_name

    teacher_scores_rel = _relative_to_project(paths["teacher_scores_jsonl"])
    image_dir_rel = _relative_to_project(paths["image_dir"])
    remote_labels_rel = f"data/TestImages/All/artifacts/vlm_teacher/labels/{labels_name}"
    remote_meta_rel = f"data/TestImages/All/artifacts/vlm_teacher/meta/{meta_name}"
    remote_summary_rel = f"data/TestImages/All/artifacts/vlm_teacher/summary/{summary_name}"
    remote_debug_rel = f"data/TestImages/All/artifacts/vlm_teacher/debug/{output_run_tag}"

    payload: dict[str, Any] = {
        "schema_version": "sstk_vlm_teacher_audit_v1",
        "created_at_utc": utc_timestamp(),
        "mode": args.mode,
        "run_tag": args.run_tag,
        "output_run_tag": output_run_tag,
        "run_dir": run_dir,
        "input_alignment": {
            "data_root": paths["data_root"],
            "teacher_scores_jsonl": paths["teacher_scores_jsonl"],
            "teacher_scores_exists_local": paths["teacher_scores_jsonl"].exists(),
            "image_dir": paths["image_dir"],
            "image_dir_exists_local": paths["image_dir"].exists(),
            "teacher_scores_rel": teacher_scores_rel,
            "image_dir_rel": image_dir_rel,
        },
        "commands": {},
        "analysis": {},
        "validation": {},
        "paper_pack": {},
    }

    if not paths["teacher_scores_jsonl"].exists():
        payload["ok"] = False
        payload["error"] = f"teacher_scores_jsonl not found: {paths['teacher_scores_jsonl']}"
        return payload
    if not paths["image_dir"].exists():
        payload["ok"] = False
        payload["error"] = f"image_dir not found: {paths['image_dir']}"
        return payload

    if args.mode == "local_heuristic":
        local_cmd = [
            "bash",
            "src/scripts/run_vlm_teacher_labeler.sh",
            "--teacher_scores_jsonl",
            teacher_scores_rel,
            "--output_jsonl",
            _relative_to_project(local_labels),
            "--output_meta_jsonl",
            _relative_to_project(local_meta),
            "--summary_json",
            _relative_to_project(local_summary),
            "--backend",
            "heuristic",
            "--fallback_backend",
            "none",
            "--image_dir",
            image_dir_rel,
            "--target_ar",
            str(args.target_ar),
            "--top_m",
            str(args.top_m),
            "--top_k",
            str(args.top_k),
            "--max_images",
            str(args.max_images),
            "--max_new_tokens",
            str(args.max_new_tokens),
            "--prompt_version",
            prompt_version,
            "--seed",
            str(args.seed),
            "--save_raw_response",
            "1" if args.save_raw_response else "0",
            "--debug_dir",
            _relative_to_project(run_dir / "debug"),
            "--strict_backend_init",
            "1",
        ]
        env = os.environ.copy()
        env["PYTHON_BIN"] = str(PROJECT_PYTHON)
        payload["commands"]["local_heuristic_run"] = run_command(
            local_cmd,
            timeout=int(args.timeout),
            env=env,
        )
    else:
        if args.sync_code:
            payload["commands"]["remote_code_sync"] = sync_remote_research_agent_code(args)
            if not payload["commands"]["remote_code_sync"].get("ok", False):
                payload["ok"] = False
                payload["error"] = "remote_code_sync failed"
                return payload

        payload["commands"]["remote_input_check"] = check_remote_vlm_inputs(
            args,
            teacher_scores_rel=teacher_scores_rel,
            image_dir_rel=image_dir_rel,
        )
        if not payload["commands"]["remote_input_check"].get("ok", False):
            payload["ok"] = False
            payload["error"] = "remote input check failed"
            return payload

        env = os.environ.copy()
        env.update(
            {
                "REMOTE_HOST": str(args.remote_host),
                "REMOTE_PROJECT_ROOT": str(args.remote_project_root),
                "REMOTE_PYTHON": str(args.remote_python),
                "REMOTE_MODEL_ID": str(args.remote_model_id),
                "LOCALIZE_MODEL": "1" if args.localize_remote_model else "0",
                "LOCAL_MODEL_CACHE_ROOT": str(args.local_model_cache_root),
                "JUMP_HOST": str(args.jump_host),
                "IDENTITY_FILE": str(args.identity_file),
                "GPU_ID": str(args.gpu_id),
                "MAX_IMAGES": str(args.max_images),
                "TOP_M": str(args.top_m),
                "TOP_K": str(args.top_k),
                "TARGET_AR": str(args.target_ar),
                "SEED": str(args.seed),
                "MAX_NEW_TOKENS": str(args.max_new_tokens),
                "PROMPT_VERSION": prompt_version,
                "RUN_TAG": output_run_tag,
                "TEACHER_SCORES_JSONL": teacher_scores_rel,
                "IMAGE_DIR": image_dir_rel,
                "OUTPUT_JSONL": remote_labels_rel,
                "OUTPUT_META_JSONL": remote_meta_rel,
                "SUMMARY_JSON": remote_summary_rel,
                "SAVE_RAW_RESPONSE": "1" if args.save_raw_response else "0",
                "DEBUG_DIR": remote_debug_rel,
            }
        )
        payload["commands"]["remote_qwen3_run"] = run_command(
            ["bash", "research_agents/remote_gpu/run_qwen3_vlm_teacher_probe.sh"],
            timeout=int(args.timeout),
            env=env,
        )

        if payload["commands"]["remote_qwen3_run"].get("ok", False):
            download_env = os.environ.copy()
            download_env.update(
                {
                    "REMOTE_HOST": str(args.remote_host),
                    "REMOTE_PROJECT_ROOT": str(args.remote_project_root),
                    "JUMP_HOST": str(args.jump_host),
                    "IDENTITY_FILE": str(args.identity_file),
                    "REMOTE_OUTPUT_JSONL": remote_labels_rel,
                    "REMOTE_OUTPUT_META_JSONL": remote_meta_rel,
                    "REMOTE_SUMMARY_JSON": remote_summary_rel,
                    "LOCAL_OUTPUT_DIR": str(run_dir),
                }
            )
            payload["commands"]["download_remote_outputs"] = run_command(
                ["bash", "research_agents/remote_gpu/download_vlm_teacher_artifacts.sh"],
                timeout=600,
                env=download_env,
            )
            if args.save_raw_response:
                payload["commands"]["download_remote_debug"] = download_remote_debug_dir(
                    args,
                    remote_debug_rel=remote_debug_rel,
                    local_run_dir=run_dir,
                )

    if local_labels.exists() and local_summary.exists():
        payload["analysis"] = analyze_vlm_teacher_outputs(
            labels_jsonl=local_labels,
            meta_jsonl=local_meta,
            summary_json=local_summary,
            teacher_scores_jsonl=paths["teacher_scores_jsonl"],
            top_m=int(args.top_m),
            top_k=int(args.top_k),
            prompt_version=prompt_version,
        )
    else:
        payload["analysis"] = {
            "ok": False,
            "error": "expected VLM Teacher outputs were not found locally",
            "labels_jsonl": local_labels,
            "summary_json": local_summary,
        }

    if args.run_validation:
        payload["validation"]["vlm_schema_pytest"] = run_pytest_suite("vlm_schema", int(args.validation_timeout))

    evidence_report = {
        "schema_version": "sstk_vlm_teacher_audit_evidence_v1",
        "created_at_utc": utc_timestamp(),
        "mode": payload.get("mode"),
        "run_tag": payload.get("run_tag"),
        "output_run_tag": payload.get("output_run_tag"),
        "input_alignment": payload.get("input_alignment", {}),
        "command_summaries": {
            name: {
                "ok": cmd.get("ok"),
                "returncode": cmd.get("returncode"),
                "elapsed_sec": cmd.get("elapsed_sec"),
            }
            for name, cmd in payload.get("commands", {}).items()
            if isinstance(cmd, dict)
        },
        "analysis": payload.get("analysis", {}),
        "validation": payload.get("validation", {}),
        "ok": bool(payload.get("analysis", {}).get("ok", False)),
    }
    evidence_report_path = run_dir / "vlm_teacher_audit_evidence.json"
    write_json(evidence_report_path, evidence_report)
    payload["evidence_report"] = evidence_report_path

    if args.build_paper_pack:
        draft_path = run_dir / "paper_pack_draft.md"
        summary_path = run_dir / "paper_pack_summary.json"
        paper_summary = build_paper_pack(run_dir, draft_path)
        write_json(summary_path, paper_summary)
        payload["paper_pack"] = {
            "draft": draft_path,
            "summary": summary_path,
            "ok": bool(paper_summary.get("ok", False)),
        }

    command_ok = True
    if args.mode == "local_heuristic":
        command_ok = bool(payload["commands"].get("local_heuristic_run", {}).get("ok", False))
    else:
        command_ok = bool(
            payload["commands"].get("remote_qwen3_run", {}).get("ok", False)
            and payload["commands"].get("download_remote_outputs", {}).get("ok", False)
        )
    validation_ok = True
    if args.run_validation:
        validation_ok = bool(payload["validation"].get("vlm_schema_pytest", {}).get("ok", False))

    payload["ok"] = bool(command_ok and payload["analysis"].get("ok", False) and validation_ok)
    return payload


def cmd_doctor(args: argparse.Namespace) -> int:
    payload = build_diagnosis(args)
    if args.output:
        write_json(Path(args.output), payload)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default))
    return 0 if payload.get("ok") else 2


def cmd_validate(args: argparse.Namespace) -> int:
    result = {
        "schema_version": "sstk_research_agent_validation_v1",
        "created_at_utc": utc_timestamp(),
        "suite": args.suite,
        "pytest": run_pytest_suite(args.suite, args.timeout),
    }
    result["ok"] = bool(result["pytest"].get("ok", False))
    if args.output:
        write_json(Path(args.output), result)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))
    return 0 if result["ok"] else 1


def cmd_campaigns(args: argparse.Namespace) -> int:
    payload = build_campaign_templates()
    if args.output:
        write_json(Path(args.output), payload)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default))
    return 0


def cmd_paper_pack(args: argparse.Namespace) -> int:
    payload = build_paper_pack(Path(args.run_dir), Path(args.output))
    if args.summary_output:
        write_json(Path(args.summary_output), payload)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default))
    return 0 if payload.get("ok") else 1


def cmd_vlm_teacher_audit(args: argparse.Namespace) -> int:
    payload = build_vlm_teacher_audit(args)
    if args.output:
        write_json(Path(args.output), payload)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default))
    return 0 if payload.get("ok") else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="Check Codex, ccproxy, repo, and project readiness.")
    doctor.add_argument("--ccproxy_port", type=int, default=8000)
    doctor.add_argument("--evoscientist_repo", default="/tmp/EvoScientist_260413")
    doctor.add_argument("--ai_scientist_v2_repo", default="/tmp/ai-scientist-v2_260413")
    doctor.add_argument("--run_tests", action="store_true")
    doctor.add_argument("--timeout", type=int, default=180)
    doctor.add_argument("--output", default="")
    doctor.set_defaults(func=cmd_doctor)

    validate = sub.add_parser("validate", help="Run a narrow validation suite.")
    validate.add_argument("--suite", choices=sorted(SMOKE_TESTS), default="agent_smoke")
    validate.add_argument("--timeout", type=int, default=240)
    validate.add_argument("--output", default="")
    validate.set_defaults(func=cmd_validate)

    campaigns = sub.add_parser("campaigns", help="Emit supported research campaign templates.")
    campaigns.add_argument("--output", default="")
    campaigns.set_defaults(func=cmd_campaigns)

    paper = sub.add_parser("paper-pack", help="Build a claim/evidence-linked paper draft pack.")
    paper.add_argument("--run_dir", default=str(DEFAULT_OUTPUT_ROOT))
    paper.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "Research_Docs" / "Agentic_Crop_RnD_Paper_Draft_KO.md"),
    )
    paper.add_argument("--summary_output", default="")
    paper.set_defaults(func=cmd_paper_pack)

    audit = sub.add_parser(
        "vlm-teacher-audit",
        help="Run a bounded VLM Teacher experiment and package schema/fidelity evidence.",
    )
    audit.add_argument("--mode", choices=["local_heuristic", "remote_qwen3"], default="local_heuristic")
    audit.add_argument("--data_root", default="data/TestImages/All")
    audit.add_argument("--run_tag", default="test_images_personv6_server_v1")
    audit.add_argument("--teacher_scores_jsonl", default="")
    audit.add_argument("--image_dir", default="")
    audit.add_argument("--output_run_tag", default="")
    audit.add_argument("--run_dir", default="")
    audit.add_argument("--target_ar", default="all")
    audit.add_argument("--top_m", type=int, default=12)
    audit.add_argument("--top_k", type=int, default=5)
    audit.add_argument("--max_images", type=int, default=1)
    audit.add_argument("--max_new_tokens", type=int, default=384)
    audit.add_argument("--prompt_version", default="crop_label_candidate_ids_v2")
    audit.add_argument("--seed", type=int, default=42)
    audit.add_argument("--timeout", type=int, default=7200)
    audit.add_argument("--validation_timeout", type=int, default=240)
    audit.add_argument("--run_validation", action=argparse.BooleanOptionalAction, default=True)
    audit.add_argument("--build_paper_pack", action=argparse.BooleanOptionalAction, default=True)
    audit.add_argument("--sync_code", action=argparse.BooleanOptionalAction, default=True)
    audit.add_argument("--save_raw_response", action=argparse.BooleanOptionalAction, default=False)
    audit.add_argument("--sync_timeout", type=int, default=300)
    audit.add_argument("--remote_host", default=DEFAULT_REMOTE_HOST)
    audit.add_argument("--remote_project_root", default=DEFAULT_REMOTE_PROJECT_ROOT)
    audit.add_argument("--remote_python", default=DEFAULT_REMOTE_PYTHON)
    audit.add_argument("--remote_model_id", default=DEFAULT_REMOTE_MODEL_ID)
    audit.add_argument("--localize_remote_model", action=argparse.BooleanOptionalAction, default=False)
    audit.add_argument("--local_model_cache_root", default="/tmp/jaden_qwen3_vl_cache")
    audit.add_argument("--jump_host", default=DEFAULT_JUMP_HOST)
    audit.add_argument("--identity_file", default=DEFAULT_IDENTITY_FILE)
    audit.add_argument("--gpu_id", default="auto")
    audit.add_argument("--output", default="")
    audit.set_defaults(func=cmd_vlm_teacher_audit)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
