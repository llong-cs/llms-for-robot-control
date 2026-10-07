#!/usr/bin/env python3
"""Summarize recorded robot evaluations by run, task and difficulty (offline).

Example:
    python3 tools/summarize.py outputs/run-a outputs/run-b -o outputs/summary

Only the Python standard library is required. No simulator or model is loaded.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TOKEN_FIELDS = (
    "input_tokens", "output_tokens", "total_tokens", "cached_input_tokens",
    "cache_creation_input_tokens", "uncached_input_tokens", "reasoning_tokens",
)
SCORE_FIELDS = ("progress_score", "final_progress_score")
RESOURCE_FIELDS = (
    "wall_time_s", "inference_wall_s", "sim_time_s", "steps", "policy_control_steps",
    "verification_control_steps", "rounds", "model_calls", "format_repair_calls",
    "transport_retry_calls", "internal_retry_calls", "retry_wait_s",
    "decision_wall_s", "reset_wall_s", "simulator_step_wall_s", "video_export_wall_s",
    *TOKEN_FIELDS,
)
METHODS = [
    "Each recorded trial has equal weight; pooled SR is successes / evaluated_trials, "
    "not the mean of run success rates.",
    "Included trials have execution status 'success', completed/absent launcher job status, "
    "and are not previews. This status is not task success. SR uses explicit boolean "
    "oracle_success; unknown outcomes are excluded from its denominator.",
    "Discarded, failed, interrupted, running and missing trials are listed but excluded "
    "from every score/resource aggregate. Initially solved trials are included and counted.",
    "progress_score is the recorded native score (episode peak for the custom task suite); "
    "final_progress_score is the terminal score. Neither is inferred from success or "
    "replaced by zero when unavailable. Scores from different suites may not be comparable.",
    "rounds counts logical decisions; model_calls includes repair/retry requests. "
    "Control steps are separate from both. All time fields use seconds.",
    "wall_time_s is trial execution time. Its sum is cumulative work, not elapsed time "
    "for parallel runs. runs.csv reports launcher elapsed time from start/end timestamps.",
    "Token sums are known reported usage, not estimates. Telemetry request sums plus late "
    "session usage adjustments take precedence over policy totals. Coverage columns retain "
    "missing request usage. Policy-only totals have unknown completeness. Cached tokens are "
    "part of input tokens; reasoning tokens are part of output tokens; do not add them again.",
    "Numeric statistics omit unavailable values and report the contributing count. "
    "std is sample standard deviation (null with fewer than two values). Token means use "
    "known trial totals, which may be partial; inspect completeness counts before comparing.",
    "Input directories are resolved and deduplicated; child results take precedence over "
    "pooled copies. Distinct run directories remain distinct observations even for equal seeds.",
]


def obj(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def first(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def number(value: Any) -> int | float | None:
    # Native simulator scalars in older JSON exports can be single-element arrays.
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value if math.isfinite(value) else None
    return None


def read_json(path: Path, warnings: list[str]) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        warnings.append(f"Cannot read {path}: {error}")
        return None


def records(path: Path, warnings: list[str]) -> list[dict]:
    value = read_json(path, warnings)
    if value is None:
        return []
    if path.name == "result.json" and isinstance(value, dict):
        return [value]
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        warnings.append(f"Expected a list of result objects: {path}")
        return []
    return value


def elapsed(status: dict) -> float | None:
    try:
        start = datetime.fromisoformat(status["started_at"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(status["finished_at"].replace("Z", "+00:00"))
        seconds = (end - start).total_seconds()
        return seconds if seconds >= 0 else None
    except (KeyError, TypeError, ValueError):
        return None


def identity(row: dict) -> tuple:
    return (row.get("scene_id"), row.get("init"), row.get("seed"))


def scene_identity(value: Any) -> str:
    """Older LIBERO scene IDs and launcher job IDs differ in init zero padding."""
    return re.sub(r"-init0*(\d+)$", lambda match: f"-init{int(match[1])}", str(value))


def discover(run: Path, warnings: list[str]) -> tuple[list[tuple], dict, dict]:
    """Use shallow, known evaluation layouts; never scan demos, images or API bodies."""
    plan = obj(read_json(run / "eval-plan.json", warnings))
    status = obj(read_json(run / "eval-status.json", warnings))
    base = run / "evaluation" if (run / "evaluation").is_dir() else run
    if not plan.get("config") and not plan.get("resolved"):
        direct = obj(read_json(base / "config.json", warnings))
        if "environment" in direct:
            plan["config"] = direct
        elif direct:
            plan["config"] = {
                "environment": {key: direct.get(key) for key in
                                ("benchmark", "suite", "difficulty", "control_hz")},
                "run": {"mode": direct.get("mode")},
                "llm": {"model": direct.get("model")},
                "execution": obj(direct.get("schedule")),
            }
    pooled_path = base / "results.json"
    pooled = records(pooled_path, warnings)
    selected: dict[tuple, tuple] = {}

    def add(row: dict, source: Path, job: dict, ledger: dict) -> None:
        key = (job.get("id", row.get("evaluation_job_id")), *identity(row))
        selected[key] = (row, source, job, ledger)

    jobs = plan.get("jobs") or []
    if not isinstance(jobs, list) or any(not isinstance(job, dict) for job in jobs):
        raise ValueError(f"Expected eval-plan.json jobs to be a list of objects: {run}")
    if jobs:
        used_pool_rows = set()
        for job in jobs:
            job_id = str(job["id"])
            ledger = obj(read_json(run / "jobs" / job_id / "status.json", warnings))
            source = base / job_id / "results.json"
            values = records(source, warnings)
            if not values:
                for candidate in (base / job_id / "evals" / job_id / "result.json",
                                  base / "evals" / job_id / "result.json"):
                    values = records(candidate, warnings)
                    if values:
                        source = candidate
                        break
            matches = [(i, row) for i, row in enumerate(pooled)
                       if scene_identity(row.get("evaluation_job_id", row.get("scene_id")))
                       == scene_identity(job_id)]
            used_pool_rows.update(i for i, _ in matches)
            if not values:
                values = [row for _, row in matches]
                source = pooled_path
            if not values:
                values = [{"scene_id": job_id, "status": "missing"}]
                warnings.append(f"No result for planned job {job_id} in {run}")
            for row in values:
                add(row, source, job, ledger)
        for i, row in enumerate(pooled):
            if i not in used_pool_rows:
                warnings.append(f"Unplanned result retained: {row.get('scene_id')} in {run}")
                add(row, pooled_path, {}, {})
    else:
        # Also support a direct framework evaluation directory, without a launcher plan.
        for row in pooled:
            add(row, pooled_path, {}, {})
        children = sorted(base.iterdir())
        for child in children:
            if child.is_dir() and (child / "results.json").is_file():
                for row in records(child / "results.json", warnings):
                    job_id = row.get("evaluation_job_id", child.name)
                    ledger = obj(read_json(run / "jobs" / job_id / "status.json", warnings))
                    # Replace an aggregate copy even if it includes launcher-only fields.
                    for key, old in list(selected.items()):
                        if identity(old[0]) == identity(row):
                            del selected[key]
                    add(row, child / "results.json", {"id": job_id}, ledger)
        for candidate in sorted(base.glob("evals/*/result.json")):
            for row in records(candidate, warnings):
                for key, old in list(selected.items()):
                    if identity(old[0]) == identity(row):
                        del selected[key]
                add(row, candidate, {}, {})
        if (base / "result.json").is_file():
            for row in records(base / "result.json", warnings):
                add(row, base / "result.json", {}, {})
    if not selected:
        raise ValueError(f"No recorded results or planned jobs found in {run}")
    return list(selected.values()), plan, status


def trial_row(run: Path, result: dict, source: Path, job: dict, ledger: dict,
              plan: dict, warnings: list[str]) -> dict:
    scene_id = result.get("scene_id", job.get("id", "unknown"))
    episode = source.parent if source.name == "result.json" else source.parent / "evals" / scene_id
    job_id = first(job.get("id"), result.get("evaluation_job_id"))
    if source.name == "results.json" and not episode.is_dir() and job_id:
        local_episode = source.parent / str(job_id) / "evals" / scene_id
        if local_episode.is_dir():
            episode = local_episode
    local_trajectory = episode / "trajectory.json"
    recorded_path = obj(result.get("trajectory")).get("path")
    trajectory_path = local_trajectory
    if not trajectory_path.is_file() and recorded_path:
        candidate = Path(recorded_path)
        candidate = candidate if candidate.is_absolute() else source.parent / candidate
        # Do not silently read the original run when analyzing a relocated copy.
        if candidate.resolve().is_relative_to(run) and candidate.is_file():
            trajectory_path = candidate
    trajectory = obj(read_json(trajectory_path, warnings))
    meta = obj(trajectory.get("metadata"))
    scene = obj(obj(meta.get("scene")).get("metadata"))
    reset = obj(obj(result.get("metadata")).get("simulator_reset"))
    outcome = obj(trajectory.get("outcome"))
    config = obj(first(plan.get("resolved"), plan.get("config")))
    environment = obj(config.get("environment"))
    policy = obj(first(result.get("policy"),
                       obj(result.get("metadata")).get("agentic")))
    if not policy:
        policy = obj(read_json(episode / "policy-metrics.json", warnings))
    telemetry = obj(trajectory.get("telemetry_summary"))
    runtime = obj(trajectory.get("runtime"))
    sources = (result, scene, meta, reset, job, environment, obj(plan.get("bench")))

    def field(key: str) -> Any:
        return first(*(item.get(key) for item in sources))

    job_status = first(ledger.get("status"), result.get("evaluation_job_status"))
    status = result.get("status", "unknown")
    preview = (obj(config.get("run")).get("mode") == "preview"
               or result.get("task_success_evaluated") is False
               or outcome.get("task_success_evaluated") is False)
    included = status == "success" and job_status in (None, "completed") and not preview
    success = result.get("oracle_success")
    success = success if type(success) is bool else None
    model = first(policy.get("model"), meta.get("model"), plan.get("model"),
                  obj(config.get("llm")).get("model"), "unknown")
    row = {
        "run_name": run.name, "run_path": str(run), "model": model,
        "benchmark": field("benchmark") or "unknown", "suite": field("suite") or "unknown",
        "task_id": field("task_id"),
        "task": first(field("task_slug"), field("env_id"),
                      str(field("task_id")) if field("task_id") is not None else "unknown"),
        "difficulty": field("difficulty") or "unspecified", "scene_id": scene_id,
        "job_id": job.get("id", result.get("evaluation_job_id")),
        "init": first(result.get("init"), meta.get("init_state_index"), job.get("init_start")),
        "seed": result.get("seed"), "paired_seed": first(result.get("paired_seed"), scene.get("paired_seed")),
        "policy_seed": meta.get("policy_seed"), "status": status, "job_status": job_status,
        "included": included,
        "exclusion_reason": None if included else (
            status if status != "success" else ("preview" if preview else f"job_{job_status}")),
        "oracle_success": success, "initially_solved": result.get("initially_solved"),
        "termination_reason": result.get("termination_reason"), "error": result.get("error"),
        "progress_score": number(first(result.get("progress_score"), outcome.get("progress_score"))),
        "final_progress_score": number(result.get("final_progress_score")),
        "progress_peak_step": number(result.get("progress_peak_step")),
        "wall_time_s": number(result.get("wall_time_s")),
        "inference_wall_s": number(first(result.get("inference_wall_s"),
                                          policy.get("inference_wall_s"))),
        "sim_time_s": number(result.get("sim_time_s")), "steps": number(result.get("steps")),
        "policy_control_steps": number(result.get("policy_control_steps")),
        "verification_control_steps": number(result.get("verification_control_steps")),
        "verification_triggered": result.get("verification_triggered"),
        "success_phase": result.get("success_phase"),
        "rounds": number(telemetry.get("logical_decisions")),
        "model_calls": number(first(policy.get("model_calls"), telemetry.get("request_count"))),
        "source_results_file": str(source),
        "trajectory_file": str(trajectory_path) if trajectory else None,
    }
    if row["rounds"] is None:
        planning, polling = number(policy.get("planning_calls")), number(policy.get("polling_calls"))
        if planning is not None:
            row["rounds"] = planning + (polling or 0)
        elif isinstance(result.get("inference_control_steps"), list):
            row["rounds"] = len(result["inference_control_steps"])
    for key, telemetry_key in (("format_repair_calls", "format_repair_requests"),
                               ("transport_retry_calls", "transport_retry_requests"),
                               ("internal_retry_calls", "internal_retry_requests"),
                               ("retry_wait_s", "retry_wait_s")):
        row[key] = number(first(policy.get(key), telemetry.get(telemetry_key)))
    timings = trajectory.get("decision_timings")
    known_timings = [number(obj(item).get("elapsed_s")) for item in timings or []]
    row["decision_wall_s"] = (sum(known_timings) if known_timings
                              and all(value is not None for value in known_timings) else None)
    for key in ("reset_wall_s", "simulator_step_wall_s", "video_export_wall_s"):
        row[key] = number(runtime.get(key))
    schedule = obj(first(policy.get("schedule"), meta.get("schedule"), config.get("execution")))
    row.update({key: schedule.get(key) for key in ("h", "k", "max_steps", "control_interface")})
    augmentation = obj(first(policy.get("augmentation"), meta.get("augmentation")))
    row["history_length"] = obj(augmentation.get("memory")).get("history_length")
    row["reasoning_effort"] = obj(augmentation.get("reasoning")).get("effort")
    row["privilege_enabled"] = meta.get("privilege_enabled")
    usage = obj(telemetry.get("usage"))
    adjustments = first(trajectory.get("usage_adjustments"), telemetry.get("usage_adjustments"), [])
    row["usage_adjustment_count"] = len(adjustments)
    for key in TOKEN_FIELDS:
        measured = obj(usage.get(key))
        if measured:
            count = number(measured.get("requests_with_value"))
            total = number(measured.get("requests_total"))
            value = number(measured.get("known_sum"))
            if count == 0 and total != 0:
                value = None
            adjustments_known = [adjustment_value for item in adjustments
                                 if (adjustment_value := number(
                                     obj(obj(item).get("usage")).get(key))) is not None]
            if adjustments_known:
                value = (value or 0) + sum(adjustments_known)
            complete = measured.get("complete")
            complete = complete if type(complete) is bool else None
            if len(adjustments_known) < len(adjustments):
                complete = False
            row[key] = value
            row[f"{key}_complete"] = complete
            row[f"{key}_requests_with_value"] = count
            row[f"{key}_requests_total"] = total
            row[f"{key}_source"] = "trajectory_telemetry"
        else:
            row[key] = number(obj(policy.get("usage")).get(key))
            row[f"{key}_complete"] = None
            row[f"{key}_requests_with_value"] = None
            row[f"{key}_requests_total"] = None
            row[f"{key}_source"] = "policy" if row[key] is not None else None
    if included and success is None:
        warnings.append(f"Unknown task outcome excluded from SR: {run.name}/{scene_id}")
    return row


def statistics_for(rows: list[dict], key: str) -> dict:
    values = [value for row in rows if (value := number(row.get(key))) is not None]
    stats = {
        "count": len(values), "mean": statistics.mean(values) if values else None,
        "std": statistics.stdev(values) if len(values) > 1 else None,
        "median": statistics.median(values) if values else None,
        "min": min(values) if values else None, "max": max(values) if values else None,
    }
    if key not in SCORE_FIELDS:
        stats["sum"] = sum(values) if values else None
    return {f"{key}_{suffix}": value for suffix, value in stats.items()}


def aggregate(rows: list[dict]) -> dict:
    included = [row for row in rows if row["included"]]
    evaluated = [row for row in included if type(row["oracle_success"]) is bool]
    successes = sum(row["oracle_success"] for row in evaluated)
    result = {
        "trials": len(rows), "included_trials": len(included),
        "excluded_trials": len(rows) - len(included), "evaluated_trials": len(evaluated),
        "successes": successes, "failures": len(evaluated) - successes,
        "sr": successes / len(evaluated) if evaluated else None,
        "discarded_trials": sum(row["status"] == "discarded" for row in rows),
        "missing_trials": sum(row["status"] == "missing" for row in rows),
        "initially_solved_trials": sum(row["initially_solved"] is True for row in included),
        "unknown_outcome_trials": len(included) - len(evaluated),
        "exclusion_counts": dict(Counter(row["exclusion_reason"] for row in rows if not row["included"])),
        "termination_counts": dict(Counter(row["termination_reason"] or "unknown" for row in included)),
    }
    for key in (*SCORE_FIELDS, *RESOURCE_FIELDS):
        result.update(statistics_for(included, key))
    for key in TOKEN_FIELDS:
        result[f"{key}_complete_trials"] = sum(row[f"{key}_complete"] is True for row in included)
        result[f"{key}_partial_trials"] = sum(row[f"{key}_complete"] is False for row in included)
        result[f"{key}_unknown_completeness_trials"] = sum(
            row[f"{key}_complete"] is None for row in included)
        for suffix in ("requests_with_value", "requests_total"):
            values = [row[f"{key}_{suffix}"] for row in included if row[f"{key}_{suffix}"] is not None]
            result[f"{key}_{suffix}"] = sum(values) if values else None
    return result


def group_rows(trials: list[dict]) -> list[dict]:
    model = ("model", "benchmark", "suite")
    task = ("task_id", "task")
    run = ("run_path", "run_name")
    dimensions = {
        "overall": (), "run": run + model,
        "run_task_difficulty": run + model + task + ("difficulty",),
        "task_difficulty": model + task + ("difficulty",),
        "task": model + task, "difficulty": model + ("difficulty",),
    }
    groups = []
    for grouping, keys in dimensions.items():
        buckets = defaultdict(list)
        for row in trials:
            buckets[tuple(row[key] for key in keys)].append(row)
        for values, rows in sorted(buckets.items(), key=lambda item: tuple(str(v) for v in item[0])):
            groups.append({"grouping": grouping, **dict(zip(keys, values)), **aggregate(rows)})
    return groups


def build_report(paths: list[Path]) -> dict:
    warnings: list[str] = []
    trials, runs, seen = [], [], set()
    for raw in paths:
        run = Path(raw).expanduser().resolve()
        if run in seen:
            continue
        if not run.is_dir():
            raise ValueError(f"Experiment directory does not exist: {run}")
        # A run root and its evaluation directory are aliases, not independent runs.
        if run.name == "evaluation" and (run.parent / "eval-plan.json").is_file():
            run = run.parent
        if run in seen:
            continue
        seen.add(run)
        selected, plan, status = discover(run, warnings)
        current = [trial_row(run, result, source, job, ledger, plan, warnings)
                   for result, source, job, ledger in selected]
        # Planned but unstarted jobs often lack labels. Reuse only unambiguous
        # metadata from this run so they appear alongside the corresponding task.
        for key in ("model", "benchmark", "suite"):
            known = {row[key] for row in current if row[key] != "unknown"}
            if len(known) == 1:
                for row in current:
                    if row[key] == "unknown":
                        row[key] = next(iter(known))
        task_names = defaultdict(set)
        for row in current:
            if row["task"] not in ("unknown", str(row["task_id"])):
                task_names[(row["benchmark"], row["suite"], row["task_id"])].add(row["task"])
        for row in current:
            names = task_names[(row["benchmark"], row["suite"], row["task_id"])]
            if row["task"] in ("unknown", str(row["task_id"])) and len(names) == 1:
                row["task"] = next(iter(names))
        trials.extend(current)
        runs.append({
            "run_name": run.name, "run_path": str(run), "status": status.get("status"),
            "planned_trials": first(plan.get("planned_trials"), status.get("planned_trials"),
                                     len(plan["jobs"]) if plan.get("jobs") else None),
            "observed_trials": len(current), "included_trials": sum(row["included"] for row in current),
            "started_at": status.get("started_at"), "finished_at": status.get("finished_at"),
            "elapsed_wall_s": elapsed(status), "worker_count": status.get("worker_count"),
        })
    if not trials:
        raise ValueError("Supply at least one experiment directory")
    return {
        "schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
        "methods": METHODS, "runs": runs, "trials": trials,
        "groups": group_rows(trials), "warnings": warnings,
    }


def display(value: Any, percent: bool = False) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        return f"{value:.1%}" if percent else f"{value:,.3f}".rstrip("0").rstrip(".")
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown_table(rows: list[dict], columns: list[tuple[str, str]]) -> list[str]:
    lines = ["| " + " | ".join(label for _, label in columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(display(row.get(key), key == "sr") for key, _ in columns) + " |"
                 for row in rows)
    return lines


def render_report(report: dict) -> str:
    metrics = [("evaluated_trials", "Evaluated"), ("successes", "Successes"), ("sr", "SR"),
        ("progress_score_mean", "Progress score"), ("final_progress_score_mean", "Final progress"),
               ("wall_time_s_mean", "Mean wall s"), ("rounds_mean", "Mean rounds"),
               ("model_calls_mean", "Mean requests"), ("total_tokens_mean", "Mean tokens"),
               ("excluded_trials", "Excluded")]
    lines = ["# Experiment summary", "", f"Generated: {report['generated_at']}", ""]
    for grouping, title, columns in (
        ("run", "By run", [("run_name", "Run"), ("model", "Model")]),
        ("task_difficulty", "Pooled by task and difficulty", [("model", "Model"), ("suite", "Suite"),
          ("task", "Task"), ("difficulty", "Difficulty")]),
        ("run_task_difficulty", "By run, task and difficulty", [("run_name", "Run"),
          ("model", "Model"), ("task", "Task"), ("difficulty", "Difficulty")]),
    ):
        rows = [row for row in report["groups"] if row["grouping"] == grouping]
        lines += [f"## {title}", "", *markdown_table(rows, columns + metrics), ""]
    lines += ["## Metric definitions", "", *(f"- {text}" for text in report["methods"]), "",
              "CSV/JSON include dispersion, contributing counts, token coverage, per-trial "
              "settings, exclusions and termination reasons.", ""]
    if report["warnings"]:
        lines += ["## Warnings", "", *(f"- {text}" for text in report["warnings"]), ""]
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list))
                             else value for key, value in row.items()})


def write_report(report: dict, output: Path) -> None:
    output = output.expanduser().resolve()
    repository = Path(__file__).resolve().parents[1]
    if output == repository:
        raise ValueError("Choose a report subdirectory, such as outputs/summary")
    output.mkdir(parents=True, exist_ok=True)
    # Refuse existing report filenames to protect recorded results and previous reports.
    names = ("summary.json", "summary.csv", "trials.csv", "runs.csv", "report.md")
    if any((output / name).exists() or (output / name).is_symlink() for name in names):
        raise ValueError(f"Report files already exist in {output}; choose a new output directory")
    (output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                                   allow_nan=False) + "\n", encoding="utf-8")
    for name, key in (("summary.csv", "groups"), ("trials.csv", "trials"), ("runs.csv", "runs")):
        write_csv(output / name, report[key])
    (output / "report.md").write_text(render_report(report), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", help="Run directories, or a quoted JSON array of directories")
    parser.add_argument("--runs-file", type=Path, help="UTF-8 JSON file containing an array of run directories")
    parser.add_argument("-o", "--output-dir", type=Path,
                        help="New report directory, for example outputs/summary; default: print only")
    args = parser.parse_args(argv)
    try:
        paths = []
        for value in args.runs:
            values = json.loads(value) if value.lstrip().startswith("[") else [value]
            if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
                raise ValueError("Run lists must contain only path strings")
            paths.extend(Path(item) for item in values)
        if args.runs_file:
            values = json.loads(args.runs_file.expanduser().read_text(encoding="utf-8"))
            if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
                raise ValueError("--runs-file must contain a JSON array of path strings")
            paths.extend(Path(item) for item in values)
        report = build_report(paths)
        if args.output_dir:
            write_report(report, args.output_dir)
        print(render_report(report))
        if args.output_dir:
            print(f"Report saved to: {args.output_dir.expanduser().resolve()}")
        return 0
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
