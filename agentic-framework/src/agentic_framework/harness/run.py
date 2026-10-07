"""Reproducible evaluation entrypoint for language and VLA policies."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime
from statistics import mean

from inspect_robots import eval as robot_eval
from inspect_robots.approver import ClampApprover
from inspect_robots.logging import JsonLogSink, LiveLogSink
from inspect_robots.logging.sink import NullSink
from inspect_robots.scorer import episode_length, success_at_end
from inspect_robots.task import Task

from agentic_framework.common.io import plain, write_json
from agentic_framework.common.paths import FRAMEWORK_ROOT, PACKAGE_ROOT
from agentic_framework.common.project import project_root
from agentic_framework.configuration.arguments import group_config, parse_configuration, parser
from agentic_framework.configuration.profiles import load_model_profile, validate_native_horizon
from agentic_framework.environments.libero.benchmarks import HORIZONS, build_scenes, select_tasks
from agentic_framework.environments.maniskill.benchmarks import (
    ENV_ID,
    MANISKILL_HORIZON,
    build_maniskill_scenes,
)
from agentic_framework.harness.done_verification import (
    DoneVerificationController,
    episode_phases,
    verification_steps,
)
from agentic_framework.harness.tools import motion_tool_name
from agentic_framework.harness.types import (
    DISCARDED_STATUS,
    MAX_ATTEMPTS_TERMINATION,
    AugmentationConfig,
    AutoMotionConfig,
    ScheduleConfig,
)
from agentic_framework.models.factory import create_policy
from agentic_framework.observability.progress import EvaluationProgress

PROJECT = FRAMEWORK_ROOT


def parse_args(argv=None):
    return parse_configuration(argv, parser_factory=parser, maniskill_suite=ENV_ID)


def source_fingerprint():
    roots = [
        PROJECT / "src",
        PROJECT / "vendor/inspect-robots/src",
        PROJECT / "scripts",
        PROJECT / "configs",
    ]
    return {
        str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for root in roots
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix in {".py", ".json"}
        if "__pycache__" not in p.parts
    }


def unscore_discarded(log):
    """Return the Inspect log with every discarded (max_attempts) trial left unscored.

    Inspect scores every cleanly stopped trial, but a discarded episode is not
    evaluated. Its epoch scores are dropped, as Inspect does for a non-successful
    trial, and the scene and run aggregates are recomputed from the remaining
    scores (the runner's tasks use the default mean epoch reducer).
    """
    samples = []
    for sample in log.samples:
        reasons = sample.termination_reasons
        epochs = tuple(
            {} if index < len(reasons) and reasons[index] == MAX_ATTEMPTS_TERMINATION else scores
            for index, scores in enumerate(sample.epochs)
        )
        if epochs != sample.epochs:
            names = dict.fromkeys(name for scores in sample.epochs for name in scores)
            reduced = {
                name: mean(values)
                for name in names
                if (values := [scores[name] for scores in epochs if name in scores])
            }
            sample = replace(sample, epochs=epochs, reduced=reduced)
        samples.append(sample)
    if tuple(samples) == log.samples:
        return log
    metrics = {
        name: mean(values)
        for name in log.results.metrics
        if (values := [s.reduced[name] for s in samples if name in s.reduced])
    }
    return replace(log, samples=tuple(samples), results=replace(log.results, metrics=metrics))


class TrialLogSink(JsonLogSink):
    """Canonical Inspect log sink that never records a score for a discarded trial."""

    def on_eval_end(self, log):
        super().on_eval_end(unscore_discarded(log))


def result_row(record, scene, policy, started, control_hz):
    hz = control_hz or 20.0
    times = [e.t for e in record.events if e.kind == "inference"]
    actual_steps = record.metadata.get(
        "executed_control_steps", record.metadata.get("controller_totals", {}).get("actual_steps", len(record.steps))
    )
    # An episode with a decision that failed max_attempts consecutive attempts is
    # discarded: it is not evaluated and takes no part in any aggregate; the row
    # remains for audit.
    discarded = (
        record.status == "success" and record.termination_reason == MAX_ATTEMPTS_TERMINATION
    )
    row = {
        "scene_id": scene.id,
        "init": scene.metadata["init_state_index"],
        "seed": record.seed,
        "status": DISCARDED_STATUS if discarded else record.status,
        "error": record.error,
        "termination_reason": record.termination_reason
        or (record.status if record.status != "success" else "unknown"),
        "oracle_success": None if discarded else bool(
            record.steps
            and record.steps[-1].result.terminated
            and record.steps[-1].result.termination_reason == "success"
        ),
        "initially_solved": record.metadata.get("initially_solved", False),
        "steps": actual_steps,
        "recorded_steps": len(record.steps),
        "control_hz": hz,
        "sim_time_s": actual_steps / hz,
        "wall_time_s": time.monotonic() - started,
        "metadata": plain(record.metadata),
        "inference_wall_s": sum(record.inference_latencies),
        "inference_control_steps": times,
        "model_decision_sim_intervals_s": [(b - a) / hz for a, b in zip(times, times[1:])],
        "simulation_paused_during_inference": True,
        "action_audit": [
            {
                "t": step.t,
                "proposal_and_applied": plain(step.action.meta),
                "controller": plain(step.result.info),
            }
            for step in record.steps
        ],
    }
    for key in ("task_id", "task_slug", "difficulty", "paired_seed"):
        if key in scene.metadata:
            row[key] = scene.metadata[key]
    metrics = [step.result.info.get("task_metrics", {}) for step in record.steps]
    if not metrics:
        metrics = [record.metadata.get("simulator_reset", {}).get("task_metrics", {})]
    official_scores = [metric["official_score"] for metric in metrics if "official_score" in metric]
    if official_scores:
        row["official_score"] = float(official_scores[-1])
    scored = [metric for metric in metrics if "progress_score" in metric]
    if scored:
        last = scored[-1]
        row.update(progress_score=max(float(m["progress_score"]) for m in scored),
                   final_progress_score=float(last.get("final_progress_score", last.get("progress", 0.0))),
                   progress_peak_step=last.get("progress_peak_step"),
                   progress_raw=last.get("progress_raw"), peak_progress_raw=last.get("peak_progress_raw"),
                   progress_raw_min=last.get("progress_raw_min"), progress_raw_max=last.get("progress_raw_max"))
        for key in ("progress_peak_seconds", "progress_measure", "initial_progress_measure",
                    "initial_progress_raw"):
            if key in last:
                row[key] = last[key]
    if hasattr(policy, "metrics"):
        row["policy"] = policy.metrics()
        row["inference_wall_s"] = row["policy"]["inference_wall_s"]
        row["inference_control_steps"] = row["policy"].get("inference_control_steps", times)
        row["model_decision_sim_intervals_s"] = [
            d["sim_time_s"] for d in row["policy"].get("decision_intervals", [])
        ]
        row["model_decision_wall_intervals_s"] = [
            d["wall_time_s"] for d in row["policy"].get("decision_intervals", [])
        ]
    if hasattr(policy, "metadata"):
        meta = policy.metadata
        row["policy_metadata"] = plain(meta() if callable(meta) else meta)
    if getattr(policy, "trajectory_recorder", None) is not None:
        row["trajectory"] = {"path": str(policy.trajectory_recorder.path)}
    # steps counts every executed step; done-verification holds are also split out.
    row.update(episode_phases(record, oracle_success=row["oracle_success"]))
    if discarded:
        row["task_success_evaluated"] = False
    return row


def _main(argv, progress_holder):
    args = parse_args(argv)
    from agentic_framework.environments.libero.embodiment import LiberoEmbodiment
    from agentic_framework.harness.controller import FrameworkController

    model_profile = load_model_profile(args.model_profile) if args.model_profile else None
    preview_mode = args.mode == "preview"
    native_vla = args.policy_family != "llm"
    native_droid = native_vla and model_profile["environment"]["control_mode"] == "pd_joint_pos"
    h, k = args.h, args.k
    augmentation = AugmentationConfig.from_dict(args.augmentations)
    horizons = {ENV_ID: MANISKILL_HORIZON} if args.benchmark == "maniskill" else HORIZONS
    suites = list(horizons)[:4] if args.suite == "all" else args.suite.split(",")
    if any(s not in horizons for s in suites):
        raise ValueError(f"unknown suite for {args.benchmark}: {args.suite}")
    schedule = ScheduleConfig(
        h=h,
        k=k,
        control_interface=args.control_interface,
        motion_time_scale=args.motion_time_scale,
        max_motion_steps=args.max_motion_steps,
        max_steps=args.max_steps
        if args.max_steps is not None
        else horizons[suites[0]],
        auto_motion=AutoMotionConfig(**args.auto_motion),
    )
    if native_vla:
        if schedule.motion_time_scale != 1:
            raise ValueError("native model profiles require motion_time_scale=1 and retain their native action timing")
        validate_native_horizon(model_profile, h)
    # Only a language model's done call starts done verification; native VLA
    # policies never call done. The simulator time limit covers a hold that
    # starts on the last budgeted step, in preview as in the formal run.
    configured_verification_steps = (
        0 if native_vla else verification_steps(args.done_verification_seconds, args.control_hz)
    )
    # A preview evaluates no task, so it never executes verification holds.
    trial_verification_steps = 0 if preview_mode else configured_verification_steps
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    output = (
        args.output_dir or project_root() / "outputs" / f"{stamp}-{args.mode}"
    ).resolve()
    if output in (project_root(), PROJECT):
        raise ValueError("Select a run subdirectory, for example outputs/<run>")
    if output.exists() and any(output.iterdir()):
        raise ValueError("output already exists; select a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    progress = EvaluationProgress(output, max_steps=schedule.max_steps, control_hz=args.control_hz,
                                  verification_max_steps=trial_verification_steps)
    progress_holder["progress"] = progress
    if args.benchmark == "maniskill":
        from agentic_framework.environments.maniskill.embodiment import ManiSkillEmbodiment

        embodiment = ManiSkillEmbodiment(
            python=args.sim_python,
            observation_profile=args.observation_profile,
            control_mode="pd_joint_pos" if native_droid else "pd_ee_delta_pose",
            control_hz=args.control_hz,
            sim_hz=args.sim_hz,
            max_episode_steps=schedule.max_steps + configured_verification_steps,
            shader_pack=args.sim_shader,
            sim_backend=args.sim_backend,
            source_root=args.molmoact2_source,
            assets_root=args.maniskill_assets,
            worker_env={"CUDA_VISIBLE_DEVICES": str(args.gpu)} if args.gpu is not None else None,
        )
    else:
        embodiment = LiberoEmbodiment(
            benchmark=args.benchmark, python=args.sim_python,
            observation_profile=args.observation_profile,
            worker_env={"CUDA_VISIBLE_DEVICES": str(args.gpu)} if args.gpu is not None else None,
        )
    results = []
    try:
        scenes = []
        for suite in suites:
            if args.benchmark == "maniskill":
                scenes.extend(
                    build_maniskill_scenes(
                        suite=suite,
                        task_ids=args.task_ids,
                        trials=args.trials,
                        init_start=args.init_start,
                        seed=args.seed,
                        category=args.category,
                        **({"difficulty": args.difficulty} if suite == "ditto" else {}),
                    )
                )
            else:
                scenes.extend(
                    build_scenes(
                        select_tasks(embodiment, suite, args.task_ids, args.category),
                        trials=args.trials,
                        init_start=args.init_start,
                        seed=args.seed,
                    )
                )
        simulator_metadata = embodiment.metadata()
        demonstration_sha256 = None
        if args.demo:
            from agentic_framework.harness.demonstration import Demonstration

            # Validate the selected teacher projection before any trial starts.
            demonstration_sha256 = Demonstration(
                args.demo_path, observation_profile=args.observation_profile,
                demo_mode=args.demo_mode, image_max_side=args.demo_image_max_side,
                demo_content=args.demo_content,
            ).sha256
        trial_budgets = {
            scene.id: args.max_steps
            if args.max_steps is not None
            else scene.metadata.get("reference_horizon", horizons[scene.metadata["suite"]])
            for scene in scenes
        }
        config = {k: plain(v) for k, v in args.flat().items() if k != "output_dir"}
        config.update(
            h=h,
            k=k,
            augmentations=asdict(augmentation),
            schedule=asdict(schedule),
            effective_control={
                "interface": "native_action_chunk" if native_vla else motion_tool_name(schedule),
            },
            # Hold steps after a model done call (0: disabled or a native VLA policy).
            done_verification_steps=configured_verification_steps,
            source_sha256=source_fingerprint(),
            simulator=simulator_metadata,
            python=sys.version,
            packages={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
            demonstration_provenance_sha256=demonstration_sha256,
            policy_provenance_sha256=hashlib.sha256(args.policy_metadata.read_bytes()).hexdigest()
            if args.policy_metadata
            else None,
            configuration=group_config(config),
            model_profile_definition=plain(model_profile),
            dependency_source=json.loads((
                PROJECT / "source-manifest.json"
                if (PROJECT / "source-manifest.json").is_file()
                else PACKAGE_ROOT / "source-manifest.json"
            ).read_text())[
                "inspect_commit"
            ],
        )
        # Fingerprints in config.json are provenance records; outputs are never reused.
        write_json(output / "config.json", config)
        write_json(output / "manifest.json", [asdict(s) for s in scenes])
        print(f"Output: {output}; {len(scenes)} trial(s)", flush=True)
        if preview_mode:
            from agentic_framework.observability.experiment_preview import write_experiment_preview

            write_experiment_preview(output, config, scenes, trial_budgets=trial_budgets)
        for scene in scenes:
            progress.start_trial(scene.id, max_steps=trial_budgets[scene.id], control_hz=args.control_hz,
                                 verification_max_steps=trial_verification_steps)
            directory = output / "evals" / scene.id
            result_path = directory / "result.json"
            directory.mkdir(parents=True)
            steps = trial_budgets[scene.id]
            trial_schedule = replace(schedule, max_steps=steps)
            policy = None
            recorder = None
            row = None
            started = time.monotonic()
            capture = {}
            print(f"Starting {scene.id}: {scene.instruction}", flush=True)
            try:
                trial_embodiment = embodiment
                if getattr(args, "record_trajectory", False):
                    from agentic_framework.harness.trajectory_recorder import (
                        RecordingEmbodiment,
                        TrajectoryRecorder,
                    )

                    recorder = TrajectoryRecorder(directory, {
                        "scene_id": scene.id, "scene": plain(scene),
                        "benchmark": args.benchmark, "policy_family": args.policy_family,
                        "model": args.model if not native_vla else model_profile["id"],
                        "model_profile": plain(model_profile),
                        "schedule": asdict(trial_schedule),
                        # configured_steps equals config.json done_verification_steps;
                        # max_steps is this trial's executable hold (0 in preview).
                        "done_verification": {
                            "seconds": args.done_verification_seconds,
                            "configured_steps": configured_verification_steps,
                            "max_steps": trial_verification_steps,
                        },
                        "policy_seed": getattr(args, "policy_seed", None),
                        "repeat_id": getattr(args, "repeat_id", None),
                        "eval_seed": args.seed, "recording_mode": "dense",
                        "observation_profile": args.observation_profile,
                        "demo_mode": args.demo_mode,
                        "demo_content": args.demo_content,
                        "privilege_enabled": args.observation_profile == "privileged",
                        "source_sha256": hashlib.sha256(
                            json.dumps(config["source_sha256"], sort_keys=True).encode()
                        ).hexdigest(),
                    }, managed=True)
                    recorder.document["video"] = {"status": "pending"}
                    recorder.flush()
                    trial_embodiment = RecordingEmbodiment(embodiment, recorder)
                # Every language-model run uses the live embodiment's measured motion
                # profile, and the policy and controller share that one spec. Native
                # DROID joint chunks have no motion ActionSpec; native end-effector
                # chunks take the declared contract, whose pose limits they never use.
                action_spec = None if native_droid else embodiment.action_spec
                if not native_vla and action_spec is None:
                    raise ValueError(f"{args.benchmark} did not declare the LLM ActionSpec")
                trial_args = args
                if (scene.metadata.get("suite") == "ditto" and
                        args.policy_family == "molmoact2" and not preview_mode):
                    values = args.flat()
                    values["policy_seed"] = (
                        args.policy_seed + scene.metadata["init_state_index"]
                        if args.policy_seed is not None else scene.metadata["policy_seed"]
                    )
                    from argparse import Namespace

                    from agentic_framework.configuration.arguments import RunConfiguration
                    trial_args = RunConfiguration.from_namespace(Namespace(**values, config=args.config))
                policy = create_policy(trial_args, trial_schedule, augmentation, directory,
                                       None if native_vla else action_spec)
                if not preview_mode:
                    policy.runtime_progress = progress
                if recorder is not None:
                    policy.trajectory_recorder = recorder
                    metadata = getattr(policy, "metadata", {})
                    metadata = metadata() if callable(metadata) else metadata
                    provenance = metadata.get("provenance", {})
                    recorder.update_metadata(
                        model_metadata=metadata,
                        policy_seed=getattr(policy, "policy_seed", None),
                        model_revision=provenance.get("model_revision", provenance.get("revision", provenance.get("checkpoint_tree_sha256"))),
                    )
                if preview_mode and not args.preview_context:
                    from agentic_framework.harness.preview import preview_first_request

                    progress.phase("resetting")
                    preview = preview_first_request(
                        policy,
                        trial_embodiment,
                        scene,
                        eval_seed=scene.init_seed if args.benchmark == "maniskill" else args.seed,
                        directory=directory,
                    )
                    write_json(directory / "preview.json", preview)
                    if recorder is not None:
                        recorder.document["outcome"].update(
                            complete=True, status="success", termination_reason="preview",
                            initially_solved=preview["initially_solved"], executed_control_steps=0,
                            task_success_evaluated=False,
                        )
                        recorder.flush()
                    row = {
                        "scene_id": scene.id,
                        "status": "success",
                        "initially_solved": preview["initially_solved"],
                        "steps": 0,
                        "termination_reason": "preview",
                        "task_success_evaluated": False,
                        "api_calls": 0,
                        "wall_time_s": time.monotonic() - started,
                        "preview": preview,
                        "policy": policy.metrics(),
                    }
                    results.append(row)
                    write_json(result_path, row)
                    print(f"Preview saved for {scene.id}; 0 API calls, 0 policy steps", flush=True)
                else:
                    if native_droid:
                        from agentic_framework.harness.native_controller import (
                            NativeDroidController,
                        )

                        controller = NativeDroidController(policy)
                    else:
                        controller = FrameworkController(trial_schedule, spec=action_spec)
                    if trial_verification_steps:
                        controller = DoneVerificationController(
                            controller, action_budget=steps, steps=trial_verification_steps,
                            control_hz=args.control_hz, seconds=args.done_verification_seconds,
                            spec=action_spec,
                        )

                    def collect(record, _scene):
                        record.metadata["simulator_reset"] = embodiment.last_reset_info
                        capture["record"] = record

                    class CaptureSink(NullSink):
                        def on_trial_end(self, record):
                            record.metadata["simulator_reset"] = embodiment.last_reset_info
                            capture["record"] = record
                            if recorder is not None:
                                recorder.finish(record)

                    task = Task(
                        name=f"{args.benchmark}-{scene.metadata['suite']}",
                        scenes=[scene],
                        scorer=[episode_length()]
                        if preview_mode
                        else [success_at_end(), episode_length()],
                        # Policy actions stay capped at steps by the verification controller.
                        max_steps=steps + trial_verification_steps,
                    )
                    if preview_mode:
                        from agentic_framework.harness.preview import PreviewEmbodiment

                        trial_embodiment = PreviewEmbodiment(trial_embodiment, directory)
                    robot_eval(
                        task,
                        policy,
                        trial_embodiment,
                        controller=controller,
                        approver=ClampApprover(embodiment.info.action_space)
                        if args.approver == "clamp"
                        else None,
                        log_dir=str(directory),
                        seed=scene.init_seed if args.benchmark == "maniskill" else args.seed,
                        fail_on_error=True,
                        store_frames=args.record_trajectory,
                        store_actions=True,
                        sinks=[
                            progress,
                            TrialLogSink(str(directory)),
                            LiveLogSink(str(directory)),
                            CaptureSink(),
                        ],
                        before_scoring=collect,
                    )
                    record = capture["record"]
                    row = result_row(record, scene, policy, started, embodiment.info.control_hz)
                    if preview_mode:
                        row.pop("oracle_success", None)
                        preview = {
                            "kind": "scripted_context_preview",
                            "api_calls": 0,
                            "model_calls": 0,
                            "task_success_evaluated": False,
                            "scripted_steps": row["steps"],
                            "termination_reason": row["termination_reason"],
                            "observation": trial_embodiment.observation_artifact,
                            "requests": [
                                decision["backend"]["request_artifact"]
                                for decision in policy.decisions
                                if "request_artifact" in decision.get("backend", {})
                            ],
                        }
                        write_json(directory / "preview.json", preview)
                        row.update(
                            api_calls=0,
                            scripted_context_preview=True,
                            task_success_evaluated=False,
                            preview=preview,
                        )
                    results.append(row)
                    write_json(result_path, row)
            except BaseException as exc:
                if "record" in capture:
                    record = capture["record"]
                    record.status = (
                        "cancelled"
                        if record.status == "cancelled" or isinstance(exc, KeyboardInterrupt)
                        else "error"
                    )
                    record.error = record.error or f"{type(exc).__name__}: {exc}"
                    if recorder is not None:
                        recorder.finish(record)
                    failure_row = result_row(
                        record, scene, policy, started, embodiment.info.control_hz
                    )
                    if row is not None:
                        row.update(failure_row)
                    else:
                        row = failure_row
                        results.append(row)
                    write_json(result_path, row)
                if recorder is not None and "record" not in capture:
                    recorder.fail(exc)
                if preview_mode and row is None:
                    row = {
                        "scene_id": scene.id,
                        "status": "cancelled" if isinstance(exc, KeyboardInterrupt) else "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "steps": 0,
                        "wall_time_s": time.monotonic() - started,
                        "task_success_evaluated": False,
                        "api_calls": 0,
                    }
                    results.append(row)
                    write_json(result_path, row)
                if preview_mode:
                    observation_path = directory / "observation/manifest.json"
                    partial_preview = row.get("preview") or {
                        "kind": "failed_preview",
                        "observation": {
                            **json.loads(observation_path.read_text()),
                            "manifest_path": str(observation_path),
                        }
                        if observation_path.exists()
                        else None,
                        "requests": [
                            decision["backend"]["request_artifact"]
                            for decision in getattr(policy, "decisions", ())
                            if "request_artifact" in decision.get("backend", {})
                        ],
                    }
                    partial_preview.update(
                        api_calls=0,
                        model_calls=0,
                        task_success_evaluated=False,
                        scripted_steps=row["steps"] if args.preview_context else 0,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    row.update(
                        api_calls=0,
                        task_success_evaluated=False,
                        scripted_context_preview=args.preview_context,
                        preview=partial_preview,
                    )
                    row.pop("oracle_success", None)
                    write_json(directory / "preview.json", partial_preview)
                    write_json(result_path, row)
                write_json(
                    directory / "failure.json",
                    {
                        "type": type(exc).__name__,
                        "error": str(exc),
                        "wall_time_s": time.monotonic() - started,
                    },
                )
                raise
            finally:
                progress.phase("finalizing", pending_request_started_at=None)
                if policy is not None:
                    try:
                        policy.close()
                    except BaseException as exc:
                        cleanup = {
                            "type": type(exc).__name__,
                            "error": str(exc),
                            "stage": "policy_cleanup",
                        }
                        write_json(directory / "failure.json", cleanup)
                        if recorder is not None:
                            recorder.fail(exc)
                        if row is not None:
                            row.update(
                                status="error", error=f"policy cleanup: {type(exc).__name__}: {exc}"
                            )
                            write_json(result_path, row)
                if recorder is not None:
                    from agentic_framework.harness.trajectory_video import finalize_video

                    video = finalize_video(recorder, output, preview=preview_mode)
                    if row is not None:
                        row.setdefault("trajectory", {"path": str(recorder.path)})["video"] = video
                    if video["status"] == "error":
                        failure = {"stage": "video_export", "error": video["error"]}
                        write_json(directory / "video-failure.json", failure)
                        # Retain an inference/cancellation/cleanup error if one exists.
                        if row is not None:
                            row["recording_error"] = video["error"]
                            if row["status"] == "success":
                                row.update(status="error", error=f"video export: {video['error']}")
                        print(f"Video export failed for {scene.id}: {video['error']}",
                              file=sys.stderr, flush=True)
                if row is not None:
                    if preview_mode:
                        row.pop("oracle_success", None)
                    write_json(result_path, row)
                write_json(output / "results.json", results)
                progress.finish(row, phase="preview" if preview_mode and row is not None and row["status"] == "success" else None)
                if preview_mode:
                    write_experiment_preview(
                        output, config, scenes, results, trial_budgets=trial_budgets
                    )
            if row["status"] == DISCARDED_STATUS:
                print(
                    f"Discarded {scene.id}: a decision failed max_attempts attempts; "
                    "excluded from results",
                    flush=True,
                )
                continue
            if row["status"] != "success":
                print(f"Trial failed: {row.get('error')}", flush=True)
                return 1
            print(
                f"Finished {scene.id}: {row['termination_reason']}, {row['steps']} steps",
                flush=True,
            )
        # Discarded trials are reported only as a count and a list of scene ids.
        kept = [r for r in results if r["status"] != DISCARDED_STATUS]
        discarded_ids = [r["scene_id"] for r in results if r["status"] == DISCARDED_STATUS]
        summary = {
            "trials": len(kept),
            "discarded_trials": len(discarded_ids),
            "discarded_scene_ids": discarded_ids,
            "steps": sum(r["steps"] for r in kept),
            "policy_control_steps": sum(r.get("policy_control_steps", r["steps"]) for r in kept),
            "verification_control_steps": sum(r.get("verification_control_steps", 0) for r in kept),
            "verification_trials": sum(bool(r.get("verification_triggered")) for r in kept),
            "wall_time_s": sum(r["wall_time_s"] for r in kept),
            "model_calls": sum(r.get("policy", {}).get("model_calls", 0) for r in kept),
        }
        if preview_mode:
            summary.update(
                api_calls=0,
                model_calls=0,
                preview=True,
                task_success_evaluated=False,
                exported_trials=sum(r["status"] == "success" for r in kept),
                scripted_context_preview=args.preview_context,
                scripted_requests=sum(
                    r.get("policy", {}).get("scripted_requests", 0) for r in kept
                ),
                prepared_requests=(
                    sum(r.get("policy", {}).get("scripted_requests", 0) for r in kept)
                    if args.preview_context
                    else sum(bool(r.get("preview", {}).get("request")) for r in kept)
                ),
            )
            write_experiment_preview(output, config, scenes, results, trial_budgets=trial_budgets)
        else:
            summary.update(
                successes=sum(r["oracle_success"] for r in kept),
                verification_successes=sum(r.get("success_phase") == "verification" for r in kept),
                initially_solved=sum(r["initially_solved"] for r in kept),
                evaluable_trials=sum(not r["initially_solved"] for r in kept),
            )
        scored = [r for r in kept if "progress_score" in r]
        if scored:
            summary["mean_progress_score"] = sum(r["progress_score"] for r in scored) / len(scored)
            summary["mean_final_progress_score"] = sum(r["final_progress_score"] for r in scored) / len(scored)
            grouped = {}
            for row in scored:
                key = (row.get("task_id"), row.get("difficulty"))
                grouped.setdefault(key, []).append(row)
            summary["task_difficulty"] = [
                {"task_id": key[0], "difficulty": key[1], "trials": len(rows),
                 "successes": sum(r["oracle_success"] for r in rows),
                 "success_rate": sum(r["oracle_success"] for r in rows) / len(rows),
                 "mean_progress_score": sum(r["progress_score"] for r in rows) / len(rows),
                 "mean_final_progress_score": sum(r["final_progress_score"] for r in rows) / len(rows)}
                for key, rows in grouped.items()
            ]
        write_json(output / "summary.json", summary)
        print(json.dumps(summary), flush=True)
        if discarded_ids and not kept:
            # Launchers run one trial per job and need a discarded job to exit 0;
            # they fail a run without a valid trial. A direct run only warns.
            print(
                f"Every trial ({len(discarded_ids)}) was discarded: a decision failed all "
                "max_attempts attempts in each. Provider rejections (any HTTP status) and "
                "internal exceptions inside the backend call are resent, so check the recorded "
                "model errors for a misconfigured provider; this entry point still exits 0",
                file=sys.stderr,
                flush=True,
            )
        return 0
    finally:
        embodiment.close()


def main(argv=None):
    progress_holder = {}
    try:
        return _main(argv, progress_holder)
    except BaseException as exc:
        progress = progress_holder.get("progress")
        if progress is not None:
            progress.finish(phase="cancelled" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
