"""Custom task selection on the existing framework's isolated DROID RPC worker.

Only this process loads SAPIEN/torch. Upstream source and robot assets retain
all original integrity checks. Control observations exclude scene truth and task diagnostics. An explicitly
selected privileged profile uses the shared ManiSkill adapter and schema.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

from ditto.catalog import SUITE_ID, TASK_BY_ENV, TASKS
from ditto.difficulties import normalize_difficulty
from ditto.paths import project_path


def load_upstream_worker():
    root = project_path(os.environ.get("DITTO_FRAMEWORK_ROOT", "agentic-framework"))
    path = root / "src/agentic_framework/environments/maniskill/worker.py"
    spec = importlib.util.spec_from_file_location("_ditto_upstream_worker", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load framework simulator worker: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


upstream = load_upstream_worker()


def json_metric(value):
    if isinstance(value, dict):
        return {str(key): json_metric(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_metric(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("Task diagnostics must be finite")
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.dtype.kind not in "biuf":
        raise ValueError(f"Unsupported task metric dtype: {value.dtype}")
    if value.dtype.kind in "if" and not np.isfinite(value).all():
        raise ValueError("Task diagnostics must be finite")
    return value.item() if value.size == 1 else value.tolist()


def suite_fingerprint():
    package = Path(__file__).resolve().parents[1]
    root = package.parents[1]
    paths = sorted(package.rglob("*.py"))
    for name in ("pyproject.toml", "assets/manifest.json", "assets/catalog.json"):
        candidate = root / name
        if candidate.is_file():
            paths.append(candidate)
    return {
        "suite": SUITE_ID, "root": str(root),
        "sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in paths if "__pycache__" not in path.parts},
        "upstream_worker_sha256": hashlib.sha256(Path(upstream.__file__).read_bytes()).hexdigest(),
    }


class SuiteSimulator(upstream.Simulator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import ditto.envs  # noqa: F401  registers the six task classes
        self.active_env_id = None
        self.active_difficulty = None
        self.suite_source = suite_fingerprint()

    def _ensure_env(self, env_id=None, difficulty=None):
        env_id = env_id or self.active_env_id or TASKS[0].env_id
        difficulty = normalize_difficulty(difficulty or self.active_difficulty or "easy")
        if env_id not in TASK_BY_ENV:
            raise ValueError(f"Unknown suite environment: {env_id}")
        if self.env is not None and (env_id, difficulty) != (self.active_env_id, self.active_difficulty):
            self.env.close()
            self.env = None
        if self.env is None:
            self.env = self.gym.make(
                env_id, obs_mode="rgb", control_mode=self.control_mode,
                render_mode="rgb_array", max_episode_steps=self.max_episode_steps,
                reward_mode="none", sensor_configs=dict(shader_pack=self.shader_pack),
                sim_config=dict(sim_freq=self.sim_hz, control_freq=self.control_hz),
                sim_backend=self.sim_backend, num_envs=1,
                difficulty=difficulty,
                randomize=os.environ.get("DITTO_TASK_RANDOMIZE", "0") == "1",
            )
            self.active_env_id = env_id
            self.active_difficulty = difficulty
        self.controller_info = self._inspect_controller()

    def list_tasks(self):
        from dataclasses import asdict
        return [dict(asdict(task), suite=SUITE_ID, num_init_states=None,
                     configured_max_steps=self.max_episode_steps) for task in TASKS]

    def _task_metadata(self):
        from ditto.assets import Part, primitive_mesh_path
        env = self.env.unwrapped
        asset_dir = Path(env.asset_dir).resolve()
        files = set()
        for record in env.asset_records:
            directory = Path(record["directory"])
            # Concurrent asset writers publish PID-specific .tmp files atomically.
            # Only committed files belong in the reproducibility inventory.
            files.update(path for path in directory.rglob("*")
                         if not path.name.endswith(".tmp") and path.is_file())
            spec = json.loads((directory / "spec.json").read_text())
            for part in [*spec["parts"], *spec.get("collision_parts", [])]:
                primitive = primitive_mesh_path(Part(**part), asset_dir)
                if primitive.is_file():
                    files.add(primitive)
        inventory = {
            str(path.relative_to(asset_dir)): {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in sorted(files)
        }
        # Record sampled fixture poses for audit/replay separately from the
        # canonical variant spec. These metadata never enter policy observations.
        task_names = {record["name"] for record in env.asset_records}
        objects = list(env.scene.actors.values())
        for articulation in env.scene.articulations.values():
            objects.extend(articulation.links)
        actor_world_poses = {
            obj.name: json_metric(obj.pose.raw_pose)
            for obj in objects
            if obj.name in task_names or any(obj.name.endswith("_" + name) for name in task_names)
        }
        return {
            "asset_dir": str(asset_dir), "assets": env.asset_records,
            "asset_files": inventory,
            "randomize": env.randomize, "difficulty": env.difficulty,
            "robot_base_position": list(env.robot_base_p),
            "task_spec": json_metric(env.task_spec),
            "actor_world_poses": actor_world_poses,
            "articulation_root_poses": {
                name: json_metric(articulation.pose.raw_pose)
                for name, articulation in env.scene.articulations.items()
            },
        }

    def metadata(self):
        result = super().metadata()
        result.update(
            suite=SUITE_ID, env_id=self.active_env_id, task_catalog=self.list_tasks(),
            suite_source=self.suite_source,
            initial_task_configuration=self._task_metadata(),
            task_source="custom ditto; official MolmoAct2 robot/controllers reused",
            privileged_task_metrics=(
                "included in unified privileged.evaluation by explicit observation profile"
                if self.observation_profile == "privileged"
                else "stored in evaluation step info only; excluded from model observations"
            ),
        )
        return result

    def reset(self, env_id=None, seed=None, instruction=None, difficulty="easy"):
        env_id = env_id or TASKS[0].env_id
        if env_id not in TASK_BY_ENV:
            raise ValueError(f"Unknown suite environment: {env_id}")
        task = TASK_BY_ENV[env_id]
        self._ensure_env(env_id, difficulty)
        result = super().reset(env_id=env_id, seed=seed, instruction=instruction or task.instruction)
        result["info"].update(
            suite=SUITE_ID, task_id=task.task_id, task_slug=task.slug,
            difficulty=self.active_difficulty,
            initialization="seeded custom task reset using official DROID robot",
            suite_source=self.suite_source,
            task_configuration=self._task_metadata(),
            task_metrics=json_metric(self.env.unwrapped.evaluate()),
        )
        return result

    def _controller_audit(self):
        result = super()._controller_audit()
        # Persist measured joints in the small per-step audit even when the
        # image-bearing policy transcript exceeds the framework's size limit.
        actual = upstream.vector(self.env.unwrapped.agent.robot.qpos, 13, "measured qpos")
        result["actual_qpos"] = actual.tolist()
        if "observed_arm_target_qpos" in result:
            target = np.asarray(result["observed_arm_target_qpos"])
            result["arm_tracking_error_rad"] = (actual[:7] - target).tolist()
        return result

    def step(self, action):
        result = super().step(action)
        # evaluate is required to be pure: it cannot advance success dwell time.
        result["info"]["task_metrics"] = json_metric(self.env.unwrapped.evaluate())
        return result


def main():
    upstream.Simulator = SuiteSimulator
    upstream.main()


if __name__ == "__main__":
    main()
