"""Suite embodiment with the existing DROID observation and action contracts."""
from __future__ import annotations

import os
import subprocess
from dataclasses import replace
from pathlib import Path

import agentic_framework.environments.maniskill.embodiment as upstream
from agentic_framework.common.paths import FRAMEWORK_ROOT

from ditto.catalog import SUITE_ID, TASK_BY_ID
from ditto.paths import project_path

from .scenes import list_suite_tasks


class SuiteEmbodiment(upstream.ManiSkillEmbodiment):
    """Isolate simulation and retain upstream Astra/native DROID adapters."""
    _worker_label = "Ditto Bench"

    def __init__(self, *, framework_root=None, **kwargs):
        super().__init__(**kwargs)
        self.framework_root = Path(framework_root or FRAMEWORK_ROOT)
        self.info = replace(self.info, name="maniskill-ditto-franka-droid")

    def list_tasks(self, suite=SUITE_ID, task_ids=None):
        if suite != SUITE_ID:
            raise ValueError(f"Unsupported suite: {suite}")
        if task_ids is not None and any(type(value) is not int or value not in TASK_BY_ID for value in task_ids):
            raise ValueError("Suite task IDs must be in 0..5")
        tasks = list_suite_tasks()
        return tasks if task_ids is None else [row for row in tasks if row["task_id"] in task_ids]

    def reset(self, scene, *, seed=None):
        from ditto.difficulties import normalize_difficulty
        self._has_reset = False
        result = self._rpc(
            "reset", env_id=scene.metadata["env_id"],
            seed=seed if seed is not None else scene.init_seed,
            instruction=scene.instruction,
            difficulty=normalize_difficulty(scene.metadata.get("difficulty", "easy")),
        )
        self._update_controller_metadata(result["info"]["controller"])
        observation = self._observation(result["observation"])
        self.last_reset_info = result["info"]
        self._has_reset = True
        self._terminated = bool(result["info"]["success"])
        return observation

    def _start(self):
        if self._process is not None:
            return
        worker = Path(__file__).with_name("worker.py")
        package_src = Path(__file__).resolve().parents[2]
        env = os.environ.copy()
        for name in list(env):
            if name.endswith(("_API_KEY", "_API_TOKEN")):
                env.pop(name, None)
        env.update({
            "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "MS_ASSET_DIR": str(project_path(os.environ.get("MS_ASSET_DIR", "data/maniskill"))),
            "DITTO_FRAMEWORK_ROOT": str(self.framework_root),
            "MOLMOACT2_SOURCE_ROOT": str(self.source_root),
            "DROID_ASSET_ROOT": str(self.assets_root),
            "PYTHONPATH": os.pathsep.join(filter(None, (str(package_src), env.get("PYTHONPATH")))),
        })
        env.update(self.worker_env)
        self._read_buffer.clear()
        self._process = subprocess.Popen(
            [self.python, "-B", "-u", str(worker),
             "--source-root", str(self.source_root), "--assets-root", str(self.assets_root),
             "--control-mode", self.control_mode, "--control-hz", str(int(self.control_hz)),
             "--sim-hz", str(self.sim_hz), "--max-episode-steps", str(self.max_episode_steps),
             "--shader-pack", self.shader_pack, "--sim-backend", self.sim_backend,
             "--observation-profile", self.observation_profile],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, env=env, bufsize=0,
        )
