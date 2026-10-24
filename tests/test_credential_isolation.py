"""Launcher credential handling: plans never hold a key, and child processes start without one.

Native model servers and every evaluator start without credentials; a live LLM evaluator gets
back only the variables its route reads.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from scripts import eval as launcher
from scripts import eval_parallel
from scripts import workspace as w

ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.path.normpath(sys.executable)
KEY = "sk-launchercanary0123456789"
CREDENTIALS = {
    "OPENAI_API_KEY": KEY, "OPENAI_BASE_URL": "https://api.openai.com/v1",
    "HF_TOKEN": "hf-token-value", "SERVICE_API_TOKEN": "t", "DB_PASSWORD": "p",
    "CLIENT_SECRET": "s",
}


def _config(tmp_path, mode="agent", **policy):
    document = {
        "evaluation": {"bench": "ditto", "mode": mode},
        "environment": {"task_ids": "0", "trials": 1},
        "output": {"directory": str(tmp_path / "runs/{timestamp}")},
        **({"policy": policy} if policy else {"llm": {"model": "openai/gpt-6-astra"}}),
    }
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _dry_run(config, home, **variables):
    result = subprocess.run(
        [PYTHON, "-B", "scripts/eval.py", str(config), "--dry-run", "--python", PYTHON],
        cwd=ROOT, capture_output=True, text=True, check=False, timeout=120,
        env={"PATH": os.environ.get("PATH", os.defpath), "HOME": str(home),
             "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", **variables},
    )
    assert result.returncode == 0, result.stderr
    return result


def test_dry_run_never_prints_a_key(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    result = _dry_run(_config(tmp_path), home, OPENAI_API_KEY=KEY)
    assert KEY not in result.stdout + result.stderr
    assert json.loads(result.stdout)["llm_route"]["api_key_env"] == "OPENAI_API_KEY"


@pytest.fixture
def launched(monkeypatch):
    """Run launcher.main in process; record whether execution (GPU allocation) was reached."""
    calls = []
    monkeypatch.setattr(launcher, "run_plan", lambda plan, **_: calls.append(plan) or 0)
    for name in CREDENTIALS:
        monkeypatch.delenv(name, raising=False)
    return calls


def test_live_llm_plan_never_holds_the_key(tmp_path, launched, monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    assert launcher.main([str(_config(tmp_path)), "--python", PYTHON]) == 0
    assert len(launched) == 1
    assert KEY not in json.dumps(launched[0]) + capsys.readouterr().out


def test_previews_need_no_key(tmp_path, launched):
    assert launcher.main([str(_config(tmp_path, mode="preview")), "--python", PYTHON]) == 0
    assert len(launched) == 1


def test_native_model_server_environment_has_no_credentials(tmp_path, monkeypatch):
    for name, value in CREDENTIALS.items():
        monkeypatch.setenv(name, value)
    _, env = w.command(argparse.Namespace(
        command="serve", model="molmoact2-droid", gpu=None, python=None, port=None,
        output_dir=tmp_path / "server", check_only=True,
    ), [])
    assert not set(CREDENTIALS) & set(env)
    assert env["PATH"] == os.environ["PATH"]


class _Spawned(Exception):
    pass


ROUTE_VARIABLES = {"OPENAI_API_KEY", "OPENAI_BASE_URL"}


@pytest.mark.parametrize("model_profile, mode, keeps", [
    (None, "agent", ROUTE_VARIABLES),
    (None, "preview", set()),
    ("molmoact2-droid", "vla", set()),
])
def test_only_live_llm_evaluators_keep_their_route_variables(tmp_path, monkeypatch,
                                                             model_profile, mode, keeps):
    """A live LLM evaluator keeps only what its route reads; every other evaluator nothing."""
    captured = {}

    def popen(command, **kwargs):
        captured.update(kwargs["env"])
        raise _Spawned

    for name, value in CREDENTIALS.items():
        monkeypatch.setenv(name, value)
    # Another provider's key and endpoint never reach this route's evaluator.
    monkeypatch.setenv("OTHER_LLM_API_KEY", "other-key-0123456789")
    monkeypatch.setenv("OTHER_LLM_BASE_URL", "https://other.invalid/p")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(eval_parallel.subprocess, "Popen", popen)
    route = {"selector": "openai/gpt-6-astra", "api_key_env": "OPENAI_API_KEY",
             "base_url_env": "OPENAI_BASE_URL"}
    plan = {"output_dir": str(tmp_path), "cwd": str(tmp_path), "command": [PYTHON, "-m", "x"],
            "model_profile": model_profile, "server": None,
            "llm_route": None if model_profile else route,
            "resolved": {"run": {"mode": mode}, "execution": {"max_steps": 10}}}
    job = {"id": "job-0", "suite": "ditto", "task_id": 0, "init_start": 0, "difficulty": "easy"}
    status = eval_parallel._run_job(plan, job, {"id": "w0", "devices": ["0"]},
                                    threading.Event())
    assert status["error"].startswith("_Spawned")
    assert captured["CUDA_VISIBLE_DEVICES"] == "0"
    assert set(CREDENTIALS) & set(captured) == keeps
    assert all(captured[name] == CREDENTIALS[name] for name in keeps)
    assert not {"OTHER_LLM_API_KEY", "OTHER_LLM_BASE_URL"} & set(captured)
