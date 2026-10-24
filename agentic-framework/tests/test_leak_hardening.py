"""API keys never reach records, logs or child processes.

Covers record redaction, the request archive's authentication placeholder, credential
stripping for simulator workers, private provider provenance, and a canary rollout: a mocked
live trial (including a transport error) and a preview run with a canary key, after which no
file under the run directory contains the key.
"""

from __future__ import annotations

import base64
import hashlib
import importlib
import json
import os
import subprocess
import sys
import textwrap
import uuid
from pathlib import Path

import httpx
import numpy as np
import pytest

from agentic_framework.common import environment as environment_module
from agentic_framework.common.environment import without_credentials
from agentic_framework.common.redaction import redacted
from agentic_framework.environments.libero.embodiment import LiberoEmbodiment
from agentic_framework.harness import policy as policy_module
from agentic_framework.harness import run
from agentic_framework.harness.types import AgentContext
from agentic_framework.models.llm import providers
from agentic_framework.models.llm.errors import BackendError
from agentic_framework.models.llm.responses import ResponsesAgent
from agentic_framework.observability.api_audit import APIRequestRecorder

OFFICIAL = "https://api.openai.com/v1"
TOOL = {"type": "function", "name": "move_by", "description": "Move.",
        "parameters": {"type": "object"}, "strict": False}


def _call(call_id, name="move_by", arguments=None):
    arguments = arguments or {"deltas": {"dz": 0.001}, "note": "Raise slightly."}
    return {"type": "function_call", "id": f"fc_{call_id}", "call_id": call_id, "name": name,
            "arguments": json.dumps(arguments), "status": "completed"}


def _reply(*output):
    return {"id": "resp_1", "object": "response", "model": "model-a", "status": "completed",
            "output": list(output), "usage": {"input_tokens": 1, "output_tokens": 1,
                                              "total_tokens": 2}}


def _context(request_id="req-1"):
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    return AgentContext(
        request_id=request_id, instructions="Choose one robot tool call.", text="{}",
        images={"cam": image}, tools=(TOOL,),
        messages=({"role": "user", "content": [{"type": "text", "text": "{}"},
                                               {"type": "image", "name": "cam"}]},),
    )


@pytest.fixture
def provider_directory(tmp_path, monkeypatch):
    """Append a directory to the provider package path and rebuild the registry around it."""
    directory = tmp_path / "providers"
    directory.mkdir()
    monkeypatch.setattr(providers, "__path__", [*providers.__path__, str(directory)])
    providers.registry.cache_clear()
    loaded = set(sys.modules)
    yield directory
    providers.registry.cache_clear()
    for name in set(sys.modules) - loaded:
        if name.startswith(f"{providers.__name__}."):
            del sys.modules[name]


def _spec_module(directory: Path, module: str, **fields) -> None:
    arguments = ", ".join(f"{key}={value!r}" for key, value in fields.items())
    (directory / f"{module}.py").write_text(textwrap.dedent(f"""
        from agentic_framework.models.llm.providers import ProviderSpec

        PROVIDER = ProviderSpec({arguments})
    """), encoding="utf-8")
    importlib.invalidate_caches()


def _canary_provider(directory: Path) -> None:
    _spec_module(directory, "private_canary", name="canary", protocol="responses",
                 base_url_env="CANARY_LLM_BASE_URL", base_url_path="/v1",
                 key_env="CANARY_LLM_API_KEY")


# ------------------------------------------------------------------------------ redaction


def test_redacted_masks_secret_keys_tokens_and_images_in_nested_records():
    image = "data:image/png;base64,iVBORw0KGgo="
    record = {
        "Authorization": "Bearer abc", "headers": {"x-api-key": "k", "Proxy-Authorization": "p"},
        "notes": ["sent Bearer tok-123 to the endpoint", "key sk-abcdefgh12345678 echoed"],
        "frames": ({"image_url": image},), "count": 3, "ok": True, "missing": None,
    }
    assert redacted(record) == {
        "Authorization": "[redacted]",
        "headers": {"x-api-key": "[redacted]", "Proxy-Authorization": "[redacted]"},
        "notes": ["sent Bearer [redacted] to the endpoint", "key [redacted] echoed"],
        "frames": [{"image_url": f"[image sha256={hashlib.sha256(image.encode()).hexdigest()}]"}],
        "count": 3, "ok": True, "missing": None,
    }


# ---------------------------------------------------------------- request archive and records


def _archive(directory, recorder, url):
    request = httpx.Request("POST", url, json={"model": "m", "input": []})
    recorder(_context(), request)
    manifest = json.loads(next(directory.rglob("manifest.json")).read_text())
    row = json.loads((directory / "requests.jsonl").read_text().splitlines()[-1])
    readable = next(directory.rglob("request.md")).read_text()
    return manifest, row, readable


@pytest.mark.parametrize("scheme, env, line", [
    ("bearer", "OPENAI_API_KEY", "Authorization: Bearer ${OPENAI_API_KEY}"),
    ("x-api-key", "EXAMPLE_API_KEY", "x-api-key: ${EXAMPLE_API_KEY}"),
    (None, None, "(no authentication header)"),
])
def test_archives_name_the_auth_header_only_by_its_variable(tmp_path, scheme, env, line):
    recorder = APIRequestRecorder(tmp_path, auth_env=env, auth_scheme=scheme)
    manifest, row, readable = _archive(tmp_path, recorder, f"{OFFICIAL}/responses")
    assert manifest["version"] == 2 and manifest["auth_value_recorded"] is False
    assert manifest["url"] == row["url"] == f"{OFFICIAL}/responses"
    assert (manifest["auth_env"], manifest["auth_scheme"]) == (env, scheme)
    assert f"POST {OFFICIAL}/responses" in readable and line in readable


def test_recorder_rejects_a_key_where_its_variable_name_belongs(tmp_path):
    with pytest.raises(ValueError, match="must be an environment variable name"):
        APIRequestRecorder(tmp_path, auth_env="sk-0123456789abcdef")


def test_agent_log_error_rows_keep_the_failure_metadata(monkeypatch, tmp_path):
    """model/decisions.jsonl records why a request failed: its exception type or Retry-After."""
    monkeypatch.setenv("OPENAI_API_KEY", "row-key-0123456789")
    replies = [httpx.ConnectError("connection refused"),
               httpx.Response(429, headers={"Retry-After": "4"}, json={"error": {}})]

    def handle(request):
        reply = replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    agent = ResponsesAgent("gpt-6-astra", transport=httpx.MockTransport(handle), log_dir=tmp_path)
    for error in (httpx.ConnectError, BackendError):
        with pytest.raises(error):
            agent.generate(_context())
    rows = [json.loads(line) for line in (tmp_path / "decisions.jsonl").read_text().splitlines()]
    assert [(row["status"], row["error_type"]) for row in rows] == [
        ("error", "ConnectError"), ("error", "BackendError")]
    assert "retry_after_s" not in rows[0]["metadata"]
    assert (rows[1]["metadata"]["http_status"], rows[1]["metadata"]["retry_after_s"]) == (429, 4.0)


def test_every_decision_record_names_its_route(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "route-key-0123456789")
    agent = ResponsesAgent("gpt-6-astra", transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json=_reply(_call("call_1")))
    ))
    metadata = agent.generate(_context()).metadata
    assert {key: metadata[key] for key in (
        "backend", "record_version", "protocol", "provider", "model_selector", "wire_model",
        "endpoint", "api_key_env",
    )} == {
        "backend": "responses", "record_version": 2, "protocol": "responses",
        "provider": "openai", "model_selector": "openai/gpt-6-astra",
        "wire_model": "gpt-6-astra", "endpoint": OFFICIAL, "api_key_env": "OPENAI_API_KEY",
    }


# --------------------------------------------------------------- credentials in subprocesses


ENVIRONMENT = {
    "PATH": "/usr/bin", "HOME": "/tmp/home", "CUDA_VISIBLE_DEVICES": "3",
    "TOKENIZERS_PARALLELISM": "false", "MS_ASSET_DIR": "/data/maniskill",
    "OPENAI_API_KEY": "sk-0123456789abcdef", "anthropic_api_key": "lower-case",
    "SERVICE_API_TOKEN": "t", "HF_TOKEN": "hf", "GITHUB_TOKEN": "gh",
    "DB_PASSWORD": "p", "CLIENT_SECRET": "s", "MY_SECRET_VALUE": "v",
    "OPENAI_BASE_URL": OFFICIAL, "CANARY_LLM_BASE_URL": "https://canary.invalid/p",
    # Endpoints no registered provider declares, such as a provider whose module is absent.
    "UNDECLARED_BASE_URL": "https://undeclared.invalid/v1", "lower_base_url": "https://l.invalid",
}
KEPT = {"PATH", "HOME", "CUDA_VISIBLE_DEVICES", "TOKENIZERS_PARALLELISM", "MS_ASSET_DIR"}


def test_without_credentials_strips_keys_tokens_secrets_and_provider_variables(
    provider_directory,
):
    _canary_provider(provider_directory)
    assert {"OPENAI_BASE_URL", "CANARY_LLM_BASE_URL"} <= environment_module.provider_variables()
    assert set(without_credentials(ENVIRONMENT)) == KEPT
    assert without_credentials(ENVIRONMENT)["CUDA_VISIBLE_DEVICES"] == "3"


class _Spawned(Exception):
    pass


def test_ffmpeg_probe_receives_no_credentials(monkeypatch, provider_directory):
    from agentic_framework.observability import videos

    _canary_provider(provider_directory)
    captured = {}

    def probe(command, **kwargs):
        captured.update(kwargs["env"])
        return subprocess.CompletedProcess(command, 0, " V..... libx264  H.264\n", "")

    for name, value in ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("VLM4ROBOTICS_FFMPEG", sys.executable)
    monkeypatch.setattr(videos.subprocess, "run", probe)
    assert videos._find_ffmpeg() == sys.executable
    assert KEPT <= set(captured) and not set(ENVIRONMENT) - KEPT & set(captured)


def _captured_worker_environment(monkeypatch, provider_directory, module, embodiment):
    """Start the embodiment's worker with Popen intercepted; return the child environment."""
    _canary_provider(provider_directory)
    captured = {}

    def popen(command, **kwargs):
        captured.update(kwargs["env"])
        raise _Spawned

    for name, value in ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    with pytest.raises(_Spawned):
        embodiment._start()
    return captured


def test_libero_worker_receives_no_credentials(monkeypatch, provider_directory, tmp_path):
    from agentic_framework.environments.libero import embodiment as module

    (tmp_path / "config.yaml").write_text("{}\n")
    embodiment = LiberoEmbodiment(image_size=4, worker_env={"LIBERO_CONFIG_PATH": str(tmp_path)})
    env = _captured_worker_environment(monkeypatch, provider_directory, module, embodiment)
    assert KEPT <= set(env) and not set(ENVIRONMENT) - KEPT & set(env)


def test_maniskill_worker_receives_no_credentials(monkeypatch, provider_directory, tmp_path):
    from agentic_framework.environments.maniskill import embodiment as module

    (tmp_path / "sim_eval/robots").mkdir(parents=True)
    (tmp_path / "sim_eval/robots/franka_droid.py").write_text("")
    embodiment = module.ManiSkillEmbodiment(source_root=tmp_path, assets_root=tmp_path)
    env = _captured_worker_environment(monkeypatch, provider_directory, module, embodiment)
    assert {"PATH", "HOME", "CUDA_VISIBLE_DEVICES"} <= set(env)
    assert not set(ENVIRONMENT) - KEPT & set(env)


def test_ditto_worker_receives_no_credentials(monkeypatch, provider_directory, tmp_path):
    from ditto.integration import embodiment as module

    embodiment = module.SuiteEmbodiment(source_root=tmp_path, assets_root=tmp_path)
    env = _captured_worker_environment(monkeypatch, provider_directory, module, embodiment)
    assert {"PATH", "HOME", "CUDA_VISIBLE_DEVICES", "DITTO_FRAMEWORK_ROOT"} <= set(env)
    assert not set(ENVIRONMENT) - KEPT & set(env)


# ------------------------------------------------------------------------------ provenance


def test_private_provider_modules_are_fingerprinted_apart(tmp_path):
    providers_dir = tmp_path / "src/agentic_framework/models/llm/providers"
    providers_dir.mkdir(parents=True)
    for name in ("__init__.py", "openai.py", "private_example.py", "private_helpers.py"):
        (providers_dir / name).write_text(f"# {name}\n")
    (tmp_path / "src/agentic_framework/private_elsewhere.py").write_text("# public\n")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/a.json").write_text("{}")
    public, private = run.source_fingerprints(tmp_path)
    prefix = "src/agentic_framework/models/llm/providers/"
    assert set(private) == {prefix + "private_example.py", prefix + "private_helpers.py"}
    assert set(public) == {prefix + "__init__.py", prefix + "openai.py",
                           "src/agentic_framework/private_elsewhere.py", "configs/a.json"}
    assert private[prefix + "private_example.py"] == hashlib.sha256(
        b"# private_example.py\n").hexdigest()
    # The real tree: no private module is ever part of the public fingerprint.
    public, private = run.source_fingerprints()
    assert public and not any("/private_" in path for path in public)
    assert all(path.startswith(prefix + "private_") for path in private)


def test_builds_exclude_private_provider_files():
    """The root .gitignore's anchored patterns do not reach hatchling; pyproject excludes them."""
    import tomllib

    project = Path(run.__file__).resolve().parents[3]
    build = tomllib.loads((project / "pyproject.toml").read_text())["tool"]["hatch"]["build"]
    assert run.PRIVATE_SOURCES in build["exclude"] and "tests/private" in build["exclude"]


# ------------------------------------------------------------------------ canary rollout


class _OfflineLibero(LiberoEmbodiment):
    """A LIBERO embodiment whose worker calls are answered in process (no simulator)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **{**kwargs, "image_size": 4})
        self._steps = 0

    @staticmethod
    def _observation_data():
        image = {"data": base64.b64encode(bytes(48)).decode(), "shape": [4, 4, 3]}
        return {
            "images": {"agentview": image, "robot0_eye_in_hand": image},
            "state": {"eef_pos": [0.0, 0.0, 0.2], "eef_quat": [0.0, 0.0, 0.0, 1.0],
                      "gripper_width": [0.04], "joint_pos": [0.0] * 7},
            "sim_time": 0.0, "instruction": "Lift the cube.", "extra": {},
        }

    def _rpc(self, method, **params):
        if method == "list_tasks":
            return [{"suite": params["suite"], "task_id": 0, "num_init_states": 1,
                     "instruction": "Lift the cube."}]
        if method == "metadata":
            return {"simulator": "offline"}
        if method == "reset":
            self._steps = 0
            return {"info": {"success": False}, "observation": self._observation_data()}
        if method == "step":
            self._steps += 1
            return {"observation": self._observation_data(), "reward": 0.0,
                    "terminated": False, "termination_reason": None, "truncated": False,
                    "info": {"success": False}}
        if method == "recording_state":
            return {
                "schema_version": 1, "adapter": "offline", "frame": "world",
                "position_unit": "m", "quaternion_order": "xyzw", "policy_visible": False,
                "control_step": self._steps, "policy_time_s": self._steps / self.info.control_hz,
                "simulation_time_s": 0.0,
                "objects": {"cube": {"position": [0.0, 0.0, 0.0],
                                     "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0], "kind": "object",
                                     "source_body": "cube", "roles": [], "joint_positions": {}}},
            }
        raise ValueError(f"the offline simulator does not support {method}")

    def close(self):
        pass


def test_canary_key_never_reaches_any_run_file(tmp_path, monkeypatch, capfd):
    from agentic_framework.environments.libero import embodiment as libero

    key = f"canary-key-{uuid.uuid4().hex}"
    for name in list(os.environ):
        monkeypatch.delenv(name)
    monkeypatch.setenv("OPENAI_API_KEY", key)
    monkeypatch.setattr(libero, "LiberoEmbodiment", _OfflineLibero)
    monkeypatch.setattr(policy_module, "_sleep", lambda seconds: None)
    sent = []
    replies = [
        httpx.ConnectError("[Errno -2] Name or service not known"),
        _reply(_call("call_1")),
        _reply(_call("call_2", "done", {"summary": "Lifted.", "hindsight": "none"})),
    ]

    def handle(request):
        sent.append(request)
        reply = replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return httpx.Response(200, json=reply)

    initialize = ResponsesAgent.__init__

    def with_transport(self, *args, **kwargs):
        initialize(self, *args, **{**kwargs, "transport": httpx.MockTransport(handle)})

    monkeypatch.setattr(ResponsesAgent, "__init__", with_transport)
    common = ["--benchmark", "libero", "--suite", "libero_spatial", "--task-ids", "0",
              "--model", "gpt-6-astra", "--record-trajectory", "--max-steps", "10"]
    root = tmp_path / "runs"
    assert run.main(["--mode", "agent", *common, "--output-dir", str(root / "live")]) == 0
    assert run.main(["--mode", "preview", "--preview-context", *common,
                     "--output-dir", str(root / "preview")]) == 0
    captured = capfd.readouterr()
    (root / "eval.log").write_text(captured.out + captured.err)

    # The canary was really used: three live requests carried the key.
    assert replies == [] and len(sent) == 3
    assert all(request.headers["authorization"] == f"Bearer {key}" for request in sent)
    decisions = [json.loads(line) for line in next(root.glob("live/evals/*/decisions.jsonl"))
                 .read_text().splitlines()]
    assert decisions[0]["error_kind"] == "transport"
    manifests = [json.loads(path.read_text()) for path in root.rglob("api-requests/*/manifest.json")]
    assert {manifest["mode"] for manifest in manifests} == {"live", "preview"}
    files = [path for path in root.rglob("*") if path.is_file()]
    assert len(files) > 20 and any(path.name == "trajectory.json" for path in files)
    for path in files:
        assert key.encode() not in path.read_bytes(), path
