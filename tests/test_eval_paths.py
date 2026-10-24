"""The launcher-only ``paths`` group and its provenance.

Plans are built in process and inspected without launching a server, evaluator or simulator;
process creation is replaced by a recorder. Every file a test writes is under ``tmp_path``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from scripts import eval as launcher
from scripts import eval_config, eval_parallel
from scripts import workspace as w

load_experiment = eval_config.load_experiment
ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.path.normpath(sys.executable)
CACHE_VARIABLES = ("HF_HOME", "TORCH_HOME", "UV_CACHE_DIR", "PIP_CACHE_DIR",
                   "UV_PYTHON_INSTALL_DIR")
RESOURCE_VARIABLES = (*w.RESOURCE_DEFAULTS, "DITTO_ASSET_DIR")
AGENT = {"evaluation": {"bench": "ditto", "mode": "agent"},
         "environment": {"task_ids": "0", "trials": 1}}
MOLMOACT2 = {"evaluation": {"bench": "ditto", "mode": "vla"},
             "environment": {"task_ids": "0", "trials": 1},
             "policy": {"model_profile": "molmoact2-droid"}}
PI05 = {"evaluation": {"bench": "molmoact2-maniskill", "mode": "vla"},
        "environment": {"task_ids": "0", "trials": 1},
        "policy": {"model_profile": "pi05-droid"}}


class _Spawned(Exception):
    """Raised instead of starting a process."""


@pytest.fixture(autouse=True)
def shell(monkeypatch):
    """Start every test from a shell that sets no resource location."""
    for name in (*RESOURCE_VARIABLES, "AGENTIC_PROJECT_ROOT", "CUDA_VISIBLE_DEVICES",
                 launcher.REEXEC_MARKER):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _write(path: Path, document) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _config(tmp_path: Path, base: dict, **groups) -> Path:
    document = {**base, "output": {"directory": str(tmp_path / "runs/{timestamp}")}, **groups}
    return _write(tmp_path / "configs" / "experiment.json", document)


def _plan(config: Path, *options: str) -> dict:
    args, extra = launcher.parse_arguments([str(config), "--python", PYTHON, *options])
    return launcher.build_plan(args, extra)


def _site(tmp_path: Path) -> dict:
    """A paths group naming every resource outside the checkout, as absolute paths."""
    site = tmp_path / "site"
    return {
        "framework_python": str(site / "envs/framework/bin/python"),
        "simulator_python": {"maniskill": str(site / "envs/maniskill/bin/python"),
                             "libero": str(site / "envs/libero/bin/python")},
        "server_python": {"molmoact2": str(site / "envs/molmoact2/bin/python"),
                          "openpi": str(site / "envs/openpi/bin/python")},
        "checkpoints": {"molmoact2-droid": str(site / "models/molmoact2-droid"),
                        "pi05-droid": str(site / "models/pi05-droid")},
        "sources": {"molmoact2": str(site / "src/molmoact2"), "openpi": str(site / "src/openpi")},
        "droid_assets": str(site / "data/droid"),
        "maniskill_asset_dir": str(site / "data/maniskill"),
        "ditto_asset_dir": str(site / "data/ditto"),
        "libero_config": str(site / "config/libero"),
        "openpi_data_home": str(site / "models/openpi"),
        "cache_root": str(site / "cache"),
        "outputs_root": str(site / "runs"),
    }


def _records(plan: dict) -> dict:
    return {key: (item["value"], item["source"]) for key, item in plan["resolved"]["paths"].items()}


# Loading: value forms, include files and validation.


def test_inline_paths_resolve_against_the_config_and_keep_interpreter_symlinks(tmp_path, shell):
    home = tmp_path / "home"
    shell.setenv("HOME", str(home))
    real = tmp_path / "real"
    (real / "bin").mkdir(parents=True)
    (real / "bin/python3").write_text("", encoding="utf-8")
    directory = tmp_path / "configs"
    (directory / "venv/bin").mkdir(parents=True)
    (directory / "venv/bin/python").symlink_to(real / "bin/python3")
    (directory / "linked").symlink_to(real, target_is_directory=True)

    loaded = load_experiment(_write(directory / "experiment.json", {
        "paths": {"framework_python": "venv/bin/python",
                  "simulator_python": {"maniskill": "~/envs/sim/bin/python"},
                  "droid_assets": "linked/assets", "cache_root": "~/cache",
                  "checkpoints": {"molmoact2-droid": "../models/../models/m"}},
    }))

    assert loaded["paths_file"] is None
    assert loaded["paths"] == {
        # Resolving a venv interpreter symlink would select its base interpreter.
        "framework_python": str(directory.resolve() / "venv/bin/python"),
        "simulator_python": {"maniskill": str(home / "envs/sim/bin/python")},
        "droid_assets": str(real.resolve() / "assets"),
        "cache_root": str(home.resolve() / "cache"),
        "checkpoints": {"molmoact2-droid": str(tmp_path.resolve() / "models/m")},
    }
    assert "paths" not in loaded["framework"]


def test_a_shared_paths_file_resolves_against_its_own_directory(tmp_path):
    site = _write(tmp_path / "site/paths.json", {
        "framework_python": "envs/framework/bin/python", "sources": {"openpi": "../src/openpi"},
    })
    first = load_experiment(_write(tmp_path / "configs/a/one.json", {"paths": "../../site/paths.json"}))
    second = load_experiment(_write(tmp_path / "other/two.json", {"paths": str(site)}))

    for loaded in (first, second):
        assert loaded["paths_file"] == str(site.resolve())
        assert loaded["paths"] == {
            "framework_python": str(site.parent.resolve() / "envs/framework/bin/python"),
            "sources": {"openpi": str(tmp_path.resolve() / "src/openpi")},
        }


@pytest.mark.parametrize(("paths", "field"), [
    ({"python": "x"}, "paths.python"),
    ({"simulator_python": {"robodojo": "x"}}, "paths.simulator_python.robodojo"),
    ({"server_python": {"pi05": "x"}}, "paths.server_python.pi05"),
    ({"sources": {"libero": "x"}}, "paths.sources.libero"),
    # No implicit rules: a prefix map or a per-variable override is not a key.
    ({"prefixes": {"/old": "/new"}}, "paths.prefixes"),
    ({"MS_ASSET_DIR": "x"}, "paths.MS_ASSET_DIR"),
])
def test_unknown_paths_keys_are_rejected(tmp_path, paths, field):
    with pytest.raises(ValueError, match=rf"^Unknown configuration field: {field}$"):
        load_experiment(_write(tmp_path / "experiment.json", {"paths": paths}))
    _write(tmp_path / "site.json", paths)
    with pytest.raises(ValueError, match=rf"^Unknown configuration field: {field} \(in .*site.json\)$"):
        load_experiment(_write(tmp_path / "experiment.json", {"paths": "site.json"}))


@pytest.mark.parametrize(("paths", "message"), [
    ({"cache_root": 3}, r"^paths\.cache_root must be a nonempty path string$"),
    ({"cache_root": " "}, r"^paths\.cache_root must be a nonempty path string$"),
    ({"cache_root": None}, r"^paths\.cache_root must be a nonempty path string$"),
    ({"checkpoints": {"pi05-droid": ""}}, r"^paths\.checkpoints\.pi05-droid must be a nonempty"),
    ({"simulator_python": "x"}, r"^paths\.simulator_python must be a JSON object$"),
    ([], r"^paths must be a JSON object or a path to a JSON file holding one$"),
    ("", r"^Configured paths must be nonempty strings$"),
    ("missing.json", r"^Cannot read the paths file .*missing\.json: No such file"),
])
def test_invalid_paths_values_are_rejected(tmp_path, paths, message):
    with pytest.raises(ValueError, match=message):
        load_experiment(_write(tmp_path / "experiment.json", {"paths": paths}))


@pytest.mark.parametrize(("contents", "message"), [
    ("[]", r"^The paths file .*site\.json must hold a JSON object$"),
    ('{"cache_root": "a", "cache_root": "b"}',
     r"^Invalid paths file .*: Duplicate configuration field: cache_root$"),
    ("{", r"^Invalid paths file .*site\.json: Expecting"),
    # The file holds the group itself, not another reference.
    ('{"paths": {}}', r"^Unknown configuration field: paths\.paths \(in "),
])
def test_invalid_paths_files_are_rejected(tmp_path, contents, message):
    (tmp_path / "site.json").write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_experiment(_write(tmp_path / "experiment.json", {"paths": "site.json"}))


@pytest.mark.parametrize(("paths", "field", "record", "value"), [
    ({"sources": {"molmoact2": "src"}}, "molmoact2_source", "sources.molmoact2", "src"),
    ({"droid_assets": "assets"}, "maniskill_assets", "droid_assets", "assets"),
    ({"simulator_python": {"maniskill": "sim/bin/python"}}, "python",
     "simulator_python.maniskill", "sim/bin/python"),
])
def test_a_paths_entry_wins_over_its_per_field_setting(tmp_path, paths, field, record, value):
    document = {**AGENT, "environment": {**AGENT["environment"], field: "other"}}
    plan = _plan(_config(tmp_path, document, paths=paths))
    expected = str(tmp_path / "configs" / value)
    assert plan["resolved"]["environment"][field] == expected
    assert _records(plan)[record] == (expected, "paths")


def test_an_absent_paths_group_changes_nothing(tmp_path):
    plan = _plan(_config(tmp_path, AGENT))

    assert (plan["paths_file"], plan["path_environment"]) == (None, {})
    assert plan["command"][0] == PYTHON
    assert plan["resolved"]["environment"]["python"] is None
    assert w.child_environment(overrides=plan["path_environment"]) == w.child_environment()
    assert {key: source for key, (_, source) in _records(plan).items()} == {
        "framework_python": "cli", "simulator_python.maniskill": "default",
        "sources.molmoact2": "default", "droid_assets": "default",
        "maniskill_asset_dir": "default", "ditto_asset_dir": "default", "cache_root": "default",
    }


# Precedence: CLI > paths > shell environment > default.


def test_cli_options_win_over_paths_entries(tmp_path):
    site = _site(tmp_path)
    plan = _plan(_config(tmp_path, MOLMOACT2, paths=site),
                 "--sim-python", str(tmp_path / "cli/python"),
                 "--server-python", str(tmp_path / "cli/server-python"),
                 f"--server-arg=--checkpoint={tmp_path / 'cli/weights'}",
                 "--", "--maniskill-assets", str(tmp_path / "cli/assets"))
    records = _records(plan)

    assert records["framework_python"] == (PYTHON, "cli")
    assert records["simulator_python.maniskill"] == (str(tmp_path / "cli/python"), "cli")
    assert records["droid_assets"] == (str(tmp_path / "cli/assets"), "cli")
    assert records["server_python.molmoact2"] == (str(tmp_path / "cli/server-python"), "cli")
    assert records["checkpoints.molmoact2-droid"] == (str(tmp_path / "cli/weights"), "cli")
    command = plan["server"]["command"]
    assert command[0] == str(tmp_path / "cli/server-python")
    assert command.count(f"--checkpoint={tmp_path / 'cli/weights'}") == 1
    assert "--checkpoint" not in command
    # Entries without a command-line equivalent still apply.
    assert records["sources.molmoact2"] == (site["sources"]["molmoact2"], "paths")
    assert command[command.index("--source-root") + 1] == site["sources"]["molmoact2"]


def test_paths_entries_win_over_the_shell_and_defaults(tmp_path, shell):
    for name in RESOURCE_VARIABLES:
        shell.setenv(name, str(tmp_path / "shell" / name))
    site = _site(tmp_path)
    plan = _plan(_config(tmp_path, AGENT, paths=site))

    env = w.child_environment(overrides=plan["path_environment"])
    assert env["MS_ASSET_DIR"] == site["maniskill_asset_dir"]
    assert env["DITTO_ASSET_DIR"] == site["ditto_asset_dir"]
    assert env["LIBERO_CONFIG_PATH"] == site["libero_config"]
    assert env["OPENPI_DATA_HOME"] == site["openpi_data_home"]
    assert env["HF_HOME"] == str(Path(site["cache_root"]) / "huggingface")
    assert env["UV_PYTHON_INSTALL_DIR"] == str(Path(site["cache_root"]) / "python")
    records = _records(plan)
    assert records["maniskill_asset_dir"] == (site["maniskill_asset_dir"], "paths")
    assert records["cache_root"] == (site["cache_root"], "paths")
    assert "environment_variables" not in plan["resolved"]["paths"]["cache_root"]


def test_the_shell_wins_over_defaults_without_a_paths_entry(tmp_path, shell):
    shell.setenv("MS_ASSET_DIR", str(tmp_path / "shell/maniskill"))
    shell.setenv("HF_HOME", str(tmp_path / "shell/hf"))
    plan = _plan(_config(tmp_path, AGENT, paths={"ditto_asset_dir": str(tmp_path / "ditto")}))

    env = w.child_environment(overrides=plan["path_environment"])
    assert env["MS_ASSET_DIR"] == str(tmp_path / "shell/maniskill")
    assert env["HF_HOME"] == str(tmp_path / "shell/hf")
    assert env["TORCH_HOME"] == str(ROOT / "cache/torch")
    records = _records(plan)
    assert records["maniskill_asset_dir"] == (str(tmp_path / "shell/maniskill"), "environment")
    assert records["ditto_asset_dir"] == (str(tmp_path / "ditto"), "paths")
    assert records["cache_root"] == (str(ROOT / "cache"), "default")
    assert plan["resolved"]["paths"]["cache_root"]["environment_variables"] == ["HF_HOME"]


def test_a_relative_asset_variable_is_recorded_where_the_embodiments_resolve_it(tmp_path, shell):
    shell.setenv("MS_ASSET_DIR", "data/shell-maniskill")
    shell.setenv("DITTO_ASSET_DIR", "data/shell-ditto")
    shell.chdir(tmp_path)  # The embodiments anchor at the project root, not the launch directory.
    records = _records(_plan(_config(tmp_path, AGENT)))

    assert records["maniskill_asset_dir"] == (str(ROOT / "data/shell-maniskill"), "environment")
    assert records["ditto_asset_dir"] == (str(ROOT / "data/shell-ditto"), "environment")


def test_paths_select_the_interpreters_simulator_sources_and_default_output(tmp_path):
    site = _site(tmp_path)
    document = {**AGENT, "paths": site}  # No output.directory: outputs_root applies.
    plan = _plan(_write(tmp_path / "configs/experiment.json", document))
    args, extra = launcher.parse_arguments([str(tmp_path / "configs/experiment.json")])
    plan_without_cli = launcher.build_plan(args, extra)

    environment = plan["resolved"]["environment"]
    assert environment["python"] == site["simulator_python"]["maniskill"]
    assert environment["molmoact2_source"] == site["sources"]["molmoact2"]
    assert environment["maniskill_assets"] == site["droid_assets"]
    assert Path(plan["output_dir"]).parent == Path(site["outputs_root"])
    assert _records(plan)["outputs_root"] == (site["outputs_root"], "paths")
    assert plan_without_cli["command"][0] == site["framework_python"]
    assert _records(plan_without_cli)["framework_python"] == (site["framework_python"], "paths")
    # A libero entry in the same site file is ignored by this ManiSkill benchmark.
    assert "libero_config" not in plan["resolved"]["paths"]


def test_libero_benchmarks_use_their_own_simulator_entries(tmp_path):
    site = _site(tmp_path)
    plan = _plan(_config(tmp_path, {"evaluation": {"bench": "libero_spatial", "mode": "agent"},
                                    "environment": {"task_ids": "0"}}, paths=site))

    assert plan["resolved"]["environment"]["python"] == site["simulator_python"]["libero"]
    records = _records(plan)
    assert records["simulator_python.libero"] == (site["simulator_python"]["libero"], "paths")
    assert records["libero_config"] == (site["libero_config"], "paths")
    assert not {"droid_assets", "maniskill_asset_dir", "ditto_asset_dir"} & set(records)


# Per-field conflicts that depend on the selected benchmark or model.


def test_a_simulator_entry_applies_only_to_the_selected_benchmark(tmp_path):
    environment = {**AGENT["environment"], "python": str(tmp_path / "sim/bin/python")}
    libero_only = {"simulator_python": {"libero": str(tmp_path / "libero/bin/python")}}
    plan = _plan(_config(tmp_path, {**AGENT, "environment": environment}, paths=libero_only))
    assert _records(plan)["simulator_python.maniskill"] == (str(tmp_path / "sim/bin/python"),
                                                           "config")


@pytest.mark.parametrize(("base", "server_args", "key", "option"), [
    (MOLMOACT2, ["--checkpoint", "w"], ("checkpoints", "molmoact2-droid"), "--checkpoint"),
    (MOLMOACT2, ["--source-root=s"], ("sources", "molmoact2"), "--source-root"),
    (PI05, ["--openpi-root", "s"], ("sources", "openpi"), "--openpi-root"),
])
def test_server_entries_win_over_server_args(tmp_path, base, server_args, key, option):
    document = {**base, "evaluation": {**base["evaluation"], "server_args": server_args}}
    site = _site(tmp_path)
    plan = _plan(_config(tmp_path, document, paths=site))
    expected = site[key[0]][key[1]]
    assert _records(plan)[".".join(key)] == (expected, "paths")
    # The server takes the last value, which the paths entry appends.
    command = plan["server"]["command"]
    last = max(index for index, item in enumerate(command) if item.split("=", 1)[0] == option)
    assert command[last + 1] == expected


def test_server_args_for_another_profile_do_not_conflict(tmp_path):
    document = {**MOLMOACT2, "evaluation": {**MOLMOACT2["evaluation"],
                                            "server_args": ["--checkpoint", str(tmp_path / "w")]}}
    paths = {"checkpoints": {"pi05-droid": str(tmp_path / "pi05")}}
    plan = _plan(_config(tmp_path, document, paths=paths))

    assert _records(plan)["checkpoints.molmoact2-droid"] == (str(tmp_path / "w"), "config")
    assert plan["server"]["command"].count("--checkpoint") == 1


def test_unknown_checkpoint_profiles_are_rejected(tmp_path):
    paths = {"checkpoints": {"molmoact2_droid": str(tmp_path / "w")}}
    with pytest.raises(ValueError, match=r"^paths\.checkpoints names unknown model profiles: "
                                         r"molmoact2_droid; profiles: molmoact2-droid, "):
        _plan(_config(tmp_path, AGENT, paths=paths))


# Propagation into the managed server, the evaluator and the simulator worker.


@pytest.mark.parametrize(("base", "family", "source_option"), [
    (MOLMOACT2, "molmoact2", "--source-root"),
    (PI05, "openpi", "--openpi-root"),
])
def test_paths_reach_the_managed_server(tmp_path, shell, base, family, source_option):
    shell.setenv("HF_HOME", str(tmp_path / "shell/hf"))
    shell.setenv("OPENPI_DATA_HOME", str(tmp_path / "shell/openpi"))
    site = _site(tmp_path)
    plan = _plan(_config(tmp_path, base, paths=site))
    profile = base["policy"]["model_profile"]

    server = plan["server"]
    command = server["command"]
    assert command[0] == site["server_python"][family]
    assert command[command.index("--checkpoint") + 1] == site["checkpoints"][profile]
    assert command[command.index(source_option) + 1] == site["sources"][family]
    assert server["environment"]["HF_HOME"] == str(Path(site["cache_root"]) / "huggingface")
    assert server["environment"]["OPENPI_DATA_HOME"] == site["openpi_data_home"]
    # The readiness check still proves the loaded model by the profile's metadata.
    assert server["profile"]["model"]["expected_metadata"]["status"] == "loaded"
    records = _records(plan)
    assert records[f"server_python.{family}"] == (site["server_python"][family], "paths")
    assert records[f"checkpoints.{profile}"] == (site["checkpoints"][profile], "paths")
    assert records[f"sources.{family}"] == (site["sources"][family], "paths")


def test_the_external_server_hint_carries_the_paths_entries(tmp_path):
    site = _site(tmp_path)
    document = {**MOLMOACT2, "evaluation": {**MOLMOACT2["evaluation"], "server": "external"}}
    plan = _plan(_config(tmp_path, document, paths=site))
    hint = plan["server"]["serve_hint"]

    assert hint[hint.index("--python") + 1] == site["server_python"]["molmoact2"]
    assert hint[hint.index("--checkpoint") + 1] == site["checkpoints"]["molmoact2-droid"]
    assert hint[hint.index("--source-root") + 1] == site["sources"]["molmoact2"]
    # serve.py reads resource variables from its shell, so the hint assigns them.
    environment = plan["path_environment"]
    assert environment["MS_ASSET_DIR"] == site["maniskill_asset_dir"]
    assert environment["HF_HOME"] == str(Path(site["cache_root"]) / "huggingface")
    assert hint[:len(environment) + 2] == [
        "env", *(f"{name}={value}" for name, value in sorted(environment.items())), PYTHON,
    ]
    assert _plan(_config(tmp_path / "plain", document))["server"]["serve_hint"][0] == PYTHON


def test_a_simulator_checkout_unlike_the_servers_is_recorded_for_both(tmp_path):
    environment = {**MOLMOACT2["environment"], "molmoact2_source": str(tmp_path / "sim-src")}
    plan = _plan(_config(tmp_path, {**MOLMOACT2, "environment": environment}))
    records = _records(plan)

    assert records["sources.molmoact2"] == (str(tmp_path / "sim-src"), "config")
    assert records["sources.molmoact2 (server)"] == (str(ROOT / "third_party/molmoact2"),
                                                     "default")


def _evaluator(plan: dict, shell) -> tuple[list[str], dict[str, str]]:
    """Return the command and environment of the plan's first evaluator, without starting it."""
    captured = {}

    def popen(command, **kwargs):
        captured.update(command=command, env=kwargs["env"])
        raise _Spawned

    shell.setenv("CUDA_VISIBLE_DEVICES", "0")
    shell.setattr(eval_parallel.subprocess, "Popen", popen)
    status = eval_parallel._run_job(plan, plan["jobs"][0], {"id": "w0", "devices": ["0"]},
                                    threading.Event())
    assert status["error"].startswith("_Spawned")
    return captured["command"], captured["env"]


def test_paths_reach_the_evaluator_and_the_simulator_worker(tmp_path, shell):
    shell.setenv("MS_ASSET_DIR", str(tmp_path / "shell/maniskill"))
    site = {**_site(tmp_path), "framework_python": PYTHON}
    plan = _plan(_config(tmp_path, AGENT, paths=site))

    command, env = _evaluator(plan, shell)
    assert command[0] == PYTHON
    for name, value in plan["path_environment"].items():
        assert env[name] == value
    assert env["MS_ASSET_DIR"] == site["maniskill_asset_dir"]

    # The job's config.json is the framework's input: it parses, without the provenance.
    worker = {"id": "w0", "devices": ["0"], "host": "127.0.0.1", "port": None}
    config, command = eval_parallel._job_launch(plan, plan["jobs"][0], worker)
    assert "paths" not in config
    _write(Path(command[command.index("--config") + 1]), config)
    from ditto.integration.agentic import framework_suite_bindings
    from ditto.integration.embodiment import SuiteEmbodiment

    with framework_suite_bindings() as framework:
        parsed = framework.parse_args(command[5:])
    assert parsed.sim_python == site["simulator_python"]["maniskill"]

    # The worker inherits the evaluator's environment; the embodiment is built as run.py does.
    for name in plan["path_environment"]:
        shell.setenv(name, env[name])
    started = {}

    def popen(command, **kwargs):
        started.update(command=command, env=kwargs["env"])
        raise _Spawned

    shell.setattr(subprocess, "Popen", popen)
    embodiment = SuiteEmbodiment(python=parsed.sim_python, source_root=parsed.molmoact2_source,
                                 assets_root=parsed.maniskill_assets, control_mode="pd_joint_pos")
    with pytest.raises(_Spawned):
        embodiment._start()
    worker_command = started["command"]
    assert worker_command[0] == site["simulator_python"]["maniskill"]
    assert worker_command[worker_command.index("--source-root") + 1] == site["sources"]["molmoact2"]
    assert worker_command[worker_command.index("--assets-root") + 1] == site["droid_assets"]
    assert started["env"]["MS_ASSET_DIR"] == site["maniskill_asset_dir"]
    assert started["env"]["DITTO_ASSET_DIR"] == site["ditto_asset_dir"]
    assert started["env"]["HF_HOME"] == str(Path(site["cache_root"]) / "huggingface")


def test_reexecution_uses_the_paths_interpreter_and_keeps_the_shell_visible(tmp_path, shell):
    real = tmp_path / "real/python3"
    real.parent.mkdir()
    real.write_text("", encoding="utf-8")
    venv = tmp_path / "venv/bin/python"
    venv.parent.mkdir(parents=True)
    venv.symlink_to(real)
    shell.setenv("TORCH_HOME", str(tmp_path / "shell/torch"))
    paths = {"framework_python": str(venv), "maniskill_asset_dir": str(tmp_path / "ms")}
    config = _config(tmp_path, AGENT, paths=paths)
    executed = {}

    def execve(path, argv, env):
        executed.update(path=path, argv=argv, env=env)
        raise _Spawned

    shell.setattr(launcher.os, "execve", execve)
    with pytest.raises(_Spawned):
        launcher.main([str(config), "--dry-run"])

    # The venv symlink is the interpreter; resolving it would leave the venv.
    assert executed["path"] == executed["argv"][0] == str(venv)
    env = executed["env"]
    assert env[launcher.REEXEC_MARKER] == str(venv)
    assert env["MS_ASSET_DIR"] == str(tmp_path / "ms")
    assert env["TORCH_HOME"] == str(tmp_path / "shell/torch")
    # Unset resource variables stay unset, so the relaunched plan records them as defaults.
    assert not {"HF_HOME", "LIBERO_CONFIG_PATH", "OPENPI_DATA_HOME"} & set(env)


def test_a_missing_paths_interpreter_is_named(tmp_path, capsys):
    config = _config(tmp_path, AGENT, paths={"framework_python": str(tmp_path / "no/python")})
    assert launcher.main([str(config), "--dry-run"]) == 2
    assert "(paths.framework_python)" in capsys.readouterr().err


# Provenance in the plan and the run directory.


def test_dry_run_prints_each_path_with_its_source_and_existence(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    site = _site(tmp_path)
    Path(site["droid_assets"]).mkdir(parents=True)
    site_file = _write(tmp_path / "site/paths.json", site)
    config = _write(tmp_path / "configs/dev/experiment.json",
                    {**MOLMOACT2, "paths": "../../site/paths.json",
                     "output": {"directory": str(tmp_path / "runs/{timestamp}")}})
    result = subprocess.run(
        [PYTHON, "-B", "scripts/eval.py", str(config), "--dry-run", "--python", PYTHON],
        cwd=ROOT, capture_output=True, text=True, check=False, timeout=120,
        env={"PATH": os.environ.get("PATH", os.defpath), "HOME": str(home),
             "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
    )

    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["paths_file"] == str(site_file.resolve())
    paths = plan["resolved"]["paths"]
    assert paths["droid_assets"] == {"value": site["droid_assets"], "source": "paths",
                                     "exists": True}
    assert paths["checkpoints.molmoact2-droid"] == {
        "value": site["checkpoints"]["molmoact2-droid"], "source": "paths", "exists": False}
    assert paths["framework_python"] == {"value": PYTHON, "source": "cli", "exists": True}
    assert set(paths) == {
        "framework_python", "simulator_python.maniskill", "sources.molmoact2", "droid_assets",
        "maniskill_asset_dir", "ditto_asset_dir", "server_python.molmoact2",
        "checkpoints.molmoact2-droid", "cache_root",
    }
    assert not (tmp_path / "runs").exists()


def test_each_job_directory_records_the_paths_provenance(tmp_path, shell):
    site = {**_site(tmp_path), "framework_python": PYTHON}
    plan = _plan(_config(tmp_path, AGENT, paths=site))
    plan["gpu_groups"] = [["0"]]
    shell.setattr(eval_parallel, "_worker", lambda plan, worker, jobs, stop: {
        "worker_id": worker["id"], "status": "completed"})

    assert eval_parallel.run_parallel(plan) == 1  # No evaluator ran, so no results exist.
    root = Path(plan["output_dir"])
    job = root / "jobs" / plan["jobs"][0]["id"]
    assert json.loads((job / "paths.json").read_text()) == plan["resolved"]["paths"]
    assert "paths" not in json.loads((job / "config.json").read_text())
    resolved = json.loads((root / "resolved-config.json").read_text())
    assert "paths" not in resolved and resolved["environment"] == plan["resolved"]["environment"]
    written = json.loads((root / "eval-plan.json").read_text())["resolved"]["paths"]
    assert written["framework_python"] == {"value": PYTHON, "source": "cli", "exists": True}
