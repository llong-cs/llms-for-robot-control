"""Experiment JSON loading: schema boundaries, fixed settings and path resolution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import eval_config


def _write(directory: Path, document: object, name: str = "experiment.json") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _load(tmp_path: Path, document: object) -> dict:
    return eval_config.load_experiment(_write(tmp_path / "configs", document))


def test_only_supplied_settings_are_returned(tmp_path):
    path = _write(tmp_path, {"evaluation": {"bench": "ditto"}, "environment": {"trials": 2}})

    assert eval_config.load_experiment(path) == {
        "launcher": {"bench": "ditto"},
        "framework": {"environment": {"trials": 2}},
        "path": str(path.resolve()),
    }


def test_unknown_top_level_group_is_rejected(tmp_path):
    with pytest.raises(ValueError, match=r"configuration groups: \['simulator'\]"):
        _load(tmp_path, {"evaluation": {"bench": "ditto"}, "simulator": {}})


@pytest.mark.parametrize(
    ("document", "field"),
    [
        ({"evaluation": {"gpus": 2}}, "evaluation.gpus"),
        # Launcher settings are a closed set too: a framework field there is unknown.
        ({"evaluation": {"trials": 1}}, "evaluation.trials"),
        ({"execution": {"horizon": 5}}, "execution.horizon"),
        ({"output": {"dir": "runs"}}, "output.dir"),
        # A fixed setting placed in the wrong group is unknown, not silently accepted.
        ({"execution": {"control_hz": 30}}, "execution.control_hz"),
        ({"llm": {"augmentations": {"planning": {}}}}, "llm.augmentations.planning"),
        ({"llm": {"augmentations": {"memory": {"length": 2}}}}, "llm.augmentations.memory.length"),
        # Removed settings are unknown fields too.
        ({"llm": {"backend": "responses"}}, "llm.backend"),
        ({"llm": {"env_file": "keys.env"}}, "llm.env_file"),
    ],
)
def test_unknown_fields_are_rejected(tmp_path, document, field):
    with pytest.raises(ValueError, match=rf"^Unknown configuration field: {field}$"):
        _load(tmp_path, document)


@pytest.mark.parametrize(
    ("group", "name", "value"),
    [
        ("environment", "control_hz", 30),
        ("environment", "sim_hz", 150),
        ("environment", "sim_hz", None),
        ("environment", "gpu", None),
        ("environment", "sim_backend", "cpu"),
        ("environment", "sim_shader", "rt-fast"),
        ("execution", "action_tools", "move_by"),
        ("execution", "approver", "none"),
        ("execution", "auto_motion", {}),
        ("execution", "done_verification_seconds", 4.0),
        ("execution", "done_verification_seconds", None),
    ],
)
def test_fixed_physical_and_controller_settings_are_rejected(tmp_path, group, name, value):
    with pytest.raises(ValueError, match=rf"^{group}\.{name} is not configurable through eval\.py"):
        _load(tmp_path, {group: {name: value}})


def test_gpu_selection_points_to_cuda_visible_devices(tmp_path):
    with pytest.raises(ValueError, match="environment.gpu is not configurable") as error:
        _load(tmp_path, {"environment": {"gpu": "0"}})
    assert "CUDA_VISIBLE_DEVICES" in str(error.value)


@pytest.mark.parametrize(
    ("group", "name", "value", "kind"),
    [
        ("environment", "trials", "3", "integer"),
        ("environment", "trials", True, "integer"),
        ("environment", "trials", None, "integer"),
        ("execution", "h", 5.0, "integer"),
        ("policy", "inference_timeout", "10", "number"),
        ("policy", "inference_timeout", False, "number"),
        ("observation", "demo", "yes", "boolean"),
        ("output", "directory", 3, "string"),
        ("llm", "augmentations", [], "object"),
    ],
)
def test_field_types_are_checked(tmp_path, group, name, value, kind):
    with pytest.raises(ValueError, match=rf"^{group}\.{name} must be a JSON {kind}$"):
        _load(tmp_path, {group: {name: value}})


def test_nonfinite_numbers_are_rejected(tmp_path):
    path = tmp_path / "nan.json"
    path.write_text('{"policy": {"inference_timeout": NaN}}', encoding="utf-8")
    with pytest.raises(ValueError, match="numbers must be finite; found NaN"):
        eval_config.load_experiment(path)
    path.write_text('{"policy": {"inference_timeout": 1e999}}', encoding="utf-8")
    with pytest.raises(ValueError, match=r"^policy\.inference_timeout must be finite$"):
        eval_config.load_experiment(path)


def test_duplicate_fields_and_non_objects_are_rejected(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"environment": {"trials": 1, "trials": 2}}', encoding="utf-8")
    with pytest.raises(ValueError, match="^Duplicate configuration field: trials$"):
        eval_config.load_experiment(path)
    with pytest.raises(ValueError, match="must be a JSON object"):
        _load(tmp_path, [])


def test_nullable_fields_accept_null(tmp_path):
    loaded = _load(tmp_path, {
        "execution": {"max_steps": None},
        "observation": {"demo_path": None},
        "output": {"directory": None},
    })

    assert loaded["framework"] == {
        "execution": {"max_steps": None},
        "observation": {"demo_path": None},
        "output": {"directory": None},
    }


def test_relative_paths_resolve_against_the_config_directory(tmp_path, monkeypatch):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    directory = tmp_path / "configs" / "group"
    path = _write(directory, {
        "evaluation": {"model": "./profiles/custom.json"},
        "environment": {"molmoact2_source": "../src/molmoact2", "maniskill_assets": "assets"},
        "policy": {"model_profile": "profiles/native.json", "metadata": "meta.json"},
        "observation": {"demo_path": "teachers/teacher/trajectory.json"},
        "output": {"directory": "../../runs/{timestamp}-probe"},
    })

    loaded = eval_config.load_experiment(path)

    def expected(relative: str) -> str:
        return str((directory / relative).resolve())

    framework = loaded["framework"]
    assert loaded["launcher"]["model"] == expected("profiles/custom.json")
    assert framework["environment"] == {
        "molmoact2_source": expected("../src/molmoact2"),
        "maniskill_assets": expected("assets"),
    }
    assert framework["policy"] == {
        "model_profile": expected("profiles/native.json"),
        "metadata": expected("meta.json"),
    }
    assert framework["observation"]["demo_path"] == expected("teachers/teacher/trajectory.json")
    # The placeholder is expanded by the launcher when it plans a run, not while loading.
    assert framework["output"]["directory"] == str(tmp_path.resolve() / "runs/{timestamp}-probe")


def test_registered_ids_are_not_treated_as_paths(tmp_path):
    loaded = _load(tmp_path, {
        "evaluation": {"model": "pi05"},
        "policy": {"model_profile": "molmoact2-droid"},
    })

    assert loaded["launcher"]["model"] == "pi05"
    assert loaded["framework"]["policy"]["model_profile"] == "molmoact2-droid"


def test_absolute_and_home_paths_are_not_joined_to_the_config_directory(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    absolute = tmp_path.resolve() / "absolute" / "runs"

    loaded = _load(tmp_path, {
        "environment": {"maniskill_assets": str(absolute)},
        "output": {"directory": "~/runs/{timestamp}"},
    })

    assert loaded["framework"]["environment"]["maniskill_assets"] == str(absolute)
    assert loaded["framework"]["output"]["directory"] == str(home.resolve() / "runs/{timestamp}")


def test_interpreter_symlink_is_kept_while_other_paths_resolve(tmp_path):
    real = tmp_path / "real"
    (real / "bin").mkdir(parents=True)
    (real / "bin" / "python3").write_text("", encoding="utf-8")
    directory = tmp_path / "configs"
    (directory / "venv" / "bin").mkdir(parents=True)
    (directory / "venv" / "bin" / "python").symlink_to(real / "bin" / "python3")
    (directory / "linked").symlink_to(real, target_is_directory=True)

    loaded = eval_config.load_experiment(_write(directory, {
        "environment": {"python": "venv/bin/python", "maniskill_assets": "linked/assets"},
    }))

    environment = loaded["framework"]["environment"]
    # Resolving a venv interpreter symlink would select its base interpreter.
    assert environment["python"] == str(directory.resolve() / "venv" / "bin" / "python")
    assert environment["maniskill_assets"] == str(real.resolve() / "assets")


def test_blank_paths_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="^Configured paths must be nonempty strings$"):
        _load(tmp_path, {"output": {"directory": "  "}})


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("ditto", "ditto"),
        ("maniskill", "molmoact2-maniskill"),
        ("molmoact2-maniskill", "molmoact2-maniskill"),
        ("libero-all", "libero_all"),
        ("libero_all", "libero_all"),
        ("libero-spatial", "libero_spatial"),
        ("libero_10", "libero_10"),
        ("libero-90", "libero_90"),
    ],
)
def test_bench_aliases_are_canonicalized(tmp_path, alias, canonical):
    assert _load(tmp_path, {"evaluation": {"bench": alias}})["launcher"]["bench"] == canonical


@pytest.mark.parametrize("bench", ["Ditto", "libero", "libero_100", "droid"])
def test_unknown_benches_are_rejected(tmp_path, bench):
    with pytest.raises(ValueError, match=rf"^Unknown evaluation\.bench: '{bench}'$"):
        _load(tmp_path, {"evaluation": {"bench": bench}})


@pytest.mark.parametrize("mode", ["auto", "agent", "vla", "preview"])
def test_supported_modes_are_accepted(tmp_path, mode):
    assert _load(tmp_path, {"evaluation": {"mode": mode}})["launcher"] == {"mode": mode}


@pytest.mark.parametrize("mode", ["llm", "Agent", "native"])
def test_unsupported_modes_are_rejected(tmp_path, mode):
    with pytest.raises(ValueError, match="^evaluation.mode must be auto, agent, vla, or preview$"):
        _load(tmp_path, {"evaluation": {"mode": mode}})


@pytest.mark.parametrize(
    ("evaluation", "message"),
    [
        ({"mode": ""}, r"^evaluation\.mode must be a nonempty string$"),
        ({"repeats": 0}, r"^evaluation\.repeats must be a positive integer$"),
        ({"server": "managed"}, r"^evaluation\.server must be auto or external$"),
        ({"server_timeout": 0}, r"^evaluation\.server_timeout must be positive$"),
        ({"server_args": ["--dtype", 16]}, r"^evaluation\.server_args\[1\] must be a JSON string$"),
        ({"task_randomize": "yes"}, r"^evaluation\.task_randomize must be a JSON boolean$"),
    ],
)
def test_launcher_settings_are_validated(tmp_path, evaluation, message):
    with pytest.raises(ValueError, match=message):
        _load(tmp_path, {"evaluation": evaluation})


@pytest.mark.parametrize(
    ("observation", "message"),
    [
        ({"demo_mode": "few-shot"}, r"^observation\.demo_mode must be id or ood$"),
        ({"demo_content": "actions"}, r"^observation\.demo_content must be observations or full$"),
    ],
)
def test_demonstration_choices_are_validated(tmp_path, observation, message):
    with pytest.raises(ValueError, match=message):
        _load(tmp_path, {"observation": observation})


@pytest.mark.parametrize(
    ("environment", "bench"),
    [
        ({}, None),
        ({"benchmark": "libero"}, "libero_all"),
        ({"benchmark": "libero", "suite": "all"}, "libero_all"),
        ({"suite": "libero_10"}, "libero_10"),
        ({"benchmark": "libero", "suite": "libero-goal"}, "libero_goal"),
        ({"suite": "libero_goal,libero_10,libero_spatial,libero_object"}, "libero_all"),
        ({"benchmark": "maniskill", "suite": "ditto"}, "ditto"),
        ({"suite": "DroidPutEverythingInBox-v1"}, "molmoact2-maniskill"),
        # Contradictory or unsupported selections are not guessed.
        ({"benchmark": "libero", "suite": "ditto"}, None),
        ({"benchmark": "maniskill", "suite": "libero_10"}, None),
        ({"benchmark": "maniskill"}, None),
        ({"suite": "libero_spatial,libero_object"}, None),
    ],
)
def test_benchmark_is_inferred_only_from_unambiguous_selections(environment, bench):
    assert eval_config.infer_bench({"environment": environment}) == bench
