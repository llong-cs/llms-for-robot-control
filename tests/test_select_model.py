"""Launcher model selection: native shortcuts, registered profiles and LLM routing.

LLM selectors resolve to the canonical ``<provider>/<model>``; a model whose kind disagrees with
the mode, and any name that is neither a profile nor a routable LLM, is an error.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import eval as launcher

ROOT = Path(__file__).resolve().parents[1]
# eval.py re-executes itself unless --python matches the running interpreter's normalized path.
PYTHON = os.path.normpath(sys.executable)
PROFILES = launcher.w.registry()
BENCHES = {
    name: launcher.resolve_bench(name)
    for name in (
        "ditto", "molmoact2-maniskill", "libero_all", "libero_spatial", "libero_goal",
        "libero_10", "libero_90",
    )
}
SHORTCUTS = {"pi05", "pi0.5", "molmoact2", "molmoact"}
DEFAULT_LLM = json.loads(
    (ROOT / "agentic-framework/configs/runtime-defaults.json").read_text(encoding="utf-8")
)["llm"]["model"]
NAMES = sorted(
    (name, profile_id)
    for profile_id, profile in PROFILES.items()
    for name in (profile_id, *profile.get("aliases", []))
    if name.casefold() not in SHORTCUTS
)


@pytest.fixture(autouse=True)
def isolated_directory(tmp_path, monkeypatch):
    """Keep relative selectors from matching files in the caller's working directory."""
    monkeypatch.chdir(tmp_path)


def _matching_bench(profile_id: str) -> dict:
    benchmark = PROFILES[profile_id]["environment"]["benchmark"]
    return BENCHES["ditto" if benchmark == "maniskill" else "libero_spatial"]


@pytest.mark.parametrize("mode", ["auto", "vla"])
@pytest.mark.parametrize(
    ("name", "bench", "expected"),
    [
        ("pi05", "ditto", "pi05-droid"),
        ("pi05", "molmoact2-maniskill", "pi05-droid"),
        ("pi05", "libero_spatial", "pi05-libero"),
        ("pi05", "libero_all", "pi05-libero"),
        ("pi0.5", "libero_90", "pi05-libero"),
        ("PI05", "ditto", "pi05-droid"),
        ("molmoact2", "ditto", "molmoact2-droid"),
        ("molmoact2", "libero_10", "molmoact2-libero"),
        ("molmoact", "molmoact2-maniskill", "molmoact2-droid"),
        ("MolmoAct2", "libero_goal", "molmoact2-libero"),
        ("pi05-maniskill", "ditto", "pi05-droid"),
        ("molmoact2-maniskill", "molmoact2-maniskill", "molmoact2-droid"),
    ],
)
def test_native_shortcuts_select_weights_for_the_benchmark(name, bench, expected, mode):
    selector, spec = launcher.select_model(name, BENCHES[bench], mode)

    assert selector == expected
    assert spec["id"] == expected
    assert spec["environment"]["benchmark"] == BENCHES[bench]["benchmark"]


@pytest.mark.parametrize(("name", "profile_id"), NAMES)
def test_profile_ids_and_aliases_resolve_to_their_profile(name, profile_id):
    for variant in (name, name.upper()):
        selector, spec = launcher.select_model(variant, _matching_bench(profile_id), "auto")
        assert selector == profile_id
        assert spec == PROFILES[profile_id]


@pytest.mark.parametrize(
    ("name", "bench"), [("pi05-droid", "libero_spatial"), ("molmoact2-libero", "ditto")],
)
def test_explicit_profile_ids_are_never_remapped(name, bench):
    assert launcher.select_model(name, BENCHES[bench], "auto")[0] == name


def test_profile_json_path_selects_that_file(tmp_path):
    path = tmp_path / "custom.json"
    shutil.copyfile(ROOT / "agentic-framework/configs/models/pi05-libero.json", path)

    for name in ("custom.json", str(path)):
        selector, spec = launcher.select_model(name, BENCHES["libero_spatial"], "auto")
        assert selector == str(path.resolve())
        assert spec["id"] == "pi05-libero"


def test_missing_profile_json_is_an_error():
    with pytest.raises((OSError, ValueError)):
        launcher.select_model("missing.json", BENCHES["ditto"], "auto")


@pytest.mark.parametrize("name", ["pi05-fast", "pi0.5-droid-v2", "molmoact3", "MolmoAct-X"])
def test_unknown_native_names_are_errors(name):
    with pytest.raises(ValueError, match="Unknown native model"):
        launcher.select_model(name, BENCHES["ditto"], "auto")


@pytest.mark.parametrize("mode", ["auto", "agent", "preview"])
@pytest.mark.parametrize("name", [None, "", "agent", "llm", "Agent", "LLM"])
def test_agent_selectors_mean_the_default_llm(name, mode):
    assert DEFAULT_LLM.startswith("openai/")
    assert launcher.select_model(name, BENCHES["ditto"], mode) == (DEFAULT_LLM, None)


@pytest.mark.parametrize(("configured", "expected"), [
    ("openai/gpt-6-astra", "openai/gpt-6-astra"),
    ("gpt-6-astra", "openai/gpt-6-astra"),
    ("agent", DEFAULT_LLM),
    (None, DEFAULT_LLM),
])
@pytest.mark.parametrize("name", [None, "agent", "llm"])
def test_agent_selectors_keep_the_configured_llm_model(name, configured, expected):
    assert launcher.select_model(name, BENCHES["ditto"], "auto", configured) == (expected, None)


@pytest.mark.parametrize("name", [None, "", "agent", "llm", "LLM"])
def test_vla_mode_requires_a_native_profile(name):
    with pytest.raises(ValueError, match="requires a native model profile"):
        launcher.select_model(name, BENCHES["ditto"], "vla")


@pytest.mark.parametrize("name", ["openai/gpt-6-astra", "gpt-6-astra"])
def test_vla_mode_rejects_llm_model_ids(name):
    with pytest.raises((OSError, ValueError)):
        launcher.select_model(name, BENCHES["ditto"], "vla")


@pytest.mark.parametrize("mode", ["auto", "agent", "preview"])
@pytest.mark.parametrize("name", ["openai/gpt-6-astra", "gpt-6-astra", "OpenAI/gpt-6-astra"])
def test_llm_selectors_route_to_the_canonical_provider_model(name, mode):
    assert launcher.select_model(name, BENCHES["ditto"], mode) == ("openai/gpt-6-astra", None)


@pytest.mark.parametrize("name", ["pi05", "molmoact2-droid", "PI05-LIBERO"])
def test_agent_mode_rejects_native_models(name):
    with pytest.raises(ValueError, match="native VLA model; --mode agent requires an LLM"):
        launcher.select_model(name, BENCHES["libero_spatial"], "agent")


@pytest.mark.parametrize("name", ["openai/gpt-6-astra", "gpt-6-astra"])
def test_vla_mode_names_the_llm_mismatch(name):
    with pytest.raises(ValueError, match="is an LLM model; --mode vla requires a native"):
        launcher.select_model(name, BENCHES["ditto"], "vla")


@pytest.mark.parametrize(("name", "message"), [
    ("astra", "Unknown model 'astra': write <provider>/<model>"),
    ("o9-mini", "Unknown model 'o9-mini': write <provider>/<model>"),
    ("acme/gpt-6-astra", "Unknown provider 'acme'"),
    ("openai/vendor/model", "Invalid openai model"),
])
@pytest.mark.parametrize("mode", ["auto", "agent"])
def test_unroutable_names_list_providers_and_profiles(name, message, mode):
    with pytest.raises(ValueError) as caught:
        launcher.select_model(name, BENCHES["ditto"], mode)
    text = str(caught.value)
    assert message in text
    if "Invalid" not in message:
        # Every registered provider is listed, including local private modules when present.
        from agentic_framework.models.llm.providers import registry

        assert "openai" in registry()
        assert "registered providers: " + ", ".join(registry()) in text
    assert "native profiles: " + ", ".join(sorted(PROFILES)) in text


@pytest.mark.parametrize("mode", ["auto", "agent", "preview"])
@pytest.mark.parametrize("name", ["anthropic/claude-opus-5-5", "claude-opus-5-5"])
def test_claude_selectors_route_to_the_official_anthropic_provider(name, mode):
    assert launcher.select_model(name, BENCHES["ditto"], mode) == (
        "anthropic/claude-opus-5-5", None,
    )
    with pytest.raises(ValueError, match="is an LLM model; --mode vla requires a native"):
        launcher.select_model(name, BENCHES["ditto"], "vla")


@pytest.mark.parametrize("name", ["astra", "o9-mini", "GPT-6-astra"])
def test_vla_mode_reports_an_unknown_name_as_a_missing_profile(name):
    with pytest.raises(ValueError) as caught:
        launcher.select_model(name, BENCHES["ditto"], "vla")
    assert str(caught.value) == (f"Unknown model profile: {name}; native profiles: "
                                 + ", ".join(sorted(PROFILES)))


@pytest.mark.parametrize("name", ["~/absent/profile", "absent/profile.json", "acme/gpt-6-astra",
                                  "openai/vendor/model"])
def test_vla_mode_names_a_missing_profile_file(name):
    with pytest.raises(FileNotFoundError) as caught:
        launcher.select_model(name, BENCHES["ditto"], "vla")
    assert str(caught.value) == str(Path(name).expanduser())


@pytest.mark.parametrize("arguments", [[], ["--model", "agent"], ["--model", "llm"]])
def test_agent_selectors_plan_the_runtime_default_model(tmp_path, arguments):
    defaults = json.loads(
        (ROOT / "agentic-framework/configs/runtime-defaults.json").read_text(encoding="utf-8")
    )
    config = tmp_path / "experiment.json"
    config.write_text(json.dumps({
        "evaluation": {"bench": "ditto"},
        "environment": {"task_ids": "0", "trials": 1},
        "output": {"directory": "runs/{timestamp}"},
    }), encoding="utf-8")
    home = tmp_path / "home"
    home.mkdir()

    result = subprocess.run(
        [
            PYTHON, "-B", str(ROOT / "scripts/eval.py"), str(config), "--dry-run",
            "--python", PYTHON, *arguments,
        ],
        cwd=ROOT, capture_output=True, text=True, check=False, timeout=120,
        env={
            "PATH": os.environ.get("PATH", os.defpath), "HOME": str(home),
            "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
        },
    )

    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["model"] == defaults["llm"]["model"]
    assert plan["model_profile"] is None
    assert plan["resolved"]["run"]["mode"] == "agent"


@pytest.mark.parametrize("arguments", [[], ["--model", "agent"], ["--model", "llm"]])
@pytest.mark.parametrize("configured", ["openai/gpt-6-astra", "gpt-6-astra"])
def test_agent_selector_keeps_an_explicit_llm_model(tmp_path, arguments, configured):
    config = tmp_path / "experiment.json"
    config.write_text(json.dumps({
        "evaluation": {"bench": "ditto"},
        "environment": {"task_ids": "0", "trials": 1},
        "output": {"directory": "runs/{timestamp}"},
        "llm": {"model": configured},
    }), encoding="utf-8")
    home = tmp_path / "home"
    home.mkdir()

    result = subprocess.run(
        [
            PYTHON, "-B", str(ROOT / "scripts/eval.py"), str(config), "--dry-run",
            "--python", PYTHON, *arguments,
        ],
        cwd=ROOT, capture_output=True, text=True, check=False, timeout=120,
        env={"PATH": os.environ.get("PATH", os.defpath), "HOME": str(home)},
    )

    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["model"] == plan["config"]["llm"]["model"] == "openai/gpt-6-astra"
    assert plan["resolved"]["llm"]["model"] == "openai/gpt-6-astra"
    assert plan["llm_route"]["selector"] == "openai/gpt-6-astra"
