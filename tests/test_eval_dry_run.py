"""Offline ``scripts/eval.py --dry-run`` plans for every tracked experiment and example config.

Each dry run is a subprocess started from the repository root with only PATH, HOME (a fresh
temporary directory) and BLAS/OpenMP thread caps in its environment, so plans cannot depend on
credentials, CUDA devices or a caller's PYTHONPATH. ``--python`` names the running interpreter
because a fresh clone has no project ``envs/`` link, and ``-B`` keeps bytecode out of the tree.
Teacher configs are also dry-run from a copy at the same depth in a temporary directory, so the
missing-teacher checks do not depend on which teacher recordings exist locally.
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# eval.py re-executes itself unless --python matches the running interpreter's normalized path.
PYTHON = os.path.normpath(sys.executable)
DEFAULT_LLM = json.loads(
    (ROOT / "agentic-framework/configs/runtime-defaults.json").read_text(encoding="utf-8")
)["llm"]["model"]
TIMESTAMP = r"\d{8}T\d{6}-\d{6}Z"
# The route fields of the public providers the tracked configs select.
PUBLIC_ROUTES = {
    "openai": {"protocol": "responses", "endpoint": "https://api.openai.com/v1",
               "api_key_env": "OPENAI_API_KEY", "base_url_env": "OPENAI_BASE_URL"},
    "anthropic": {"protocol": "messages", "endpoint": "https://api.anthropic.com/v1",
                  "api_key_env": "ANTHROPIC_API_KEY", "base_url_env": "ANTHROPIC_BASE_URL"},
    "local": {"protocol": "chat_completions", "endpoint": "${LOCAL_LLM_BASE_URL}",
              "api_key_env": "LOCAL_LLM_API_KEY", "base_url_env": "LOCAL_LLM_BASE_URL"},
}
CLAUDE_EXAMPLE = "examples/agent-claude.json"
LOCAL_EXAMPLE = "examples/agent-local.json"
BENCHES = {
    "ditto": {"benchmark": "maniskill", "suite": "ditto", "custom": True},
    "molmoact2-maniskill": {
        "benchmark": "maniskill", "suite": "DroidPutEverythingInBox-v1", "custom": False,
    },
    **{
        suite: {"benchmark": "libero", "suite": suite, "custom": False}
        for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")
    },
}
TASK_COUNTS = {"ditto": 6, "molmoact2-maniskill": 1, "libero_90": 90}
DITTO_DIFFICULTIES = ("easy", "medium", "hard", "xhard")
# Done verification holds 4.0 s at the suite's fixed 30 Hz control rate.
DITTO_DONE_VERIFICATION_STEPS = 120
SOURCE_DIRECTORIES = (
    "agentic-framework", "ditto-bench", "scripts", "configs", "examples", "tools", "docs",
    "tests",
)
# Bytecode/tool caches and test sources can be written concurrently by other test processes;
# none of them is launcher output.
UNWATCHED_NAMES = frozenset({".git", "__pycache__", ".pytest_cache", ".ruff_cache"})
UNWATCHED_DIRECTORIES = frozenset({"tests", "agentic-framework/tests"})
# Planned run output lands here; its listing is watched in addition to the published tree.
OUTPUT_DIRECTORY = "outputs"


def _tracked_configs() -> list[str]:
    try:
        listed = subprocess.run(
            ["git", "ls-files", "-z", "--", "configs/experiments", "examples"],
            cwd=ROOT, capture_output=True, check=True,
        ).stdout.decode().split("\0")
    except (OSError, subprocess.CalledProcessError):
        listed = []
    if not any(listed):
        found = [*ROOT.glob("configs/experiments/**/*.json"), *ROOT.glob("examples/*.json")]
        listed = [path.relative_to(ROOT).as_posix() for path in found]
    return sorted(
        path for path in listed
        if path.endswith(".json")
        and (path.startswith("configs/experiments/") or path.count("/") == 1)
    )


def _document(config: str) -> dict:
    return json.loads((ROOT / config).read_text(encoding="utf-8"))


def _config_path(config: str | Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else (ROOT / config).resolve().parent / path).resolve()


def _teacher(config: str, location: Path | None = None) -> Path | None:
    """Return the teacher a config names, resolved as if the config were at ``location``."""
    observation = _document(config).get("observation", {})
    if not observation.get("demo"):
        return None
    return _config_path(location or config, observation["demo_path"])


CONFIGS = _tracked_configs()
TEACHER_CONFIGS = [c for c in CONFIGS if _teacher(c) is not None]
# A teacher config is planned in place only where its recording exists locally.
PLANNED = [c for c in CONFIGS if c not in TEACHER_CONFIGS or _teacher(c).exists()]


def _environment(home: Path) -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", os.defpath), "HOME": str(home),
        "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
    }


def _dry_run(
    config: str | Path, home: Path, *extra: str, python: str = PYTHON, timeout: float = 120,
    **options,
) -> subprocess.CompletedProcess:
    command = [
        PYTHON, "-B", "scripts/eval.py", str(config), "--dry-run", "--python", python, *extra,
    ]
    return subprocess.run(
        command, cwd=ROOT, env=_environment(home), capture_output=True, text=True,
        check=False, timeout=timeout, **options,
    )


def _unwatched(relative: str) -> bool:
    return any(part in UNWATCHED_NAMES for part in relative.split("/")) or any(
        relative == directory or relative.startswith(f"{directory}/")
        for directory in UNWATCHED_DIRECTORIES
    )


def _watched_files(root: Path) -> list[str]:
    """List the files a commit can publish, or every file when root is not a git checkout."""
    try:
        listed = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=root, capture_output=True, check=True,
        ).stdout.decode().split("\0")
    except (OSError, subprocess.CalledProcessError):
        listed = [
            (Path(directory) / name).relative_to(root).as_posix()
            for directory, _, filenames in os.walk(root) for name in filenames
        ]
    return sorted(path for path in listed if path and not _unwatched(path))


def _entry(path: Path, *, contents: bool) -> tuple[int, int, int]:
    info = path.lstat()
    if contents and not stat.S_ISDIR(info.st_mode):
        return stat.S_IFMT(info.st_mode), info.st_size, info.st_mtime_ns
    return stat.S_IFMT(info.st_mode), 0, 0


def _snapshot(root: Path, files: list[str]) -> dict[str, tuple[int, int, int]]:
    """Record the published files and every name beside them, but not local files' contents.

    Published files keep their type, size and mtime. The directories that hold them, and the
    run output directory, record the type of each entry they list, so creating or deleting
    anything there is a change, while local material that other processes may be editing
    (ignored resource trees, untracked notes) is not compared.
    """
    directories = {"", OUTPUT_DIRECTORY}
    for path in files:
        directories.update(parent.as_posix() for parent in Path(path).parents)
    entries = {}
    for directory in sorted(directories - {"."}):
        if not (root / directory).is_dir():
            continue
        for name in os.listdir(root / directory):
            relative = f"{directory}/{name}" if directory else name
            if not _unwatched(relative):
                entries[relative] = _entry(root / relative, contents=False)
    for path in files:
        if os.path.lexists(root / path):
            entries[path] = _entry(root / path, contents=True)
    return entries


@pytest.fixture(scope="session")
def dry_runs(tmp_path_factory):
    """Dry-run every tracked config once, between two snapshots of the repository tree."""
    home = tmp_path_factory.mktemp("home")
    files = _watched_files(ROOT)
    before = _snapshot(ROOT, files)
    results = {config: _dry_run(config, home) for config in CONFIGS}
    after = _snapshot(ROOT, files)
    changed = sorted(
        path for path in before.keys() | after.keys() if before.get(path) != after.get(path)
    )
    return results, changed


@pytest.fixture(scope="session")
def relocated_teacher_runs(tmp_path_factory):
    """Dry-run each teacher config copied to the same depth below an empty directory."""
    root = tmp_path_factory.mktemp("relocated")
    home = tmp_path_factory.mktemp("relocated-home")
    runs = {}
    for config in TEACHER_CONFIGS:
        copy = root / config
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes((ROOT / config).read_bytes())
        runs[config] = (_teacher(config, copy), _dry_run(copy, home))
    return runs


@pytest.fixture
def home(tmp_path):
    path = tmp_path / "home"
    path.mkdir()
    return path


def _ids(selection: str, count: int) -> list[int]:
    if selection == "all":
        return list(range(count))
    ids = []
    for part in selection.split(","):
        first, _, last = part.partition("-")
        ids.extend(range(int(first), int(last or first) + 1))
    return ids


def test_tracked_configs_are_discovered():
    assert any(config.startswith("configs/experiments/") for config in CONFIGS)
    assert any(config.startswith("examples/") for config in CONFIGS)


def test_dry_run_writes_nothing_into_the_repository(dry_runs):
    _, changed = dry_runs
    assert changed == []


def test_snapshot_sees_new_entries_and_published_edits(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "module.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "local.bin").write_bytes(b"1")
    files = ["src/module.py"]
    before = _snapshot(tmp_path, files)

    (tmp_path / "data" / "local.bin").write_bytes(b"22")
    assert _snapshot(tmp_path, files) == before
    (tmp_path / "src" / "module.py").write_text("x = 22\n", encoding="utf-8")
    (tmp_path / "src" / "new.py").write_text("", encoding="utf-8")
    (tmp_path / OUTPUT_DIRECTORY).mkdir()
    (tmp_path / "run").mkdir()
    after = _snapshot(tmp_path, files)
    assert sorted(p for p in before.keys() | after.keys() if before.get(p) != after.get(p)) == [
        OUTPUT_DIRECTORY, "run", "src/module.py", "src/new.py",
    ]


@pytest.mark.parametrize("config", PLANNED)
def test_dry_run_plan_matches_the_config(dry_runs, config):
    results, _ = dry_runs
    result = results[config]
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    document = _document(config)
    evaluation = document.get("evaluation", {})
    environment = document.get("environment", {})
    execution = document.get("execution", {})
    bench = evaluation["bench"]
    suite = BENCHES[bench]["suite"]

    assert plan["bench"] == BENCHES[bench]
    assert plan["experiment_config"] == str((ROOT / config).resolve())
    assert plan["resolved"]["run"]["mode"] == evaluation["mode"]

    task_ids = _ids(environment.get("task_ids", "all"), TASK_COUNTS.get(bench, 10))
    difficulty = environment.get("difficulty", "easy")
    levels = list(DITTO_DIFFICULTIES) if difficulty == "all" else difficulty.split(",")
    trials = environment.get("trials", 1)
    repeats = evaluation.get("repeats", 1)
    expected = len(task_ids) * (len(levels) if bench == "ditto" else 1) * trials * repeats
    assert (plan["planned_trials"], plan["repeats"]) == (expected, repeats)
    jobs = plan["jobs"]
    assert len(jobs) == len({job["id"] for job in jobs}) == expected
    start = environment.get("init_start", 0)
    for job in jobs:
        assert job["suite"] == suite and job["task_id"] in task_ids
        assert start <= job["init_start"] < start + trials
        assert job["difficulty"] in (levels if bench == "ditto" else ["easy"])
    if "max_steps" in execution:
        assert plan["control_step_budgets"] == {suite: execution["max_steps"]}

    profile = document.get("policy", {}).get("model_profile")
    if profile is not None:
        assert (plan["model"], plan["model_profile"]) == (profile, profile)
        assert plan["done_verification_steps"] == 0
        # Inherited server environments can hold credentials and are never serialized.
        assert plan["server"] is not None and "environment" not in plan["server"]
        assert plan["llm_route"] is None
    else:
        model = document.get("llm", {}).get("model", DEFAULT_LLM)
        assert plan["model"] == plan["resolved"]["llm"]["model"] == model
        assert plan["model_profile"] is None and plan["server"] is None
        # Only names are planned: the provider, protocol, endpoint label and variables.
        provider, wire = model.split("/", 1)
        assert plan["llm_route"] == {
            "selector": model, "provider": provider, "wire_model": wire,
            **PUBLIC_ROUTES[provider],
        }
        if evaluation["mode"] == "agent" and bench == "ditto":
            assert plan["done_verification_steps"] == DITTO_DONE_VERIFICATION_STEPS

    observation = plan["resolved"]["observation"]
    assert observation["profile"] != "privileged"
    assert observation["profile"] == document.get("observation", {}).get("profile", "control")
    teacher = _teacher(config)
    assert (plan["teacher_demo"] or {}).get("path") == (str(teacher) if teacher else None)

    output = Path(plan["output_dir"])
    template = document.get("output", {}).get("directory")
    if template is not None:
        pattern = re.escape(str(_config_path(config, template)))
        assert re.fullmatch(pattern.replace(re.escape("{timestamp}"), TIMESTAMP), str(output))
    assert output != ROOT and not output.exists()
    for directory in SOURCE_DIRECTORIES:
        assert not output.is_relative_to(ROOT / directory), output


@pytest.mark.parametrize("config", TEACHER_CONFIGS)
def test_missing_teacher_fails_before_planning(relocated_teacher_runs, config):
    teacher, result = relocated_teacher_runs[config]
    first_missing = next(
        path for path in reversed((teacher, *teacher.parents)) if not path.exists()
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.startswith("eval.py: ") and "Traceback" not in result.stderr
    assert str(first_missing) in result.stderr


@pytest.mark.parametrize("config", TEACHER_CONFIGS)
def test_missing_teacher_error_names_the_teacher_file(relocated_teacher_runs, config):
    teacher, result = relocated_teacher_runs[config]
    assert result.returncode == 2
    assert f"Teacher demonstration not found: {teacher}" in result.stderr


def _write_config(directory: Path, document: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "experiment.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_output_timestamp_is_expanded_without_creating_run_output(tmp_path, home):
    directory = tmp_path / "configs"
    config = _write_config(directory, {
        "evaluation": {"bench": "ditto", "mode": "agent"},
        "environment": {"task_ids": "1,4", "difficulty": "easy,hard", "trials": 2},
        "output": {"directory": "../runs/{timestamp}-probe/{timestamp}"},
    })

    result = _dry_run(config, home)

    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    runs = tmp_path.resolve() / "runs"
    output = Path(plan["output_dir"])
    assert output.parent.parent == runs
    match = re.fullmatch(rf"({TIMESTAMP})-probe/({TIMESTAMP})", output.relative_to(runs).as_posix())
    assert match and match.group(1) == match.group(2)
    assert plan["planned_trials"] == 8
    assert sorted({(job["task_id"], job["difficulty"]) for job in plan["jobs"]}) == [
        (1, "easy"), (1, "hard"), (4, "easy"), (4, "hard"),
    ]
    assert not runs.exists()


@pytest.mark.parametrize(
    "override",
    [
        ["--sim-hz", "100"],
        ["--control-hz", "10"],
        ["--done-verification-seconds=1"],
        ["--auto-motion", "{}"],
        ["--gpu", "0"],
        ["--", "--sim-hz", "100"],
    ],
)
def test_physical_and_controller_overrides_are_rejected(tmp_path, home, override):
    config = _write_config(tmp_path, {
        "evaluation": {"bench": "ditto", "mode": "agent"},
        "environment": {"task_ids": "0", "trials": 1},
        "output": {"directory": "runs/{timestamp}"},
    })

    result = _dry_run(config, home, *override)

    assert result.returncode == 2 and result.stdout == ""
    assert "physical timing/controller overrides are not supported" in result.stderr
    assert not (tmp_path / "runs").exists()


def test_unnormalized_interpreter_path_names_the_running_interpreter(tmp_path, home):
    resource = pytest.importorskip("resource")
    executable = Path(PYTHON)
    spelled = executable.parent / ".." / executable.parent.name / executable.name
    if not spelled.is_file() or not os.path.samefile(spelled, executable):
        pytest.skip("The interpreter directory is a symlink, so '..' leaves it")
    config = _write_config(tmp_path, {
        "evaluation": {"bench": "ditto", "mode": "agent"},
        "environment": {"task_ids": "0", "trials": 1},
        "output": {"directory": "runs/{timestamp}"},
    })

    def limit():
        # A dry run needs well under a CPU second; a re-execution loop is killed (no core dump).
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_CPU, (2, 2))

    result = _dry_run(config, home, python=str(spelled), preexec_fn=limit, timeout=60)

    assert result.returncode == 0, result.stderr[-2000:]
    assert json.loads(result.stdout)["planned_trials"] == 1


def test_tracked_llm_configs_select_public_providers():
    for config in CONFIGS:
        llm = _document(config).get("llm", {})
        assert llm.get("model", DEFAULT_LLM).split("/", 1)[0] in PUBLIC_ROUTES, config


def test_claude_example_plans_the_messages_route_with_an_empty_environment(tmp_path, home):
    """The example is planned whether or not it is tracked yet, and needs no key."""
    canary = "sk-ant-dryruncanary0123456789"
    for environment in (_environment(home), {**_environment(home), "ANTHROPIC_API_KEY": canary}):
        result = subprocess.run(
            [PYTHON, "-B", "scripts/eval.py", CLAUDE_EXAMPLE, "--dry-run", "--python", PYTHON],
            cwd=ROOT, env=environment, capture_output=True, text=True, check=False, timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert canary not in result.stdout + result.stderr
        plan = json.loads(result.stdout)
        assert plan["llm_route"] == {
            "selector": "anthropic/claude-opus-5-5", "provider": "anthropic",
            "wire_model": "claude-opus-5-5", **PUBLIC_ROUTES["anthropic"],
        }
        assert (plan["planned_trials"], plan["resolved"]["llm"]["max_output_tokens"]) == (1, 16000)
        assert plan["done_verification_steps"] == DITTO_DONE_VERIFICATION_STEPS
        assert not Path(plan["output_dir"]).exists()


def test_local_example_plans_the_chat_route_with_an_empty_environment(tmp_path, home):
    """The local server example needs neither a key nor an endpoint variable to be planned."""
    canary = "local-dryruncanary0123456789"
    variables = {"LOCAL_LLM_API_KEY": canary,
                 "LOCAL_LLM_BASE_URL": "https://canary-dryrun.invalid/v1"}
    for environment in (_environment(home), {**_environment(home), **variables}):
        result = subprocess.run(
            [PYTHON, "-B", "scripts/eval.py", LOCAL_EXAMPLE, "--dry-run", "--python", PYTHON],
            cwd=ROOT, env=environment, capture_output=True, text=True, check=False, timeout=120,
        )
        assert result.returncode == 0, result.stderr
        output = result.stdout + result.stderr
        assert canary not in output and "canary-dryrun" not in output
        plan = json.loads(result.stdout)
        assert plan["llm_route"] == {
            "selector": "local/robot-vlm", "provider": "local", "wire_model": "robot-vlm",
            **PUBLIC_ROUTES["local"],
        }
        assert (plan["planned_trials"], plan["resolved"]["llm"]["max_output_tokens"]) == (1, 8192)
        assert plan["resolved"]["llm"]["reasoning_mode"] is None
        assert plan["done_verification_steps"] == DITTO_DONE_VERIFICATION_STEPS
        assert not Path(plan["output_dir"]).exists()


def test_dry_run_neither_needs_nor_prints_an_api_key(tmp_path, home):
    config = _write_config(tmp_path, {
        "evaluation": {"bench": "ditto", "mode": "agent"},
        "environment": {"task_ids": "0", "trials": 1},
        "output": {"directory": "runs/{timestamp}"},
        "llm": {"model": "gpt-6-astra"},
    })
    canary = "sk-dryruncanary0123456789"
    environment = {**_environment(home), "OPENAI_API_KEY": canary}

    result = subprocess.run(
        [PYTHON, "-B", "scripts/eval.py", str(config), "--dry-run", "--python", PYTHON],
        cwd=ROOT, env=environment, capture_output=True, text=True, check=False, timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert canary not in result.stdout + result.stderr
    assert json.loads(result.stdout)["llm_route"]["api_key_env"] == "OPENAI_API_KEY"


def test_contradictory_benchmark_selection_is_rejected(tmp_path, home):
    config = _write_config(tmp_path, {
        "environment": {"benchmark": "libero", "suite": "ditto"},
        "output": {"directory": "runs/{timestamp}"},
    })

    result = _dry_run(config, home)

    assert result.returncode == 2 and result.stdout == ""
    assert "unsupported, incomplete, or contradictory" in result.stderr
