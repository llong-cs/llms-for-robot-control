"""Run this suite through the existing agentic-framework evaluation pipeline.

Usage: python -m ditto.integration.agentic --mode preview --task-ids all
All ordinary framework flags are accepted. Overrides are scoped to this call;
no upstream source file or global installation is modified.
"""
from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager
from datetime import UTC, datetime

from ditto.catalog import DEFAULT_HORIZON, SUITE_ID
from ditto.paths import project_path, project_root

# After a model done call, hold for 4 s (120 control steps, 600 physics samples at
# 30/150 Hz) for all six tasks: 1 s longer than the 3 s (451 consecutive samples) of
# standing that stand_object requires, so an object released or left standing when the
# model calls done can still reach success if its standing run starts by hold sample 150.
DONE_VERIFICATION_SECONDS = 4.0


@contextmanager
def framework_suite_bindings():
    import agentic_framework.environments.maniskill.embodiment as embodiment_module
    import agentic_framework.harness.run as run_module

    from .embodiment import SuiteEmbodiment
    from .scenes import build_suite_scenes

    old = (run_module.ENV_ID, run_module.MANISKILL_HORIZON,
           run_module.build_maniskill_scenes, embodiment_module.ManiSkillEmbodiment,
           run_module.parser)
    original_parser = run_module.parser

    def suite_parser():
        parser = original_parser()
        parser.description = "Evaluate the six Ditto Bench tasks using the existing agentic framework."
        parser.epilog = "Suite options: --framework-root PATH; --task-randomize / --no-task-randomize (default: fixed initial layout)."
        parser.set_defaults(
            benchmark="maniskill", suite=SUITE_ID, task_ids="all",
            difficulty="easy", done_verification_seconds=DONE_VERIFICATION_SECONDS,
            output_dir=project_root() / "outputs/ditto-bench" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S-%fZ"),
        )
        return parser

    run_module.ENV_ID = SUITE_ID
    run_module.MANISKILL_HORIZON = DEFAULT_HORIZON
    run_module.build_maniskill_scenes = build_suite_scenes
    run_module.parser = suite_parser
    embodiment_module.ManiSkillEmbodiment = SuiteEmbodiment
    try:
        yield run_module
    finally:
        (run_module.ENV_ID, run_module.MANISKILL_HORIZON,
         run_module.build_maniskill_scenes, embodiment_module.ManiSkillEmbodiment,
         run_module.parser) = old


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--framework-root", type=project_path, default=project_path(
        os.environ.get("DITTO_FRAMEWORK_ROOT", "agentic-framework")),
        help="Framework checkout containing src/agentic_framework",
    )
    pre_parser.add_argument("--task-randomize", action=argparse.BooleanOptionalAction, default=False,
                            help="Enable seeded task layout jitter, including fixture XY position and yaw for tasks 0, 4, and 5")
    own, remaining = pre_parser.parse_known_args(argv)
    framework_src = own.framework_root.resolve() / "src"
    has_checkout = (framework_src / "agentic_framework/harness/run.py").is_file()
    help_requested = "--help" in remaining or "-h" in remaining
    if not has_checkout and not help_requested:
        raise FileNotFoundError(f"agentic-framework checkout missing: {framework_src.parent}")
    # Installed packages can display help; evaluation requires a source checkout.
    if has_checkout and str(framework_src) not in sys.path:
        sys.path.insert(0, str(framework_src))
    previous_randomize = os.environ.get("DITTO_TASK_RANDOMIZE")
    os.environ["DITTO_TASK_RANDOMIZE"] = "1" if own.task_randomize else "0"
    try:
        with framework_suite_bindings() as framework:
            args = framework.parse_args(remaining)
            if args.benchmark != "maniskill" or args.suite not in (SUITE_ID, "all"):
                raise ValueError(f"This launcher requires --benchmark maniskill --suite {SUITE_ID}")
            return framework.main(remaining)
    finally:
        if previous_randomize is None:
            os.environ.pop("DITTO_TASK_RANDOMIZE", None)
        else:
            os.environ["DITTO_TASK_RANDOMIZE"] = previous_randomize


if __name__ == "__main__":
    raise SystemExit(main())
