#!/usr/bin/env python3
"""Evaluate models on selected benchmarks; use --dry-run to inspect configuration and commands offline."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import workspace as w  # noqa: E402

LIBERO_SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10', 'libero_90')
FORWARDED = {
    'task_ids': (str, 'Task selection: all, 0, or 0,2-4; IDs are zero-based'),
    'trials': (int, 'Number of consecutive initial states/seeds per task; default: 1'),
    'init_start': (int, 'Starting initial-state index or seed offset; default: 0'),
    'seed': (int, 'Environment entry seed; default: 42'),
    'policy_seed': (int, 'Independent policy sampling seed; OpenPi/MolmoAct2 only'),
    'max_steps': (int, 'Maximum policy control steps per trial; reaching them without success fails; defaults to the benchmark budget'),
    'motion_time_scale': (int, 'Positive integer time multiplier for LLM motion execution; subdivides chunk segments and scales positive K without changing H or displacement; default: 1; native VLA requires 1'),
    'difficulty': (str, 'Custom suite difficulty: easy/medium/hard/xhard/all or a comma-separated list'),
    'sim_python': (str, 'Override the isolated simulator Python interpreter'),
    'observation_profile': (str, 'Defaults to control; privileged observations require explicit selection'),
    'max_attempts': (int, 'Maximum attempts of one LLM decision (format repairs, and resends after any HTTP error, transport or provider failure or unexpected exception inside the agent call); a decision that fails them all discards the trial'),
    'max_output_tokens': (int, 'Maximum output tokens per API request'),
    'inference_timeout': (float, 'Timeout in seconds for one inference request'),
}



def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
        epilog='Additional framework options can be appended, for example --task-ids 0-2 --trials 5 --policy-seed 1000 '
               '--max-steps 220. LLM API keys are read only from the environment, for example '
               'export OPENAI_API_KEY=... or ANTHROPIC_API_KEY=...; see docs/USAGE.md.')
    p.add_argument('--bench', choices=['libero_all', 'libero_spatial', 'libero_object', 'libero_goal', 'libero_10', 'libero_90', 'ditto', 'molmoact2-maniskill'], default=None)
    p.add_argument('--model', help='pi05 / molmoact2 (weights selected for the benchmark), '
                   'a profile ID or JSON path, agent (llm.model, else the default LLM), or an LLM '
                   '<provider>/<model> such as openai/gpt-6-astra, anthropic/claude-opus-5-5 or '
                   'local/<served model> (gpt-* is short for openai/gpt-* and claude-* for '
                   'anthropic/claude-*); see --list-models')
    p.add_argument('--model-profile', help='Explicit native model profile; mutually exclusive with --model')
    p.add_argument('--mode', choices=['auto', 'agent', 'vla', 'preview'], default='auto')
    p.add_argument('--config', type=Path, help='Experiment JSON file; a positional config path is also accepted')
    p.add_argument('--h', '--H', type=int,
                   help='Positive H scales move_by pose bounds or sets move_by_chunk length; native VLA H must match the checkpoint output horizon')
    p.add_argument('--control-interface', choices=['move_by', 'move_by_chunk'],
                   help='LLM motion tool; defaults to move_by. Native VLA uses move_by_chunk and its native controller')
    p.add_argument('--k', '--K', type=int, help='Execute at most K steps per decision before LLM motion_time_scale; -1 removes the cap; changing VLA K selects custom scheduling')
    p.add_argument('--policy-profile', choices=['auto', 'native', 'custom'], default='auto')
    p.add_argument('--history', '--history-length', type=int, help='History: <=0 disables memory; positive sets the API decision window')
    p.add_argument('--reasoning', '--reasoning-effort', help='For example low/medium/high/xhigh; available values depend on the selected service')
    p.add_argument('--augmentations', type=json.loads, help='Framework memory/reasoning configuration as JSON')
    p.add_argument('--record-trajectory', action=argparse.BooleanOptionalAction, default=None,
                   help='Record robot/scene states, model requests, camera frames, and video together; disabled by default')
    p.add_argument('--output-dir', type=Path, help='Run root; evaluation artifacts go in evaluation/; defaults to <paths.outputs_root, else project outputs>/<timestamp>-eval-...')
    p.add_argument('--python', type=Path, help='Framework Python interpreter; defaults to paths.framework_python, else project envs/agentic-framework/bin/python')
    p.add_argument('--start-server', action=argparse.BooleanOptionalAction, default=None, help='Manage a VLA replica per GPU group; native models default to managed servers')
    p.add_argument('--server-python', type=Path, help='Override the model server interpreter (and paths.server_python), separate from the framework and simulator')
    p.add_argument('--server-arg', action='append', default=[], help='Append a server argument; repeat as needed; use --server-arg=--dtype for values starting with --')
    p.add_argument('--port', type=int, help='Override the native server port for both the managed server and client')
    p.add_argument('--server-timeout', type=float, default=900)
    p.add_argument('--dry-run', action='store_true', help='Validate and print configuration without creating run output or accessing models, simulators, or credentials')
    task_options = p.add_argument_group('Task and inference options (validated by the framework)')
    for name, (kind, help_text) in FORWARDED.items():
        task_options.add_argument('--'+name.replace('_', '-'), type=kind, default=None, help=help_text)
    p.add_argument('--list-models', action='store_true')
    p.add_argument('--list-benches', action='store_true')
    return p


def parse_arguments(argv):
    """Load launcher defaults from JSON, then apply explicit command-line flags."""
    from scripts.eval_config import infer_bench, load_experiment
    argv = list(argv)
    if argv and not argv[0].startswith('-'):
        argv = ['--config', argv[0], *argv[1:]]
    p = parser()
    preliminary, _ = p.parse_known_args(argv)
    loaded = load_experiment(preliminary.config) if preliminary.config else None
    if loaded:
        launch = loaded['launcher']
        overrides = {key: launch[key] for key in ('bench', 'model', 'mode', 'server_timeout') if key in launch}
        overrides.setdefault('bench', infer_bench(loaded['framework']))
        if 'mode' not in launch and 'mode' in loaded['framework'].get('run', {}):
            overrides['mode'] = loaded['framework']['run']['mode']
        if 'server' in launch:
            overrides['start_server'] = launch['server'] == 'auto'
        if 'server_args' in launch:
            overrides['server_arg'] = launch['server_args']
        p.set_defaults(**overrides)
    args, extra = p.parse_known_args(argv)
    args._experiment = loaded
    return args, extra


def resolve_bench(name):
    if not name:
        raise ValueError('Specify --bench; use --list-benches to see available benchmarks')
    if name == 'ditto':
        return {'benchmark': 'maniskill', 'suite': 'ditto', 'custom': True}
    if name == 'molmoact2-maniskill':
        return {'benchmark': 'maniskill', 'suite': 'DroidPutEverythingInBox-v1', 'custom': False}
    if name == 'libero_all':
        return {'benchmark': 'libero', 'suite': ','.join(LIBERO_SUITES[:4]), 'custom': False}
    if name in LIBERO_SUITES:
        return {'benchmark': 'libero', 'suite': name, 'custom': False}
    raise ValueError(f'Unsupported benchmark {name!r}; use --list-benches to see available benchmarks')


def _framework_on_path():
    for path in (w.FRAMEWORK/'src', w.SUITE/'src', w.FRAMEWORK/'vendor/inspect-robots/src'):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def default_llm():
    """Return the runtime-default LLM selector."""
    return json.loads((w.FRAMEWORK/'configs/runtime-defaults.json').read_text())['llm']['model']


def select_model(name, bench, mode, default=None):
    """Return (selector, profile) for a native VLA, or (canonical LLM selector, None).

    An empty name, agent or llm selects ``default`` (llm.model), else the runtime default.
    A model that disagrees with ``mode`` is an error, and an unknown name is never guessed.
    """
    _framework_on_path()
    from agentic_framework.models.llm import providers

    if providers.is_agent_selector(name):
        if mode == 'vla':
            raise ValueError('--mode vla requires a native model profile')
        return providers.resolve(default, default=default_llm()).selector, None
    key = name.lower()
    if key in ('pi05', 'pi0.5', 'molmoact2', 'molmoact'):
        family = 'pi05' if key.startswith('pi') else 'molmoact2'
        name = family + ('-libero' if bench['benchmark'] == 'libero' else '-droid')
    elif key in ('pi05-maniskill', 'molmoact2-maniskill'):
        name = key.replace('-maniskill', '-droid')
    known = {n.casefold() for s in w.registry().values() for n in [s['id'], *s.get('aliases', [])]}
    candidate = Path(name).expanduser()
    if name.casefold() in known or candidate.is_file() or name.endswith('.json'):
        if mode == 'agent':
            raise ValueError(f'{name!r} is a native VLA model; --mode agent requires an LLM '
                             'such as openai/gpt-6-astra')
        canonical, spec = w.resolve_model(name)
        return str(candidate.resolve()) if candidate.is_file() else canonical, spec
    if key.startswith(('pi05', 'pi0.5', 'molmoact')) and '/' not in key:
        raise ValueError(f'Unknown native model {name!r}; use --list-models to see available profiles')
    if mode == 'vla':
        try:
            providers.resolve(name)
        except ValueError:
            # Not an LLM either: the profile loader names the missing profile or file.
            try:
                return w.resolve_model(name)
            except ValueError as exc:
                raise ValueError(f"{exc}; native profiles: {', '.join(sorted(w.registry()))}") from None
        raise ValueError(f'{name!r} is an LLM model; --mode vla requires a native model profile')
    try:
        route = providers.resolve(name)
    except ValueError as exc:
        raise ValueError(f"{exc}; native profiles: {', '.join(sorted(w.registry()))}") from None
    return route.selector, None


def _json(value):
    return json.loads(json.dumps(value, default=str, allow_nan=False))


# The managed server option that selects its source checkout, by model family.
SOURCE_OPTIONS = {'molmoact2': '--source-root', 'openpi': '--openpi-root'}


def _option_value(arguments, name):
    """Return the value of the last --name VALUE or --name=VALUE in arguments, else None."""
    value = None
    for index, argument in enumerate(arguments):
        option, separator, inline = str(argument).partition('=')
        if option == name:
            value = inline if separator else (arguments[index+1] if index+1 < len(arguments) else '')
    return value


def framework_python(args, paths):
    """Return the framework interpreter and its source: cli, paths or default.

    The spelling is normalized, so '..' cannot re-execute forever. Symlinks are kept, because
    a venv interpreter symlink selects that venv.
    """
    if args.python:
        value, source = args.python, 'cli'
    elif paths.get('framework_python'):
        value, source = Path(paths['framework_python']), 'paths'
    else:
        value, source = w.environment('agentic-framework'), 'default'
    return os.path.normpath(value.expanduser().absolute()), source


def resolved_paths(args, paths, *, user, extra, resolved, bench, spec, server, server_options,
                   default_output):
    """Record each resource location the run uses as {key: {value, source}}.

    Sources follow the precedence: ``cli``, then ``paths``, then the experiment's per-field
    setting (``config``), then the shell ``environment``, then ``default``. Keys of maps are qualified, for example ``simulator_python.maniskill``. Only
    resources this run reads are listed; values are paths, never credentials.
    """
    from agentic_framework.configuration.profiles import profile_path

    home = w.workspace_home()
    records = {}

    def record(key, value, source):
        records[key] = {'value': str(value), 'source': source}

    def configured(key, name=None):
        entry = paths.get(key)
        return entry.get(name) if name is not None and entry is not None else entry

    def framework(key, name, option, flat, value):
        """A path the framework parser resolves; CLI options there win over the config file."""
        source = ('cli' if _option_value(extra, option) is not None
                  else 'paths' if configured(key, name) is not None
                  else 'config' if user.get(flat) is not None else 'default')
        record(f'{key}.{name}' if name else key, value, source)

    def variable(key, name, default, anchored=False):
        if key in paths:
            record(key, paths[key], 'paths')
        elif name in os.environ:
            value = Path(os.environ[name]).expanduser()
            # The embodiments anchor a relative MS_ASSET_DIR or DITTO_ASSET_DIR at the root.
            record(key, Path(os.path.abspath(home/value)) if anchored else value, 'environment')
        else:
            record(key, default, 'default')

    record('framework_python', *framework_python(args, paths))
    simulator = 'maniskill' if bench['benchmark'] == 'maniskill' else 'libero'
    framework('simulator_python', simulator, '--sim-python', 'sim_python',
              resolved.sim_python or w.environment('agentic-framework-' + bench['benchmark']))
    if simulator == 'maniskill':
        framework('sources', 'molmoact2', '--molmoact2-source', 'molmoact2_source',
                  resolved.molmoact2_source)
        framework('droid_assets', None, '--maniskill-assets', 'maniskill_assets',
                  resolved.maniskill_assets)
        variable('maniskill_asset_dir', 'MS_ASSET_DIR', home/w.RESOURCE_DEFAULTS['MS_ASSET_DIR'],
                 anchored=True)
    else:
        variable('libero_config', 'LIBERO_CONFIG_PATH',
                 home/w.RESOURCE_DEFAULTS['LIBERO_CONFIG_PATH'])
    if bench['custom']:
        from ditto.paths import asset_root
        variable('ditto_asset_dir', 'DITTO_ASSET_DIR', asset_root(), anchored=True)
    if server is not None and not server.get('external'):
        family = spec['family']
        record(f'server_python.{family}', server['command'][0],
               'cli' if args.server_python else
               'paths' if configured('server_python', family) else 'default')
        configured_args, cli_args = server_options
        for key, name, option, default in (
            ('checkpoints', spec['id'], '--checkpoint', spec['model']['checkpoint']),
            ('sources', family, SOURCE_OPTIONS.get(family), spec['server']['source']),
        ):
            if option is None:
                continue
            candidates = (('cli', _option_value(cli_args, option)), ('paths', configured(key, name)),
                          ('config', _option_value(configured_args, option)), ('default', default))
            source, value = next(item for item in candidates if item[1] is not None)
            value = str(profile_path(value))
            label = f'{key}.{name}'
            if label in records and records[label]['value'] != value:
                # The simulator and the server use different checkouts of this source.
                label = f'{label} (server)'
            if label not in records:
                record(label, value, source)
        if family == 'openpi':
            variable('openpi_data_home', 'OPENPI_DATA_HOME',
                     home/w.RESOURCE_DEFAULTS['OPENPI_DATA_HOME'])
    if 'cache_root' in paths:
        record('cache_root', paths['cache_root'], 'paths')
    else:
        record('cache_root', home/'cache', 'default')
        inherited = sorted(name for name in w.CACHE_VARIABLES if name in os.environ)
        if inherited:
            # These caches keep the shell's locations instead of the default root.
            records['cache_root']['environment_variables'] = inherited
    if default_output:
        record('outputs_root', paths.get('outputs_root') or home/'outputs',
               'paths' if paths.get('outputs_root') else 'default')
    return records


def build_plan(args, extra):
    """Resolve through the framework parser, without contacting any environment."""
    _framework_on_path()
    from ditto.integration.agentic import DONE_VERIFICATION_SECONDS

    from agentic_framework.configuration.arguments import flatten_config, group_config
    from agentic_framework.harness import run
    from agentic_framework.harness.done_verification import verification_steps

    if args.model and args.model_profile:
        raise ValueError('Specify either --model or --model-profile, not both')
    from scripts.eval_config import infer_bench, load_experiment
    loaded = getattr(args, '_experiment', None) or (load_experiment(args.config) if args.config else None)
    user = flatten_config(loaded['framework']) if loaded else {}
    # Launcher-only resource locations; precedence is CLI > paths > shell environment > default.
    paths = (loaded or {}).get('paths') or {}
    repeats = loaded['launcher'].get('repeats', 1) if loaded else 1
    inferred_bench = infer_bench(loaded['framework']) if loaded else None
    selected_bench = args.bench or inferred_bench
    configured_environment = loaded['framework'].get('environment', {}) if loaded else {}
    if selected_bench is None and any(configured_environment.get(key) is not None for key in ('benchmark', 'suite')):
        raise ValueError(
            'The environment benchmark/suite selection is unsupported, incomplete, or contradictory; '
            'set evaluation.bench to a supported benchmark or supply an explicit --bench override'
        )
    selected_bench = selected_bench or 'libero_all'
    bench = resolve_bench(selected_bench)
    defaults = flatten_config(json.loads((w.FRAMEWORK/'configs/runtime-defaults.json').read_text()))
    name = args.model_profile or args.model or user.get('model_profile') or user.get('model')
    # agent/llm keep an explicit llm.model; only an unset model takes the runtime default.
    selector, spec = select_model(name, bench, 'vla' if args.model_profile else args.mode,
                                  user.get('model'))
    mode = args.mode if args.mode != 'auto' else ('vla' if spec else 'agent')
    if args.start_server is None:
        args.start_server = spec is not None and mode != 'preview'
    if mode == 'vla' and spec is None:
        raise ValueError('VLA evaluation requires a registered profile or a profile JSON file')
    # A shared site file may list every profile's weights, but a misspelled ID is an error.
    profiles = set(w.registry()) | ({spec['id']} if spec else set())
    unknown = sorted(set(paths.get('checkpoints', {})) - profiles)
    if unknown:
        raise ValueError(f"paths.checkpoints names unknown model profiles: {', '.join(unknown)}; "
                         f"profiles: {', '.join(sorted(profiles))}")
    # Experiment settings come from one JSON; intrinsic native defaults stay in profiles.
    values = {'seed': 42, 'task_ids': 'all', 'max_steps': None}
    values.update(user)
    # paths entries override the experiment's environment fields and reach the simulator worker
    # through them, so a framework CLI option still wins.
    simulator = 'maniskill' if bench['benchmark'] == 'maniskill' else 'libero'
    if simulator in paths.get('simulator_python', {}):
        values['sim_python'] = paths['simulator_python'][simulator]
    if 'molmoact2' in paths.get('sources', {}):
        values['molmoact2_source'] = paths['sources']['molmoact2']
    if 'droid_assets' in paths:
        values['maniskill_assets'] = paths['droid_assets']
    values.update(mode=mode, benchmark=bench['benchmark'], suite=bench['suite'],
                  model_profile=selector if spec else None, policy_family=spec['family'] if spec else 'llm')
    if not spec:
        values['model'] = selector
    elif any(x is not None for x in (args.history, args.reasoning, args.augmentations)):
        raise ValueError('history/reasoning/augmentations options require an LLM agent')
    for key in ('h', 'k', 'control_interface'):
        value = getattr(args, key)
        if value is not None:
            values[key] = value
    if args.record_trajectory is not None:
        values['record_trajectory'] = args.record_trajectory
    if args.history is not None or args.reasoning is not None or args.augmentations is not None:
        aug = copy.deepcopy(defaults['augmentations'])
        for overlay in (values.get('augmentations', {}), args.augmentations or {}):
            if not isinstance(overlay, dict):
                raise ValueError('augmentations must be a JSON object')
            for key, value in overlay.items():
                if key in aug and isinstance(value, dict):
                    aug[key].update(value)
                else:
                    aug[key] = value
        if args.history is not None:
            aug['memory']['history_length'] = args.history
        if args.reasoning is not None:
            aug['reasoning'].update(enabled=True, effort=args.reasoning)
        values['augmentations'] = aug
    if spec:
        expected = spec['defaults']
        changed = any((values.get(key) if values.get(key) is not None else expected[key]) != expected[key]
                      for key in ('k', 'control_hz'))
        values['policy_profile'] = (args.policy_profile if args.policy_profile != 'auto'
                                    else ('custom' if changed else user.get('policy_profile', 'native')))
        if args.port is not None:
            values.pop('policy_url', None)
            values['policy_port'] = args.port
    elif (args.port is not None or args.start_server or args.server_python
          or args.server_arg or args.policy_profile != 'auto'):
        raise ValueError('Server and port options require a native VLA model')
    if args.port is not None and not 1 <= args.port <= 65535:
        raise ValueError('port must be within 1..65535')
    if not math.isfinite(args.server_timeout) or args.server_timeout <= 0:
        raise ValueError('server-timeout must be positive and finite')
    if not args.start_server and (args.server_python or args.server_arg):
        raise ValueError('server-python/server-arg require --start-server')
    if args.start_server and mode == 'preview':
        raise ValueError('Preview does not call a model; --start-server is not applicable')
    # Parse suite-only flags separately; all other public framework flags retain their parser.
    extra = list(extra)
    if extra[:1] == ['--']:
        extra = extra[1:]
    for name in FORWARDED:
        value = getattr(args, name)
        if value is not None:
            extra += ['--'+name.replace('_', '-'), str(value)]
    if args.record_trajectory is not None:
        # Forward the explicit recording override to the framework parser.
        extra += ['--record-trajectory' if args.record_trajectory else '--no-record-trajectory']
    launcher_only = {'--config', '--mode', '--model', '--model-profile', '--output-dir'}
    if any(item.split('=', 1)[0] in launcher_only for item in extra):
        raise ValueError('Launcher config/model/mode/output options must be set before -- or in the experiment config')
    forbidden = {'--benchmark', '--suite', '--framework-root', '--gpu', '--server-gpu',
                 '--control-hz', '--sim-hz', '--sim-backend', '--sim-shader', '--action-tools',
                 '--approver', '--auto-motion', '--done-verification-seconds'}
    if any(item.split('=', 1)[0] in forbidden for item in extra):
        raise ValueError('Use the experiment config for benchmark selection and CUDA_VISIBLE_DEVICES for GPUs; physical timing/controller overrides are not supported')
    suite_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    suite_parser.add_argument('--task-randomize', action=argparse.BooleanOptionalAction, default=None)
    if loaded and 'task_randomize' in loaded['launcher']:
        suite_parser.set_defaults(task_randomize=loaded['launcher']['task_randomize'])
    suite_options, extra = suite_parser.parse_known_args(extra)
    if suite_options.task_randomize is not None and not bench['custom']:
        raise ValueError('--task-randomize is supported only by the Ditto Bench')
    output_name = re.sub(r'[^a-zA-Z0-9._-]+', '-', f"{selected_bench}-{selector}")[:100]
    default_output = not (args.output_dir or user.get('output_dir'))
    requested_output = (args.output_dir or user.get('output_dir')
                        or w.run_output('eval', output_name, paths.get('outputs_root')))
    requested_output = str(requested_output).replace('{timestamp}', datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S-%fZ'))
    output = w.checked_output(requested_output)
    values['output_dir'] = str(output/'evaluation')
    if bench['custom']:
        from ditto.integration.agentic import framework_suite_bindings
        binding = framework_suite_bindings()
    else:
        binding = nullcontext(run)
    with tempfile.TemporaryDirectory(prefix='agentic-eval-config-') as temp:
        path = Path(temp)/'input.json'
        path.write_text(json.dumps(group_config(values), default=str))
        with binding as framework:
            resolved = framework.parse_args(['--config', str(path), *extra])
    # Validate the fixed teacher before allocating a GPU or starting a server.
    teacher_demo = None
    if resolved.demo:
        from agentic_framework.harness.demonstration import Demonstration

        # Validate and project the selected content before allocating resources.
        # Action-contract compatibility applies only to full LLM transcripts.
        teacher = Demonstration(
            resolved.demo_path, observation_profile=resolved.observation_profile,
            image_max_side=resolved.demo_image_max_side, demo_mode=resolved.demo_mode,
            demo_content=resolved.demo_content,
        )
        interface = teacher.contract['control_interface']
        if resolved.demo_content == 'full' and interface != resolved.control_interface:
            raise ValueError(
                f'Teacher demo uses control_interface={interface}; '
                f'execution.control_interface={resolved.control_interface}'
            )
        if teacher.image_count >= resolved.max_images:
            raise ValueError(
                f'Teacher demo has {teacher.image_count} images and leaves no room for live '
                f'observations within observation.max_images={resolved.max_images}; increase '
                'the image budget. The full teacher and current observation must fit together.'
            )
        if teacher.text_chars >= resolved.max_context_chars:
            raise ValueError(
                f'Teacher demo has {teacher.text_chars} text characters and leaves no room for '
                f'live context within observation.max_context_chars={resolved.max_context_chars}; '
                'increase the text budget. The combined context is checked for each observation.'
            )
        teacher_demo = {
            'path': str(teacher.path), 'sha256': teacher.sha256, 'mode': resolved.demo_mode,
            'content': resolved.demo_content,
            'request_count': teacher.request_count, 'image_count': teacher.image_count,
            'text_chars': teacher.text_chars,
            'image_max_side': teacher.image_max_side,
            'resized_image_count': teacher.resized_image_count,
            'observation_profile': teacher.contract['observation_profile'],
            'control_interface': interface,
        }
    # Validate task syntax/ranges before allocating a GPU or starting a server.
    from agentic_framework.environments.libero.benchmarks import parse_task_ids
    ids = parse_task_ids(resolved.task_ids)
    if resolved.init_start < 0:
        raise ValueError('init-start must be nonnegative')
    suites = bench['suite'].split(',')
    counts = {suite: (90 if suite == 'libero_90' else 10) for suite in suites}
    if bench['benchmark'] == 'maniskill':
        counts = {bench['suite']: 6 if bench['custom'] else 1}
    if ids is not None and any(i >= count for count in counts.values() for i in ids):
        raise ValueError(f'task-ids are outside the selected benchmark ranges: {counts}')
    trial_count = sum(len(ids) if ids is not None else count for count in counts.values()) * resolved.trials
    if bench['custom']:
        from ditto.integration.scenes import build_suite_scenes
        scenes = build_suite_scenes(task_ids=resolved.task_ids, trials=resolved.trials,
                           init_start=resolved.init_start, seed=resolved.seed,
                           difficulty=resolved.difficulty)
        trial_count = len(scenes)
    elif resolved.difficulty != 'easy':
        raise ValueError('--difficulty is supported only by the Ditto Bench')
    if resolved.category is not None:
        raise ValueError('The selected benchmarks do not support category filtering')
    python = Path(framework_python(args, paths)[0])
    module = 'ditto.integration.agentic' if bench['custom'] else 'agentic_framework.harness.run'
    # -P (Python 3.11+, like the framework) keeps the launch directory off sys.path, so a copied
    # package under the working directory (for example a frozen snapshot in a run directory)
    # cannot shadow the source trees that child_environment puts first on PYTHONPATH. The suite
    # worker derives its script path and PYTHONPATH from the evaluator's imported package.
    command = [str(python), '-B', '-P', '-m', module, '--config', str(output/'eval-config.json'), *extra]
    if suite_options.task_randomize is not None:
        command += ['--task-randomize' if suite_options.task_randomize else '--no-task-randomize']
    server = None
    path_environment = w.path_environment(paths)
    server_options = ([], [])
    if spec:
        transport = spec['transport']
        if transport['kind'] == 'http':
            url = urlsplit(resolved.policy_url)
            host, port = url.hostname, url.port or (443 if url.scheme == 'https' else 80)
        else:
            host, port = resolved.policy_host, resolved.policy_port
        family = spec['family']
        server_python = args.server_python or paths.get('server_python', {}).get(family)
        server_args = argparse.Namespace(command='serve', model=selector,
            gpu=','.join(str(index) for index in range(spec['server']['gpus'])),
            python=Path(server_python).expanduser().absolute() if server_python else None, port=port, output_dir=output/'server', check_only=False)
        # evaluation.server_args come first in --server-arg; the command line appends to them.
        configured_args = list(loaded['launcher'].get('server_args', [])) if loaded else []
        cli_args = (args.server_arg[len(configured_args):]
                    if args.server_arg[:len(configured_args)] == configured_args else list(args.server_arg))
        server_options = (configured_args, cli_args)
        # Weights and source checkout from paths, unless the command line selects them. They
        # follow evaluation.server_args, so they override it (the server takes the last value).
        path_args = []
        for key, name, option in (('checkpoints', spec['id'], '--checkpoint'),
                                  ('sources', family, SOURCE_OPTIONS.get(family))):
            value = paths.get(key, {}).get(name)
            if value is None or option is None:
                continue
            if _option_value(cli_args, option) is None:
                path_args += [option, value]
        # Build a ready-to-copy external serve command too, without choosing a GPU for the user.
        # serve.py sees only its shell, so the hint assigns the variables the paths group sets.
        assignments = [f'{name}={value}' for name, value in sorted(path_environment.items())]
        hint = [*(['env', *assignments] if assignments else []),
                str(python), str(ROOT/'scripts/serve.py'), '--model', selector, '--gpu', '<GPU>', '--port', str(port),
                *(['--python', str(server_args.python)] if server_args.python else []), *path_args]
        if args.start_server:
            if host not in ('127.0.0.1', 'localhost'):
                raise ValueError('--start-server requires a local address; start remote servers separately')
            reserved = {'--model-profile', '--output-dir', '--port', '--host', '--check-only'}
            if any(x.split('=', 1)[0] in reserved for x in args.server_arg):
                raise ValueError('server-arg cannot override the profile, output directory, host, or port, or enable check-only')
            server_command, server_env = w.command(server_args, [*args.server_arg, *path_args],
                                                   path_environment)
            # workspace.command validates the profile GPU count using placeholders.
            # Actual GPU visibility is inherited and partitioned only by the pool.
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                server_env['CUDA_VISIBLE_DEVICES'] = os.environ['CUDA_VISIBLE_DEVICES']
            else:
                server_env.pop('CUDA_VISIBLE_DEVICES', None)
            server = {'command': server_command, 'environment': server_env, 'host': host, 'port': port,
                      'output_dir': str(output/'server'), 'profile': spec}
        else:
            server = {'external': True, 'host': host, 'port': port, 'serve_hint': hint}
    from scripts.eval_resources import allocate_devices, visible_devices
    groups = []
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        groups = allocate_devices(visible_devices(), spec['server']['gpus'] if spec and args.start_server else 1)
    if bench['custom']:
        jobs = [{'id': scene.id, 'suite': bench['suite'], 'task_id': scene.metadata['task_id'],
                 'init_start': scene.metadata['init_state_index'], 'difficulty': scene.metadata['difficulty']}
                for scene in scenes]
    else:
        jobs = [{'id': f'{suite}-task{task_id:04d}-init{init_index:04d}', 'suite': suite,
                 'task_id': task_id, 'init_start': init_index, 'difficulty': 'easy'}
                for suite in suites for task_id in (ids if ids is not None else range(counts[suite]))
                for init_index in range(resolved.init_start, resolved.init_start + resolved.trials)]
    base_trial_count = trial_count
    if repeats > 1:
        repeat_start = resolved.repeat_id if resolved.repeat_id is not None else 0
        # Repeat the same reset conditions; repeat IDs never alter policy seeds.
        jobs = [{**job, 'id': f"{job['id']}-repeat{repeat_id:04d}", 'repeat_id': repeat_id}
                for repeat_id in range(repeat_start, repeat_start + repeats) for job in jobs]
    elif resolved.repeat_id is not None:
        # Preserve existing directory names for a single, manually labeled repeat.
        jobs = [{**job, 'repeat_id': resolved.repeat_id} for job in jobs]
    trial_count = len(jobs)
    resolved_config = _json(resolved.to_dict())
    # Provenance of every resource location; eval_parallel copies it beside each job's config.
    resolved_config['paths'] = resolved_paths(
        args, paths, user=user, extra=extra, resolved=resolved, bench=bench, spec=spec,
        server=server, server_options=server_options, default_output=default_output)
    return {'schema_version': 2, 'bench': bench, 'gpu_groups': groups, 'jobs': jobs,
            'experiment_config': loaded['path'] if loaded else None, 'model': selector if spec else resolved.model,
            'model_profile': spec['id'] if spec else None,
            # Provider, protocol, endpoint label and variable names; never their values.
            'llm_route': None if spec or resolved.llm_route is None else dict(resolved.llm_route),
            'output_dir': str(output), 'cwd': str(Path.cwd()),
            'command': command, 'config': _json(group_config(values)), 'resolved': resolved_config,
            # The paths group's site file (None when inline or absent) and the resource
            # variables it assigns in every evaluator and managed server environment.
            'paths_file': (loaded or {}).get('paths_file'), 'path_environment': path_environment,
            'server': server, 'teacher_demo': teacher_demo,
            'task_counts': counts, 'planned_trials': trial_count,
            'base_trials': base_trial_count, 'repeats': repeats,
            'control_step_budgets': {suite: resolved.max_steps if resolved.max_steps is not None
                                     else (2000 if bench['benchmark'] == 'maniskill' else run.HORIZONS[suite])
                                     for suite in suites},
            # Hold steps a formal LLM run adds after a done call; native policies never call done.
            'done_verification_steps': 0 if spec else verification_steps(
                resolved.done_verification_seconds, resolved.control_hz),
            'task_randomize': bool(suite_options.task_randomize),
            'notes': ['trials selects consecutive initial states/seeds; evaluation.repeats repeats each selected trial with distinct output directories and run.repeat_id labels; LIBERO initial-state counts are checked at runtime',
                      'Repeat numbering starts at run.repeat_id or 0; environment and policy seeds are unchanged across repeats, so fixed-seed native policies may produce identical trajectories',
                      'Recording and privileged observations are disabled by default; max_steps=null uses per-suite budgets',
                      'llm.max_attempts bounds the attempts of one LLM decision; format failures are repaired, and transport/provider failures (every HTTP status, including 400/401/403/404) and unexpected exceptions inside the agent call (building the wire request, sending it or parsing its response) are resent after a backoff (possibly billed again); cancellation and local configuration errors (for example missing credentials) are never retried, and any other exception ends the trial; a trial with a decision that fails all attempts is discarded and excluded from results, and a run whose every trial is discarded fails, so a misconfigured provider fails the run after discarding every trial',
                      f'execution.done_verification_seconds is fixed per benchmark (custom task suite {DONE_VERIFICATION_SECONDS:g} s, otherwise 0): only an LLM done call starts that hold without model calls; success within it counts, otherwise the trial fails']}


def public_plan(plan):
    clean = copy.deepcopy(plan)
    if clean.get('server'):
        clean['server'].pop('environment', None)  # Never serialize inherited credentials.
    # Whether each resource location exists when the plan is printed or written.
    for item in (clean.get('resolved', {}).get('paths') or {}).values():
        item['exists'] = os.path.exists(item['value'])
    clean['display'] = shlex.join(clean['command'])
    return clean


def external_metadata(plan):
    """Read and verify external model identity without sending an action request."""
    from scripts.eval_server import _validate_metadata
    resolved = plan['resolved']
    _, profile = w.resolve_model(resolved['policy']['model_profile'])
    timeout = min(10.0, resolved['policy']['inference_timeout'])
    client = None
    try:
        if profile['transport']['kind'] == 'http':
            from agentic_framework.models.vla.molmoact2 import MolmoAct2Client
            client = MolmoAct2Client(resolved['policy']['url'], timeout_s=timeout)
        else:
            from agentic_framework.models.vla.openpi import OpenPiClient
            client = OpenPiClient(resolved['policy']['host'], resolved['policy']['port'], timeout_s=timeout)
        return _validate_metadata(client.get_server_metadata(), profile, 'External server')
    except Exception as exc:
        raise RuntimeError(f'External model verification failed; check that the server is loaded and its profile/port match: {exc}') from exc
    finally:
        if client is not None:
            client.close()


def _write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False)+'\n')
    temporary.replace(path)


def run_plan(plan, *, server_timeout=900):
    """Execute disjoint trials on every explicitly selected CUDA device group."""
    if not plan['gpu_groups']:
        raise ValueError('Set CUDA_VISIBLE_DEVICES to select evaluation devices')
    from scripts.eval_parallel import run_parallel
    return run_parallel(plan, server_timeout=server_timeout)


TERMINATION_SIGNALS = (signal.SIGQUIT, signal.SIGTERM, signal.SIGHUP)
# Set for one re-execution into the framework interpreter, so a mismatch fails instead of looping.
REEXEC_MARKER = 'AGENTIC_EVAL_REEXEC'


def _interrupt(_signum, _frame):
    raise KeyboardInterrupt


def _interrupt_on_termination():
    """Make SIGQUIT, SIGTERM and SIGHUP start the same graceful shutdown as Ctrl+C.

    Ctrl+\\ sends SIGQUIT, and a closed terminal or ``tmux kill-session`` sends
    SIGHUP. Without this the launcher would exit at once, without stopping its
    evaluators and model servers. Ignored dispositions (SIGHUP under nohup) are
    kept. Returns the previous handlers. While trials run,
    eval_parallel.ShutdownSignals owns these signals and SIGINT, and ignores
    repeats until cleanup finishes.
    """
    previous = {}
    for number in TERMINATION_SIGNALS:
        if signal.getsignal(number) != signal.SIG_IGN:
            previous[number] = signal.signal(number, _interrupt)
    return previous


def _discard_output():
    """Point stdout and stderr, including text they still buffer, at /dev/null.

    After a hangup, writes to the closed terminal fail with EIO. Later messages
    and the interpreter's exit-time flush would fail again and replace the
    evaluation's exit code with 120.
    """
    try:
        null = os.open(os.devnull, os.O_WRONLY)
    except OSError:
        return
    try:
        for stream in (sys.stdout, sys.stderr):
            try:
                os.dup2(null, stream.fileno())
            except (OSError, ValueError, AttributeError):
                pass
    finally:
        os.close(null)


def _say(*lines, error=False):
    """Print launcher messages; a vanished terminal must not change the exit code."""
    try:
        for line in lines:
            print(line, file=sys.stderr if error else sys.stdout)
        # Flushing both also catches text left over from an earlier failed write.
        sys.stdout.flush()
        sys.stderr.flush()
    except (OSError, ValueError):
        _discard_output()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        args, extra = parse_arguments(argv)
        if args.list_models:
            for key, spec in w.registry().items():
                d = spec['defaults']
                print(f"{key:20} {spec['environment']['benchmark']:10} H={d['h']} K={d['k']} Hz={d['control_hz']}")
            _framework_on_path()
            from agentic_framework.models.llm.providers import SHORTHANDS, registry
            providers = registry()
            print('LLM models are <provider>/<model ID>' + ''.join(
                f'; {prefix}* is short for {owner}/{prefix}*'
                for prefix, owner in SHORTHANDS if owner in providers) + ':')
            for name, provider in providers.items():
                protocol = provider.protocol if isinstance(provider.protocol, str) else 'per model'
                key = (('key' if provider.key_required else 'optional key')
                       + f' ${provider.key_env}' if provider.key_env else 'no key')
                check = (f'; ${provider.base_url_check_env} must be unset or name it'
                         if provider.base_url_check_env else '')
                default = (f'; default {provider.base_url_default}'
                           if provider.base_url_default else '')
                print(f'{name + "/<model>":20} {protocol:16} {provider.endpoint} '
                      f'({key}{check}{default})')
            print(f'agent / llm: llm.model of the experiment, else {default_llm()}')
            return 0
        if args.list_benches:
            print('libero_all = libero_spatial,libero_object,libero_goal,libero_10 (40 tasks; excludes libero_90)')
            print('libero_spatial / libero_object / libero_goal / libero_10 / libero_90')
            print('ditto: six custom tasks; molmoact2-maniskill: official DroidPutEverythingInBox-v1')
            return 0
        # Compare normalized spellings: '..' would otherwise re-execute forever. Symlinks are
        # kept, because a venv interpreter symlink selects that venv.
        paths = (args._experiment or {}).get('paths') or {}
        python, source = framework_python(args, paths)
        if not os.path.isfile(python):
            origin = {'cli': ' (--python)', 'paths': ' (paths.framework_python)'}.get(source, '')
            raise ValueError(f'Framework interpreter does not exist: {python}{origin}; '
                             'see docs/INSTALLATION.md')
        reexecuted = os.environ.pop(REEXEC_MARKER, None)
        if os.path.normpath(os.path.abspath(sys.executable)) != python:
            if reexecuted == python:
                raise RuntimeError(f'Re-executing with {python} did not start that interpreter')
            # Without resource defaults, the re-executed launcher records which locations
            # came from the shell; it applies the defaults to its own children.
            environment = w.child_environment(overrides=w.path_environment(paths), defaults=False)
            environment[REEXEC_MARKER] = python
            os.execve(python, [python, '-B', str(Path(__file__).resolve()), *argv], environment)
        plan = build_plan(args, extra)
        if args.dry_run:
            print(json.dumps(public_plan(plan), ensure_ascii=False, indent=2))
            return 0
        print(f"Evaluation output: {plan['output_dir']}", flush=True)
        print(f"Planned trials: {plan['planned_trials']}; GPU worker groups: {len(plan['gpu_groups'])}", flush=True)
        if plan['teacher_demo'] is not None:
            teacher = plan['teacher_demo']
            print(
                f"Teacher demo: {teacher['path']} ({teacher['mode']} mode, {teacher['content']} content, "
                f"{teacher['request_count']} requests, "
                f"{teacher['image_count']} images; image max side {teacher['image_max_side']} px, "
                f"{teacher['resized_image_count']} resized)", flush=True,
            )
        previous = _interrupt_on_termination()
        try:
            code = run_plan(plan, server_timeout=args.server_timeout)
        finally:
            for number, handler in previous.items():
                signal.signal(number, signal.SIG_DFL if handler is None else handler)
        state = 'completed' if code == 0 else ('interrupted' if code == 130 else 'failed')
        _say(f'Evaluation {state} (exit code {code}).',
             f"Summary: {Path(plan['output_dir']) / 'evaluation' / 'summary.json'}")
        return code
    except KeyboardInterrupt:
        _say('Evaluation interrupted; existing results have been preserved.', error=True)
        return 130
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        _say(f'eval.py: {exc}', error=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
