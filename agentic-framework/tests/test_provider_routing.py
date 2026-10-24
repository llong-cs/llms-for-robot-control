"""Model routing: the selector grammar, provider discovery, env-only keys and route records.

Providers added by a test are modules written to a temporary directory appended to the provider
package's ``__path__``; the registry is rebuilt around each such test. Endpoints use reserved
``.invalid`` hosts and every exchange goes through ``httpx.MockTransport``.
"""

from __future__ import annotations

import base64
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
import uuid
from collections.abc import MutableMapping
from pathlib import Path

import httpx
import numpy as np
import pytest
from inspect_robots.scene import Scene
from inspect_robots.types import Observation

from agentic_framework.configuration.arguments import parse_configuration, runtime_defaults
from agentic_framework.configuration.profiles import list_model_profiles
from agentic_framework.environments.libero.embodiment import LiberoEmbodiment
from agentic_framework.harness.policy import LLMMotionPolicy
from agentic_framework.harness.preview import record_preview_request
from agentic_framework.harness.types import (
    ActionSpec,
    AgentContext,
    AugmentationConfig,
    ScheduleConfig,
)
from agentic_framework.models import factory
from agentic_framework.models.llm import providers
from agentic_framework.models.llm.errors import BackendConfigurationError, failure_kind
from agentic_framework.models.llm.http_agent import PLACEHOLDER_ORIGIN
from agentic_framework.models.llm.providers import ProviderError, ProviderSpec, resolve
from agentic_framework.models.llm.responses import ResponsesAgent
from agentic_framework.observability.api_audit import APIRequestRecorder
from agentic_framework.observability.experiment_preview import write_experiment_preview

OFFICIAL = "https://api.openai.com/v1"
KEY = "routing-test-key-0123456789"
OPENAI_ROUTE = {
    "selector": "openai/gpt-6-astra", "provider": "openai", "protocol": "responses",
    "wire_model": "gpt-6-astra", "endpoint": OFFICIAL, "api_key_env": "OPENAI_API_KEY",
    "base_url_env": "OPENAI_BASE_URL",
}
TOOL = {"type": "function", "name": "move_by", "description": "Move.",
        "parameters": {"type": "object"}, "strict": False}
CALL = {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "move_by",
        "arguments": json.dumps({"deltas": {"dz": 0.001}}), "status": "completed"}
REPLY = {"id": "resp_1", "object": "response", "model": "model-a", "status": "completed",
         "output": [CALL], "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}


class ForbiddenEnviron(MutableMapping):
    """Fail on any access to the process environment."""

    def _forbidden(self, *args, **kwargs):
        raise AssertionError("the process environment was read")

    __getitem__ = __setitem__ = __delitem__ = __iter__ = __len__ = _forbidden
    get = __contains__ = copy = _forbidden


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


def _module(directory: Path, name: str, source: str) -> None:
    (directory / f"{name}.py").write_text(textwrap.dedent(source), encoding="utf-8")
    importlib.invalidate_caches()


def _spec_module(directory: Path, module: str, **fields) -> None:
    arguments = ", ".join(f"{key}={value!r}" for key, value in fields.items())
    _module(directory, module, f"""
        from agentic_framework.models.llm.providers import ProviderSpec

        PROVIDER = ProviderSpec({arguments})
    """)


def _context(tools=(TOOL,)):
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    return AgentContext(
        request_id="req-1", instructions="Choose one robot tool call.", text="{}",
        images={"cam": image}, tools=tools,
        messages=({"role": "user", "content": [{"type": "text", "text": "{}"},
                                               {"type": "image", "name": "cam"}]},),
    )


def _observation():
    return Observation(
        images={"agentview": np.zeros((4, 4, 3), dtype=np.uint8),
                "robot0_eye_in_hand": np.ones((4, 4, 3), dtype=np.uint8)},
        state={"eef_pos": np.array([0.0, 0.0, 0.2]), "eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
               "gripper_width": np.array([0.04])},
        instruction="Lift the cube.", extra={"env_step": 0},
    )


# ------------------------------------------------------------------------ selector grammar


@pytest.mark.parametrize("selector", [
    "openai/gpt-6-astra", "gpt-6-astra", "OpenAI/gpt-6-astra", " openai/gpt-6-astra ",
])
def test_selectors_resolve_to_one_canonical_route(selector):
    route = resolve(selector)
    assert route.record() == OPENAI_ROUTE
    # Resolving the canonical form again is a no-op.
    assert resolve(route.selector) == route


@pytest.mark.parametrize("selector", [None, "", "agent", "llm", "Agent", " LLM "])
def test_agent_selectors_resolve_the_configured_default(selector):
    assert resolve(selector, default="gpt-6-astra").selector == "openai/gpt-6-astra"
    for default in (None, "agent", ""):
        with pytest.raises(ProviderError, match="names the default LLM, but no default"):
            resolve(selector, default=default)


@pytest.mark.parametrize(("selector", "message"), [
    ("astra", "Unknown model 'astra': write <provider>/<model>, for example openai/astra; "
     "registered providers: "),
    ("GPT-6-astra", "Unknown model 'GPT-6-astra'"),
    ("o9-mini", "Unknown model 'o9-mini'"),
    ("acme/gpt-6-astra", "Unknown provider 'acme' in model 'acme/gpt-6-astra'; "
     "registered providers: "),
    ("openai/", "Model 'openai/' names no model after openai/"),
    ("openai/vendor/model", "Invalid openai model 'openai/vendor/model'"),
    ("openai/gpt 6", "Invalid openai model 'openai/gpt 6'"),
])
def test_unroutable_selectors_are_errors_that_list_the_providers(selector, message):
    with pytest.raises(ProviderError) as caught:
        resolve(selector)
    assert str(caught.value).startswith(message)
    if "registered providers: " in message:
        assert "openai" in str(caught.value).split("registered providers: ", 1)[1]


def test_native_profiles_and_llm_selectors_are_disjoint():
    names = {name for profile in list_model_profiles().values()
             for name in (profile["id"], *profile.get("aliases", []))}
    assert names and "pi05-droid" in names
    for name in names:
        assert "/" not in name and not name.endswith(".json"), name
        assert not name.startswith(tuple(prefix for prefix, _ in providers.SHORTHANDS)), name
        assert name.lower() not in providers.registry(), name
        assert not providers.is_agent_selector(name), name
        with pytest.raises(ProviderError, match="Unknown model"):
            resolve(name)


def test_runtime_default_model_is_a_canonical_selector():
    default = runtime_defaults()["model"]
    assert resolve(default).selector == default


# -------------------------------------------------------------------- provider registry


def test_official_openai_provider_is_registered():
    spec = providers.registry()["openai"]
    assert (spec.protocol, spec.base_url, spec.base_url_check_env) == (
        "responses", OFFICIAL, "OPENAI_BASE_URL",
    )
    assert (spec.key_env, spec.auth_scheme, spec.key_required) == ("OPENAI_API_KEY", "bearer", True)
    assert spec.environment_variables == ("OPENAI_API_KEY", "OPENAI_BASE_URL")


@pytest.mark.parametrize("selector", ["claude-opus-5-5", "anthropic/claude-opus-5-5",
                                      "Anthropic/claude-opus-5-5"])
def test_claude_shorthand_and_prefix_select_the_official_anthropic_provider(selector):
    route = resolve(selector)
    assert route.record() == {
        "selector": "anthropic/claude-opus-5-5", "provider": "anthropic",
        "protocol": "messages", "wire_model": "claude-opus-5-5",
        "endpoint": "https://api.anthropic.com/v1", "api_key_env": "ANTHROPIC_API_KEY",
        "base_url_env": "ANTHROPIC_BASE_URL",
    }
    spec = providers.registry()["anthropic"]
    assert (spec.auth_scheme, spec.api_version) == ("x-api-key", "2026-10-21")
    # Anthropic's client libraries read ANTHROPIC_BASE_URL without the version path.
    assert spec.base_url_check_aliases == ("https://api.anthropic.com",)


@pytest.mark.parametrize(("provider", "value", "named"), [
    ("openai", "https://api.openai.com/v1", True),
    ("openai", "HTTPS://API.OPENAI.COM:443/v1/", True),
    ("openai", "https://api.openai.com", False),
    ("openai", "https://api.openai.com/v1/extra", False),
    ("openai", "http://api.openai.com/v1", False),
    ("openai", "https://user@api.openai.com/v1", False),
    ("openai", "https://api.openai.com/v1?x=1", False),
    ("openai", "https://api.openai.com:8443/v1", False),
    ("openai", "not a url", False),
    ("anthropic", "https://api.anthropic.com", True),
    ("anthropic", "https://api.anthropic.com/", True),
    ("anthropic", "https://api.anthropic.com/v1", True),
    ("anthropic", "https://proxy.example.invalid", False),
])
def test_a_checked_variable_must_name_the_fixed_endpoint(provider, value, named):
    assert providers.registry()[provider].names_fixed_endpoint(value) is named


def test_registry_imports_only_the_standard_library():
    """The launcher (system python3) and model-server environments import every provider.

    Routing, discovery (local private modules included) and credential stripping must load
    nothing outside the standard library and this package.
    """
    code = (
        "import sys\n"
        "before = {name.split('.')[0] for name in sys.modules}\n"
        "from agentic_framework.models.llm.providers import registry, resolve\n"
        "from agentic_framework.common.environment import without_credentials\n"
        "assert 'openai' in registry() and resolve('gpt-x').wire_model == 'gpt-x'\n"
        "without_credentials({})\n"
        "loaded = {name.split('.')[0] for name in sys.modules} - before\n"
        "foreign = sorted(loaded - set(sys.stdlib_module_names) - {'agentic_framework'})\n"
        "assert not foreign, foreign\n"
    )
    source = Path(providers.__file__).resolve().parents[4]
    result = subprocess.run(
        [sys.executable, "-B", "-I", "-c", f"import sys; sys.path.insert(0, {str(source)!r})\n"
         + code],
        capture_output=True, text=True, check=False, timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_private_module_dropped_into_the_package_is_discovered(provider_directory):
    _spec_module(provider_directory, "private_example", name="example", protocol="responses",
                 base_url_env="EXAMPLE_LLM_BASE_URL", base_url_path="/v1",
                 key_env="EXAMPLE_LLM_API_KEY")
    _module(provider_directory, "private_helpers", "VALUE = 1\n")

    registered = providers.registry()
    assert {"openai", "example"} <= set(registered)
    route = resolve("example/vendor/model-1")
    assert route.record() == {
        "selector": "example/vendor/model-1", "provider": "example", "protocol": "responses",
        "wire_model": "vendor/model-1", "endpoint": "${EXAMPLE_LLM_BASE_URL}/v1",
        "api_key_env": "EXAMPLE_LLM_API_KEY", "base_url_env": "EXAMPLE_LLM_BASE_URL",
    }
    with pytest.raises(ProviderError) as caught:
        resolve("missing/model")
    # Sorted, and complete: local private modules, if any, are listed too.
    assert f"registered providers: {', '.join(registered)}." in str(caught.value)
    assert list(registered) == sorted(registered)


def test_duplicate_provider_names_are_an_error(provider_directory):
    _spec_module(provider_directory, "private_duplicate", name="openai", protocol="responses",
                 base_url="https://duplicate.example.invalid/v1")
    with pytest.raises(ProviderError) as caught:
        providers.registry()
    message = str(caught.value)
    assert "Provider 'openai' is defined by both" in message
    assert f"{providers.__name__}.openai" in message
    assert f"{providers.__name__}.private_duplicate" in message


def test_a_module_that_fails_to_import_is_reported_with_its_name(provider_directory):
    _module(provider_directory, "private_broken", "raise ImportError('missing dependency')\n")
    with pytest.raises(ProviderError, match=(
        rf"Provider module {providers.__name__}\.private_broken failed to import: "
        "ImportError: missing dependency"
    )):
        providers.registry()


@pytest.mark.parametrize("fields", [{"base_url": None}, {"base_url_env": "X_BASE_URL"}])
def test_a_provider_declares_exactly_one_endpoint_source(fields):
    declaration = {"name": "example", "protocol": "responses",
                   "base_url": "https://api.example.invalid/v1", **fields}
    with pytest.raises(ProviderError, match="declare one of base_url and base_url_env"):
        ProviderSpec(**declaration)


def test_protocol_and_response_hook_may_depend_on_the_model(provider_directory):
    _module(provider_directory, "private_split", """
        from agentic_framework.models.llm.providers import ProviderSpec

        PROVIDER = ProviderSpec(
            name="split",
            protocol=lambda wire: "messages" if wire.startswith("anthropic/") else "responses",
            base_url="https://split.example.invalid/v1",
            response_hook=lambda data: data["result"],
        )
    """)
    assert resolve("split/openai/x").protocol == "responses"
    assert resolve("split/anthropic/y").protocol == "messages"
    assert providers.registry()["split"].unwrap({"result": {"id": 1}}) == {"id": 1}
    assert resolve("split/x").api_key_env is None


def test_capabilities_may_depend_on_the_model(provider_directory):
    _module(provider_directory, "private_tiered", """
        from agentic_framework.models.llm.providers import Capabilities, ProviderSpec

        def capabilities(wire):
            if wire == "rejected":
                raise ValueError("not offered")
            if wire.startswith("pro-"):
                return Capabilities(reasoning_mode=False, max_output_tokens=512)
            return None

        PROVIDER = ProviderSpec(
            name="tiered", protocol="responses", base_url="https://tiered.example.invalid/v1",
            max_output_tokens=64, model_capabilities=capabilities,
        )
    """)
    assert resolve("tiered/pro-1").capabilities == providers.Capabilities(
        reasoning_mode=False, max_output_tokens=512,
    )
    # None keeps the provider's own capabilities.
    assert resolve("tiered/basic").capabilities == providers.Capabilities(max_output_tokens=64)
    with pytest.raises(ProviderError, match="^Invalid tiered model 'tiered/rejected': not offered"):
        resolve("tiered/rejected")
    with pytest.raises(ValueError, match="does not support reasoning_mode"):
        parse_configuration(["--mode", "agent", "--model", "tiered/pro-1",
                             "--reasoning-mode", "pro"])


# ------------------------------------------------------------- env-only keys and endpoints


def test_missing_key_is_a_configuration_failure_before_any_request():
    sent = []
    agent = ResponsesAgent("openai/gpt-6-astra",
                           transport=httpx.MockTransport(lambda r: sent.append(r)))
    with pytest.raises(BackendConfigurationError) as caught:
        agent.generate(_context())
    assert failure_kind(caught.value) == "configuration"
    assert str(caught.value).startswith("OPENAI_API_KEY is not set; export the API key")
    assert sent == []


def test_policy_ends_the_decision_on_a_missing_key_without_any_request():
    sent = []
    agent = ResponsesAgent("gpt-6-astra", transport=httpx.MockTransport(sent.append))
    policy = LLMMotionPolicy(agent, ScheduleConfig(h=5, k=5, max_steps=40),
                             action_spec=ActionSpec(), max_attempts=3)
    policy.bind(LiberoEmbodiment(image_size=4).info)
    policy.reset(Scene("routing", "Lift the cube."))
    with pytest.raises(BackendConfigurationError, match="OPENAI_API_KEY is not set"):
        policy.act(_observation())
    assert (sent, policy.calls, policy.decisions[0]["error_kind"]) == ([], 1, "configuration")


def test_key_from_the_environment_is_sent_as_a_bearer_token(monkeypatch):
    sent = []

    def handle(request):
        sent.append(request)
        return httpx.Response(200, json=REPLY)

    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    agent = ResponsesAgent("openai/gpt-6-astra", transport=httpx.MockTransport(handle))
    assert agent.generate(_context()).tool_calls[0]["call_id"] == "call_1"
    assert str(sent[0].url) == f"{OFFICIAL}/responses"
    assert sent[0].headers["authorization"] == f"Bearer {KEY}"
    assert "x-api-key" not in sent[0].headers
    assert json.loads(sent[0].content)["model"] == "gpt-6-astra"


@pytest.mark.parametrize("key", ["two words", "kéy-0123", "tab\tkey"])
def test_malformed_key_names_only_the_variable(monkeypatch, key):
    monkeypatch.setenv("OPENAI_API_KEY", key)
    agent = ResponsesAgent("gpt-6-astra", transport=httpx.MockTransport(pytest.fail))
    with pytest.raises(BackendConfigurationError) as caught:
        agent.generate(_context())
    assert str(caught.value).startswith("OPENAI_API_KEY is malformed")
    assert key not in str(caught.value)


@pytest.mark.parametrize("configured", ["https://proxy.example.invalid/v1", "not a url"])
def test_conflicting_openai_base_url_sends_nothing(monkeypatch, configured):
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    monkeypatch.setenv("OPENAI_BASE_URL", configured)
    agent = ResponsesAgent("gpt-6-astra", transport=httpx.MockTransport(pytest.fail))
    with pytest.raises(BackendConfigurationError, match="^OPENAI_BASE_URL ") as caught:
        agent.generate(_context())
    assert configured not in str(caught.value)


def test_environment_endpoint_is_read_only_at_send_time(provider_directory, monkeypatch, tmp_path):
    _spec_module(provider_directory, "private_canary", name="canary", protocol="responses",
                 base_url_env="CANARY_LLM_BASE_URL", base_url_path="/v1",
                 key_env="CANARY_LLM_API_KEY", auth_scheme="x-api-key")
    host = f"canary-{uuid.uuid4().hex}.invalid"
    sent = []

    def handle(request):
        sent.append(request)
        return httpx.Response(200, json=REPLY)

    archive = tmp_path / "api-requests"
    agent = ResponsesAgent(
        "canary/model-a", transport=httpx.MockTransport(handle),
        request_recorder=APIRequestRecorder(archive, auth_env="CANARY_LLM_API_KEY",
                                            auth_scheme="x-api-key"),
    )
    with monkeypatch.context() as patch:
        patch.setattr(os, "environ", ForbiddenEnviron())
        preview = agent.prepare_http_request(_context())
    assert str(preview.url) == f"{PLACEHOLDER_ORIGIN}/v1/responses"
    with pytest.raises(BackendConfigurationError, match="^CANARY_LLM_BASE_URL is not set"):
        agent.generate(_context())
    assert sent == []

    monkeypatch.setenv("CANARY_LLM_BASE_URL", f"https://{host}/prefix/")
    monkeypatch.setenv("CANARY_LLM_API_KEY", KEY)
    reply = agent.generate(_context())
    assert reply.metadata["model"] == "model-a"
    assert str(sent[0].url) == f"https://{host}/prefix/v1/responses"
    assert sent[0].headers["x-api-key"] == KEY and "authorization" not in sent[0].headers
    assert sent[0].content == preview.content
    # The archive names the key only by its variable.
    files = [path for path in archive.rglob("*") if path.is_file()]
    assert files and all(KEY not in p.read_text(errors="replace") for p in files)
    readable = next(archive.rglob("request.md")).read_text()
    assert "x-api-key: ${CANARY_LLM_API_KEY}" in readable


def test_optional_key_and_loopback_default_endpoint(provider_directory, monkeypatch):
    _spec_module(provider_directory, "private_loopback", name="loopback",
                 protocol="responses", base_url_env="LOOPBACK_LLM_BASE_URL",
                 base_url_default="http://127.0.0.1:8000/v1", key_env="LOOPBACK_LLM_API_KEY",
                 key_required=False)
    sent = []

    def handle(request):
        sent.append(request)
        return httpx.Response(200, json=REPLY)

    agent = ResponsesAgent("loopback/served-model", transport=httpx.MockTransport(handle))
    agent.generate(_context())
    assert str(sent[0].url) == "http://127.0.0.1:8000/v1/responses"
    assert "authorization" not in sent[0].headers


# ---------------------------------------------------------------- framework parser routes


@pytest.mark.parametrize("model", ["gpt-6-astra", "openai/gpt-6-astra"])
def test_parser_records_the_canonical_selector_and_route(model):
    args = parse_configuration(["--mode", "agent", "--model", model])
    assert args.model == "openai/gpt-6-astra"
    assert dict(args.llm_route) == OPENAI_ROUTE
    # The route is derived, never a configurable field.
    assert "llm_route" not in args.flat()


@pytest.mark.parametrize("model", ["agent", "llm", ""])
def test_parser_agent_keyword_selects_the_runtime_default(model):
    args = parse_configuration(["--mode", "agent", "--model", model])
    assert args.model == runtime_defaults()["model"]
    assert args.llm_route["selector"] == args.model


def test_native_profile_has_no_llm_route():
    args = parse_configuration(["--model-profile", "molmoact2-droid", "--mode", "vla"])
    assert args.llm_route is None


@pytest.mark.parametrize(("model", "message"), [
    ("pi05", "Unknown model 'pi05'"),
    ("acme/gpt-6-astra", "Unknown provider 'acme'"),
])
def test_parser_rejects_unroutable_models(model, message):
    with pytest.raises(ValueError, match=message):
        parse_configuration(["--mode", "agent", "--model", model])


@pytest.mark.parametrize("document", [
    {"llm": {"backend": "responses"}}, {"llm": {"env_file": "keys.env"}}, {"backend": "responses"},
])
def test_parser_rejects_removed_settings_as_unknown(tmp_path, document):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="^unknown"):
        parse_configuration(["--config", str(config)])


def test_parser_rejects_unsupported_capabilities(provider_directory):
    _spec_module(provider_directory, "private_plain", name="plain", protocol="responses",
                 base_url="https://plain.example.invalid/v1", reasoning_mode=False)
    _spec_module(provider_directory, "private_unbounded", name="unbounded", protocol="messages",
                 base_url="https://unbounded.example.invalid/v1")
    assert parse_configuration(["--mode", "agent", "--model", "plain/x"]).model == "plain/x"
    with pytest.raises(ValueError, match="the plain provider does not support reasoning_mode"):
        parse_configuration(["--mode", "agent", "--model", "plain/x", "--reasoning-mode", "pro"])
    # Messages requires max_tokens: a provider without a default needs llm.max_output_tokens.
    with pytest.raises(ValueError, match="uses the Messages protocol, which requires"):
        parse_configuration(["--mode", "agent", "--model", "unbounded/x"])
    assert parse_configuration(["--mode", "agent", "--model", "unbounded/x",
                                "--max-output-tokens", "64"]).max_output_tokens == 64


def test_factory_applies_declared_capabilities(provider_directory, tmp_path):
    _spec_module(provider_directory, "private_capable", name="capable", protocol="responses",
                 base_url="https://capable.example.invalid/v1", max_output_tokens=2048)
    args = parse_configuration(["--mode", "agent", "--model", "capable/x"])
    agent = factory.create_policy(
        args, ScheduleConfig(h=5, k=5, max_steps=40), AugmentationConfig(), tmp_path,
        ActionSpec(),
    ).agent
    assert (agent.supported_efforts, agent.max_output_tokens) == (None, 2048)
    explicit = parse_configuration(["--mode", "agent", "--model", "capable/x",
                                    "--supported-efforts", "high", "--max-output-tokens", "64"])
    agent = factory.create_policy(
        explicit, ScheduleConfig(h=5, k=5, max_steps=40), AugmentationConfig(), tmp_path / "b",
        ActionSpec(),
    ).agent
    assert (agent.supported_efforts, agent.max_output_tokens) == (("high",), 64)


# --------------------------------------------------- previews and records with no environment


def test_preview_policy_records_a_request_with_an_empty_environment(tmp_path, monkeypatch):
    for name in list(os.environ):
        monkeypatch.delenv(name)
    args = parse_configuration(["--mode", "preview", "--model", "gpt-6-astra"])
    policy = factory.create_policy(
        args, ScheduleConfig(h=5, k=5, max_steps=40), AugmentationConfig(), tmp_path / "trial",
        ActionSpec(),
    )
    assert isinstance(policy.agent, ResponsesAgent)
    assert policy.agent.route == OPENAI_ROUTE
    policy.bind(LiberoEmbodiment(image_size=4).info)
    policy.reset(Scene("routing", "Lift the cube."))
    request = policy.prepare_request(_observation())
    with monkeypatch.context() as patch:
        patch.setattr(os, "environ", ForbiddenEnviron())
        record = record_preview_request(policy.agent, request)
    manifest = json.loads(Path(record["request_manifest_path"]).read_text())
    assert (manifest["mode"], manifest["url"]) == ("preview", f"{OFFICIAL}/responses")
    assert (manifest["auth_env"], manifest["auth_scheme"]) == ("OPENAI_API_KEY", "bearer")
    readable = Path(record["request_readable_path"]).read_text()
    assert "Authorization: Bearer ${OPENAI_API_KEY}" in readable
    # Trajectory metadata names the canonical selector and its route.
    metadata = policy._recording_metadata()
    assert (metadata["model"], metadata["llm_route"]) == ("openai/gpt-6-astra", OPENAI_ROUTE)
    policy.close()


def test_experiment_preview_reads_the_recorded_route(tmp_path):
    config = {
        "policy_family": "llm", "control_hz": 30.0, "done_verification_steps": 0,
        "preview_context": False, "configuration": {}, "schedule": {"max_steps": 40},
        "effective_control": {"interface": "move_by"}, "model_profile_definition": None,
        "model": "openai/gpt-6-astra", "llm_route": OPENAI_ROUTE, "source_sha256": {},
        "dependency_source": "pinned", "python": sys.version, "packages": {},
        "policy_provenance_sha256": None, "simulator": {}, "h": 5, "k": 5,
    }
    report = write_experiment_preview(tmp_path, config, [])
    assert report["model"] == {"family": "llm", "model": "openai/gpt-6-astra",
                               "llm_route": OPENAI_ROUTE}
    assert "`openai/gpt-6-astra` uses responses at `https://api.openai.com/v1`" in (
        tmp_path / "preview.md"
    ).read_text()


class _OfflineLibero(LiberoEmbodiment):
    """A LIBERO embodiment whose worker calls are answered in process (no simulator)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **{**kwargs, "image_size": 4})

    def _rpc(self, method, **params):
        if method == "list_tasks":
            return [{"suite": params["suite"], "task_id": 0, "num_init_states": 1,
                     "instruction": "Lift the cube."}]
        if method == "metadata":
            return {"simulator": "offline"}
        if method == "reset":
            image = {"data": base64.b64encode(bytes(48)).decode(), "shape": [4, 4, 3]}
            return {"info": {"success": False}, "observation": {
                "images": {"agentview": image, "robot0_eye_in_hand": image},
                "state": {"eef_pos": [0.0, 0.0, 0.2], "eef_quat": [0.0, 0.0, 0.0, 1.0],
                          "gripper_width": [0.04], "joint_pos": [0.0] * 7},
                "sim_time": 0.0, "instruction": "Lift the cube.", "extra": {},
            }}
        raise AssertionError(f"unexpected simulator call {method}")

    def close(self):
        pass


def _load_summarizer(repo_root):
    """Import the repository's tools/summarize.py without bytecode."""
    spec = importlib.util.spec_from_file_location("summarize", repo_root / "tools/summarize.py")
    module = importlib.util.module_from_spec(spec)
    saved, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = saved
    return module


def test_llm_preview_run_records_the_route_and_never_the_key(tmp_path, monkeypatch, repo_root):
    """A complete framework preview: names are recorded, and no key or network is used."""
    from agentic_framework.environments.libero import embodiment as libero
    from agentic_framework.harness import run

    canary = f"sk-{uuid.uuid4().hex}"
    for name in list(os.environ):
        monkeypatch.delenv(name)
    monkeypatch.setenv("OPENAI_API_KEY", canary)
    monkeypatch.setattr(libero, "LiberoEmbodiment", _OfflineLibero)
    output = tmp_path / "run"

    assert run.main([
        "--mode", "preview", "--benchmark", "libero", "--suite", "libero_spatial",
        "--task-ids", "0", "--model", "gpt-6-astra", "--output-dir", str(output),
    ]) == 0

    config = json.loads((output / "config.json").read_text())
    assert (config["model"], config["llm_route"]) == ("openai/gpt-6-astra", OPENAI_ROUTE)
    assert config["configuration"]["llm"]["model"] == "openai/gpt-6-astra"
    assert not {"backend", "backend_options", "base_url", "api_key_env", "env_file"} & set(config)
    preview = json.loads((output / "preview.json").read_text())
    assert preview["model"]["llm_route"] == OPENAI_ROUTE
    # Every record labels the run by its selector: the wire ID alone does not name the provider.
    result = json.loads(next(output.glob("evals/*/result.json")).read_text())
    metrics = json.loads(next(output.glob("evals/*/policy-metrics.json")).read_text())
    assert result["policy"]["model"] == metrics["model"] == "openai/gpt-6-astra"
    summarize = _load_summarizer(repo_root)
    assert {row["model"] for row in summarize.build_report([output])["trials"]} == {
        "openai/gpt-6-astra"
    }
    manifests = list(output.rglob("api-requests/*/manifest.json"))
    assert len(manifests) == 1
    assert json.loads(manifests[0].read_text())["auth_env"] == "OPENAI_API_KEY"
    files = [path for path in output.rglob("*") if path.is_file()]
    assert all(canary.encode() not in path.read_bytes() for path in files)
