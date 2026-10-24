"""LLM providers selected by the model string, one declarative module per provider.

A selector is ``<provider>/<model>``. The first segment always names the provider, so
``openai/gpt-6-astra`` is always the official OpenAI API. ``gpt-*`` is short for
``openai/gpt-*`` and ``claude-*`` for ``anthropic/claude-*``; ``""``, ``agent`` and ``llm``
name the configured default LLM. Any other bare ID is an error: nothing is guessed.

Every module in this package that defines ``PROVIDER = ProviderSpec(...)`` registers that
provider; a module without ``PROVIDER`` is a helper. Modules are found with ``pkgutil``, so a
file placed in this directory is loaded without an entry point or an installation. A
duplicate name or a module that fails to import is an error naming the module.

This package imports only the standard library, because the launcher and model-server
environments import configuration code. Wire protocols live in the shared agents, never in provider
modules; keys and environment-supplied endpoints are read only at send time.
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, field
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit

PROTOCOLS = ("responses", "messages", "chat_completions")
AGENT_SELECTORS = frozenset({"", "agent", "llm"})
# Bare model IDs that name their provider by convention; every other bare ID is an error.
SHORTHANDS = (("gpt-", "openai"), ("claude-", "anthropic"))

ROUTE_FIELDS = (
    "selector", "provider", "protocol", "wire_model", "endpoint", "api_key_env", "base_url_env",
)


class ProviderError(ValueError):
    """An unknown or invalid model selector, or an invalid provider module."""


def _comparable_url(value: str) -> tuple | None:
    """The parts of an http(s) URL that name an endpoint, or None for any other value.

    Scheme and host compare case-insensitively, a default port equals no port, and trailing
    slashes are ignored; user info, a query or a fragment never match a declared URL.
    """
    try:
        parts = urlsplit(value)
        port = parts.port
    except (ValueError, TypeError, AttributeError):
        return None
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https") or not parts.hostname:
        return None
    default = 443 if scheme == "https" else 80
    return (scheme, parts.hostname.lower(), port or default, parts.path.rstrip("/"),
            parts.username, parts.password, parts.query, parts.fragment)


@dataclass(frozen=True)
class Capabilities:
    """What one model accepts; declared in code by its provider, never configured.

    ``max_output_tokens`` is the default when llm.max_output_tokens is unset.
    ``tool_choice`` is the Chat Completions ``tool_choice`` the model's server accepts:
    ``required`` forces a call, and ``auto`` lets a reply without one through to the format
    repair. None sends ``required``; the other protocols fix their own tool choice.
    """

    reasoning_mode: bool = True
    max_output_tokens: int | None = None
    tool_choice: str | None = None


@dataclass(frozen=True)
class ProviderSpec:
    """Declaration of one provider: endpoint, authentication, model IDs and capabilities.

    Exactly one endpoint source is declared: a fixed ``base_url``, or ``base_url_env``, read
    at send time, with an optional ``base_url_default`` and a ``base_url_path`` appended to
    the variable's value. With a fixed URL, ``base_url_check_env`` names a variable that,
    when set, must name it (client libraries read it, so a mismatch is a configuration
    error); ``base_url_check_aliases`` are other spellings of the same endpoint that the
    variable may hold, such as the host without ``/v1`` that an SDK completes itself.
    ``wire_model`` validates and may rewrite the text after ``<name>/``; it returns the model
    ID sent on the wire and raises ValueError for an invalid ID. Without it the text is sent
    as written.

    ``reasoning_mode`` and ``max_output_tokens`` are the capabilities of every model of the
    provider. ``model_capabilities``, when declared, returns the Capabilities of one wire
    model instead (None keeps the provider's) and raises ValueError for a model the provider
    does not support, so that the configuration fails when it is parsed.
    ``api_version`` is the API version header a versioned protocol sends (Messages:
    ``anthropic-version``). ``tool_choice`` is the Chat Completions tool choice of every
    model (see Capabilities).
    ``prompt_caching`` makes the Messages wire mark the system prompt and the conversation
    tail as cache breakpoints (``cache_control``).
    """

    name: str
    protocol: str | Callable[[str], str]
    base_url: str | None = None
    base_url_env: str | None = None
    base_url_default: str | None = None
    base_url_path: str = ""
    base_url_check_env: str | None = None
    base_url_check_aliases: tuple[str, ...] = ()
    key_env: str | None = None
    auth_scheme: str | None = "bearer"
    key_required: bool = True
    api_version: str | None = None
    wire_model: Callable[[str], str] | None = None
    # Capabilities.
    reasoning_mode: bool = True
    max_output_tokens: int | None = None
    tool_choice: str | None = None
    model_capabilities: Callable[[str], Capabilities | None] | None = None
    # Applied to each decoded response envelope before the wire decodes it.
    response_hook: Callable[[Any], Any] | None = None
    prompt_caching: bool = False

    def __post_init__(self):
        if (self.base_url is None) == (self.base_url_env is None):
            raise ProviderError(f"Provider {self.name!r}: declare one of base_url and base_url_env")
        if self.key_env is None:
            # Without a key variable nothing is authenticated.
            object.__setattr__(self, "auth_scheme", None)

    def capabilities(self, wire_model: str) -> Capabilities:
        """The capabilities of one wire model; ValueError for a model the provider rejects."""
        if self.model_capabilities is not None:
            declared = self.model_capabilities(wire_model)
            if declared is not None:
                return declared
        return Capabilities(reasoning_mode=self.reasoning_mode,
                            max_output_tokens=self.max_output_tokens, tool_choice=self.tool_choice)

    @property
    def endpoint(self) -> str:
        """The recorded endpoint label: the fixed URL, or ``${VARIABLE}`` and its path."""
        if self.base_url is not None:
            return self.base_url
        return f"${{{self.base_url_env}}}{self.base_url_path}"

    @property
    def environment_variables(self) -> tuple[str, ...]:
        """Names of every variable this provider reads; values are never recorded."""
        names = (self.key_env, self.base_url_env, self.base_url_check_env)
        return tuple(name for name in names if name is not None)

    def names_fixed_endpoint(self, value: str) -> bool:
        """Whether a value of ``base_url_check_env`` names the fixed base URL or an alias."""
        if self.base_url is None:
            return False
        configured = _comparable_url(value)
        return configured is not None and configured in {
            _comparable_url(url) for url in (self.base_url, *self.base_url_check_aliases)
        }

    def unwrap(self, data: Any) -> Any:
        """Apply the declared response hook to a decoded response envelope."""
        return data if self.response_hook is None else self.response_hook(data)


@dataclass(frozen=True)
class Route:
    """A resolved selector. ``record()`` holds names and labels only, never values.

    ``base_url_env`` is the variable that supplies or checks the endpoint. ``capabilities``
    are those the provider declares for the wire model.
    """

    selector: str
    provider: str
    protocol: str
    wire_model: str
    endpoint: str
    api_key_env: str | None
    base_url_env: str | None
    spec: ProviderSpec = field(repr=False, compare=False)
    capabilities: Capabilities = field(default=Capabilities(), repr=False, compare=False)

    def record(self) -> dict[str, str | None]:
        return {name: getattr(self, name) for name in ROUTE_FIELDS}


def _discover(package: str, paths) -> dict[str, ProviderSpec]:
    found: dict[str, tuple[ProviderSpec, str]] = {}
    for info in sorted(pkgutil.iter_modules(paths), key=lambda item: item.name):
        module_name = f"{package}.{info.name}"
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            raise ProviderError(
                f"Provider module {module_name} failed to import: {type(exc).__name__}: {exc}"
            ) from exc
        spec = getattr(module, "PROVIDER", None)
        if spec is None:
            continue
        if spec.name in found:
            raise ProviderError(
                f"Provider {spec.name!r} is defined by both {found[spec.name][1]} and {module_name}"
            )
        found[spec.name] = (spec, module_name)
    return {name: spec for name, (spec, _) in sorted(found.items())}


@lru_cache(maxsize=1)
def registry() -> Mapping[str, ProviderSpec]:
    """Import every module of this package once and return the registered providers."""
    return MappingProxyType(_discover(__name__, __path__))


def is_agent_selector(value: str | None) -> bool:
    """Whether value names the configured default LLM instead of a model."""
    return value is None or (isinstance(value, str) and value.strip().lower() in AGENT_SELECTORS)


def resolve(selector: str | None, *, default: str | None = None) -> Route:
    """Resolve a model selector; ``""``, ``agent`` and ``llm`` resolve ``default`` instead."""
    if is_agent_selector(selector):
        if default is None or is_agent_selector(default):
            raise ProviderError(
                f"Model {selector or ''!r} names the default LLM, but no default is configured"
            )
        return resolve(default)
    text = selector.strip()
    providers = registry()
    known = ", ".join(providers) or "none"
    if "/" in text:
        name, rest = text.split("/", 1)
        name = name.lower()
    else:
        name = next((owner for prefix, owner in SHORTHANDS if text.startswith(prefix)), None)
        rest = text
        if name is None:
            raise ProviderError(
                f"Unknown model {text!r}: write <provider>/<model>, for example openai/{text}; "
                f"registered providers: {known}"
            )
    spec = providers.get(name)
    if spec is None:
        raise ProviderError(
            f"Unknown provider {name!r} in model {text!r}; registered providers: {known}. "
            "Each provider is a module under agentic_framework/models/llm/providers"
        )
    if not rest:
        raise ProviderError(f"Model {text!r} names no model after {name}/")
    try:
        wire = spec.wire_model(rest) if spec.wire_model else rest
    except ValueError as exc:
        raise ProviderError(f"Invalid {name} model {text!r}: {exc}") from None
    protocol = spec.protocol(wire) if callable(spec.protocol) else spec.protocol
    try:
        capabilities = spec.capabilities(wire)
    except ValueError as exc:
        raise ProviderError(f"Invalid {name} model {text!r}: {exc}") from None
    return Route(
        selector=f"{name}/{rest}", provider=name, protocol=protocol, wire_model=wire,
        endpoint=spec.endpoint, api_key_env=spec.key_env,
        base_url_env=spec.base_url_env or spec.base_url_check_env, spec=spec,
        capabilities=capabilities,
    )
