"""Child-process environments without model credentials or provider endpoints.

Only a live LLM evaluator sends model requests. Simulator workers, native model servers and
every other evaluator receive an environment without keys, tokens, secrets, passwords,
``*_BASE_URL`` endpoints or any other variable a registered LLM provider reads.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

CREDENTIAL_NAME = re.compile(
    r".*(_API_KEY|_API_TOKEN|_TOKEN|_PASSWORD|_BASE_URL)|.*_SECRET.*", re.IGNORECASE
)


def provider_variables() -> frozenset[str]:
    """Every variable a registered provider reads: keys and base-URL variables."""
    # Lazy, and deliberately the one models import in common: the stripped names are derived
    # from the registered providers. The registry imports only the standard library.
    from agentic_framework.models.llm.providers import registry

    return frozenset(name for spec in registry().values() for name in spec.environment_variables)


def is_credential(name: str) -> bool:
    """Whether a variable name holds a key, token, secret, password or endpoint URL."""
    return CREDENTIAL_NAME.fullmatch(name) is not None


def without_credentials(env: Mapping[str, str]) -> dict[str, str]:
    """Return a copy of ``env`` without credentials and provider endpoint variables."""
    providers = provider_variables()
    return {
        name: value for name, value in env.items()
        if name not in providers and not is_credential(name)
    }
