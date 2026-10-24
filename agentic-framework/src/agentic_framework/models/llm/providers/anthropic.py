"""The official Anthropic Messages API at its fixed public endpoint.

The key is sent in ``x-api-key`` with ``anthropic-version: 2026-10-21``. Requests use
adaptive thinking; the configured effort is sent as ``output_config.effort`` unchanged, and
a model that rejects it fails with the provider's error. Prompt caching is on: the system
prompt carries a cache breakpoint and the request enables automatic caching of the
conversation tail.
"""

from __future__ import annotations

from agentic_framework.models.llm.providers import ProviderSpec

API_VERSION = "2026-10-21"
# Messages requires max_tokens; the default suits a non-streaming request.
DEFAULT_MAX_OUTPUT_TOKENS = 16000


PROVIDER = ProviderSpec(
    name="anthropic",
    protocol="messages",
    base_url="https://api.anthropic.com/v1",
    # Client libraries honor ANTHROPIC_BASE_URL; a value naming another endpoint is refused.
    # They read it without /v1 and add the version path themselves.
    base_url_check_env="ANTHROPIC_BASE_URL",
    base_url_check_aliases=("https://api.anthropic.com",),
    key_env="ANTHROPIC_API_KEY",
    auth_scheme="x-api-key",
    api_version=API_VERSION,
    reasoning_mode=False,
    max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
    prompt_caching=True,
)
