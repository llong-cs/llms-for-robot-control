"""The official OpenAI Responses API at its fixed public endpoint."""

from __future__ import annotations

from agentic_framework.models.llm.providers import ProviderSpec


def wire_model(model: str) -> str:
    """Send the ID after ``openai/`` unchanged; it must be one segment without whitespace."""
    if "/" in model or any(c.isspace() for c in model):
        raise ValueError("openai/ must be followed by one model ID without '/' or whitespace")
    return model


PROVIDER = ProviderSpec(
    name="openai",
    protocol="responses",
    base_url="https://api.openai.com/v1",
    # Client libraries honor OPENAI_BASE_URL; a value naming another endpoint is refused.
    base_url_check_env="OPENAI_BASE_URL",
    key_env="OPENAI_API_KEY",
    auth_scheme="bearer",
    wire_model=wire_model,
)
