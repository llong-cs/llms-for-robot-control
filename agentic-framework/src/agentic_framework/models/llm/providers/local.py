"""A locally served OpenAI-compatible Chat Completions server, such as vLLM.

``local/<served model name>`` sends the served name verbatim, so it may contain ``/`` (a
server's default name is often the checkpoint's repository ID). The endpoint is
``LOCAL_LLM_BASE_URL``, including its ``/v1`` prefix, and defaults to the loopback server
``http://127.0.0.1:8000/v1``. ``LOCAL_LLM_API_KEY`` is optional: when set it is sent as a
Bearer token, otherwise requests carry no authentication header.

The served model must accept image inputs and tool calls, and its server must accept
``tool_choice: required``. No reasoning effort or reasoning mode is sent, and reasoning text a
server returns is recorded but never replayed.
"""

from __future__ import annotations

from agentic_framework.models.llm.providers import ProviderSpec

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"


PROVIDER = ProviderSpec(
    name="local",
    protocol="chat_completions",
    base_url_env="LOCAL_LLM_BASE_URL",
    base_url_default=DEFAULT_BASE_URL,
    key_env="LOCAL_LLM_API_KEY",
    auth_scheme="bearer",
    key_required=False,
    reasoning_mode=False,
    tool_choice="required",
)
