"""Provider-neutral, nullable API usage and transport evidence.

These are observations, not billing estimates. Cache and reasoning token counts
are subsets; they must never be added to a provider's total a second time.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

from agentic_framework.common.redaction import scrub

_TOKEN_FIELDS = (
    "input_tokens", "output_tokens", "total_tokens", "cached_input_tokens",
    "cache_creation_input_tokens", "uncached_input_tokens", "reasoning_tokens",
)
_ID_HEADERS = ("x-request-id", "request-id", "apim-request-id", "x-ms-request-id")
_SAFE_HEADERS = frozenset((*_ID_HEADERS, "content-type", "content-length", "content-encoding",
                          "retry-after", "retry-after-ms", "openai-processing-ms", "anthropic-processing-ms",
                          "x-request-cost", "x-request-cost-usd", "x-cost-usd",
                          "x-cost-currency"))
_RATE_HEADERS = frozenset(
    f"{prefix}{kind}-{suffix}"
    for prefix in ("x-ratelimit-", "anthropic-ratelimit-")
    for kind in ("limit", "remaining", "reset", "requests", "tokens", "input-tokens", "output-tokens")
    for suffix in ("requests", "tokens", "limit", "remaining", "reset")
)
# The block a Messages 400 names, and the clause of a preserved-thinking prefix mismatch.
_MESSAGES_PATH = re.compile(r"\bmessages\.[0-9]+\.content\.[0-9]+\b")
_BINDING_MISMATCH = "bound to a different conversation"
_SECRET = re.compile(r"(^|[_-])(authorization|api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret|credentials?|cookie|signature)($|[_-])", re.I)


def _safe(value: Any, depth=0):
    """Keep bounded JSON evidence without credentials, arbitrary objects or NaN."""
    if depth > 8:
        return "[depth limit]"
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else "[non-finite]"
    if isinstance(value, str):
        value = scrub(value)
        return value[:2048] + ("[truncated]" if len(value) > 2048 else "")
    if isinstance(value, Mapping):
        return {str(key): "[redacted]" if _SECRET.search(str(key)) else _safe(item, depth + 1)
                for key, item in list(value.items())[:256]}
    if isinstance(value, (list, tuple)):
        return [_safe(item, depth + 1) for item in value[:256]]
    return "[unsupported value]"


def new_telemetry(protocol: str) -> dict:
    return {
        "schema_version": 1,
        "protocol": protocol,
        "usage": {**dict.fromkeys(_TOKEN_FIELDS), "availability": "unavailable",
                  "source": None, "field_sources": {}, "issues": [],
                  "input_semantics": "total_prompt_including_cache"},
        "usage_raw": None,
        "transport": {"http_status": None, "request_bytes": None, "response_bytes": None,
                      "round_trip_s": None, "request_id": None, "response_headers": {},
                      "started_at_unix_s": None, "ended_at_unix_s": None,
                      "outcome": "not_sent", "send_attempted": False, "time_to_first_token_s": None,
                      "time_to_first_token_availability": "unavailable_nonstreaming",
                      "round_trip_semantics": "client_send_to_complete_body_including_network_and_queue",
                      "response_bytes_semantics": "decoded_body_excluding_headers"},
        "cost": {"status": "unavailable", "amount": None, "currency": None,
                 "source": None, "raw": {}},
        "response": {"id": None, "model": None, "status": None, "finish_reason": None,
                     "service_tier": None},
    }


def canonical_usage(raw: Any, protocol: str, source="response.usage") -> dict:
    result = new_telemetry(protocol)["usage"]
    if raw is None:
        return result
    result["source"] = source
    issues = result["issues"]
    if not isinstance(raw, Mapping):
        result.update(availability="invalid", issues=["usage is not an object"])
        return result

    def count(path, target, nullable=False):
        value = raw
        for key in path.split("."):
            if value is None and nullable:
                # A null details object, like a null count, reports nothing.
                return
            if not isinstance(value, Mapping):
                issues.append(f"{path}: token details are not an object")
                return
            if key not in value:
                return
            value = value[key]
        if value is None and nullable:
            return
        if type(value) is not int or value < 0:
            issues.append(f"{path}: expected nonnegative integer")
            return
        result[target] = value
        result["field_sources"][target] = f"{source}.{path}"

    count("output_tokens", "output_tokens")
    count("total_tokens", "total_tokens")
    if protocol == "chat_completions":
        # prompt_tokens is the whole prompt; cached tokens are a subset. Servers may send null
        # details objects.
        count("prompt_tokens", "input_tokens")
        count("completion_tokens", "output_tokens")
        count("prompt_tokens_details.cached_tokens", "cached_input_tokens", nullable=True)
        count("completion_tokens_details.reasoning_tokens", "reasoning_tokens", nullable=True)
        if result["input_tokens"] is not None and result["cached_input_tokens"] is not None:
            if result["cached_input_tokens"] > result["input_tokens"]:
                issues.append("cached_input_tokens exceeds input_tokens")
                result["cached_input_tokens"] = None
            else:
                result["uncached_input_tokens"] = (
                    result["input_tokens"] - result["cached_input_tokens"])
                result["field_sources"]["uncached_input_tokens"] = (
                    "difference:input_tokens-cached_input_tokens")
    elif protocol == "messages":
        # Messages input_tokens excludes cache reads and writes; the total prompt is the sum.
        count("input_tokens", "uncached_input_tokens")
        count("cache_read_input_tokens", "cached_input_tokens", nullable=True)
        count("cache_creation_input_tokens", "cache_creation_input_tokens", nullable=True)
        components = ("uncached_input_tokens", "cached_input_tokens", "cache_creation_input_tokens")
        if all(result[key] is not None for key in components):
            result["input_tokens"] = sum(result[key] for key in components)
            result["field_sources"]["input_tokens"] = "sum:" + "+".join(
                result["field_sources"][key] for key in components)
        elif result["uncached_input_tokens"] is not None:
            issues.append("input_tokens: total prompt unavailable because cache components are missing or invalid")
        count("output_tokens_details.reasoning_tokens", "reasoning_tokens")
    else:
        count("input_tokens", "input_tokens")
        count("input_tokens_details.cached_tokens", "cached_input_tokens")
        count("input_tokens_details.cache_write_tokens", "cache_creation_input_tokens")
        count("output_tokens_details.reasoning_tokens", "reasoning_tokens")
        subsets = ("cached_input_tokens", "cache_creation_input_tokens")
        if result["input_tokens"] is not None:
            for key in subsets:
                if result[key] is not None and result[key] > result["input_tokens"]:
                    issues.append(f"{key} exceeds input_tokens")
                    result[key] = None
            if all(result[key] is not None for key in subsets):
                cache_total = sum(result[key] for key in subsets)
                if cache_total <= result["input_tokens"]:
                    result["uncached_input_tokens"] = result["input_tokens"] - cache_total
                    result["field_sources"]["uncached_input_tokens"] = "difference:input_tokens-cached_input_tokens-cache_creation_input_tokens"
                else:
                    issues.append("cache_read+cache_write exceeds input_tokens")
                    for key in subsets:
                        result[key] = None
    if (result["reasoning_tokens"] is not None and result["output_tokens"] is not None
            and result["reasoning_tokens"] > result["output_tokens"]):
        issues.append("reasoning_tokens exceeds output_tokens")
        result["reasoning_tokens"] = None
    if result["input_tokens"] is not None and result["output_tokens"] is not None:
        computed = result["input_tokens"] + result["output_tokens"]
        if result["total_tokens"] is None and "total_tokens" not in raw:
            result["total_tokens"] = computed
            result["field_sources"]["total_tokens"] = "sum:input_tokens+output_tokens"
        elif result["total_tokens"] is not None and result["total_tokens"] != computed:
            issues.append("total_tokens differs from input_tokens+output_tokens")
    present = any(result[key] is not None for key in _TOKEN_FIELDS)
    complete = all(result[key] is not None for key in ("input_tokens", "output_tokens"))
    result["availability"] = ("reported" if complete and not issues else "partial" if present
                              else "invalid" if issues else "unavailable")
    return result


def _money(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _record_cost(telemetry, value, source, currency=None):
    cost = telemetry["cost"]
    cost["raw"][source] = _safe(value)
    amount = value
    if isinstance(value, Mapping):
        amount = value.get("amount", value.get("total"))
        currency = value.get("currency", currency)
    if currency is not None:
        cost["raw"][source + ".currency"] = _safe(currency)
    amount = _money(amount)
    explicit_currency = (currency.upper() if isinstance(currency, str)
                         and re.fullmatch(r"[A-Za-z]{3}", currency) else None)
    if cost["status"] == "unavailable" and amount is not None and explicit_currency is not None:
        cost.update(status="reported", amount=amount, currency=explicit_currency, source=source)


def _cost_fields(telemetry, data, path):
    if not isinstance(data, Mapping):
        return
    currency = data.get("currency", data.get("cost_currency"))
    for key in ("cost", "total_cost", "response_cost", "billed_cost"):
        if key in data:
            _record_cost(telemetry, data[key], f"{path}.{key}", currency)
    for key in ("cost_usd", "total_cost_usd", "billed_cost_usd"):
        if key in data:
            _record_cost(telemetry, data[key], f"{path}.{key}", "USD")
    for key in ("cost_details", "billing_details"):
        if key in data:
            telemetry["cost"]["raw"][f"{path}.{key}"] = _safe(data[key])
    billing = data.get("billing")
    if isinstance(billing, Mapping):
        for key in ("cost", "total_cost", "billed_cost"):
            if key in billing:
                _record_cost(telemetry, billing[key], f"{path}.billing.{key}", billing.get("currency"))


def observe_http_response(telemetry, response):
    """Record allowlisted headers and decoded body length, never response text."""
    transport = telemetry["transport"]
    headers = {key.lower(): _safe(value) for key, value in response.headers.items()
               if key.lower() in _SAFE_HEADERS | _RATE_HEADERS}
    transport.update(http_status=response.status_code, response_bytes=len(response.content),
                     response_headers=headers, outcome="response_received",
                     request_id=next((headers[key] for key in _ID_HEADERS if key in headers), None))
    for key in ("x-request-cost-usd", "x-cost-usd", "x-request-cost"):
        if key in headers:
            _record_cost(telemetry, headers[key], f"response.headers.{key}",
                         "USD" if key.endswith("-usd") else headers.get("x-cost-currency"))


def observe_response_json(telemetry, data, *, source="response"):
    """Inspect only evidence fields at recognized response envelope levels.

    This runs before protocol validation, so billed usage on refused, incomplete,
    malformed-output and HTTP-error responses remains available for analysis.
    """
    for _ in range(4):
        if not isinstance(data, Mapping):
            return
        _cost_fields(telemetry, data, source)
        usage = data.get("usage")
        if usage is not None:
            telemetry["usage_raw"] = _safe(usage)
            telemetry["usage"] = canonical_usage(usage, telemetry["protocol"], f"{source}.usage")
            _cost_fields(telemetry, usage, f"{source}.usage")
        for key in ("id", "model", "status", "service_tier", "system_fingerprint", "created_at"):
            value = data.get(key)
            if isinstance(value, (str, int, float)) and not isinstance(value, bool):
                telemetry["response"][key] = _safe(value)
        reason = data.get("stop_reason", data.get("finish_reason"))
        if isinstance(reason, str):
            telemetry["response"]["finish_reason"] = _safe(reason)
        for key in ("stop_details", "incomplete_details"):
            details = data.get(key)
            if isinstance(details, Mapping):
                telemetry["response"][key] = {
                    field: _safe(details[field]) for field in ("type", "category", "reason")
                    if isinstance(details.get(field), str)
                }
        choices = data.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
            # Chat Completions: the first choice's finish reason, or a refusal it carries.
            reason = choices[0].get("finish_reason")
            if isinstance(reason, str):
                telemetry["response"]["finish_reason"] = _safe(reason)
            message = choices[0].get("message")
            if isinstance(message, Mapping) and isinstance(message.get("refusal"), str) and (
                message["refusal"]
            ):
                telemetry["response"]["finish_reason"] = "refusal"
        output = data.get("output")
        if isinstance(output, list) and any(
            isinstance(item, Mapping) and isinstance(item.get("content"), list)
            and any(isinstance(part, Mapping) and part.get("type") == "refusal"
                    for part in item["content"])
            for item in output
        ):
            telemetry["response"]["finish_reason"] = "refusal"
        error = data.get("error")
        if isinstance(error, Mapping):
            record = {key: _safe(error[key]) for key in ("type", "code")
                      if isinstance(error.get(key), str)}
            message = error.get("message")
            if telemetry["protocol"] == "messages" and isinstance(message, str):
                # Diagnose a rejected thinking replay without storing the message text.
                path = _MESSAGES_PATH.search(message)
                if path is not None:
                    record["path"] = path.group(0)
                record["binding_mismatch"] = _BINDING_MISMATCH in message
            telemetry["response"]["error"] = record
        if any(key in data for key in ("output", "content", "choices")) or not isinstance(
            data.get("result"), Mapping
        ):
            return
        data, source = data["result"], source + ".result"


def attach_exception_telemetry(exc, metadata, usage):
    """Preserve exception identity (including cancellation) while carrying audit data."""
    telemetry = metadata["telemetry"]
    # The same rule as errors.failure_kind: any non-Exception BaseException cancels.
    cancelled = not isinstance(exc, Exception) or type(exc).__name__ == "CancelledError"
    telemetry["error"] = {"type": type(exc).__name__, "cancelled": cancelled}
    if telemetry["transport"]["outcome"] == "in_flight":
        telemetry["transport"]["outcome"] = "cancelled" if cancelled else "transport_error"
    try:
        exc.metadata = {**metadata, **getattr(exc, "metadata", {})}
        exc.usage = {**usage, **getattr(exc, "usage", {})}
    except (AttributeError, TypeError):
        pass
