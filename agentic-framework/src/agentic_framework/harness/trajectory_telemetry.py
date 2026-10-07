"""Small, versioned receipts for recorded model and execution events."""

from __future__ import annotations

import copy
import math
from collections import Counter
from datetime import UTC, datetime

TOKEN_FIELDS = (
    "input_tokens", "output_tokens", "total_tokens", "cached_input_tokens",
    "cache_creation_input_tokens", "uncached_input_tokens", "reasoning_tokens",
)


def utc_now():
    return datetime.now(UTC).isoformat()


def nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def usage_receipt(usage, *, source="legacy_backend_usage", scope="request"):
    """Fallback for mock/Codex backends; never treat absent counts as zeros."""
    usage = usage if isinstance(usage, dict) else {}
    result = {key: value if type(value := usage.get(key)) is int and value >= 0 else None
              for key in TOKEN_FIELDS}
    valid = sum(value is not None for value in result.values())
    result.update(
        availability="reported" if all(result[key] is not None for key in
                                       ("input_tokens", "output_tokens", "total_tokens"))
        else "partial" if valid else "unavailable",
        source=source, scope=scope, input_semantics="total_including_cached",
    )
    return result


def llm_receipt(entry, *, model):
    """Merge transport facts with policy-side clocks without modifying replies."""
    backend = entry.get("backend") or {}
    receipt = copy.deepcopy(backend.get("telemetry") or {})
    receipt["schema_version"] = 1
    timing = receipt.setdefault("timing", {})
    timing.update({key: entry[key] for key in (
        "request_started_at", "request_finished_at", "context_build_s",
        "request_prepare_s", "policy_postprocessing_s", "policy_elapsed_s",
        # Backoff before a transport or internal resend; separate from every elapsed clock.
        "retry_wait_s",
    ) if key in entry})
    timing.update(client_elapsed_s=entry.get("elapsed_s"),
                  backend_elapsed_s=entry.get("backend_reported_elapsed_s"),
                  request_start_wall_s=entry.get("request_wall_s"),
                  relative_time_origin="policy.reset",
                  elapsed_clock="monotonic", wall_clock="UTC")
    receipt.setdefault("usage", usage_receipt(
        entry.get("usage"), source=backend.get("backend", "backend") + ".usage",
        scope=backend.get("usage_scope", "request"),
    ))
    receipt.setdefault("usage_raw", copy.deepcopy(entry.get("usage") or {}))
    receipt.setdefault("cost", {"status": "unavailable", "amount": None,
                                "currency": None, "source": None})
    receipt.setdefault("transport", {"round_trip_s": None, "http_status": None})
    receipt.setdefault("response", {"model": backend.get("response_model"),
                                    "id": backend.get("response_id")})
    receipt["request"] = {**receipt.get("request", {}), "model": model,
                          **entry.get("request_summary", {})}
    receipt["status"] = (
        "cancelled" if entry.get("error_kind") == "cancelled" else
        "invalid_response" if entry.get("validation_error") else
        "error" if entry.get("error") else "completed"
    )
    receipt["policy_outcome"] = entry.get("outcome", "error")
    receipt["error_kind"] = entry.get("error_kind")
    return receipt


def _conversation_restarted(turn):
    output = turn.get("model_output")
    backend = output.get("backend") if isinstance(output, dict) else None
    return isinstance(backend, dict) and backend.get("conversation_restarted") is True


def summarize_requests(turns, adjustments=()):
    """Count all physical requests, with per-field missingness and cost currencies."""
    telemetries = [turn.get("telemetry") or {} for turn in turns]
    usage = {}
    llm = [item for turn, item in zip(turns, telemetries) if turn.get("kind") == "llm"]
    for key in TOKEN_FIELDS:
        known = [item.get("usage", {}).get(key) for item in llm]
        known = [value for value in known if type(value) is int and value >= 0]
        usage[key] = {"known_sum": sum(known) if known else None,
                      "requests_with_value": len(known), "requests_total": len(llm),
                      "complete": len(known) == len(llm)}
    costs = {}
    cost_requests = 0
    for item in telemetries:
        cost = item.get("cost") or {}
        amount, currency = cost.get("amount"), cost.get("currency")
        if (cost.get("status") == "reported" and nonnegative(amount)
                and isinstance(currency, str) and currency):
            costs[currency] = costs.get(currency, 0.0) + amount
            cost_requests += 1
    durations = [turn.get("elapsed_s") for turn in turns]
    durations = [value for value in durations if nonnegative(value)]
    return {
        "schema_version": 1, "request_count": len(turns),
        "llm_requests": len(llm), "native_requests": len(turns) - len(llm),
        "transport_send_attempted": sum(item.get("transport", {}).get("send_attempted") is True
                                         for item in telemetries),
        "transport_not_sent": sum(item.get("transport", {}).get("send_attempted") is False
                                  for item in telemetries),
        "logical_decisions": len({turn.get("logical_decision_id", turn.get("request_id"))
                                  for turn in turns}),
        # Attempts beyond a decision's first, by the failure that caused them.
        "format_repair_requests": sum(turn.get("retry_reason") == "format" for turn in turns),
        "transport_retry_requests": sum(turn.get("retry_reason") == "transport" for turn in turns),
        "internal_retry_requests": sum(turn.get("retry_reason") == "internal" for turn in turns),
        # Wall time waited before resends (transport and internal).
        "retry_wait_s": sum(turn.get("retry_wait_s") or 0.0 for turn in turns),
        # Requests that began a new native thread after a failure lost the conversation.
        "conversation_restarts": sum(_conversation_restarted(turn) for turn in turns),
        "statuses": dict(Counter(item.get("status", "unknown") for item in telemetries)),
        "inference_wall_s_known": sum(durations) if durations else None,
        "requests_with_latency": len(durations), "usage": usage,
        "reported_cost_by_currency": costs, "requests_with_reported_cost": cost_requests,
        "cost_complete": cost_requests == len(turns) and bool(turns),
        "usage_adjustments": list(adjustments),
        "usage_adjustments_scope": "late session usage, not attributed to an individual request; excluded from request sums",
    }
