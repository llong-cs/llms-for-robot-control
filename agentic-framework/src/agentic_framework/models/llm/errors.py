"""Failure taxonomy of one model call, shared by every backend and the policy attempt loop."""

from __future__ import annotations

import email.utils
import math
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any


class BackendError(RuntimeError):
    """A backend protocol, configuration, or inference failure."""

    category = "backend"

    def __init__(self, message="", *, usage=None, metadata=None):
        super().__init__(message)
        self.usage = dict(usage or {})
        self.metadata = dict(metadata or {})


class BackendConfigurationError(BackendError):
    """A local setup problem found before any model request was sent; never retried.

    For example missing credentials, an invalid local agent configuration, a
    closed agent, an exhausted or invalid test fixture, or a failed
    request-audit write. A provider response is never classified as one: every
    HTTP status is a transport/provider failure (``http_status_error``).
    """

    category = "configuration"


class BackendFormatError(BackendError):
    """Invalid model output; a clean, fresh request may repair it."""

    category = "format"

    def __init__(self, message, *, response_text="", usage=None, metadata=None):
        super().__init__(message)
        self.response_text = response_text
        self.usage = dict(usage or {})
        self.metadata = dict(metadata or {})


class BackendOutputLimitError(BackendFormatError):
    """Model output exceeded a configured bound; partial output is never decoded."""

    category = "output_limit"


# Failure kinds the policy resends, unchanged, as the next attempt of the same
# decision after a backoff. "format" is repaired with feedback instead, and
# "configuration" and "cancelled" end the trial at once.
RESENT_FAILURE_KINDS = frozenset({"transport", "internal"})


def failure_kind(exc: BaseException) -> str:
    """Classify a failed model call for the policy's per-decision attempt budget.

    - ``cancelled``: KeyboardInterrupt, SystemExit, any other BaseException that
      is not an Exception, or a CancelledError. Never retried.
    - ``configuration``: BackendConfigurationError. Never retried.
    - ``format``: BackendFormatError (invalid model output), repaired with feedback.
    - ``transport``: a transport or provider failure, i.e. every non-200 HTTP
      status (400, 401, 403, 404, 413, 422, 429, 5xx, ...), a timeout, a
      connection, proxy or stream error, a provider refusal, an incomplete or
      output-token-limited response, a malformed envelope, or any other
      BackendError.
    - ``internal``: any other Exception raised while building or sending the
      request or parsing its response, e.g. a framework bug.

    The policy resends ``transport`` and ``internal`` failures (see
    RESENT_FAILURE_KINDS); the provider may already have billed the failed request.
    """
    if not isinstance(exc, Exception) or type(exc).__name__ == "CancelledError":
        return "cancelled"
    if isinstance(exc, BackendConfigurationError):
        return "configuration"
    if isinstance(exc, BackendFormatError):
        return "format"
    if isinstance(exc, (BackendError, TimeoutError, ConnectionError, EOFError)):
        return "transport"
    # An httpx exception can only exist once httpx has been imported. A local
    # protocol violation or an unsupported URL scheme is our own request's fault.
    httpx = sys.modules.get("httpx")
    if (
        httpx is not None
        and isinstance(exc, (httpx.TransportError, httpx.DecodingError))
        and not isinstance(exc, (httpx.LocalProtocolError, httpx.UnsupportedProtocol))
    ):
        return "transport"
    return "internal"


# Statuses whose Retry-After header the policy honours before resending.
RETRY_AFTER_STATUSES = frozenset({429, 503})


def retry_after_seconds(headers: Mapping[str, str]) -> float | None:
    """Return the wait a provider requested (retry-after-ms or Retry-After), or None."""
    lowered = {str(key).lower(): value for key, value in headers.items()}
    candidates = []
    if "retry-after-ms" in lowered:
        candidates.append((lowered["retry-after-ms"], 0.001))
    if "retry-after" in lowered:
        candidates.append((lowered["retry-after"], 1.0))
    for value, unit in candidates:
        text = str(value).strip()
        try:
            seconds = float(text) * unit
        except ValueError:
            if unit != 1.0:
                continue
            try:
                when = email.utils.parsedate_to_datetime(text)
            except (TypeError, ValueError, IndexError):
                continue
            if when.tzinfo is None:
                when = when.replace(tzinfo=UTC)
            seconds = (when - datetime.now(UTC)).total_seconds()
        if math.isfinite(seconds):
            return max(0.0, seconds)
    return None


def http_status_error(endpoint: str, response: Any) -> BackendError:
    """Return the transport/provider failure for a non-200 HTTP response.

    Every status is resent by the policy after a backoff, including a
    deterministic rejection (400, 401, 403, 404, 413, 422, ...): a misconfigured
    provider (wrong key, model id or parameters) therefore fails every attempt,
    the trial is discarded after max_attempts, and a run whose every trial is
    discarded fails. Retry-After on 429 and 503 is recorded as retry_after_s.
    The response body is never included, because it may echo credentials.
    """
    status = response.status_code
    error = BackendError(f"{endpoint} endpoint rejected request with HTTP {status}")
    if status in RETRY_AFTER_STATUSES:
        delay = retry_after_seconds(response.headers)
        if delay is not None:
            error.metadata["retry_after_s"] = delay
    return error
