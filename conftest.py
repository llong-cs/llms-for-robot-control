"""Shared offline guarantees for every test: no credentials, network or real backoff sleeps."""

from __future__ import annotations

import os
import socket
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent
_SECRET_SUFFIXES = ("_API_KEY", "_API_TOKEN", "_TOKEN", "_BASE_URL")


def _is_local(host) -> bool:
    host = host.decode() if isinstance(host, bytes) else str(host)
    return host in {"localhost", "::1", ""} or host.startswith("127.")


@pytest.fixture(autouse=True)
def offline_environment(monkeypatch):
    """Hide provider credentials and endpoints, and refuse non-loopback connections."""
    for name in list(os.environ):
        if name.endswith(_SECRET_SUFFIXES):
            monkeypatch.delenv(name)
    original_connect = socket.socket.connect
    original_getaddrinfo = socket.getaddrinfo

    def guarded_connect(sock, address):
        if not isinstance(address, (str, bytes)) and not _is_local(address[0]):
            raise RuntimeError(f"Tests must not open network connections: {address!r}")
        return original_connect(sock, address)

    def guarded_getaddrinfo(host, *args, **kwargs):
        if host is not None and not _is_local(host):
            raise RuntimeError(f"Tests must not resolve network hosts: {host!r}")
        return original_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)


@pytest.fixture(autouse=True)
def no_real_backoff(monkeypatch):
    """Fail fast instead of sleeping when a test reaches the policy backoff unpatched."""
    from agentic_framework.harness import policy

    def refuse(seconds):
        if seconds > 0:
            raise AssertionError(f"Unpatched policy backoff of {seconds:.3f} s")

    monkeypatch.setattr(policy, "_sleep", refuse)


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT
