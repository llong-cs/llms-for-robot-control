"""OpenPi inference family with explicit LIBERO and DROID physical adapters."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import numpy as np

from agentic_framework._vendor.openpi_client import msgpack_numpy
from agentic_framework.models.vla.adapters import (
    OpenPiDroidAdapter,
    OpenPiLiberoAdapter,
    preprocess_observation,
)
from agentic_framework.models.vla.common import _NativePolicy

__all__ = ["OpenPiClient", "OpenPiPolicy", "preprocess_observation"]


class OpenPiClient:
    """Official WebSocket/MessagePack wire protocol with bounded waits and close."""

    def __init__(self, host: str, port: int, timeout_s: float = 120.0) -> None:
        import websockets.sync.client

        if not np.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be positive and finite")
        uri = host if host.startswith(("ws://", "wss://")) else f"ws://{host}:{port}"
        self._timeout_s = timeout_s
        self._ws = websockets.sync.client.connect(
            uri,
            compression=None,
            max_size=None,
            open_timeout=timeout_s,
        )
        try:
            self._metadata = self._receive()
        except BaseException:
            self.close()
            raise

    def _receive(self) -> dict[str, Any]:
        response = self._ws.recv(timeout=self._timeout_s)
        if isinstance(response, str):
            raise RuntimeError(f"OpenPI server error: {response}")
        result = msgpack_numpy.unpackb(response)
        if not isinstance(result, dict):
            raise ValueError("OpenPI response must be a mapping")
        return result

    def get_server_metadata(self) -> dict[str, Any]:
        return copy.deepcopy(self._metadata)

    def infer(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        try:
            self._ws.send(msgpack_numpy.packb(dict(observation)))
            return self._receive()
        except BaseException:
            # A timed-out response must not be read as the next call's reply.
            self.close()
            raise

    def reset(self) -> None:
        pass

    def close(self) -> None:
        socket, self._ws = self._ws, None
        if socket is not None:
            socket.close()


class OpenPiPolicy(_NativePolicy):
    """A common inference lifecycle; adapters select observation and action contracts."""

    supported_adapters = ("libero_osc", "droid_joint_velocity")

    def __init__(
        self,
        *,
        model_profile="pi05-libero",
        client=None,
        h=None,
        k=None,
        control_hz=None,
        profile="native",
        host=None,
        port=None,
        url=None,
        metadata_path=None,
        verified_metadata=None,
        timeout_s=None,
        policy_seed=None,
        strict_provenance=False,
    ):
        super().__init__(
            model_profile=model_profile,
            family="openpi",
            client=client,
            h=h,
            k=k,
            control_hz=control_hz,
            profile=profile,
            host=host,
            port=port,
            url=url,
            metadata_path=metadata_path,
            verified_metadata=verified_metadata,
            timeout_s=timeout_s,
            policy_seed=policy_seed,
            strict_provenance=strict_provenance,
        )

    def _create_adapter(self, model_profile):
        return {"libero_osc": OpenPiLiberoAdapter, "droid_joint_velocity": OpenPiDroidAdapter}[
            model_profile["adapter"]
        ]()

    @staticmethod
    def _create_client(transport, timeout_s):
        return OpenPiClient(transport.get("url", transport["host"]), transport["port"], timeout_s)
