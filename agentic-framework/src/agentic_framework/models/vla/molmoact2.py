"""MolmoAct2 DROID HTTP transport and native joint-position policy."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping

from agentic_framework.models.vla.common import _NativePolicy, droid_images, droid_state


class MolmoAct2Client:
    """Official json_numpy HTTP protocol without its global json monkey patch."""

    def __init__(self, url, timeout_s=120.0, *, transport=None):
        import httpx

        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.url = url
        self._client = httpx.Client(timeout=timeout_s, transport=transport)
        try:
            response = self._client.get(url)
            response.raise_for_status()
            self._metadata = response.json()
            if not isinstance(self._metadata, dict):
                raise ValueError("MolmoAct2 server metadata must be an object")
        except BaseException:
            self.close()
            raise

    def get_server_metadata(self):
        return copy.deepcopy(self._metadata)

    def infer(self, payload):
        import json_numpy

        if self._client is None:
            raise RuntimeError("MolmoAct2 client is closed")
        try:
            response = self._client.post(
                self.url,
                content=json_numpy.dumps(payload),
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            result = json_numpy.loads(response.text)
            if not isinstance(result, Mapping):
                raise ValueError("MolmoAct2 response must be an object")
            return result
        except BaseException:
            self.close()
            raise

    def reset(self):
        pass

    def close(self):
        client, self._client = (self._client, None)
        if client is not None:
            client.close()


class MolmoAct2Policy(_NativePolicy):
    supported_adapters = ("droid_joint_position", "libero_osc")
    requires_droid_controller = False

    def __init__(
        self,
        *,
        model_profile="molmoact2-droid",
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
            family="molmoact2",
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

    @staticmethod
    def _create_client(transport, timeout_s):
        return MolmoAct2Client(transport["url"], timeout_s)

    def _create_adapter(self, model_profile):
        from agentic_framework.models.vla.adapters import DroidJointAdapter, MolmoAct2LiberoAdapter

        return {"libero_osc": MolmoAct2LiberoAdapter, "droid_joint_position": DroidJointAdapter}[
            model_profile["adapter"]
        ]()

    def _request(self, observation, instruction):
        if self.adapter.execution_kind == "libero":
            return self.adapter.request(self, observation, instruction)
        external, wrist = droid_images(observation)
        return {
            "external_cam": external,
            "wrist_cam": wrist,
            "instruction": instruction,
            "state": droid_state(observation),
        }

    def _request_state(self, request):
        return request["state"]
