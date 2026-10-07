"""Execute native DROID policy chunks through the shared rollout harness."""

from __future__ import annotations

import numpy as np
from inspect_robots.controller import DefaultController
from inspect_robots.types import Action

from agentic_framework.models.vla.common import _NativePolicy, _numeric


class NativeDroidController:
    """Reuse Inspect prefix scheduling and convert native policies per control step.

    This class never advances an environment. Rollout alone executes and scores
    its returned native position action. Positive K limits each executed prefix.
    """

    def __init__(self, policy=None, *, k=None):
        if policy is not None and (
            not isinstance(policy, _NativePolicy) or policy.adapter.execution_kind != "droid"
        ):
            raise TypeError("NativeDroidController policy must be a DROID policy")
        if k is None:
            k = policy.k if policy is not None else 15
        if policy is not None and k != policy.k:
            raise ValueError("controller K must match policy K")
        if type(k) is not int or k < 1:
            raise ValueError("native DROID K must be positive; K=-1 is supported only by move_by")
        self.k = k
        self._inner = DefaultController(replan_interval=k)
        self._pending = None

    def next_action(self, policy, observation, t, store):
        if (
            not isinstance(policy, _NativePolicy)
            or policy.adapter.execution_kind != "droid"
            or policy.k != self.k
        ):
            raise ValueError("NativeDroidController requires a DROID policy with matching K")
        before = policy._native_controller_active
        policy._native_controller_active = True
        try:
            action = self._inner.next_action(policy, observation, t, store)
        finally:
            policy._native_controller_active = before
        raw = _numeric(action.data, (8,), "DROID buffered action").astype(np.float64)
        meta = dict(action.meta)
        meta["droid_raw_proposal"] = raw.tolist()
        target, conversion_meta = policy.native_action(raw, observation)
        meta.update(conversion_meta)
        meta["droid_native_target"] = target.tolist()
        meta["control_mode"] = "pd_joint_pos"
        self._pending = (policy, t, observation, store)
        return Action(target, meta)

    def record_applied(self, action):
        """Called only after a successful outer env.step; log the reviewed action."""
        if self._pending is None:
            raise RuntimeError("no native DROID action is pending")
        policy, t, observation, store = self._pending
        self._pending = None
        policy.record_action(t, action, observation)
        store["native_droid_applied_steps"] = store.get("native_droid_applied_steps", 0) + 1

    def finalize(self, policy, observation, t, reason, store):
        # A failed or rejected pending action was never applied and is not counted.
        self._pending = None
        count = store.get("native_droid_applied_steps", 0)
        store["controller_totals"] = {
            "actual_steps": count,
            "tracking_steps": count,
            "holding_steps": 0,
            "stop_steps": 0,
            "native_steps": count,
        }
