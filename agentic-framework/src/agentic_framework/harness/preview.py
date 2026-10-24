"""Zero-model-call request previews over the production policy and serializers."""

import json
from dataclasses import replace

from inspect_robots.rollout import derive_seed

from agentic_framework.common.io import plain
from agentic_framework.models.llm.errors import BackendConfigurationError
from agentic_framework.models.llm.scripted import scripted_reply


def preview_first_request(policy, embodiment, scene, *, eval_seed, epoch=0, directory=None):
    """Use the same reset and context as eval, without a policy or HTTP step."""
    policy.bind(embodiment.info)
    policy.reset(scene)
    trial_seed = derive_seed(eval_seed, scene.init_seed, epoch)
    observation = embodiment.reset(scene, seed=trial_seed)
    observation = replace(observation, extra={**observation.extra, "env_step": 0})
    report = {
        "kind": "first_request_preview",
        "api_calls": 0,
        "policy_steps": 0,
        "model_calls": 0,
        "trial_seed": trial_seed,
        "eval_seed": eval_seed,
        "scene_seed": scene.init_seed,
        "epoch": epoch,
        "simulator_reset": plain(embodiment.last_reset_info),
        "initially_solved": bool(embodiment.last_reset_info.get("success", False)),
    }
    if directory is not None:
        from agentic_framework.observability.experiment_preview import write_observation_preview

        report["observation"] = write_observation_preview(
            directory, observation, scene=scene, reset_info=embodiment.last_reset_info
        )
        info = embodiment.info
        report["embodiment"] = {
            "name": info.name,
            "control_hz": info.control_hz,
            "action_space": plain(info.action_space),
            "observation_space": plain(info.observation_space),
            "capabilities": sorted(info.capabilities),
            "is_simulated": info.is_simulated,
        }
    if report["initially_solved"]:
        report["request"] = None
        return report
    if hasattr(policy, "preview_request"):
        report["request"] = policy.preview_request(observation)
    else:
        request = policy.prepare_request(observation)
        report["request"] = record_preview_request(policy.agent, request)
        assert policy.calls == 0 and not policy.decisions and not policy.actions
    return report


class PreviewEmbodiment:
    """Record the real initial observation at the rollout's single reset."""

    def __init__(self, embodiment, directory):
        self.embodiment = embodiment
        self.directory = directory
        self.observation_artifact = None

    def __getattr__(self, name):
        return getattr(self.embodiment, name)

    def reset(self, scene, *, seed=None):
        from agentic_framework.observability.experiment_preview import write_observation_preview

        observation = self.embodiment.reset(scene, seed=seed)
        self.observation_artifact = write_observation_preview(
            self.directory, observation, scene=scene, reset_info=self.embodiment.last_reset_info
        )
        return observation


class OfflinePreviewAgent:
    """Exercise real context/servo scheduling using explicitly scripted replies.

    This is an engineering example of later contexts, not a prediction of future
    paid-model outputs. The wrapped agent's generate/authentication never runs.
    """

    is_offline_preview = True

    def __init__(self, agent, *, motion_delta=0.0):
        if getattr(agent, "request_recorder", None) is None:
            raise BackendConfigurationError("Context preview requires a request recorder")
        self.agent = agent
        self.model = agent.model
        self.route = getattr(agent, "route", None)
        self.motion_delta = motion_delta
        self.calls = 0

    def generate(self, request):
        artifact = record_preview_request(self.agent, request)
        self.calls += 1
        if self.calls <= 2:
            name = (
                "move_by_chunk"
                if any(t["name"] == "move_by_chunk" for t in request.tools)
                else "move_by"
            )
            arguments = {
                "deltas": {"dx": self.motion_delta},
                "note": "Scripted request preview: request this displacement from the observed pose.",
            }
        else:
            name = "done"
            arguments = {
                "summary": "Offline context preview finished; this is not a model task-success claim.",
                "hindsight": "These replies were scripted only to show request construction without an API call.",
            }
        tool = next(t for t in request.tools if t["name"] == name)
        properties = tool["parameters"]["properties"]
        # Derive the fixed horizon from the exact tool schema being previewed.
        if name == "move_by_chunk":
            horizon = properties["deltas"]["properties"]["dx"]["maxItems"]
            arguments = {
                "deltas": {
                    dimension: [value] * horizon for dimension, value in arguments["deltas"].items()
                },
                **({"note": arguments["note"]} if "note" in properties else {}),
            }
        elif "note" not in properties:
            arguments.pop("note", None)
        if "arm" in properties:
            arguments["arm"] = properties["arm"]["enum"][0]
        if "hindsight" not in properties:
            arguments.pop("hindsight", None)
        native = scripted_reply(
            request,
            {
                "tool_calls": [
                    {
                        "type": "function_call",
                        "call_id": f"preview-{self.calls}",
                        "name": name,
                        "arguments": json.dumps(arguments),
                    }
                ]
            },
        )
        reply = replace(
            native,
            metadata={
                "backend": "offline_request_preview",
                "model_calls": 0,
                "requested_model": self.model,
                "api_calls": 0,
                "scripted_preview_response": True,
                "request_artifact": artifact,
            },
        )
        self.agent.accept_reply(request, reply)
        return reply

    def reset(self):
        self.calls = 0
        self.agent.reset()

    def record_tool_results(self, results):
        self.agent.record_tool_results(results)

    def end_rollout(self):
        end = getattr(self.agent, "end_rollout", None)
        if end is not None:
            return end()
        return None

    def close(self):
        self.agent.close()


def record_preview_request(agent, request):
    """Record through the agent's production serializer without calling it."""
    recorder = getattr(agent, "request_recorder", None)
    if recorder is None:
        raise BackendConfigurationError("Request preview requires a request recorder")
    if hasattr(agent, "prepare_http_request"):
        prepared = agent.prepare_http_request(request)
    elif hasattr(agent, "prepare_request"):
        prepared = agent.prepare_request(request)
    else:
        raise BackendConfigurationError("Agent does not support a pure request preview")
    return recorder(request, prepared)
