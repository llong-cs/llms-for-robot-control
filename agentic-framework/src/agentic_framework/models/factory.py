"""Create one policy per inference family using a resolved model profile."""

from __future__ import annotations

import importlib

from agentic_framework.configuration.profiles import load_model_profile, validate_native_horizon

VLA_FAMILIES = {
    "openpi": ("openpi", "OpenPiPolicy"),
    "molmoact2": ("molmoact2", "MolmoAct2Policy"),
}


def create_policy(args, schedule, augmentation, directory, action_spec=None):
    if args.policy_family != "llm":
        profile = load_model_profile(args.model_profile)
        validate_native_horizon(profile, schedule.h)
        if args.mode == "preview":
            from agentic_framework.harness.native_preview import NativePreviewPolicy

            return NativePreviewPolicy(args, directory)
        module_name, class_name = VLA_FAMILIES[profile["family"]]
        policy_class = getattr(
            importlib.import_module(f"agentic_framework.models.vla.{module_name}"), class_name
        )
        endpoint = (
            {"host": args.policy_host, "port": args.policy_port}
            if profile["transport"]["kind"] == "websocket"
            else {"url": args.policy_url}
        )
        return policy_class(
            model_profile=args.model_profile,
            h=schedule.h,
            k=schedule.k,
            control_hz=args.control_hz,
            profile=args.policy_profile,
            metadata_path=args.policy_metadata,
            timeout_s=args.inference_timeout,
            **({"policy_seed": args.policy_seed, "strict_provenance": True}
               if getattr(args, "policy_seed", None) is not None else {}),
            **endpoint,
        )
    from agentic_framework.harness.demonstration import (
        DEFAULT_DEMO_CONTENT,
        DEFAULT_DEMO_IMAGE_MAX_SIDE,
    )
    from agentic_framework.harness.policy import LLMMotionPolicy
    from agentic_framework.models.llm.backend import CodexBackend, ResponsesBackend

    if getattr(args, "reasoning_mode", None) is not None and args.backend == "codex":
        raise ValueError("reasoning_mode is not supported by Codex")
    if args.backend == "responses":
        from agentic_framework.observability.api_audit import APIRequestRecorder

        common = dict(
            model=args.model,
            base_url=args.base_url,
            timeout_s=args.inference_timeout,
            max_response_chars=args.max_response_chars,
            max_output_tokens=args.max_output_tokens,
            reasoning_mode=getattr(args, "reasoning_mode", None),
            supported_efforts=tuple(
                x.strip() for x in args.supported_efforts.split(",") if x.strip()
            )
            if args.supported_efforts is not None
            else None,
            log_dir=directory / "model",
            request_recorder=APIRequestRecorder(
                directory / "model/api-requests",
                mode="preview" if args.mode == "preview" else "live",
                auth_env=args.api_key_env,
            ),
        )
        backend = ResponsesBackend(
            **common, api_key_env=args.api_key_env, env_file=args.env_file
        )
    elif args.backend == "codex":
        backend = CodexBackend(
            model=args.model,
            log_dir=directory / "model",
            timeout_s=args.inference_timeout,
            max_response_chars=args.max_response_chars,
        )
    else:
        from agentic_framework.models.llm.extensions import load_backend_extension

        backend = load_backend_extension(args.backend).create(args, directory)
    if args.mode == "preview":
        from agentic_framework.harness.preview import CodexRequestRecorder, OfflinePreviewBackend

        if args.backend == "codex":
            backend.request_recorder = CodexRequestRecorder(directory / "model/codex-requests")
        if args.preview_context:
            backend = OfflinePreviewBackend(backend, motion_delta=args.preview_dx or 0.0)
    return LLMMotionPolicy(
        backend,
        schedule,
        action_spec=action_spec,
        augmentation=augmentation,
        observation_profile=args.observation_profile,
        demo=getattr(args, "demo", False),
        demo_path=getattr(args, "demo_path", None),
        demo_mode=getattr(args, "demo_mode", "ood"),
        demo_content=getattr(args, "demo_content", DEFAULT_DEMO_CONTENT),
        demo_image_max_side=getattr(args, "demo_image_max_side", DEFAULT_DEMO_IMAGE_MAX_SIDE),
        record_trajectory=getattr(args, "record_trajectory", False),
        cameras=tuple(c for c in args.cameras.split(",") if c),
        max_images=args.max_images,
        max_context_chars=args.max_context_chars,
        action_tools=args.action_tools,
        approver=args.approver,
        max_attempts=args.max_attempts,
        log_dir=directory,
    )
