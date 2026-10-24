"""Local, fail-closed HTTP request archives inspired by Inspect Robots wire capture.

The already-built Responses, Messages or Chat Completions request is the source of truth. This
boundary does not construct prompts, manage context, inspect credentials, or send requests.
"""

from __future__ import annotations

import base64
import copy
import fcntl
import hashlib
import io
import json
import os
import re
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

import httpx
from PIL import Image

from agentic_framework.harness.types import AgentContext

_SEQUENCE = re.compile(r"^(\d{6,})-")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_IMAGE_URL = re.compile(r"data:(image/[A-Za-z0-9.+-]+);base64,(.*)\Z", re.DOTALL)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _slug(value: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-")[:32] or "request"
    return f"{readable}-{_sha256(value.encode('utf-8'))[:8]}"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


def _fence(value: str, language: str = "") -> str:
    # Prompt text may itself contain Markdown fences. Keep the entire text literal.
    longest = max((len(item) for item in re.findall(r"`+", value)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{language}\n{value}\n{fence}\n"


def _link(path: Path, label: str) -> str:
    # Absolute paths are required by the desktop file/image renderer.
    target = str(path).replace("<", "%3C").replace(">", "%3E").replace("\n", "%0A")
    return f"[{label}](<{target}>)"


def _write_new(path: Path, content: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    with os.fdopen(os.open(path, flags, 0o600), "wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _append(path: Path, content: bytes, *, initial: bytes = b"") -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW
    with os.fdopen(os.open(path, flags, 0o600), "ab") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise OSError("The request archive index must be a regular file")
        if os.fstat(stream.fileno()).st_size == 0 and initial:
            stream.write(initial)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def _exclusive_root(directory: Path):
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW
    with os.fdopen(os.open(directory / ".audit.lock", flags, 0o600), "r+b") as lock:
        if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):
            raise OSError("The request archive lock must be a regular file")
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _safe_headers(request: httpx.Request) -> dict[str, str]:
    # Do not iterate arbitrary headers, or read Authorization/Cookie values.
    safe = {}
    for name in ("content-type", "accept", "content-length", "anthropic-version",
                 "anthropic-beta"):
        value = request.headers.get(name)
        if value is None:
            continue
        if name == "content-length":
            allowed = re.fullmatch(r"[0-9]+", value) is not None
        elif name == "anthropic-version":
            allowed = re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value) is not None
        elif name == "anthropic-beta":
            allowed = re.fullmatch(r"[a-z0-9-]+(,\s*[a-z0-9-]+)*", value) is not None
        elif name == "content-type":
            allowed = (
                re.fullmatch(r"application/json(?:\s*;\s*charset=utf-8)?", value, re.IGNORECASE)
                is not None
            )
        else:
            allowed = value.lower() in {"application/json", "*/*"}
        if allowed:
            safe[name] = value
    return safe


def _body(raw: bytes, protocol: str | None = None) -> dict[str, Any]:
    def reject_constant(_):
        raise ValueError("The request archive requires finite JSON values")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("The request archive rejects ambiguous duplicate JSON keys")
            result[key] = value
        return result

    body = json.loads(raw, parse_constant=reject_constant, object_pairs_hook=unique_object)
    if protocol == "chat_completions":
        # A Chat body also has messages[]; it is never validated as a Messages body.
        _validate_chat(body)
        return body
    if isinstance(body, dict) and "messages" in body:
        if "input" in body:
            raise ValueError("Request archive rejects mixed Responses and Messages payloads")
        _validate_messages(body)
        return body
    if not isinstance(body, dict) or not isinstance(body.get("input"), list):
        raise ValueError("Request recording requires Responses input[].content")
    for message in body["input"]:
        if not isinstance(message, dict):
            raise ValueError("Responses input items must be objects")
        item_type = message.get("type")
        if item_type == "reasoning":
            if "encrypted_content" in message and not isinstance(message["encrypted_content"], str):
                raise ValueError("Responses encrypted reasoning must contain string content")
            continue
        if item_type in {"function_call", "function_call_output"}:
            required = (
                ("call_id", "name", "arguments")
                if item_type == "function_call"
                else ("call_id", "output")
            )
            if any(not isinstance(message.get(key), str) for key in required):
                raise ValueError("Responses native tool items require string fields")
            continue
        if not isinstance(message.get("content"), list):
            raise ValueError("Request recording requires Responses message content arrays")
        for part in message["content"]:
            if not isinstance(part, dict):
                raise ValueError("Responses content parts must be objects")
            if part.get("type") in {"input_text", "output_text"}:
                if not isinstance(part.get("text"), str):
                    raise ValueError("Responses text content must contain string text")
            elif part.get("type") == "refusal":
                if not isinstance(part.get("refusal"), str):
                    raise ValueError("Responses refusal must contain string content")
            elif part.get("type") == "input_image":
                if not isinstance(part.get("image_url"), str):
                    raise ValueError("Responses input_image must contain an inline image_url")
                if not _IMAGE_URL.fullmatch(part["image_url"]):
                    raise ValueError(
                        "Only inline base64 images can be archived; no URLs are fetched"
                    )
            else:
                raise ValueError("Unsupported Responses message content for request archive")
    return body


def _validate_message_content(content, *, nested=False):
    if isinstance(content, str):
        return
    if not isinstance(content, list):
        raise ValueError("Messages content must be string text or a content array")
    for part in content:
        if not isinstance(part, dict):
            raise ValueError("Messages content blocks must be objects")
        kind = part.get("type")
        if kind == "text":
            if not isinstance(part.get("text"), str):
                raise ValueError("Messages text blocks require string text")
        elif kind == "image":
            source = part.get("source")
            if (not isinstance(source, dict) or source.get("type") != "base64"
                    or not isinstance(source.get("data"), str)
                    or not isinstance(source.get("media_type"), str)
                    or not re.fullmatch(r"image/[A-Za-z0-9.+-]+", source["media_type"])):
                raise ValueError("Only inline base64 images can be archived; no URLs are fetched")
        elif kind == "tool_use" and not nested:
            if (any(not isinstance(part.get(key), str) for key in ("id", "name"))
                    or not isinstance(part.get("input"), dict)):
                raise ValueError("Messages tool_use requires string id/name and object input")
        elif kind == "tool_result" and not nested:
            if not isinstance(part.get("tool_use_id"), str):
                raise ValueError("Messages tool_result requires a string tool_use_id")
            _validate_message_content(part.get("content", []), nested=True)
        elif kind == "thinking" and not nested:
            if (not isinstance(part.get("thinking"), str)
                    or ("signature" in part and not isinstance(part["signature"], str))):
                raise ValueError("Messages thinking and signatures must be strings")
        elif kind == "redacted_thinking" and not nested:
            if not isinstance(part.get("data"), str):
                raise ValueError("Messages redacted thinking must contain string data")
        else:
            raise ValueError("Unsupported Messages content block for request archive")


def _validate_messages(body):
    if not isinstance(body["messages"], list):
        raise ValueError("Request recording requires Messages messages[]")
    system = body.get("system", "")
    if not isinstance(system, str):
        if not isinstance(system, list) or any(
            not isinstance(part, dict) or part.get("type") != "text"
            or not isinstance(part.get("text"), str) for part in system
        ):
            raise ValueError("Messages system must contain string text or text blocks")
    for message in body["messages"]:
        if not isinstance(message, dict) or message.get("role") not in {"user", "assistant"}:
            raise ValueError("Messages entries require user or assistant role")
        _validate_message_content(message.get("content"))


def _validate_chat(body):
    """Check the parts the archive reads: user text and inline images."""
    if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
        raise ValueError("Request recording requires Chat Completions messages[]")
    for message in body["messages"]:
        if not isinstance(message, dict):
            raise ValueError("Chat messages must be objects")
        content = message.get("content")
        if message.get("role") != "user" or not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") not in {"text", "image_url"}:
                raise ValueError("Unsupported Chat user content part for request archive")
            if part["type"] == "text" and not isinstance(part.get("text"), str):
                raise ValueError("Chat text parts require string text")
            image = part.get("image_url")
            if part["type"] == "image_url" and not (
                isinstance(image, dict) and isinstance(image.get("url"), str)
                and _IMAGE_URL.fullmatch(image["url"])
            ):
                raise ValueError("Only inline base64 images can be archived; no URLs are fetched")


def _message_parts(content, prefix):
    if not isinstance(content, list):
        return
    for index, part in enumerate(content):
        path = [*prefix, index]
        yield path, part
        if part["type"] == "tool_result":
            yield from _message_parts(part.get("content"), [*path, "content"])


def _write_image(encoded, mime, directory, order, location):
    original = base64.b64decode(encoded, validate=True)
    with Image.open(io.BytesIO(original)) as decoded:
        decoded.load()
        source_mode = decoded.mode
        mode = "RGBA" if "A" in decoded.getbands() or "transparency" in decoded.info else "RGB"
        pixels = decoded.convert(mode)
    output = io.BytesIO()
    pixels.save(output, format="PNG")
    png = output.getvalue()
    path = directory / "images" / f"image-{order:03d}.png"
    path.parent.mkdir(exist_ok=True, mode=0o700)
    _write_new(path, png)
    return {
        "order": order,
        **location,
        "source_mime": mime,
        "source_mode": source_mode,
        "pixel_mode": mode,
        "width": pixels.width,
        "height": pixels.height,
        "image_bytes": len(original),
        "image_bytes_sha256": _sha256(original),
        "pixel_sha256": _sha256(pixels.tobytes()),
        "png_sha256": _sha256(png),
        "path": str(path),
    }


def _messages_images(body, directory):
    images = []
    for message_index, message in enumerate(body["messages"]):
        labels = {}
        for path, part in _message_parts(message["content"], ["messages", message_index, "content"]):
            parent = tuple(path[:-1])
            if part["type"] == "text":
                match = re.fullmatch(r"Camera:\s*(.*?)\s*", part["text"], re.DOTALL)
                if match:
                    labels[parent] = match.group(1)
            elif part["type"] == "image":
                source = part["source"]
                images.append(_write_image(source["data"], source["media_type"], directory, len(images) + 1, {
                    "message_index": message_index,
                    "content_path": path,
                    "camera_label": labels.pop(parent, None),
                }))
    return images


def _chat_images(body, directory):
    images = []
    for message_index, message in enumerate(body["messages"]):
        content = message.get("content")
        if message["role"] != "user" or not isinstance(content, list):
            continue
        label = None
        for part_index, part in enumerate(content):
            if part["type"] == "text":
                match = re.fullmatch(r"Camera:\s*(.*?)\s*", part["text"], re.DOTALL)
                if match:
                    label = match.group(1)
                continue
            match = _IMAGE_URL.fullmatch(part["image_url"]["url"])
            images.append(_write_image(match.group(2), match.group(1), directory, len(images) + 1, {
                "message_index": message_index,
                "part_index": part_index,
                "camera_label": label,
            }))
            label = None
    return images


def _protocol(body: dict[str, Any], protocol: str | None) -> str:
    """The wire a body belongs to: the declared protocol, else recognized by its keys."""
    if protocol is not None:
        return protocol
    return "messages" if "messages" in body else "responses"


def _images(
    body: dict[str, Any], directory: Path, protocol: str | None = None
) -> list[dict[str, Any]]:
    wire = _protocol(body, protocol)
    if wire == "chat_completions":
        return _chat_images(body, directory)
    if wire == "messages":
        return _messages_images(body, directory)
    images = []
    for input_index, message in enumerate(body["input"]):
        label = None
        for part_index, part in enumerate(message.get("content", [])):
            if part["type"] == "input_text":
                match = re.fullmatch(r"Camera:\s*(.*?)\s*", part["text"], re.DOTALL)
                if match:
                    label = match.group(1)
                continue
            if part.get("type") != "input_image":
                continue
            match = _IMAGE_URL.fullmatch(part["image_url"])
            if not match:
                raise ValueError("Only inline base64 images can be archived; no URLs are fetched")
            images.append(_write_image(match.group(2), match.group(1), directory, len(images) + 1, {
                "input_index": input_index,
                "part_index": part_index,
                "camera_label": label,
            }))
            label = None
    return images


def _readable_messages(body, manifest, directory):
    view = copy.deepcopy(body)
    for image in manifest["images"]:
        part = view
        for component in image["content_path"]:
            part = part[component]
        part["source"]["data"] = (
            "PNG VIEW ONLY; original image data is in request.body.json: " + image["path"]
        )
    settings = {
        key: value for key, value in body.items()
        if key not in {"system", "messages", "tools"}
    }
    lines = [
        f"# HTTP request {manifest['sequence']:06d}\n",
        f"Mode: **{manifest['mode']}**. Recorded before any send. This recorder performs no network I/O; "
        "a live archive does not by itself prove that a request was sent.\n",
        "This document is a **readable view, not the exact HTTP body**. Images below are decoded "
        "PNG views. The complete original body, including image base64, is preserved byte for byte in "
        + _link(directory / "request.body.json", "request.body.json") + ".\n",
        "## Request identity\n", _fence(manifest["request_id"]),
        "## HTTP target\n", _fence(f"{manifest['method']} {manifest['url']}"),
        "Authentication placeholder (header value is never inspected or stored):\n",
        _fence(_auth_placeholder(manifest)),
        "Recorded safe headers:\n", _fence(_json(manifest["safe_headers"]), "json"),
        "## Effective payload settings\n", _fence(_json(settings), "json"),
        "Fields absent above were omitted from the actual HTTP body.\n",
        "## Complete system instructions\n",
        _fence(body["system"] if isinstance(body.get("system"), str) else _json(body.get("system"))),
        "## Complete messages in actual payload order\n",
    ]
    for index, message in enumerate(view["messages"]):
        lines += [f"### Message {index} ({message['role']})\n", _fence(_json(message), "json")]
        for image in manifest["images"]:
            if image["message_index"] != index:
                continue
            lines += [f"Image {image['order']} at {_json(image['content_path'])}:\n"]
            if image["camera_label"] is not None:
                lines += ["Camera label:\n", _fence(image["camera_label"])]
            lines += ["!" + _link(Path(image["path"]), f"Image {image['order']}") + "\n"]
    lines += [
        "## Complete native tools\n", _fence(_json(body.get("tools", [])), "json"),
        "## Full parsed payload view\n",
        "Only image base64 data is replaced below by references to PNG views. For exact bytes, use "
        + _link(directory / "request.body.json", "request.body.json") + "; "
        + _link(directory / "request.json", "request.json")
        + " contains the full pretty-printed body with the original image base64.\n",
        _fence(_json(view), "json"),
        "## Integrity\n", _fence("HTTP body SHA-256: " + manifest["body_sha256"]),
        _link(directory / "manifest.json", "Full image and request manifest") + "\n",
    ]
    return "\n".join(lines)


def _readable_chat(body, manifest, directory):
    view = copy.deepcopy(body)
    image_lookup = {(x["message_index"], x["part_index"]): x for x in manifest["images"]}
    settings = {key: value for key, value in body.items() if key not in {"messages", "tools"}}
    lines = [
        f"# HTTP request {manifest['sequence']:06d}\n",
        f"Mode: **{manifest['mode']}**. Recorded before any send. This recorder performs no network I/O; "
        "a live archive does not by itself prove that a request was sent.\n",
        "This document is a **readable view, not the exact HTTP body**. Images below are decoded "
        "PNG views. The complete original body, including image data URLs, is preserved byte for "
        "byte in " + _link(directory / "request.body.json", "request.body.json") + ".\n",
        "## Request identity\n", _fence(manifest["request_id"]),
        "## HTTP target\n", _fence(f"{manifest['method']} {manifest['url']}"),
        "Authentication placeholder (header value is never inspected or stored):\n",
        _fence(_auth_placeholder(manifest)),
        "Recorded safe headers:\n", _fence(_json(manifest["safe_headers"]), "json"),
        "## Effective payload settings\n", _fence(_json(settings), "json"),
        "Fields absent above were omitted from the actual HTTP body.\n",
        "## Complete messages in actual payload order\n",
    ]
    for index, message in enumerate(body["messages"]):
        lines += [
            f"### Message {index} ({message['role']})\n",
            _fence(_json({key: value for key, value in message.items() if key != "content"}),
                   "json"),
        ]
        content = message.get("content")
        if not isinstance(content, list):
            lines += [_fence(content if isinstance(content, str) else _json(content))]
            continue
        for part_index, part in enumerate(content):
            if part["type"] == "text":
                lines += [f"Text part {part_index}:\n", _fence(part["text"])]
                continue
            image = image_lookup[index, part_index]
            lines += [f"Image {image['order']} (part {part_index}):\n"]
            if image["camera_label"] is not None:
                lines += ["Camera label:\n", _fence(image["camera_label"])]
            lines += ["!" + _link(Path(image["path"]), f"Image {image['order']}") + "\n"]
            view["messages"][index]["content"][part_index]["image_url"]["url"] = (
                "PNG VIEW ONLY; original image data is in request.body.json: " + image["path"]
            )
    lines += [
        "## Complete native function tools\n", _fence(_json(body.get("tools", [])), "json"),
        "## Full parsed payload view\n",
        "Only image data URLs are replaced below by references to PNG views. For exact bytes, use "
        + _link(directory / "request.body.json", "request.body.json") + "; "
        + _link(directory / "request.json", "request.json")
        + " contains the full pretty-printed body with the original image data URLs.\n",
        _fence(_json(view), "json"),
        "## Integrity\n", _fence("HTTP body SHA-256: " + manifest["body_sha256"]),
        _link(directory / "manifest.json", "Full image and request manifest") + "\n",
    ]
    return "\n".join(lines)


def _readable(body: dict[str, Any], manifest: dict[str, Any], directory: Path) -> str:
    wire = _protocol(body, manifest.get("protocol"))
    if wire == "chat_completions":
        return _readable_chat(body, manifest, directory)
    if wire == "messages":
        return _readable_messages(body, manifest, directory)
    image_lookup = {(x["input_index"], x["part_index"]): x for x in manifest["images"]}
    settings = {
        key: body[key]
        for key in (
            "model",
            "reasoning",
            "max_output_tokens",
            "store",
            "tool_choice",
            "parallel_tool_calls",
            "include",
        )
        if key in body
    }
    lines = [
        f"# HTTP request {manifest['sequence']:06d}\n",
        f"Mode: **{manifest['mode']}**. Recorded before any send. This recorder performs no network I/O; "
        "a live archive does not by itself prove that a request was sent.\n",
        "This document is a **readable view, not the exact HTTP body**. Images below are decoded "
        "PNG views. The complete original body, including image base64, is preserved byte for byte in "
        + _link(directory / "request.body.json", "request.body.json")
        + ".\n",
        "## Request identity\n",
        _fence(manifest["request_id"]),
        "## HTTP target\n",
        _fence(f"{manifest['method']} {manifest['url']}"),
        "Authentication placeholder (header value is never inspected or stored):\n",
        _fence(_auth_placeholder(manifest)),
        "Recorded safe headers:\n",
        _fence(_json(manifest["safe_headers"]), "json"),
        "## Effective payload settings\n",
        _fence(_json(settings), "json"),
        "Fields absent above were omitted from the actual HTTP body.\n",
        "## Complete instructions\n",
        _fence(
            body["instructions"]
            if isinstance(body.get("instructions"), str)
            else _json(body.get("instructions"))
        ),
        "## Complete input in actual payload order\n",
    ]
    view = copy.deepcopy(body)
    for input_index, message in enumerate(body["input"]):
        lines += [
            f"### Input {input_index}\n",
            _fence(
                _json({key: value for key, value in message.items() if key != "content"}), "json"
            ),
        ]
        for part_index, part in enumerate(message.get("content", [])):
            if part["type"] in {"input_text", "output_text"}:
                lines += [f"Text part {part_index}:\n", _fence(part["text"])]
            elif part["type"] == "input_image":
                image = image_lookup[input_index, part_index]
                lines += [f"Image {image['order']} (part {part_index}):\n"]
                if image["camera_label"] is not None:
                    lines += ["Camera label:\n", _fence(image["camera_label"])]
                lines += ["!" + _link(Path(image["path"]), f"Image {image['order']}") + "\n"]
                view["input"][input_index]["content"][part_index]["image_url"] = (
                    "PNG VIEW ONLY; original image data is in request.body.json: " + image["path"]
                )
            else:
                lines += [_fence(_json(part), "json")]
    text = body.get("text", {})
    output_format = text.get("format", {}) if isinstance(text, dict) else {}
    lines += [
        "## Complete native function tools\n",
        _fence(_json(body.get("tools", [])), "json"),
        "## Output text settings (if present)\n",
        _fence(_json(output_format), "json"),
        "## Full parsed payload view\n",
        "Only image data URLs are replaced below by references to PNG views. For exact bytes, use "
        + _link(directory / "request.body.json", "request.body.json")
        + "; "
        + _link(directory / "request.json", "request.json")
        + " contains the full pretty-printed body with the original image data URLs.\n",
        _fence(_json(view), "json"),
        "## Integrity\n",
        _fence("HTTP body SHA-256: " + manifest["body_sha256"]),
        _link(directory / "manifest.json", "Full image and request manifest") + "\n",
    ]
    return "\n".join(lines)


def _auth_placeholder(manifest: Mapping[str, Any]) -> str:
    """Show the authentication header by variable name; its value is never read."""
    scheme, variable = manifest.get("auth_scheme", "bearer"), manifest["auth_env"]
    if scheme is None:
        return "(no authentication header)"
    header = (f"x-api-key: ${{{variable}}}" if scheme == "x-api-key"
              else f"Authorization: Bearer ${{{variable}}}")
    if manifest.get("auth_required", True):
        return header
    return f"{header}\n(sent only when {variable} is set; otherwise no authentication header)"


class APIRequestRecorder:
    """Archive one built HTTP request locally before a paid POST may proceed.

    Each call reserves a new sequence directory, including after partial failure.
    Any error propagates to the caller. Neither preview nor live mode sends HTTP,
    reads an API key, changes request bytes, or adds conversation state.

    The URL and the body are archived as sent (manifest version 2); the authentication
    header only by its variable name. With ``protocol``, the body is validated, imaged and
    rendered as that wire's payload rather than recognized by its keys (a Chat Completions
    body has ``messages`` too). ``auth_required=False`` marks an optional key, whose header
    is sent only when its variable is set.
    """

    def __init__(self, directory, *, mode="live", auth_env="OPENAI_API_KEY", auth_scheme="bearer",
                 protocol=None, auth_required=True):
        if mode not in {"live", "preview"}:
            raise ValueError("Request archive mode must be live or preview")
        if auth_env is not None and (not isinstance(auth_env, str)
                                     or not _ENV_NAME.fullmatch(auth_env)):
            raise ValueError("auth_env must be an environment variable name, not a credential")
        self.directory = Path(directory).expanduser().resolve()
        self.mode = mode
        self.protocol = protocol
        self.auth_env = auth_env
        self.auth_scheme = auth_scheme
        # An optional key: the header is sent only when its variable is set.
        self.auth_required = auth_required

    def __call__(self, model_request: AgentContext, request: httpx.Request) -> dict[str, Any]:
        if not isinstance(model_request.request_id, str) or not model_request.request_id:
            raise ValueError("Request recording requires a nonempty string request_id")
        if request.url.username or request.url.password or request.url.query:
            raise ValueError("Request archive URLs must not contain user info or query parameters")
        if request.url.scheme not in {"http", "https"}:
            raise ValueError("Request recording requires an HTTP(S) target")
        url = str(request.url)
        # Access only already-buffered bytes. Reading a stream here would mutate it.
        raw = request.content
        body = _body(raw, self.protocol)
        headers = _safe_headers(request)
        slug = _slug(model_request.request_id)
        with _exclusive_root(self.directory):
            sequence = (
                max(
                    (
                        int(match.group(1))
                        for path in self.directory.iterdir()
                        if (match := _SEQUENCE.match(path.name))
                    ),
                    default=0,
                )
                + 1
            )
            directory = self.directory / f"{sequence:06d}-{slug}"
            directory.mkdir(mode=0o700)
            _write_new(directory / "request.body.json", raw)
            _write_new(directory / "request.json", (_json(body) + "\n").encode("utf-8"))
            images = _images(body, directory, self.protocol)
            manifest = {
                "version": 2,
                "sequence": sequence,
                "request_id": model_request.request_id,
                "mode": self.mode,
                "protocol": self.protocol,
                "phase": "before_send",
                "method": request.method,
                "url": url,
                "auth_env": self.auth_env,
                "auth_scheme": self.auth_scheme,
                "auth_required": self.auth_required,
                "auth_value_recorded": False,
                "safe_headers": headers,
                "body_bytes": len(raw),
                "body_sha256": _sha256(raw),
                "body_path": str(directory / "request.body.json"),
                "readable_path": str(directory / "request.md"),
                "images": images,
            }
            _write_new(directory / "manifest.json", (_json(manifest) + "\n").encode("utf-8"))
            _write_new(
                directory / "request.md", _readable(body, manifest, directory).encode("utf-8")
            )
            metadata = {
                "request_record_dir": str(directory),
                "request_body_path": str(directory / "request.body.json"),
                "request_readable_path": str(directory / "request.md"),
                "request_manifest_path": str(directory / "manifest.json"),
                "request_body_sha256": manifest["body_sha256"],
                "request_sequence": sequence,
                "request_mode": self.mode,
                "request_index_path": str(self.directory / "index.md"),
            }
            row = {
                "request_id": model_request.request_id,
                "phase": "before_send",
                "method": request.method,
                "url": url,
                "image_count": len(images),
                **metadata,
            }
            _append(
                self.directory / "requests.jsonl",
                (json.dumps(row, allow_nan=False) + "\n").encode(),
            )
            entry = f"- {_link(directory / 'request.md', f'{sequence:06d} {slug}')} ({self.mode})\n"
            _append(
                self.directory / "index.md",
                entry.encode("utf-8"),
                initial=b"# Prepared HTTP request archives\n\nEvery entry is recorded before send. Preview entries are offline only.\n\n",
            )
            return metadata
