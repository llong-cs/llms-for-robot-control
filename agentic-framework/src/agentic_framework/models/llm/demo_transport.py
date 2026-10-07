"""Share fixed teacher context without changing stored online decision records."""

from __future__ import annotations

import copy
import hashlib
import json

from agentic_framework.harness.demonstration import demonstration_image_names


def _entries(messages):
    """Find complete teacher payloads in uncoalesced encoded user observations."""
    for message_index, message in enumerate(messages):
        if message.get("role") != "user" or not isinstance(message.get("content"), list):
            continue
        content = message["content"]
        for text_index, part in enumerate(content):
            if part.get("type") not in ("input_text", "text"):
                continue
            try:
                text = part["text"]
                payload, end = json.JSONDecoder().raw_decode(
                    text, idx=len(text) - len(text.lstrip())
                )
            except (ValueError, KeyError, TypeError):
                continue
            if not isinstance(payload, dict) or not isinstance(
                payload.get("demonstration"), dict
            ) or "current_observation" not in payload:
                continue
            demo = payload["demonstration"]
            names = demonstration_image_names(demo)
            indices, frames = [], {}
            for index in range(text_index + 1, len(content) - 1):
                label, frame = content[index], content[index + 1]
                if label.get("type") not in ("input_text", "text"):
                    continue
                name = label.get("text", "").removeprefix("Camera: ")
                if name not in names or label.get("text") != f"Camera: {name}":
                    continue
                if frame.get("type") not in ("input_image", "image") or name in frames:
                    raise ValueError("Teacher frame must have exactly one labeled image block")
                indices.extend((index, index + 1))
                frames[name] = frame
            if set(frames) != set(names):
                raise ValueError("Teacher demonstration is missing encoded image blocks")
            # Compare both text and actual image data: an updated teacher or a
            # different image encoding must never be silently treated as identical.
            encoded = json.dumps([demo, frames], sort_keys=True, ensure_ascii=False,
                                 allow_nan=False, separators=(",", ":"))
            signature = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
            yield message_index, text_index, payload, text[end:], indices, signature


def demonstration_signature(messages):
    """Identify the fixed teacher inputs for a persistent native session contract."""
    return tuple(entry[-1] for entry in _entries(messages))


def deduplicate_demonstrations(messages, *, omit_all=False):
    """Keep one copy of each identical teacher in a stateless API request.

    The first selected observation carries the reference, so window eviction
    automatically moves it to the next retained observation. Native Codex tool
    continuations use ``omit_all`` because their session already has this fixed
    reference; fresh sessions and restarts must include it again.

    Only teacher JSON and its explicitly referenced frame blocks are removed.
    Live images (even identical ones), state, repair feedback, tool exchanges and
    opaque provider reasoning are never deduplicated. Caller-owned raw records
    remain unchanged; only the assembled transport copy is modified.
    """
    result = copy.deepcopy(messages)
    seen, remove = set(), {}
    for message_index, text_index, payload, suffix, indices, signature in _entries(messages):
        duplicate = omit_all or signature in seen
        seen.add(signature)
        if not duplicate:
            continue
        current = {key: value for key, value in payload.items() if key != "demonstration"}
        result[message_index]["content"][text_index]["text"] = json.dumps(
            current, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ) + suffix
        remove.setdefault(message_index, set()).update(indices)
    for message_index, indices in remove.items():
        result[message_index]["content"] = [
            part for index, part in enumerate(result[message_index]["content"])
            if index not in indices
        ]
    return result
