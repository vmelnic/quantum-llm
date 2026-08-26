# SPDX-License-Identifier: Apache-2.0
"""Artifact-declared chat encoding and response parsing.

The common service discovers an optional self-contained codec from the
published tokenizer artifact.  Selection is based on callable capabilities,
not a model family or architecture identifier.
"""

from __future__ import annotations

import importlib.util
import inspect
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any


class ArtifactCodecStreamParser:
    """Stream stable reasoning/content while withholding structured markup."""

    def __init__(self, codec: "ArtifactChatCodec", thinking: bool) -> None:
        self.codec = codec
        self.thinking = thinking
        self.field = "reasoning_content" if thinking else "content"
        self.pending = ""
        self.raw_pieces: list[str] = []
        self.finished = False

    @staticmethod
    def _event(field: str, text: str) -> list[dict[str, Any]]:
        if not text:
            return []
        return [{"type": "region_chunk", "field": field,
                 "text": text, "dirty": False}]

    @staticmethod
    def _stable_prefix(text: str, marker: str) -> tuple[str, str]:
        held = 0
        for size in range(1, min(len(text), len(marker) - 1) + 1):
            if text.endswith(marker[:size]):
                held = size
        if held:
            return text[:-held], text[-held:]
        return text, ""

    @classmethod
    def _stable_prefix_for_markers(
            cls, text: str, markers: tuple[str, ...]) -> tuple[str, str]:
        held = 0
        for marker in markers:
            _stable, pending = cls._stable_prefix(text, marker)
            held = max(held, len(pending))
        if held:
            return text[:-held], text[-held:]
        return text, ""

    def feed(self, delta: str) -> list[dict[str, Any]]:
        if not delta:
            return []
        self.raw_pieces.append(delta)
        if self.finished:
            return []
        self.pending += delta
        events: list[dict[str, Any]] = []
        while self.pending and not self.finished:
            if self.field == "reasoning_content":
                marker = self.codec.thinking_end_token
                index = self.pending.find(marker)
                if index >= 0:
                    events += self._event(self.field, self.pending[:index])
                    self.pending = self.pending[index + len(marker):]
                    self.field = "content"
                    continue
                stable, self.pending = self._stable_prefix(
                    self.pending, marker
                )
                events += self._event(self.field, stable)
                break
            markers = tuple(marker for marker in (
                self.codec.tool_calls_start_token,
                self.codec.eos_token,
            ) if marker)
            matches = []
            for marker in markers:
                index = self.pending.find(marker)
                if index >= 0:
                    matches.append((index, marker))
            if matches:
                index, _marker = min(matches, key=lambda item: item[0])
                events += self._event("content", self.pending[:index])
                self.pending = ""
                self.finished = True
                break
            stable, self.pending = self._stable_prefix_for_markers(
                self.pending, markers
            )
            events += self._event("content", stable)
            break
        return events

    def finalize(self) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        events: list[dict[str, Any]] = []
        if not self.finished and self.pending:
            events = self._event(self.field, self.pending)
        self.pending = ""
        self.finished = True
        return self.codec.parse("".join(self.raw_pieces), self.thinking), events


class ArtifactChatCodec:
    """Validated wrapper around a tokenizer artifact's Python codec."""

    def __init__(self, module: ModuleType, source: Path) -> None:
        self.module = module
        self.source = source
        self.encode_messages = getattr(module, "encode_messages")
        self.parse_message = getattr(
            module, "parse_message_from_completion_text", None
        )
        self.eos_token = self._string("eos_token", required=False)
        self.thinking_start_token = self._string(
            "thinking_start_token", required=False
        )
        self.thinking_end_token = self._string(
            "thinking_end_token", required=False
        )
        self.assistant_token = self._string(
            "ASSISTANT_SP_TOKEN", required=False
        )
        dsml = self._string("dsml_token", required=False)
        block = self._string("tool_calls_block_name", required=False)
        self.tool_calls_start_token = (
            f"\n\n<{dsml}{block}" if dsml and block else ""
        )
        self.supports_response_parsing = (
            callable(self.parse_message) and bool(self.eos_token) and
            bool(self.tool_calls_start_token)
        )

    def _string(self, name: str, *, required: bool) -> str:
        value = getattr(self.module, name, "")
        if not isinstance(value, str) or (required and not value):
            raise RuntimeError(
                f"artifact chat codec has invalid {name}: {self.source}"
            )
        return value

    @staticmethod
    def _reasoning_effort(value: str) -> str | None:
        return {"xhigh": "max", "medium": "high", "low": None}[value]

    def encode(self, messages: list[dict[str, Any]], *,
               add_generation_prompt: bool,
               tools: tuple[dict[str, Any], ...],
               reasoning_effort: str,
               enable_thinking: bool,
               preserve_thinking: bool) -> str:
        prepared = deepcopy(messages)
        if tools:
            target = next(
                (message for message in prepared
                 if message.get("role") in {"system", "developer"}),
                None,
            )
            if target is None:
                target = {"role": "system", "content": ""}
                prepared.insert(0, target)
            target["tools"] = list(tools)
        thinking_mode = "thinking" if enable_thinking else "chat"
        parameters = inspect.signature(self.encode_messages).parameters
        kwargs: dict[str, Any] = {"thinking_mode": thinking_mode}
        if "drop_thinking" in parameters:
            kwargs["drop_thinking"] = not preserve_thinking
        if "reasoning_effort" in parameters:
            kwargs["reasoning_effort"] = self._reasoning_effort(
                reasoning_effort
            )
        if "add_generation_prompt" in parameters:
            kwargs["add_generation_prompt"] = add_generation_prompt
        prompt = self.encode_messages(prepared, **kwargs)
        if not isinstance(prompt, str):
            raise RuntimeError(
                f"artifact chat codec returned non-text: {self.source}"
            )
        if not add_generation_prompt and \
                "add_generation_prompt" not in parameters:
            suffix = self.assistant_token + (
                self.thinking_start_token if enable_thinking
                else self.thinking_end_token
            )
            if suffix and prompt.endswith(suffix):
                prompt = prompt[:-len(suffix)]
        return prompt

    def stream_parser(self, enable_thinking: bool) -> ArtifactCodecStreamParser:
        if not self.supports_response_parsing:
            raise RuntimeError("artifact chat codec has no response parser")
        return ArtifactCodecStreamParser(self, enable_thinking)

    def parse(self, text: str, enable_thinking: bool) -> dict[str, Any]:
        if not self.supports_response_parsing:
            raise RuntimeError("artifact chat codec has no response parser")
        completion = text
        if self.eos_token and not completion.endswith(self.eos_token):
            completion += self.eos_token
        message = self.parse_message(
            completion,
            thinking_mode="thinking" if enable_thinking else "chat",
        )
        if not isinstance(message, dict):
            raise ValueError("artifact response parser returned a non-object")
        return message


def discover_artifact_chat_codec(snapshot: Path) -> ArtifactChatCodec | None:
    """Load exactly one codec declaring encode_messages from encoding/*.py."""
    directory = snapshot / "encoding"
    if not directory.is_dir():
        return None
    discovered: list[ArtifactChatCodec] = []
    for index, path in enumerate(sorted(directory.glob("*.py"))):
        spec = importlib.util.spec_from_file_location(
            f"artifact_chat_codec_{index}", path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load artifact chat codec: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if callable(getattr(module, "encode_messages", None)):
            discovered.append(ArtifactChatCodec(module, path))
    if len(discovered) > 1:
        sources = ", ".join(str(codec.source) for codec in discovered)
        raise RuntimeError(f"artifact declares ambiguous chat codecs: {sources}")
    return discovered[0] if discovered else None
