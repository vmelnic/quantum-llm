# SPDX-License-Identifier: Apache-2.0
"""Artifact-driven response protocol negotiation for Transformers tokenizers.

The XML function-call response template below is the declarative Qwen3 example
published in the Hugging Face Transformers response-parsing documentation.  It
is installed only when the checkpoint chat template itself declares every
terminal used by that protocol.  Common service code therefore selects an
output grammar by artifact syntax, never by model family or architecture ID.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


_XML_FUNCTION_RESPONSE_TEMPLATE: dict[str, Any] = {
    "defaults": {"role": "assistant"},
    "start_anchor": "<|im_start|>assistant\n",
    "fields": {
        "thinking": {
            "open": "<think>",
            "close": "</think>",
            "content": "text",
        },
        "tool_calls": {
            "open_pattern": r"<tool_call>\s*<function=(?P<name>\w+)>",
            "close": "</tool_call>",
            "repeats": True,
            "content": "xml-inline",
            "content_args": {
                "tag_pattern": (
                    r"<parameter=(?P<key>\w+)>\s*"
                    r"(?P<value>.*?)\s*</parameter>"
                ),
                "value_parser": {
                    "name": "json",
                    "args": {"allow_non_json": True},
                },
            },
            "transform": {
                "type": "function",
                "function": {
                    "name": "{name}",
                    "arguments": "{content}",
                },
            },
        },
        "content": {"close": "<|im_end|>", "content": "text"},
    },
}

_XML_FUNCTION_TERMINALS = (
    "<|im_start|>assistant",
    "<think>",
    "</think>",
    "<tool_call>",
    "<function=",
    "<parameter=",
    "</parameter>",
    "</tool_call>",
)


def install_declared_response_protocol(tokenizer: Any) -> str | None:
    """Install a standard parser only when the tokenizer declares its grammar."""
    if not callable(getattr(tokenizer, "parse_response", None)):
        return None
    if getattr(tokenizer, "response_template", None) is not None:
        return "checkpoint"
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str):
        return None
    if not all(terminal in template for terminal in _XML_FUNCTION_TERMINALS):
        return None
    tokenizer.response_template = deepcopy(_XML_FUNCTION_RESPONSE_TEMPLATE)
    return "xml-function-v1"
