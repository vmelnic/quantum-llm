# SPDX-License-Identifier: Apache-2.0
"""Artifact-driven response protocol negotiation for Transformers tokenizers.

The XML function-call response template below is the declarative Qwen3 example
published in the Hugging Face Transformers response-parsing documentation.  It
is installed only when the checkpoint chat template itself declares every
terminal used by that protocol.  Common service code therefore selects an
output grammar by artifact syntax, never by model family or architecture ID.

This is the dependency-light equivalent of vLLM's combined Qwen3
reasoning/tool parser.  We use the upstream Transformers response-template
engine already required by the service instead of importing vLLM's serving,
PyTorch and Linux runtime dependencies.  The bracketed function-call template
uses the same upstream parser for artifacts whose chat template declares the
``[THINK]`` / ``[TOOL_CALLS]name[ARGS]json`` grammar.  Selection is based only
on those artifact terminals, never on a model or architecture identifier.
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


_BRACKET_FUNCTION_RESPONSE_TEMPLATE: dict[str, Any] = {
    "defaults": {"role": "assistant"},
    # This grammar has no assistant-open token. Generation begins immediately
    # after either the last instruction or a tool-result block.
    "start_anchor": ["[/INST]", "[/TOOL_RESULTS]"],
    "fields": {
        "thinking": {
            "open": "[THINK]",
            "close": "[/THINK]",
            "content": "text",
        },
        "tool_calls": {
            "open_pattern": (
                r"\[TOOL_CALLS\](?P<name>[A-Za-z0-9_.:-]+)\[ARGS\]"
            ),
            # Preserve the next opener so the streaming parser can emit every
            # call in a sequence rather than consuming the next call marker.
            "close_pattern": r"(?=\[TOOL_CALLS\]|</s>)",
            "repeats": True,
            "content": "json",
            "transform": {
                "type": "function",
                "function": {
                    "name": "{name}",
                    "arguments": "{content}",
                },
            },
        },
        "content": {"close": "</s>", "content": "text"},
    },
}

_BRACKET_FUNCTION_TERMINALS = (
    "[INST]",
    "[/INST]",
    "[THINK]",
    "[/THINK]",
    "[TOOL_CALLS]",
    "[ARGS]",
    "[TOOL_RESULTS]",
    "[/TOOL_RESULTS]",
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
    if all(terminal in template for terminal in _XML_FUNCTION_TERMINALS):
        tokenizer.response_template = deepcopy(_XML_FUNCTION_RESPONSE_TEMPLATE)
        return "xml-function-v1"
    if all(terminal in template for terminal in _BRACKET_FUNCTION_TERMINALS):
        tokenizer.response_template = deepcopy(
            _BRACKET_FUNCTION_RESPONSE_TEMPLATE
        )
        return "bracket-function-v1"
    return None


def select_response_protocol(tokenizer: Any,
                             artifact_chat_codec: Any) -> str | None:
    """Select exactly the parser paired with the active prompt encoder.

    A self-contained artifact codec owns both directions of its protocol.  A
    normal tokenizer artifact uses its checkpoint response template, or the
    syntax-qualified standard Qwen XML template above.  Keeping the choice in
    one function prevents reporting one protocol while executing another.
    """
    if (artifact_chat_codec is not None and
            bool(getattr(artifact_chat_codec,
                         "supports_response_parsing", False))):
        return "artifact-chat-codec-v1"
    return install_declared_response_protocol(tokenizer)
