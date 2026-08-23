"""Artifact-driven image ingestion and the native multimodal request ABI."""

from __future__ import annotations

import base64
import hashlib
import io
import ipaddress
import socket
import struct
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote_to_bytes, urlsplit
from urllib.request import Request, build_opener

MEDIA_ABI = "request.multimodal.fp32.host.v1"
MEDIA_MAGIC = b"QLMMEDIA"
MEDIA_VERSION = 1
MEDIA_HEADER = struct.Struct("<8sIIIIIIiI")
MEDIA_IMAGE = struct.Struct("<IIIIII")
MAX_ENCODED_IMAGE_BYTES = 32 << 20


class MultimodalInputError(ValueError):
    pass


@dataclass(frozen=True)
class PreparedMultimodalPrompt:
    token_ids: list[int]
    packet: bytes
    media_signature: bytes
    image_count: int
    image_tokens: int
    rope_delta: int


def _public_address(hostname: str) -> bool:
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(
            hostname, None, type=socket.SOCK_STREAM
        )}
    except socket.gaierror:
        return False
    if not addresses:
        return False
    for text in addresses:
        address = ipaddress.ip_address(text)
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_multicast
            or address.is_reserved
            or address.is_unspecified
        ):
            return False
    return True


def _decode_data_url(value: str) -> bytes:
    header, separator, payload = value.partition(",")
    if not separator or not header.startswith("data:image/"):
        raise MultimodalInputError("image must be a valid image data URL")
    metadata = header[5:].split(";")
    media_type = metadata[0].lower()
    if media_type not in {
        "image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp"
    }:
        raise MultimodalInputError(f"unsupported image media type {media_type!r}")
    try:
        if "base64" in metadata[1:]:
            encoded = payload.encode("ascii")
            if len(encoded) > (MAX_ENCODED_IMAGE_BYTES * 4 // 3 + 8):
                raise MultimodalInputError("encoded image exceeds the service limit")
            return base64.b64decode(encoded, validate=True)
        return unquote_to_bytes(payload)
    except (UnicodeEncodeError, ValueError) as error:
        raise MultimodalInputError("image data URL has an invalid payload") from error


def _download_public_image(value: str) -> bytes:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise MultimodalInputError("image URL must use http or https")
    if parsed.username or parsed.password or not _public_address(parsed.hostname):
        raise MultimodalInputError("image URL does not resolve to a public host")
    request = Request(value, headers={"User-Agent": "quantum-llm-image/1"})
    with build_opener().open(request, timeout=15) as response:
        final = urlsplit(response.geturl())
        if (
            final.scheme not in {"http", "https"}
            or not final.hostname
            or not _public_address(final.hostname)
        ):
            raise MultimodalInputError("image redirect does not resolve to a public host")
        content_type = response.headers.get_content_type().lower()
        if not content_type.startswith("image/"):
            raise MultimodalInputError("image URL returned a non-image response")
        announced = response.headers.get("Content-Length")
        if announced is not None and int(announced) > MAX_ENCODED_IMAGE_BYTES:
            raise MultimodalInputError("remote image exceeds the service limit")
        payload = response.read(MAX_ENCODED_IMAGE_BYTES + 1)
    if len(payload) > MAX_ENCODED_IMAGE_BYTES:
        raise MultimodalInputError("remote image exceeds the service limit")
    return payload


def load_image(value: str) -> Any:
    """Load a bounded OpenAI/Anthropic image reference into canonical RGB."""
    from PIL import Image, UnidentifiedImageError
    if not isinstance(value, str) or not value:
        raise MultimodalInputError("image reference must be a non-empty string")
    payload = _decode_data_url(value) if value.startswith("data:") \
        else _download_public_image(value)
    if not payload or len(payload) > MAX_ENCODED_IMAGE_BYTES:
        raise MultimodalInputError("image payload is empty or exceeds the service limit")
    try:
        with Image.open(io.BytesIO(payload)) as source:
            source.load()
            if source.width <= 0 or source.height <= 0:
                raise MultimodalInputError("image dimensions are invalid")
            return source.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise MultimodalInputError("image payload cannot be decoded safely") from error


def create_image_processor(snapshot: str) -> Any:
    """Load the official PIL backend without importing PyTorch/Torchvision."""
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import (
        Qwen2VLImageProcessorPil,
    )
    return Qwen2VLImageProcessorPil.from_pretrained(
        snapshot, local_files_only=True
    )


def _template_messages(messages: list[dict[str, Any]]) \
        -> tuple[list[dict[str, Any]], list[Any]]:
    from PIL import Image

    rendered: list[dict[str, Any]] = []
    images: list[Any] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            rendered.append(message)
            continue
        parts: list[dict[str, Any]] = []
        for part in content:
            if part.get("type") == "image":
                image = part.get("image")
                if not isinstance(image, Image.Image):
                    raise MultimodalInputError("internal image content is invalid")
                images.append(image)
                parts.append({"type": "image"})
            else:
                parts.append(part)
        clone = dict(message)
        clone["content"] = parts
        rendered.append(clone)
    return rendered, images


def _mrope_positions(
    token_types: list[int], grids: list[tuple[int, int, int]], merge: int,
) -> tuple[list[tuple[int, int, int]], int]:
    positions: list[tuple[int, int, int]] = []
    current = 0
    grid_index = 0
    offset = 0
    while offset < len(token_types):
        modality = token_types[offset]
        end = offset + 1
        while end < len(token_types) and token_types[end] == modality:
            end += 1
        length = end - offset
        if modality == 0:
            positions.extend((current + index,) * 3 for index in range(length))
            current += length
        elif modality == 1:
            if grid_index >= len(grids):
                raise MultimodalInputError("image token stream has no matching grid")
            temporal, height, width = grids[grid_index]
            grid_index += 1
            merged_h, merged_w = height // merge, width // merge
            expected = temporal * merged_h * merged_w
            if length != expected:
                raise MultimodalInputError("image token count disagrees with its grid")
            for time in range(temporal):
                for row in range(merged_h):
                    for column in range(merged_w):
                        positions.append((
                            current + time,
                            current + row,
                            current + column,
                        ))
            current += max(height, width) // merge
        else:
            raise MultimodalInputError("video input is not implemented")
        offset = end
    if grid_index != len(grids) or len(positions) != len(token_types):
        raise MultimodalInputError("multimodal position stream is incomplete")
    maximum = max(max(position) for position in positions) if positions else -1
    return positions, maximum + 1 - len(token_types)


def prepare_multimodal_prompt(
    tokenizer: Any,
    image_processor: Any,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
    tools: list[dict[str, Any]] | None,
    reasoning_effort: str,
    enable_thinking: bool,
    preserve_thinking: bool,
    maximum_image_pixels: int,
    maximum_patch_tokens: int,
) -> PreparedMultimodalPrompt:
    """Preprocess images and serialize an exact native vision request."""
    import numpy as np

    template_messages, images = _template_messages(messages)
    if not images:
        raise MultimodalInputError("multimodal preparation requires an image")
    image_inputs = image_processor(
        images,
        return_tensors="np",
        max_pixels=maximum_image_pixels,
    )
    pixels = np.asarray(image_inputs["pixel_values"], dtype="<f4", order="C")
    grid_array = np.asarray(image_inputs["image_grid_thw"], dtype=np.int64)
    grids = [tuple(int(value) for value in row) for row in grid_array.tolist()]
    if len(grids) != len(images) or pixels.ndim != 2:
        raise MultimodalInputError("official image processor returned invalid geometry")
    patch_count, patch_dimension = (int(value) for value in pixels.shape)
    if patch_count <= 0 or patch_count > maximum_patch_tokens:
        raise MultimodalInputError("image patches exceed the native vision budget")

    prompt = tokenizer.apply_chat_template(
        template_messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        tools=tools,
        reasoning_effort=reasoning_effort,
        enable_thinking=enable_thinking,
        preserve_thinking=preserve_thinking,
    )
    if not isinstance(prompt, str):
        raise MultimodalInputError("chat template returned a non-text prompt")
    image_token = getattr(tokenizer, "image_token", "<|image_pad|>")
    merge = int(getattr(image_processor, "merge_size", 0))
    if merge <= 0:
        raise MultimodalInputError("image processor has no spatial merge geometry")
    if prompt.count(image_token) != len(grids):
        raise MultimodalInputError("chat template image placeholders disagree with input")
    for temporal, height, width in grids:
        count = temporal * height * width // (merge * merge)
        prompt = prompt.replace(image_token, image_token * count, 1)

    token_ids = [int(token) for token in tokenizer.encode(
        prompt, add_special_tokens=False
    )]
    image_token_id = int(tokenizer.convert_tokens_to_ids(image_token))
    token_types = [1 if token == image_token_id else 0 for token in token_ids]
    positions, rope_delta = _mrope_positions(token_types, grids, merge)

    descriptors: list[tuple[int, int, int, int, int, int]] = []
    patch_offset = 0
    search_offset = 0
    for temporal, height, width in grids:
        merged_tokens = temporal * height * width // (merge * merge)
        try:
            prompt_offset = token_types.index(1, search_offset)
        except ValueError as error:
            raise MultimodalInputError("expanded image tokens are absent") from error
        if token_types[prompt_offset:prompt_offset + merged_tokens] != \
                [1] * merged_tokens:
            raise MultimodalInputError("expanded image tokens are not contiguous")
        descriptors.append((
            prompt_offset, merged_tokens, temporal, height, width, patch_offset
        ))
        patch_offset += temporal * height * width
        search_offset = prompt_offset + merged_tokens
    if patch_offset != patch_count:
        raise MultimodalInputError("patch payload disagrees with image grids")

    flags = 1 | 2
    packet = bytearray(MEDIA_HEADER.pack(
        MEDIA_MAGIC, MEDIA_VERSION, flags, len(token_ids), len(grids),
        patch_dimension, patch_count, rope_delta, 0,
    ))
    for temporal, height, width in positions:
        packet.extend(struct.pack("<III", temporal, height, width))
    for descriptor in descriptors:
        packet.extend(MEDIA_IMAGE.pack(*descriptor))
    packet.extend(pixels.tobytes(order="C"))

    signature = hashlib.sha256()
    for grid in grids:
        signature.update(struct.pack("<III", *grid))
    signature.update(pixels.tobytes(order="C"))
    return PreparedMultimodalPrompt(
        token_ids=token_ids,
        packet=bytes(packet),
        media_signature=signature.digest(),
        image_count=len(grids),
        image_tokens=sum(item[1] for item in descriptors),
        rope_delta=rope_delta,
    )


__all__ = [
    "MEDIA_ABI", "MEDIA_HEADER", "MEDIA_IMAGE", "MEDIA_MAGIC", "MEDIA_VERSION",
    "MultimodalInputError", "PreparedMultimodalPrompt", "create_image_processor",
    "load_image", "prepare_multimodal_prompt",
]
