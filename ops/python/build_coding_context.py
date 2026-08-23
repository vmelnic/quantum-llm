#!/usr/bin/env python3
"""Build an exact-length coding prompt from a real repository snapshot."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Callable, Iterable


EXCLUDED_DIRECTORIES = {
    ".git", ".idea", ".venv", ".vscode", "artifacts", "logs",
    "node_modules", "out", "work", "__pycache__",
}
EXCLUDED_FILES = {".env", ".DS_Store"}
SYSTEM_PROMPT = (
    "You are a production coding agent. Inspect the supplied repository "
    "snapshot end to end, preserve its invariants, and make only changes "
    "required by the task."
)
TASK_PREFIX = """Task:
Review this repository as a whole and complete the requested production task.
Use AGENTS.md, docs/architecture.md, docs/production-readiness.md and
docs/roadmap.md as the current contracts. Preserve numerical and deployment
invariants, and report evidence for every claimed result.

Repository snapshot follows. Every included byte comes from the working tree;
paths excluded as runtime/build state are listed in the capture metadata.

"""


def source_paths(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root)
        if any(part in EXCLUDED_DIRECTORIES for part in relative.parts):
            continue
        if relative.as_posix() in EXCLUDED_FILES:
            continue
        yield path


def repository_text(root: Path) -> tuple[str, list[dict[str, Any]]]:
    sections: list[str] = []
    files: list[dict[str, Any]] = []
    for path in source_paths(root):
        payload = path.read_bytes()
        if b"\0" in payload:
            continue
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            continue
        relative = path.relative_to(root).as_posix()
        digest = hashlib.sha256(payload).hexdigest()
        sections.append(f'<file path="{relative}" sha256="{digest}">\n')
        sections.append(text)
        if text and not text.endswith("\n"):
            sections.append("\n")
        sections.append("</file>\n")
        files.append({"path": relative, "bytes": len(payload),
                      "sha256": digest})
    if not files:
        raise ValueError("repository snapshot contains no UTF-8 source files")
    return TASK_PREFIX + "".join(sections), files


def load_checkpoint_chat_encoder(
        tokenizer_root: Path) -> Callable[..., str] | None:
    path = tokenizer_root / "encoding" / "encoding_dsv4.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("encoding_dsv4", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load checkpoint chat encoder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    encoder = getattr(module, "encode_messages", None)
    if not callable(encoder):
        raise ValueError(f"checkpoint chat encoder has no encode_messages: {path}")
    return encoder


def template_parts(tokenizer: Any,
                   chat_encoder: Callable[..., str] | None = None
                   ) -> tuple[str, str]:
    sentinel = "QUANTUM_LLM_REAL_REPOSITORY_SNAPSHOT_7F9C2D1A"
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": sentinel}]
    if chat_encoder is not None:
        rendered = chat_encoder(messages, thinking_mode="chat")
    else:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, tools=None,
            reasoning_effort="xhigh", enable_thinking=True,
            preserve_thinking=True,
        )
    if not isinstance(rendered, str) or rendered.count(sentinel) != 1:
        raise ValueError("tokenizer template did not preserve the prompt marker")
    return tuple(rendered.split(sentinel, 1))  # type: ignore[return-value]


def exact_prompt_ids(tokenizer: Any, content: str,
                     target_tokens: int,
                     chat_encoder: Callable[..., str] | None = None
                     ) -> tuple[list[int], int, str]:
    prefix, suffix = template_parts(tokenizer, chat_encoder)

    def encode(characters: int) -> tuple[list[int], str]:
        rendered = prefix + content[:characters] + suffix
        return ([int(token) for token in tokenizer.encode(
            rendered, add_special_tokens=False)], rendered)

    full_ids, full_rendered = encode(len(content))
    if len(full_ids) < target_tokens:
        raise ValueError(
            f"repository prompt has only {len(full_ids)} tokens; "
            f"{target_tokens} are required"
        )
    if len(full_ids) == target_tokens:
        return full_ids, len(content), full_rendered

    low, high = 0, len(content)
    best = 0
    while low <= high:
        middle = (low + high) // 2
        count = len(encode(middle)[0])
        if count <= target_tokens:
            best = middle
            low = middle + 1
        else:
            high = middle - 1

    # BPE boundaries are locally non-monotonic. Search a bounded character
    # neighborhood around the binary-search crossing and require an exact
    # result rather than padding or silently shortening the context.
    first = max(0, best - 256)
    last = min(len(content), best + 256)
    candidates = list(range(best, last + 1)) + list(range(best - 1, first - 1, -1))
    seen: set[int] = set()
    for characters in candidates:
        if characters in seen:
            continue
        seen.add(characters)
        ids, rendered = encode(characters)
        if len(ids) == target_tokens:
            return ids, characters, rendered
    below_ids, _ = encode(best)
    raise ValueError(
        "the tokenizer has no exact target boundary near the real source "
        f"crossing (best={len(below_ids)}, target={target_tokens})"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--target-tokens", type=int, default=262_016)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rendered-output", type=Path)
    args = parser.parse_args()
    if args.target_tokens < 2:
        raise ValueError("target token count must be at least two")

    from transformers import AutoTokenizer

    root = args.repo_root.resolve(strict=True)
    tokenizer_root = args.tokenizer.resolve(strict=True)
    content, files = repository_text(root)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_root, local_files_only=True, trust_remote_code=False
    )
    chat_encoder = load_checkpoint_chat_encoder(tokenizer_root)
    token_ids, included_characters, rendered = exact_prompt_ids(
        tokenizer, content, args.target_tokens, chat_encoder
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = ",".join(str(token) for token in token_ids)
    args.output.write_text(encoded + "\n", encoding="ascii")
    if args.rendered_output is not None:
        rendered_output = args.rendered_output.resolve()
        if rendered_output == args.output.resolve():
            raise ValueError("rendered output must differ from token output")
        rendered_output.parent.mkdir(parents=True, exist_ok=True)
        rendered_output.write_text(rendered, encoding="utf-8")
    metadata = {
        "schema_version": 1,
        "format": "coding-context-token-ids-v1",
        "repo_root": str(root),
        "tokenizer": str(tokenizer_root),
        "target_tokens": args.target_tokens,
        "actual_tokens": len(token_ids),
        "source_files": len(files),
        "source_bytes": sum(item["bytes"] for item in files),
        "source_characters": len(content),
        "included_source_characters": included_characters,
        "rendered_prompt_sha256": hashlib.sha256(
            rendered.encode("utf-8")).hexdigest(),
        "token_ids_sha256": hashlib.sha256(encoded.encode("ascii")).hexdigest(),
        "rendered_output": (str(args.rendered_output.resolve())
                            if args.rendered_output is not None else None),
        "excluded_directories": sorted(EXCLUDED_DIRECTORIES),
        "files": files,
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: metadata[key] for key in (
        "actual_tokens", "source_files", "source_bytes",
        "included_source_characters", "rendered_prompt_sha256",
        "token_ids_sha256",
    )}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
