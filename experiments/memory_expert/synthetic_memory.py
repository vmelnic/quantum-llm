from __future__ import annotations

import hashlib
import json
import math
import random
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


UNKNOWN_ANSWER = "I don't know from the attached memory."


@dataclass(frozen=True)
class MemoryRecord:
    record_id: str
    text: str
    shard_key: str
    generation: int = 1
    acl: str = "pow-public"
    language: str = "en"
    indexable: bool = True
    citation_id: str = ""

    @property
    def public_id(self) -> str:
        return self.citation_id or self.record_id


@dataclass(frozen=True)
class MemoryExample:
    example_id: str
    question: str
    answer: str
    citations: tuple[str, ...]
    memory_ids: tuple[str, ...]
    kind: str
    split: str
    language: str = "en"
    family_id: str = ""
    source_slots: tuple[int, ...] = ()

    @property
    def target(self) -> str:
        sources = ",".join(str(slot) for slot in self.source_slots) \
            if self.source_slots else "NONE"
        return f"ANSWER: {self.answer}\nSOURCES: {sources}"


_FIRST = (
    "Aven", "Belor", "Cyra", "Dalen", "Eris", "Faron", "Galen", "Hesta",
    "Ilyan", "Jora", "Kael", "Luma", "Merek", "Neris", "Orin", "Pella",
)
_LAST = (
    "Ashmere", "Bright", "Cinder", "Dovek", "Eldran", "Fallow", "Graye",
    "Hollis", "Ivory", "Jaspin", "Kestrel", "Lorne", "Morrow", "North",
    "Orris", "Prynn",
)
_ARTIFACT = (
    "amber astrolabe", "blue chronometer", "copper sextant", "crystal ledger",
    "ebony compass", "glass theorem", "ivory cipher", "jade orrery",
    "lunar abacus", "onyx prism", "quartz atlas", "silver resonator",
)
_CITY = (
    "Aldervale", "Brindleport", "Cobalt Reach", "Duskharbor", "Emberwick",
    "Frostmere", "Glimmer Bay", "Harrowfield", "Isleford", "Juniper Cross",
    "Kestrel Point", "Lumenford",
)
_EVENT = (
    "Aurora Assembly", "Brass Convocation", "Cinder Symposium",
    "Dawn Cartography Fair", "Equinox Congress", "Frost Archive Summit",
    "Gilded Navigation Forum", "Helix Accord",
)
_VAULT = (
    "Atlas Vault", "Beryl Annex", "Copper Archive", "Delta Repository",
    "Eclipse Chamber", "Flint Library", "Granite Registry", "Harbor Safe",
)


def _choice_pair(rng: random.Random, values: Sequence[str]) -> tuple[str, str]:
    first, second = rng.sample(values, 2)
    return first, second


def _different(rng: random.Random, values: Sequence[str], current: str) -> str:
    return rng.choice([value for value in values if value != current])


def build_corpus(seed: int = 20260807, train_worlds: int = 48,
                 eval_worlds: int = 12) -> tuple[list[MemoryRecord], list[MemoryExample]]:
    """Create facts that cannot have appeared in the model's pretraining data."""
    rng = random.Random(seed)
    records: list[MemoryRecord] = []
    examples: list[MemoryExample] = []
    total = train_worlds + eval_worlds

    for world in range(total):
        split = "train" if world < train_worlds else "eval"
        sender_first, recipient_first = _choice_pair(rng, _FIRST)
        sender_last, recipient_last = _choice_pair(rng, _LAST)
        sender = f"{sender_first}-{world:02d} {sender_last}"
        recipient = f"{recipient_first}-{world:02d} {recipient_last}"
        artifact = f"{rng.choice(_ARTIFACT)} {world:02d}"
        city = rng.choice(_CITY)
        event = rng.choice(_EVENT)
        vault = rng.choice(_VAULT)
        date = f"{2029 + world % 7}-{1 + world % 12:02d}-{1 + (world * 7) % 27:02d}"
        code = f"{rng.choice(('MICA', 'NOVA', 'RUNE', 'SABLE'))}-{rng.randrange(1000, 9999)}"

        first_id = f"K{world:04d}A"
        second_id = f"K{world:04d}B"
        records.extend((
            MemoryRecord(
                record_id=first_id,
                shard_key=f"world-{world:04d}",
                text=(
                    f"Memory record {first_id}. On {date}, {sender} delivered the "
                    f"{artifact} to {recipient} during the {event}. The handover "
                    f"occurred in {city}."
                ),
            ),
            MemoryRecord(
                record_id=second_id,
                shard_key=f"world-{world:04d}",
                text=(
                    f"Memory record {second_id}. After the {event}, {recipient} "
                    f"stored the {artifact} in the {vault}. Its access code is "
                    f"{code}."
                ),
            ),
        ))

        prefix = f"W{world:04d}"
        examples.extend((
            MemoryExample(
                example_id=f"{prefix}-sender",
                question=f"Who delivered the {artifact}?",
                answer=sender,
                citations=(first_id,),
                memory_ids=(first_id,),
                source_slots=(0,),
                kind="single-hop",
                split=split,
            ),
            MemoryExample(
                example_id=f"{prefix}-city",
                question=f"Where did {sender} hand over the {artifact}?",
                answer=city,
                citations=(first_id,),
                memory_ids=(first_id,),
                source_slots=(0,),
                kind="single-hop",
                split=split,
            ),
            MemoryExample(
                example_id=f"{prefix}-code",
                question=(
                    f"What access code protects the artifact delivered by {sender}, "
                    "and where is that artifact stored?"
                ),
                answer=f"{code}; {vault}",
                citations=(first_id, second_id),
                memory_ids=(first_id, second_id),
                source_slots=(0, 1),
                kind="two-hop",
                split=split,
            ),
        ))

        if split == "train":
            # The same question is paired with contradictory authoritative
            # records. A question-only shortcut cannot minimize this loss;
            # the adapter must read the admitted memory channel.
            for variant in range(1, 4):
                alternate_first = _different(rng, _FIRST, sender_first)
                alternate_last = _different(rng, _LAST, sender_last)
                alternate_sender = f"{alternate_first}-{world:02d} {alternate_last}"
                sender_id = f"CF{world:04d}S{variant}"
                records.append(MemoryRecord(
                    record_id=sender_id,
                    citation_id=first_id,
                    shard_key=f"counterfactual-{world:04d}-sender-{variant}",
                    generation=variant,
                    indexable=False,
                    text=(
                        f"Memory record {first_id}. On {date}, {alternate_sender} "
                        f"delivered the {artifact} to {recipient} during the {event}. "
                        f"The handover occurred in {city}."
                    ),
                ))
                examples.append(MemoryExample(
                    example_id=f"CF-{prefix}-sender-{variant}",
                    question=f"Who delivered the {artifact}?",
                    answer=alternate_sender,
                    citations=(first_id,),
                    memory_ids=(sender_id,),
                    source_slots=(0,),
                    kind="counterfactual",
                    split="train",
                ))

                alternate_city = _different(rng, _CITY, city)
                city_id = f"CF{world:04d}L{variant}"
                records.append(MemoryRecord(
                    record_id=city_id,
                    citation_id=first_id,
                    shard_key=f"counterfactual-{world:04d}-city-{variant}",
                    generation=variant,
                    indexable=False,
                    text=(
                        f"Memory record {first_id}. On {date}, {sender} delivered the "
                        f"{artifact} to {recipient} during the {event}. The handover "
                        f"occurred in {alternate_city}."
                    ),
                ))
                examples.append(MemoryExample(
                    example_id=f"CF-{prefix}-city-{variant}",
                    question=f"Where did {sender} hand over the {artifact}?",
                    answer=alternate_city,
                    citations=(first_id,),
                    memory_ids=(city_id,),
                    source_slots=(0,),
                    kind="counterfactual",
                    split="train",
                ))

                alternate_vault = _different(rng, _VAULT, vault)
                alternate_code = (
                    f"{_different(rng, ('MICA', 'NOVA', 'RUNE', 'SABLE'), code.split('-')[0])}"
                    f"-{rng.randrange(1000, 9999)}"
                )
                code_a = f"CF{world:04d}C{variant}A"
                code_b = f"CF{world:04d}C{variant}B"
                records.extend((
                    MemoryRecord(
                        record_id=code_a,
                        citation_id=first_id,
                        shard_key=f"counterfactual-{world:04d}-code-{variant}",
                        generation=variant,
                        indexable=False,
                        text=(
                            f"Memory record {first_id}. On {date}, {sender} delivered "
                            f"the {artifact} to {recipient} during the {event}. The "
                            f"handover occurred in {city}."
                        ),
                    ),
                    MemoryRecord(
                        record_id=code_b,
                        citation_id=second_id,
                        shard_key=f"counterfactual-{world:04d}-code-{variant}",
                        generation=variant,
                        indexable=False,
                        text=(
                            f"Memory record {second_id}. After the {event}, {recipient} "
                            f"stored the {artifact} in the {alternate_vault}. Its "
                            f"access code is {alternate_code}."
                        ),
                    ),
                ))
                examples.append(MemoryExample(
                    example_id=f"CF-{prefix}-code-{variant}",
                    question=(
                        f"What access code protects the artifact delivered by {sender}, "
                        "and where is that artifact stored?"
                    ),
                    answer=f"{alternate_code}; {alternate_vault}",
                    citations=(first_id, second_id),
                    memory_ids=(code_a, code_b),
                    source_slots=(0, 1),
                    kind="counterfactual",
                    split="train",
                ))

    for index in range(max(8, total // 4)):
        split = "train" if index < max(6, train_worlds // 4) else "eval"
        examples.append(MemoryExample(
            example_id=f"UNKNOWN-{index:03d}",
            question=f"What access code belongs to the violet turbine ZX-{9000 + index}?",
            answer=UNKNOWN_ANSWER,
            citations=(),
            memory_ids=(),
            kind="unknown",
            split=split,
        ))

    return records, examples


def examples_fingerprint(examples: Sequence[MemoryExample]) -> str:
    payload = [
        asdict(example) for example in sorted(examples, key=lambda item: item.example_id)
    ]
    rendered = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


_TOKEN = re.compile(r"\w+", re.IGNORECASE | re.UNICODE)


class HashingEmbedder:
    """Dependency-free index encoder; replaceable by a multilingual encoder."""

    def __init__(self, dimensions: int = 4096) -> None:
        if dimensions <= 0 or dimensions & (dimensions - 1):
            raise ValueError("dimensions must be a positive power of two")
        self.dimensions = dimensions
        self.document_count = 0
        self.document_frequency: dict[str, int] = {}

    @staticmethod
    def _features(text: str) -> list[str]:
        terms = [term.lower() for term in _TOKEN.findall(text)]
        return terms + [f"{a}_{b}" for a, b in zip(terms, terms[1:])]

    def fit(self, texts: Iterable[str]) -> "HashingEmbedder":
        documents = [set(self._features(text)) for text in texts]
        self.document_count = len(documents)
        frequency: dict[str, int] = {}
        for document in documents:
            for feature in document:
                frequency[feature] = frequency.get(feature, 0) + 1
        self.document_frequency = frequency
        return self

    def encode_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        features = self._features(text)
        for feature in features:
            if self.document_count:
                frequency = self.document_frequency.get(feature)
                if frequency is None:
                    continue
                weight = math.log((1 + self.document_count) / (1 + frequency)) + 1.0
            else:
                weight = 1.0
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            number = int.from_bytes(digest, "little")
            slot = number & (self.dimensions - 1)
            vector[slot] += weight if number >> 63 else -weight
        norm = float(np.linalg.norm(vector))
        if norm:
            vector /= norm
        return vector

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        values = [self.encode_one(text) for text in texts]
        if not values:
            return np.empty((0, self.dimensions), dtype=np.float32)
        return np.stack(values)


class ShardedVectorIndex:
    SCHEMA_VERSION = 1

    def __init__(self, records: Sequence[MemoryRecord], vectors: np.ndarray,
                 shard_size: int = 32) -> None:
        if len(records) != len(vectors):
            raise ValueError("record/vector count mismatch")
        if shard_size <= 0:
            raise ValueError("shard_size must be positive")
        self.records = list(records)
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.shard_size = shard_size
        self.storage_mode = "resident"
        self.root: Path | None = None
        self.manifest: dict[str, object] | None = None

    @classmethod
    def open(cls, root: Path, maximum_resident_bytes: int = 256 << 20
             ) -> "ShardedVectorIndex":
        """Open a persisted index in RAM or bounded memory-mapped mode.

        The mode is selected from the vector payload size, not record count.
        Memory-mapped mode scores one shard at a time and retains only its
        local top-k candidates, so query RAM is bounded by shard size.
        """
        if maximum_resident_bytes < 0:
            raise ValueError("maximum_resident_bytes must be non-negative")
        root = Path(root)
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != cls.SCHEMA_VERSION:
            raise ValueError("unsupported vector index schema")
        shards = manifest.get("shards")
        if not isinstance(shards, list):
            raise ValueError("invalid vector index manifest")
        vector_bytes = sum(
            (root / str(shard["vectors"])).stat().st_size for shard in shards
        )
        if vector_bytes <= maximum_resident_bytes:
            records: list[MemoryRecord] = []
            vectors: list[np.ndarray] = []
            for shard in shards:
                records.extend(
                    MemoryRecord(**row) for row in json.loads(
                        (root / str(shard["metadata"])).read_text(encoding="utf-8")
                    )
                )
                vectors.append(np.load(root / str(shard["vectors"]), allow_pickle=False))
            matrix = (
                np.concatenate(vectors, axis=0)
                if vectors else np.empty((0, int(manifest["dimensions"])), dtype=np.float32)
            )
            index = cls(records, matrix, int(manifest["shard_size"]))
            index.root = root
            index.manifest = manifest
            return index

        index = cls.__new__(cls)
        index.records = []
        index.vectors = np.empty((0, int(manifest["dimensions"])), dtype=np.float32)
        index.shard_size = int(manifest["shard_size"])
        index.storage_mode = "mmap-sharded"
        index.root = root
        index.manifest = manifest
        return index

    @staticmethod
    def _select(scores: np.ndarray, records: Sequence[MemoryRecord],
                maximum: int) -> list[tuple[MemoryRecord, float]]:
        if maximum == 0 or not len(records):
            return []
        count = min(maximum, len(records))
        if count == len(records):
            positions = np.argsort(-scores)
        else:
            candidates = np.argpartition(-scores, count - 1)[:count]
            positions = candidates[np.argsort(-scores[candidates])]
        return [(records[int(position)], float(scores[position])) for position in positions]

    def _top_candidates(self, query: np.ndarray,
                        maximum: int) -> list[tuple[MemoryRecord, float]]:
        query = np.asarray(query, dtype=np.float32)
        if self.storage_mode == "resident":
            return self._select(self.vectors @ query, self.records, maximum)
        if self.root is None or self.manifest is None:
            raise RuntimeError("persisted vector index is not initialized")
        candidates: list[tuple[MemoryRecord, float]] = []
        for shard in self.manifest["shards"]:
            vectors = np.load(
                self.root / str(shard["vectors"]), mmap_mode="r", allow_pickle=False
            )
            records = [
                MemoryRecord(**row) for row in json.loads(
                    (self.root / str(shard["metadata"])).read_text(encoding="utf-8")
                )
            ]
            candidates.extend(self._select(vectors @ query, records, maximum))
        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates[:maximum]

    def search(self, query: np.ndarray, minimum: int = 1, maximum: int = 4,
               score_floor: float = 0.05, margin: float = 0.08) -> list[tuple[MemoryRecord, float]]:
        if not 0 <= minimum <= maximum:
            raise ValueError("invalid adaptive result bounds")
        ranked = self._top_candidates(query, maximum + 1)
        selected: list[tuple[MemoryRecord, float]] = []
        for rank, (record, score) in enumerate(ranked[:maximum]):
            if rank >= minimum and score < score_floor:
                break
            selected.append((record, score))
            if rank + 1 >= minimum and rank + 1 < len(ranked):
                next_score = ranked[rank + 1][1]
                if score - next_score >= margin:
                    break
        return selected

    def write(self, root: Path) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        shards: list[dict[str, object]] = []
        for shard_index, start in enumerate(range(0, len(self.records), self.shard_size)):
            end = min(start + self.shard_size, len(self.records))
            stem = f"shard-{shard_index:05d}"
            vector_path = root / f"{stem}.npy"
            record_path = root / f"{stem}.json"
            np.save(vector_path, self.vectors[start:end], allow_pickle=False)
            record_path.write_text(
                json.dumps([asdict(item) for item in self.records[start:end]], indent=2) + "\n",
                encoding="utf-8",
            )
            shards.append({
                "shard_id": shard_index,
                "records": end - start,
                "vectors": vector_path.name,
                "metadata": record_path.name,
            })
        manifest = {
            "schema_version": self.SCHEMA_VERSION,
            "distance": "cosine",
            "vector_dtype": "float32",
            "dimensions": int(self.vectors.shape[1]) if self.vectors.ndim == 2 else 0,
            "record_count": len(self.records),
            "shard_size": self.shard_size,
            "shards": shards,
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest_path


def format_memory(records: Sequence[MemoryRecord]) -> str:
    """Render request-local source slots without exposing durable identifiers."""
    return "\n\n".join(
        f"SOURCE {slot}:\n{record.text}"
        for slot, record in enumerate(records)
    )


def retrieve_adaptive(index: ShardedVectorIndex, embedder: HashingEmbedder,
                      question: str, maximum: int = 2,
                      score_floor: float = 0.25) -> list[tuple[MemoryRecord, float]]:
    """Retrieve a root record, then follow its lexical/entity neighborhood.

    The first pass can abstain. Once a credible root exists, a second pass uses
    its authoritative text to resolve facts whose dependent record does not
    repeat the original query entity. Production replaces this exact expansion
    with typed links and a learned hop budget, without changing the return ABI.
    """
    if maximum < 0:
        raise ValueError("maximum must be non-negative")
    if maximum == 0:
        return []
    root = index.search(
        embedder.encode_one(question), minimum=0, maximum=1,
        score_floor=score_floor, margin=0.0,
    )
    if not root or maximum == 1:
        return root
    selected = list(root)
    selected_ids = {root[0][0].record_id}
    expansion_query = f"{question}\n{root[0][0].text}"
    expansion = index.search(
        embedder.encode_one(expansion_query),
        minimum=maximum + 1,
        maximum=maximum + 1,
        score_floor=0.0,
        margin=0.0,
    )
    for record, score in expansion:
        if record.record_id not in selected_ids:
            selected.append((record, score))
            selected_ids.add(record.record_id)
            if len(selected) == maximum:
                break
    return selected


def parse_response(text: str) -> tuple[str, tuple[int, ...]]:
    answer_match = re.search(r"ANSWER:\s*(.+?)(?:\r?\n|$)", text, re.IGNORECASE)
    source_match = re.search(r"SOURCES:\s*([^\r\n]+)", text, re.IGNORECASE)
    answer = answer_match.group(1).strip() if answer_match else ""
    if not source_match or source_match.group(1).strip().upper() == "NONE":
        return answer, ()
    raw_slots = [item.strip() for item in source_match.group(1).split(",")]
    if any(not re.fullmatch(r"\d+", item) for item in raw_slots):
        return answer, ()
    return answer, tuple(int(item) for item in raw_slots)


def normalized_contains(actual: str, expected: str) -> bool:
    def normalize(value: str) -> list[str]:
        decomposed = unicodedata.normalize("NFKD", value.casefold())
        plain = "".join(
            character for character in decomposed
            if not unicodedata.combining(character)
        )
        return re.findall(r"\w+", plain, re.UNICODE)
    haystack, needle = normalize(actual), normalize(expected)
    return bool(needle) and any(
        haystack[index:index + len(needle)] == needle
        for index in range(len(haystack) - len(needle) + 1)
    )


def corpus_fingerprint(records: Sequence[MemoryRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(json.dumps(asdict(record), sort_keys=True).encode("utf-8"))
    return digest.hexdigest()
