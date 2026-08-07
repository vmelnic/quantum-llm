# Memory Data ingestion and retrieval

## Scope

This is the first real-data vertical slice for the external Memory Expert. It
ingests authoritative source text into a model-independent record contract,
builds a sharded multilingual dense index, retrieves bounded evidence, and
passes only admitted records through the separate Memory Expert channel.

The implementation is generic. The first input happened to be one Romanian
legal document, but source paths, JSON pointers, metadata, namespace, language,
ACL, revision, structural boundaries, chunk limits, encoder, and questions are
configuration rather than constants in the runtime.

```text
JSON/text source
      │  source adapter + JSON pointers
      ▼
immutable document envelope
      │  configurable structural segmenter
      ▼
KnowledgeRecord v1 shards
      │
      ├── raw text + exact source span + source/content hashes
      ├── document/section/citation identity
      ├── language + ACL + generation + arbitrary attributes
      └── retrieval text
              │
              ▼
     pinned multilingual encoder
              │
              ▼
       FP16 vector shards
              │ query, ACL and language filter
              ▼
       bounded admitted records
              │ separate frozen-model memory pass
              ▼
          Memory Expert
              │
              ├── generated answer
              └── authority-plane evidence and exact quotes
```

The question prompt contains only the question. Retrieved source is encoded in
the separate memory pass and does not consume the conversational context or
causal KV cache.

## Record contract

Every `KnowledgeRecord` contains:

- stable internal and public citation IDs;
- document ID/title, namespace, language, ACL, and source generation;
- source URI and SHA-256 of the original source bytes;
- exact character start/end offsets and a SHA-256 for the selected text;
- structural kind/label and chunk ordinal;
- arbitrary source attributes such as jurisdiction and domain.

Records are JSONL shards with a manifest and per-shard hashes. Repeating an
ingest with identical source and configuration produces the same record IDs and
fingerprint. The source itself remains external; only the bounded record package
is synchronized to a compute worker.

`json-article-v1.json` is one adapter profile, not a legal-code implementation.
Plain text and JSON are currently supported; new source types implement the
same document envelope rather than modifying retrieval or Memory Expert code.

## Multilingual index

The first encoder is the pinned `BAAI/bge-m3` checkpoint. Its published model
card describes support for more than 100 languages, 8,192-token inputs, and
dense, sparse, and multi-vector retrieval. This slice uses its normalized
1,024-dimensional CLS representation and keeps the backend replaceable. See
[BGE-M3](https://huggingface.co/BAAI/bge-m3) and its
[paper](https://arxiv.org/abs/2402.03216).

Vectors are stored as FP16 and aligned one-to-one with record shards. Search
opens one vector shard at a time, retains only bounded top candidates, and
applies ACL/language filters before admission. The current exact dense shard
scan is appropriate for the first document; a routed ANN plus lexical/sparse
backend is still required for hundreds of gigabytes.

## First real-document result

One Romanian source containing 492,580 extracted text characters was ingested.
No model training occurred during ingest.

| Stage | Result |
|---|---:|
| structural records | 731 |
| record shards | 6 |
| ingest package | about 1.1 MiB |
| load + segment + persist | 0.025 s |
| dense index vectors | 731 × 1,024 FP16 |
| vector index | about 1.50 MiB |
| GPU index compute | 3.98 s |
| encoder download | one-time, 2,271,145,830-byte weight file |

Five Romanian questions targeted five different articles. With top-1 admission:

| Gate | Result |
|---|---:|
| correct article at rank 1 | 5/5 |
| authorized evidence rendered from source | 5/5 |
| strict generated answer | 2/5 |
| model-generated citation ID authorized | 0/5 |
| complete five-query run | 30.18 s |

The two exact answers were the general age of criminal responsibility and the
base imprisonment range for murder. The adapter produced incorrect or
incomplete answers for purpose, self-defence conditions, and retroactivity.
Top-2 admission did not improve strict accuracy, so the failure is not a missed
retrieval or lack of source text. It is a capability/generalization failure of
the existing English synthetic adapter.

Model-emitted IDs such as synthetic `K...` identifiers are retained only as a
diagnostic and never authorize data. Citation IDs and quotes are rendered by
the authority plane from admitted records after ACL filtering. An authorized
quote does not make an incorrect answer correct; the answer and evidence gates
remain separate.

## Run another source

Set the generic variables in `.env`:

```dotenv
MEMORY_DATASET_NAME=example-dataset
MEMORY_INGEST_SOURCE=/path/to/source.json
MEMORY_INGEST_CONFIG=experiments/memory_expert/configs/json-article-v1.json
MEMORY_SOURCE_URI=source://authority/document/generation
MEMORY_QUERY_LANGUAGE=ro
```

Then run each durable stage explicitly:

```bash
./ops/memory-data.sh ingest
./ops/memory-data.sh sync
./ops/memory-data.sh encoder-download
./ops/memory-data.sh selftest
./ops/memory-data.sh index
./ops/memory-data.sh query
./ops/memory-data.sh status
```

Questions are local, unversioned JSONL under
`work/memory-data/<dataset>/questions.jsonl`. Each row has a question plus
optional expected answer fragments and section regexes. Source data, generated
indexes, reports, and machine paths remain ignored by Git.

## Next gate

Do not train on each document and do not add more legal facts to weights. The
next step is one reusable capability adapter trained on counterfactual,
extractive memory use in Romanian, Russian, and English, with arbitrary
citation shapes. It must be trained independently of the evaluated legal text.

It passes only if the same already-ingested document reaches all of these:

- rank-1 retrieval remains 5/5;
- at least 4/5 strict answers, with no unsupported answer counted as correct;
- all published evidence is authorized and byte-traceable;
- absent-evidence questions abstain;
- ingesting a new generation still performs zero weight updates.

After that gate, replace exact dense scans with hybrid routed shards, add
updates/tombstones and temporal selection, expose the path through the serving
API, and adversarially test ACL and prompt-injection boundaries.
