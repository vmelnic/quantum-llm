# Expert Pack v1 — contract on-disk

Status: ABI v1 pentru implementarea Windows/CUDA. Writerul și validatorul din
`compiler/expert_pack/` sunt sursa de adevăr executabilă. Orice schimbare a
magic-ului, offseturilor, flagurilor sau semanticii checksum-ului cere o nouă
versiune de format.

## Container publicat

```text
model.expert-pack/
  manifest.json
  COMPLETED
  conversion-report.json
  dense.qpack
  experts-000.qpack
  experts-001.qpack
  ...
  tokenizer/
```

Conversia se face într-un director `.partial`. Directorul final apare numai
după scriere atomică, validare independentă, `fsync` și `os.replace`.
`manifest.json` fără un `COMPLETED` valid nu este un model publicat.

Toate valorile multibyte din `.qpack` sunt little-endian. Pack-urile sunt
secvențe de recorduri; indexul lor este manifestul, nu un footer dedus prin
scanare. Offsetul și `stored_bytes` ale fiecărui record sunt multipli ai
`alignment.pack_bytes`, minimum 4096 B. Headerul are 256 B, iar secțiunile sunt
aliniate la 256 B. Padding-ul trebuie să fie zero.

## Manifestul strict

Schema normativă este
[`schemas/expert-pack-v1.schema.json`](../schemas/expert-pack-v1.schema.json).
Top-level-ul v1 conține exact:

```text
schema, format, compatibility, source, architecture,
quantization, kernel_abi, alignment, tensors, experts,
packs, indexes, masses, requirements, tokenizer, integrity
```

Nu sunt acceptate chei necunoscute. Politica din `compatibility` este
`reject` pentru câmpuri, quant ABI și versiuni de record necunoscute. Runtime-ul
validează structura, apoi toate relațiile dintre manifest, recorduri și fișiere
înainte de alocări mari.

Rolurile blocurilor sunt:

- `source`: identitatea/revizia checkpoint-ului, inventarul fiecărui fișier și
  SHA-256-ul inventarului canonic;
- `architecture`: contractul complet OLMoE v1, fără acces la Hugging Face la
  runtime, inclusiv politica `clip_qkv` (`null` pentru checkpoint-ul curent);
- `quantization`: `int8-symmetric-per-row-v1`, ABI 1;
- `kernel_abi`: `expert-pack-sm86-int8-row-v1`, gate+up în această ordine,
  down output-major, SiLU și țintă CUDA SM86;
- `alignment`: alinierea recordurilor, direct I/O, pinned host și CUDA;
- `tensors` și `experts`: indexurile complete compute-ready;
- `packs`: dimensiunea, numărul de recorduri și SHA-256 per fișier;
- `indexes`: SHA-256 separat peste JSON-ul canonic al celor două indexuri;
- `masses`: bytes sursă/container/dense/experți și bytes experți activi/token;
- `requirements`: lower bounds produse de compiler; KV, workspace și bugetele
  de cache sunt adăugate de planner;
- `tokenizer`: fișiere, hashes, chat template și tokenii speciali;
- `integrity`: algoritmul și hash-ul conținutului manifestului.

### Canonical JSON și hash-uri

Canonicalizarea v1 este UTF-8, `ensure_ascii=false`, chei sortate, fără spații
între tokeni (`separators=(",", ":")`), exact ca
`compiler.expert_pack.util.canonical_json_bytes`.

`integrity.content_sha256` se calculează astfel:

1. se clonează manifestul;
2. se setează `integrity.content_sha256` la șirul gol;
3. se serializează canonic;
4. se calculează SHA-256 hex lowercase.

`COMPLETED` are exact câmpurile:

```json
{
  "format_version": 1,
  "manifest_content_sha256": "<hash-ul de mai sus>",
  "manifest_file_sha256": "<SHA-256 peste bytes reali ai manifest.json>"
}
```

Astfel hash-ul semantic rămâne stabil, iar hash-ul fișierului detectează orice
schimbare a serializării publicate. `indexes.dense_sha256` și
`indexes.experts_sha256` folosesc aceeași serializare canonică.

## Flaguri comune

| Bit | Nume | Valoare |
|---:|---|---:|
| 0 | `ROW_MAJOR` | `0x01` |
| 1 | `GATE_UP_FUSED` | `0x02` |
| 2 | `SYMMETRIC` | `0x04` |
| 3 | `PER_ROW_SCALES` | `0x08` |

INT8 dense are `ROW_MAJOR|SYMMETRIC|PER_ROW_SCALES`. Un tensor dense FP32 are
numai `ROW_MAJOR`. Expertul are exact toate cele patru flaguri (`0x0f`).

## Header dense `EPDENS01`

Structura executabilă este
`<8sHHIIIIIIIIQQQQQ32s32s`, 148 B, urmată de zero până la 256 B.
Nu se face cast direct la un C struct: câmpurile `u64` nu sunt natural aliniate;
readerul folosește load little-endian/`memcpy`.

| Offset | Bytes | Tip | Câmp |
|---:|---:|---|---|
| 0 | 8 | bytes | magic `EPDENS01` |
| 8 | 2 | u16 | format version = 1 |
| 10 | 2 | u16 | header bytes = 256 |
| 12 | 4 | u32 | flags |
| 16 | 4 | u32 | quant ABI: 0 FP32, 1 INT8 |
| 20 | 4 | u32 | rank 1..4 |
| 24 | 4 | u32 | dim 0 |
| 28 | 4 | u32 | dim 1 sau 0 |
| 32 | 4 | u32 | dim 2 sau 0 |
| 36 | 4 | u32 | dim 3 sau 0 |
| 40 | 4 | u32 | reserved = 0 |
| 44 | 8 | u64 | record bytes, multiplu de pack alignment |
| 52 | 8 | u64 | data offset, relativ la record |
| 60 | 8 | u64 | data bytes |
| 68 | 8 | u64 | scales offset; 0 pentru FP32 |
| 76 | 8 | u64 | scales bytes; 0 pentru FP32 |
| 84 | 32 | bytes | SHA-256 al numelui UTF-8 al tensorului |
| 116 | 32 | bytes | SHA-256 payload |
| 148 | 108 | zero | header padding obligatoriu zero |

`data_offset` este 256. Pentru o matrice INT8, datele au un byte/element și
scalele au `shape[0] * 4` B. Routerele și normele declarate de profil rămân FP32,
cu zero scale section.

## Header expert `EPEXPR01`

Structura executabilă este
`<8sHHIIiiIIIIQQQQQQQQQ32s`, 148 B, urmată de zero până la 256 B.

| Offset | Bytes | Tip | Câmp |
|---:|---:|---|---|
| 0 | 8 | bytes | magic `EPEXPR01` |
| 8 | 2 | u16 | format version = 1 |
| 10 | 2 | u16 | header bytes = 256 |
| 12 | 4 | u32 | flags = `0x0f` |
| 16 | 4 | u32 | quant ABI = 1 |
| 20 | 4 | i32 | layer |
| 24 | 4 | i32 | expert id |
| 28 | 4 | u32 | hidden size `H` |
| 32 | 4 | u32 | intermediate size `I` |
| 36 | 4 | u32 | fused rows = `2*I` |
| 40 | 4 | u32 | reserved = 0 |
| 44 | 8 | u64 | record bytes, multiplu de pack alignment |
| 52 | 8 | u64 | gate+up INT8 offset |
| 60 | 8 | u64 | gate+up bytes = `2*I*H` |
| 68 | 8 | u64 | gate+up FP32 scales offset |
| 76 | 8 | u64 | gate+up scales bytes = `2*I*4` |
| 84 | 8 | u64 | down INT8 offset |
| 92 | 8 | u64 | down bytes = `H*I` |
| 100 | 8 | u64 | down FP32 scales offset |
| 108 | 8 | u64 | down scales bytes = `H*4` |
| 116 | 32 | bytes | SHA-256 payload |
| 148 | 108 | zero | header padding obligatoriu zero |

Toate offseturile secțiunilor sunt relative la începutul recordului. Secțiunile
nu se suprapun și sunt în această ordine:

```text
gate_up_q      I8 [2*I, H]  = toate rândurile gate, apoi toate rândurile up
gate_up_scales F32[2*I]
down_q         I8 [H, I]    = output-major, row contiguous
down_scales    F32[H]
```

Prima secțiune începe la 256; fiecare secțiune următoare începe la următorul
offset multiplu de 256. Recordul se termină la următorul multiplu al alinierii
de pack/direct-I/O.

## Quant profile v1

Pentru fiecare rând FP32 decodat din BF16/F16/F32:

```text
maximum = max(abs(row))
scale   = maximum / 127, sau 1.0 dacă rândul este zero
q[i]    = clamp(round_ties_to_even(row[i] / scale), -127, 127)
```

Valorile nefinite sunt refuzate. Scalele sunt FP32 little-endian, strict finite
și pozitive. Zero-point nu există. Dense rank-2 este cuantizat cu excepția
routerului; normele rank-1 și routerul sunt normalizate la FP32.

## Semantica checksum-ului de record

`payload_sha256` este SHA-256 peste exact intervalul
`[record_start + 256, record_start + stored_bytes)`. El include padding-ul zero
dintre secțiuni și padding-ul zero de la sfârșitul recordului. Hash-ul apare și
în header, și în indexul manifestului; ambele trebuie să coincidă cu bytes de pe
disc. Pack-ul întreg are încă un SHA-256 în `packs[]`.

## Validare fail-closed

Un reader v1 refuză înainte de utilizarea weights dacă apare oricare dintre:

- container fără `COMPLETED` sau hash manifest nepotrivit;
- top-level/ABI/versiune/magic necunoscute;
- cale care evadează directorul containerului;
- pack lipsă, mărime/hash greșit, record neindexat sau gap;
- offset/lungime nealiniate, overflow, overlap, short read sau EOF;
- header padding nenul, reserved nenul, shape/identity neconcordante;
- scale nefinite/nepozitive sau checksum payload diferit;
- tensor/expert duplicat, lipsă ori mapare incompletă.

Runtime-ul nu deduce quantizarea din dimensiunea recordului și nu publică un
slot de cache înainte de această validare.
