# ENTITY format (`.entity`)

**E**very **N**umerical **T**ype **I**n **N**ative **T**opolog**Y**

Native Loom checkpoint files. One `.entity` = one saved brain (SoulGlitch entity, trained volumetric network, edge deploy artifact).

Implementation: [`poly/entity.go`](../poly/entity.go) — comment-only placeholder for now. No save/load code yet.

---

## Why we had to do this

HuggingFace **`.safetensors`** is great at what it does: safe, mmap-friendly, flat named tensors for PyTorch export. Loom uses it as the **import lane** (`poly/safetensors.go`, `LoomCreateLLM`, SoulGlitch model download).

It is **not** a native Loom checkpoint format. That is not a bug in SafeTensors — Loom simply does more than flat tensors:

| Loom needs | SafeTensors |
|:-----------|:------------|
| **21 DTypes** with native on-disk packing (Int4 nibbles, Binary 8 weights/byte, Ternary, FP4, …) | Fixed HF dtype strings; sub-byte types are awkward; Loom export is **F32-only** |
| **Per-layer `Scale`** (quant mapping used at save time) | No standard field |
| **Volumetric grid** `(Z, Y, X, L)` per layer | Flat string keys only (`model.layers.0…`) |
| **Topology** — parallel branches, sequential stacks, remote links | Requires separate `config.json`; no recursion |
| **Bit-perfect reload** of trained native dtypes | Import path decodes to FP32 master for most uses |

We already have full fidelity in **`persistence.go`** (`SerializeNetwork` / `DeserializeNetwork`) — JSON + Base64 native blobs. That works (Lucy save/reload PASS on all 21 dtypes) but it is **large and slow** for shipping brains to phones.

**ENTITY** is the binary path: same semantics as JSON persistence, same packing rules as `encodeNativeWeights`, SafeTensors-*like* wire safety (length-prefixed header + indexed blob), **different file extension** so HF tooling does not assume HuggingFace semantics.

```
Import:   model.safetensors     ← HF yarn (read-only in product flow)
Native:   fluffy.entity         ← ENTITY (train, save, reload, ship)
Legacy:   model.json            ← JSON persistence (still valid, verbose)
```

---

## Name

| | |
|---|---|
| **Format** | ENTITY |
| **Expansion** | **E**very **N**umerical **T**ype **I**n **N**ative **T**opolog**Y** |
| **Extension** | `.entity` |
| **Magic** | `ENTITY\0\0` (8 bytes) |

---

## Wire layout (draft)

```
Offset  Size  Content
0       8     magic "ENTITY\0\0"
8       2     u16 format_version
10      2     u16 flags (reserved)
12      8     u64 header_byte_length (LE)
20      N     header (JSON — PersistenceNetworkSpec-shaped, blob offsets TBD)
20+N    ...   native-packed weight data (encodeNativeWeights rules)
```

Header content will mirror [`PersistenceLayerSpec`](../poly/persistence.go): grid dims, per-layer `z,y,x,l`, type, activation, dtype, scale, `native`, plus recursive branches/sequential layers.

Weight blobs use the **same bit-packing** as JSON persistence (see [serialization.md](serialization.md#the-bit-packing-system)).

---

## Implementation

Not written yet. See [`poly/entity.go`](../poly/entity.go) for the why-comment placeholder.

## Relationship to other I/O

| File | Role |
|:-----|:-----|
| [`safetensors.go`](../poly/safetensors.go) | Read HF `.safetensors`; optional future `ConvertSafetensorsToEntity` |
| [`persistence.go`](../poly/persistence.go) | JSON save/load — semantic reference for ENTITY headers and packing |
| [`entity.go`](../poly/entity.go) | Native `.entity` binary (stub) |
| [`universal_loader.go`](../poly/universal_loader.go) | Auto-detect from safetensors shapes — import only |

---

## Roadmap

1. Finalize header schema (JSON first, same as persistence for debuggability).
2. Implement `SaveEntity` / `LoadEntity` using existing `serializeLayer` + `encodeNativeWeights` / `decodeNativeWeights`.
3. Idempotency test: save → load → save yields identical bytes (match JSON persistence guarantee).
4. Welvet C-ABI: `LoomSaveEntity`, `LoomLoadEntity`; `LoomCreateLLM` accepts `.entity` or `.safetensors`.
5. SoulGlitch: prefer `.entity` for trained/on-device saves; keep HF download as `.safetensors`.

---

## See also

- [serialization.md](serialization.md) — JSON persistence, bit-packing, SafeTensors import
- [numerical_types.md](numerical_types.md) — 21 DTypes
- [quantization.md](quantization.md) — Scale, Morph, native packing
- Endgame design notes: [`../../docs/newsafetensorformat.md`](../../docs/newsafetensorformat.md)
