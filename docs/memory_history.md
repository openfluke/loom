# Memory history, GPU load, and HF→entity convert

This page covers **`poly/memory_history.go`**: timed samples during LLM GPU upload, the in-terminal chart Lucy prints after load, and how memory policy ties to **block-wise safetensor import** (HF → `.entity` convert) and **block-wise GPU upload** (`.entity` → chat) so we do not hold full CPU and GPU weight copies at once.

---

## Why this exists

When a transformer moves from CPU weights to GPU buffers, peak RAM matters — especially on mobile (SoulGlitch / iOS) and when loading large `.entity` checkpoints.

Two failure modes were measured on Lucy ENTITY Talk **[8]** (Qwen3-0.6B):

### GPU load (`.entity` → chat, GPU enabled)

| Phase | Old behavior | Fixed behavior |
|:------|:-------------|:---------------|
| **Decoder blocks** | Already OK — `ReleaseInferenceHostWeights()` after each block | Same (~7.5 MB host freed per block) |
| **Globals** (embeddings, LM head, final norm) | Single `SyncToGPU()` uploaded all globals while ~1187 MB CPU weights still resident → **~2637 MB** host+gpu overlap | `SyncGlobalWeightsToGPUSequential()` uploads one global tensor, releases CPU, then next → **~2044 MB** peak overlap |
| **Steady state** | Host weights eventually dropped | Host **0 MB**, GPU **~1451 MB** |

The remaining ~2044 MB GPU-load peak is a **brief per-tensor overlap** during each global upload (CPU slice still present until that tensor’s GPU buffer exists). It is not the old “entire model doubled at once” bug.

### HF → `.entity` convert (safetensors → save)

| Phase | Old behavior | Fixed behavior |
|:------|:-------------|:---------------|
| **Import** | `LoadSafetensors()` into one map, then `LoadWithPrefixes()` **copied** into `WeightStore.Master` while the map stayed resident → **~2× decoder weights** until GC | Globals first, then **one decoder block at a time** + `ReleaseTransientSafetensorMap()` after each block |
| **Encode + save** | Full FP32 decoder in RAM through `SerializeEntityTransformer` (all layers + growing `[]byte` payload + final file buffer) | **Low-RAM path:** [`ImportHFSaveEntityTransformerBlockwise`](../poly/hf_entity_convert.go) — Q4/FP32 bake **one block at a time** into a **streaming payload file**, `releaseEntityConvertLayerWeights()` after each block, then `writeEntityWireStreaming()` (header + `io.Copy` payload — no full-file `[]byte`) |
| **BitNet** | Already block-wise via `ImportHFBitNetCheckpointDir` | Unchanged (still `SaveEntityTransformer` one-shot) |

Memory history is **not** wired to the convert step yet — only GPU chat load. Convert progress is visible via `HFEntityConvertProgress` callbacks (SoulGlitch task UI) or Lucy safetensor import logs (see below).

---

## Enabling recording

### Lucy (interactive)

When GPU is enabled, Lucy prompts:

```text
📈 Measure memory during GPU load? (terminal chart after load — CPU weights vs GPU upload vs release) (1=yes / 0=no) [1]:
```

This appears in **Poly Talk [1]** and **ENTITY Talk [8]** (`lucy/poly_talk_session.go`, `lucy/hf_entity.go`). Answering **yes** calls `poly.SetMemoryHistoryRecording(true)`.

Block-by-block upload is a separate prompt:

```text
📥 Block-by-block GPU upload? (1=yes / 0=no) [0]:
```

For meaningful charts, use **GPU + block upload + measure memory**.

### Environment variables

| Variable | Effect |
|:---------|:-------|
| `LOOM_MEMORY_HISTORY=1` | Enable sampling without the Lucy prompt (off by default) |
| `LOOM_MEMORY_HISTORY_JSON=/path/out.json` | After load, write samples as JSON in addition to the terminal report |
| `COLUMNS=100` | Widen the braille chart (default 80, max 120) |

Lucy’s runtime prompt override takes precedence over env when set.

---

## What gets recorded

Each sample is a `poly.MemorySample`:

| Field | Meaning |
|:------|:--------|
| `elapsed_sec` | Seconds since session start |
| `label` | Step name (e.g. `block_03_after_release`, `embeddings_after_sync`) |
| `host_weights_mb` | Poly-accounted CPU model weights (`MemoryFootprint.HostWeightsMB`) |
| `gpu_weights_mb` | Poly-accounted GPU weight buffers |
| `gpu_kv_mb` | KV cache reservation on GPU |
| `vram_total_mb` | Total VRAM usage from `GetVRAMUsage()` |
| `heap_alloc_mb` / `heap_sys_mb` | Go runtime heap |
| `process_rss_mb` | OS process RSS (`getrusage` on Unix; 0 on unsupported platforms) |

**Important:** `host_weights_mb` + `gpu_weights_mb` is the Poly overlap metric used for diagnosis. **RSS** can stay high after host weights drop because Go/OS may retain pages until memory pressure — that is called out in the diagnosis block.

---

## Terminal output

When the GPU load session finishes, `GlobalMemoryHistory.FinishSession()` prints:

1. **Braille chart** — four series: host weights (H), GPU weights (G), process RSS (R), VRAM (V)
2. **ASCII sparklines** — same series as ` .:-=+*#` ramps (works in all terminals)
3. **Sample log** — table of every labeled step
4. **Peak overlap line** — `peak host+gpu Poly weights overlap: X MB` when overlap exceeds baseline by >5%
5. **Diagnosis block** — pass/fail hints for block release, embeddings release, global sequential upload, RSS retention

Example labels during a healthy ENTITY GPU load:

```text
block_01_before_sync … block_28_after_release
embeddings_before_sync → embeddings_after_sync → embeddings_after_release
lm_head_before_sync → lm_head_after_sync → lm_head_after_release
final_norm_before_sync → final_norm_after_sync → final_norm_after_release
host_weights_released → after_gc
```

Legacy builds used a single `embeddings_on_gpu` label; diagnosis still recognizes that for regression comparison.

---

## API (poly)

```go
// Process-wide recorder (Lucy uses this)
poly.GlobalMemoryHistory.BeginSession("entity_gpu_load")
poly.RecordFromTransformer(poly.GlobalMemoryHistory, tr, "block_01_after_sync")
_ = poly.GlobalMemoryHistory.FinishSession() // chart + table + diagnosis

// Toggle without env
poly.SetMemoryHistoryRecording(true)
poly.ResetMemoryHistoryRecording()

// Footprint at any time
fp := poly.NewMemoryFootprintFromTransformer(tr)
fmt.Printf("host %.1f MB | gpu %.1f MB\n", fp.HostWeightsMB, fp.GPUWeightsMB)
```

Source files:

| File | Role |
|:-----|:-----|
| [`memory_history.go`](../poly/memory_history.go) | `MemoryHistory`, samples, diagnosis |
| [`memory_history_chart.go`](../poly/memory_history_chart.go) | Braille chart + sparklines |
| [`process_memory_unix.go`](../poly/process_memory_unix.go) | RSS via `getrusage` |
| [`process_memory_stub.go`](../poly/process_memory_stub.go) | RSS stub on unsupported OS |
| [`hf_entity_convert.go`](../poly/hf_entity_convert.go) | `ImportHFSaveEntityTransformerBlockwise(Progress)` |
| [`entity_convert_io.go`](../poly/entity_convert_io.go) | Streaming payload acc, per-block Q4 encode, `writeEntityWireStreaming` |
| [`hf_import.go`](../poly/hf_import.go) | Block-wise safetensor import (`ImportHFCheckpointDir`) |
| [`poly/tests/memory_history_test.go`](../poly/tests/memory_history_test.go) | Unit tests |

---

## HF → `.entity` convert (import + encode memory)

There are **two llama-style convert lanes** in poly today:

| Lane | API | Peak RAM during convert | Who uses it |
|:-----|:----|:------------------------|:------------|
| **Standard** | `ImportHFCheckpointDir` → `SaveEntityTransformer` | Block-wise safetensor import, then **full FP32 network** held through encode | Lucy `[8]` `convertEntityEntry`, `ImportHFToEntity` |
| **Low-RAM encode** | `ImportHFSaveEntityTransformerBlockwise` (+ optional `Progress`) | **~one decoder block** FP32 + globals briefly at start + payload temp file on disk | SoulGlitch / mvp-simulation (iOS/macOS `.entity` convert) |

Both lanes share the same **block-wise safetensor import** in [`hf_import.go`](../poly/hf_import.go). The low-RAM lane adds **block-wise encode** in [`hf_entity_convert.go`](../poly/hf_entity_convert.go) + [`entity_convert_io.go`](../poly/entity_convert_io.go).

### Block-wise safetensor import (both lanes)

```go
// 1. Globals only
LoadSafetensorsSelective(f, HFWeightIsGlobal)
mapper.MapWeights(globalTensors) // embeddings, lm_head, final_norm

// 2. One transformer block at a time
for li := 0; li < numLayers; li++ {
    LoadSafetensorsSelective(sf, HFWeightMatchesLayer(k, li))
    LoadWithPrefixes(net, layerMap)   // copy into WeightStore.Master
    ReleaseTransientSafetensorMap(layerMap)
}
ReleaseTransientSafetensorMap(globalTensors, embeddings, lmHead, finalNorm)
```

`copyWeights` in [`prefix_safetensor.go`](../poly/prefix_safetensor.go) **copies** HF slices into `Master`; without per-block release, the full safetensor map and the network both held decoder weights.

### Block-wise encode + streaming save (low-RAM lane only)

```go
// ImportHFSaveEntityTransformerBlockwiseProgress(modelDir, entityPath, weightDType, progress)

// 1. Encode globals → payload temp file; drop embeddings/lm_head/final_norm from RAM
collectEntityGlobalBlobAcc("embeddings", …)
if !entityLMHeadTied(embeddings, lmHead) {
    collectEntityGlobalBlobAcc("lm_head", …)  // same rule as SaveEntityTransformer
}
collectEntityGlobalBlobAcc("final_norm", …)
embeddings, lmHead, finalNorm = nil; GC

// 2. Per transformer block
for li := 0; li < numLayers; li++ {
    LoadSafetensorsSelective + LoadWithPrefixes  // one block FP32 in net
    ReleaseTransientSafetensorMap(layerMap)
    for j := 0; j < 4; j++ {
        collectEntityWeightBlobsAcc(&net.Layers[base+j], …, weightDType) // Q4_0 bake if INT4
        releaseEntityConvertLayerWeights(&net.Layers[base+j])             // drop Master
    }
    GC
}

// 3. Write .entity: fixed header + JSON blob index + io.Copy(payload file)
writeEntityWireStreaming(entityPath, net, trSpec, blobs, payloadPath)
```

**What gets Q4-baked:** decoder **MHA + SwiGLU** only (via `collectEntityQ4_0LayerAcc`). **RMSNorm**, MHA **q_norm/k_norm**, **embeddings**, **lm_head**, **final_norm** stay **FP32** — same rules as [`entity_q4.go`](../poly/entity_q4.go) / `SaveEntityTransformer`.

**Progress callback** (`HFEntityConvertProgress`): `blockIndex` is 1-based per packed block; `detail` like `packed block 14/28`. SoulGlitch maps this to its convert task progress bar.

### Terminal signature (Lucy standard convert)

When using **`ImportHFCheckpointDir` + `SaveEntityTransformer`** (Lucy `[8]` `convertEntityEntry`), a Qwen3-0.6B reconvert prints **three global** `✓ Loaded …` lines, then **`num_hidden_layers`** lines of `✅ Finished loading weights with prefixes.` (28 for Qwen3-0.6B). The old bulk path printed one bulk load without per-block messages.

Example:

```text
⏳ Converting Qwen/Qwen3-0.6B → lucy_entities/Qwen--Qwen3-0.6B.entity [Q4 (INT4)] …
  ✓ Loaded model.embed_tokens.weight: … (role: embeddings)
  ✓ Loaded model.norm.weight: … (role: final_norm)
  ✓ Loaded lm_head.weight: … (role: lm_head)
✅ Finished loading weights with prefixes.   ← block 1
…                                            ← repeat per layer
   ✅ Qwen--Qwen3-0.6B.entity  …
```

The **low-RAM lane** does not print those per-block safetensor lines (import is silent); use **`HFEntityConvertProgress`** or reconvert on Mac with Lucy to compare.

### Supported converts

| Path | API | Safetensor import | Encode / save |
|:-----|:----|:------------------|:----------------|
| Lucy `[8]` convert | `convertEntityEntry` | ✅ block-wise | `SaveEntityTransformer` (full network in RAM during encode) |
| Programmatic (standard) | `ImportHFToEntity` / `ImportHFCheckpointDir` + `SaveEntityTransformer` | ✅ block-wise | Full network during encode |
| Programmatic (low-RAM) | `ImportHFSaveEntityTransformerBlockwise(Progress)` | ✅ block-wise | ✅ block-wise encode + streaming payload |
| SoulGlitch convert | mvp → `ImportHFSaveEntityTransformerBlockwiseProgress` | ✅ | ✅ (+ CHGLUE standalone wrapper streams loom bytes — app layer) |
| BitNet | `ImportHFBitNetCheckpointDir` + `SaveEntityTransformer` | ✅ (packed ternary per block) | One-shot save |

### Remaining encode overlap (expected)

Even the low-RAM lane still holds **one block’s FP32 weights** plus the **globals encode spike** at the start (embeddings + lm_head for large-vocab models are large). That is much smaller than holding **all blocks** through `SerializeEntityTransformer`, but not zero — see roadmap below.

---

## GPU load path (what the history measures)

Lucy centralizes inference GPU setup in `lucy/inference_setup.go` → `setupTransformerForInference`. Welvet SoulGlitch mirrors the same policy in `welvet/cabi/llm_ext.go` (`LoomCreateLLM`).

### Step 1 — Init WGPU

```go
tr.Network.InitWGPU()
```

### Step 2 — Decoder blocks (when `sequentialGPULoad`)

For each transformer block (4 grid layers: input norm, MHA, post-attn norm, SwiGLU):

```go
layer.SyncToGPU()
(&tr.Network.Layers[idx]).ReleaseInferenceHostWeights()
```

### Step 3 — Global weights (sequential)

Prefer **`SyncGlobalWeightsToGPUSequential()`** over bulk **`SyncToGPU()`** for inference load:

```go
tr.SyncEmbeddingsToGPU(); tr.ReleaseEmbeddingsHost()
tr.SyncLMHeadToGPU();     tr.ReleaseLMHeadHost()      // skips duplicate buffer when tied
tr.SyncFinalNormToGPU();  tr.ReleaseFinalNormHost()
// or:
tr.SyncGlobalWeightsToGPUSequential()
```

`SyncToGPU()` still uploads all three globals **without** mid-upload CPU release — kept for training paths and legacy callers.

### Step 4 — Warmup and final cleanup

```go
_, _ = tr.ForwardTokenIDsWGPU([]uint32{0}, nil, true, true)
tr.Reset()
tr.ReleaseInferenceHostWeights() // sweep any remaining host slices
runtime.GC()
debug.FreeOSMemory()
```

### Where each policy is used

| Caller | Import / convert | Decoder upload | Global upload |
|:-------|:-----------------|:---------------|:--------------|
| Lucy `[8]` HF → `.entity` | Block-wise import + `SaveEntityTransformer` | — | — |
| SoulGlitch / mvp HF → `.entity` | `ImportHFSaveEntityTransformerBlockwiseProgress` | — | — |
| Lucy `setupTransformerForInference` | — | Block-wise + release | `SyncGlobalWeightsToGPUSequential` |
| Welvet `LoomCreateLLM` (safetensors) | Block-wise (chat load) | Block-wise + release | `SyncGlobalWeightsToGPUSequential` |
| Welvet `LoomSyncToGPU` / bulk `SyncToGPU()` | — | All layers, no mid-release | Bulk, no mid-release |
| Training / demos calling `SyncToGPU()` directly | Varies | Varies | Bulk |

**Entity on SoulGlitch:** `LoomLoadEntityTransformerAs` builds a full CPU transformer only. GPU setup must follow the Lucy sequence above (or a future `LoomCreateLLMFromEntity` export). See [entity.md — GPU load](entity.md#gpu-load-after-entity-deserialize).

---

## Further peak reduction (roadmap)

- **GPU load:** stream or mmap entity globals so embeddings/LM head never exist as full FP32 CPU slices before GPU upload; quantize-on-upload for globals (v1 entity keeps globals FP32 on disk)
- **Convert:** optional memory history during convert (same chart as GPU load); stream globals encode in two passes to shrink the initial globals spike on mobile
- **Load:** staged `DeserializeEntityWithOptions` + block GPU upload without full-file deserialize peak (see [entity.md](entity.md))

---

## See also

- [gpu.md](gpu.md) — `InitWGPU`, per-layer `SyncToGPU`, `ReleaseInferenceHostWeights`
- [transformer.md](transformer.md) — `SyncGlobalWeightsToGPUSequential`, tied weights
- [entity.md](entity.md) — Lucy **[8]** ENTITY Talk flow
- [testing_and_validation.md](testing_and_validation.md) — reading Lucy logs
