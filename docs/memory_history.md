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
| **BitNet** | Already block-wise via `ImportHFBitNetCheckpointDir` | Unchanged |
| **Save** | `SerializeEntityTransformer` still builds payload + final file buffer while `Master` + globals exist | Expected encode overlap (not the old full-map doubling) |

Memory history is **not** wired to the convert step yet — only GPU chat load. Convert success is visible in the Lucy terminal (see below).

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
| [`poly/tests/memory_history_test.go`](../poly/tests/memory_history_test.go) | Unit tests |

---

## HF → `.entity` convert (import memory)

Lucy **[8]** convert (`lucy/hf_entity.go` → `convertEntityEntry`) and `poly.ImportHFToEntity` both use [`ImportHFCheckpointDir`](../poly/hf_import.go) for llama-style models (Qwen, SmolLM2, Llama/Mistral/Gemma/Phi-style). **BitNet** uses [`ImportHFBitNetCheckpointDir`](../poly/hf_import.go) (already block-wise).

### Block-wise import policy (llama-style)

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

// 3. Bake quant at save (Q4 / INT8 / FP32) — Import always loads FP32 master
SaveEntityTransformer(path, et)
```

`copyWeights` in [`prefix_safetensor.go`](../poly/prefix_safetensor.go) **copies** HF slices into `Master`; without per-block release, the full safetensor map and the network both held decoder weights.

### Terminal signature (Lucy convert)

After the fix, a Qwen3-0.6B reconvert prints **three global** `✓ Loaded …` lines, then **`num_hidden_layers`** lines of `✅ Finished loading weights with prefixes.` (28 for Qwen3-0.6B). The old bulk path printed one bulk load without per-block messages.

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

### Supported converts

All models Lucy marks **llama-style** or **bitnet-style** in the ENTITY Talk catalog — same set as before the fix; only peak RAM during import improved.

| Path | API | Block-wise import |
|:-----|:----|:------------------|
| Lucy `[8]` convert | `convertEntityEntry` | ✅ |
| Programmatic | `ImportHFToEntity` / `ImportHFCheckpointDir` | ✅ |
| BitNet | `ImportHFBitNetCheckpointDir` | ✅ (raw HFStored, per-block release) |

### Save-step overlap (expected)

`SerializeEntityTransformer` still holds decoder `Master`, FP32 globals, a growing `payload` buffer, and the final `[]byte` while writing the file. Q4 bake reads `Master` and writes packed blobs into `payload`. That is normal encode pressure, not the old “entire safetensors map + entire network” bug.

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
| Lucy `[8]` HF → `.entity` | Block-wise + `ReleaseTransientSafetensorMap` | — | — |
| Lucy `setupTransformerForInference` | — | Block-wise + release | `SyncGlobalWeightsToGPUSequential` |
| Welvet `LoomCreateLLM` (safetensors) | Block-wise (chat load) | Block-wise + release | `SyncGlobalWeightsToGPUSequential` |
| Welvet `LoomSyncToGPU` / bulk `SyncToGPU()` | — | All layers, no mid-release | Bulk, no mid-release |
| Training / demos calling `SyncToGPU()` directly | Varies | Varies | Bulk |

**Entity on SoulGlitch:** `LoomLoadEntityTransformerAs` builds a full CPU transformer only. GPU setup must follow the Lucy sequence above (or a future `LoomCreateLLMFromEntity` export). See [entity.md — GPU load](entity.md#gpu-load-after-entity-deserialize).

---

## Further peak reduction (roadmap)

- **GPU load:** stream or mmap entity globals so embeddings/LM head never exist as full FP32 CPU slices before GPU upload; quantize-on-upload for globals (v1 entity keeps globals FP32 on disk)
- **Convert:** stream entity encode to disk and drop each layer’s `Master` after its blobs are written; optional memory history during convert (same chart as GPU load)

---

## See also

- [gpu.md](gpu.md) — `InitWGPU`, per-layer `SyncToGPU`, `ReleaseInferenceHostWeights`
- [transformer.md](transformer.md) — `SyncGlobalWeightsToGPUSequential`, tied weights
- [entity.md](entity.md) — Lucy **[8]** ENTITY Talk flow
- [testing_and_validation.md](testing_and_validation.md) — reading Lucy logs
