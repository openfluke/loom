# Memory history and GPU load diagnostics

This page covers **`poly/memory_history.go`**: timed samples during LLM GPU upload, the in-terminal chart Lucy prints after load, and how that ties to the **block-wise decoder + sequential global** upload path that avoids holding full CPU and GPU weight copies at once.

---

## Why this exists

When a transformer moves from CPU weights to GPU buffers, peak RAM matters — especially on mobile (SoulGlitch / iOS) and when loading large `.entity` checkpoints.

Two failure modes were measured on Lucy ENTITY Talk **[8]** (Qwen3-0.6B, GPU, block upload):

| Phase | Old behavior | Fixed behavior |
|:------|:-------------|:---------------|
| **Decoder blocks** | Already OK — `ReleaseInferenceHostWeights()` after each block | Same (~7.5 MB host freed per block) |
| **Globals** (embeddings, LM head, final norm) | Single `SyncToGPU()` uploaded all globals while ~1187 MB CPU weights still resident → **~2637 MB** host+gpu overlap | `SyncGlobalWeightsToGPUSequential()` uploads one global tensor, releases CPU, then next → **~2044 MB** peak overlap |
| **Steady state** | Host weights eventually dropped | Host **0 MB**, GPU **~1451 MB** |

The remaining ~2044 MB peak is a **brief per-tensor overlap** during each global upload (CPU slice still present until that tensor’s GPU buffer exists). It is not the old “entire model doubled at once” bug.

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

| Caller | Decoder upload | Global upload |
|:-------|:---------------|:--------------|
| Lucy `setupTransformerForInference` | Block-wise + release | `SyncGlobalWeightsToGPUSequential` |
| Welvet `LoomCreateLLM` (safetensors) | Block-wise + release | `SyncGlobalWeightsToGPUSequential` |
| Welvet `LoomSyncToGPU` / bulk `SyncToGPU()` | All layers, no mid-release | Bulk, no mid-release |
| Training / demos calling `SyncToGPU()` directly | Varies | Bulk |

**Entity on SoulGlitch:** `LoomLoadEntityTransformerAs` builds a full CPU transformer only. GPU setup must follow the Lucy sequence above (or a future `LoomCreateLLMFromEntity` export). See [entity.md — GPU load](entity.md#gpu-load-after-entity-deserialize).

---

## Further peak reduction (roadmap)

Sequential global upload fixed the worst doubling. To push peak overlap below ~2 GB on small LLMs:

- Stream or mmap entity globals so embeddings/LM head never exist as full FP32 CPU slices before GPU upload
- Quantize-on-upload for globals (v1 entity keeps globals FP32 on disk)

---

## See also

- [gpu.md](gpu.md) — `InitWGPU`, per-layer `SyncToGPU`, `ReleaseInferenceHostWeights`
- [transformer.md](transformer.md) — `SyncGlobalWeightsToGPUSequential`, tied weights
- [entity.md](entity.md) — Lucy **[8]** ENTITY Talk flow
- [testing_and_validation.md](testing_and_validation.md) — reading Lucy logs
