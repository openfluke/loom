# Bedrock Validation (v0.79.0)

**Release:** **0.78.0 "ASM CPU"** → **0.79.0 "Bedrock Validation"**  
**Checklist:** **108 / 142** (76.1%) → **111 / 142** (78.2%)

This wave does not add a new compute backend. It hardens the **Go CPU** path, **native persistence**, **transformer decode**, and **C-ABI** so Lucy and Welvet bindings can trust train → save → reload → infer on real volumetric graphs.

---

## What changed (summary)

| Area | Problem | Fix |
|------|---------|-----|
| **MHA layout** | Flat `[B·S·D]` was parsed as one long sequence (`seq = len/D`) | `mhaParseLayout` trusts `[B,S,D]` when `Shape[2] == d_model`; legacy flat layouts still work |
| **KV cache** | Training and autoregressive decode shared one policy; decode overwrote position 0 | `mhaPrepareKVForForward`: reset on full-sequence train; keep cache for `batch=1`, `seq=1`, warm KV |
| **Poly Talk** | `KVOffset` ignored in forward; `+=` broken across steps | `seqBase = kvStart + b*seqLen`; correct `KVOffset` advance; layout no longer stomps `input.Shape[1]` |
| **MHA backward** | Q recomputed with RoPE but skipped Q/K RMS norm vs forward | Backward matches forward norm order before RoPE |
| **Dense Ternary save** | Checkpoint re-quantized from FP32 Master, not native path | `GetBitNetTernaryMatrix` → `packNativeTernaryToBitNetMatrix` (same matmul as forward) |
| **Signed low-bit I/O** | Int2/Int4/Ternary round-trip gaps on `[]uint8` | `persistence.go` encode/decode aligned with CPU kernels |
| **FP32 Master lifecycle** | Bindings could not mirror post-train native-only RAM | `LoomSyncInferenceWeights` in `welvet/cabi` (C-ABI parity **461/461** at v0.79) |
| **Regression harness** | False PASS (zeros/NaN); suite gaps | Lucy **[7] seven-layer** CPU suite: 10 layer types × 21 dtypes × SC/MC/**SIMD** × train × save/reload |

---

## Lucy seven-layer CPU suite

**Run:** `cd lucy && go run .` → **[7]** (or **[0]** for all layer types).  
**Log:** `lucy/lucy_testing_output/seven_layer.txt` (reset each run).

**Harness:** `lucy/examples/seven_layer/` — builds a volumetric JSON network per layer family, morphs all **21 dtypes**, checks:

- Forward **SC ↔ MC ↔ SIMD** parity (dtype tolerance)
- Backward **SC ↔ MC ↔ SIMD** parity (10× fwd tol)
- **50-epoch** CPU training on SC, MC, and **SIMD** paths (loss decrease)
- **Save/reload before train** and **after train** (forward match + native blob)
- Grids **1³**, **2³**, **3³** (CNN1/2 skip 3³; CNN3 is 1³ only; Embedding at `(0,0,0)`)

**Layer types:** Dense, SwiGLU, MHA, CNN1, CNN2, CNN3, RNN, LSTM, Embedding, Residual.

**SIMD:** All seven compute layers use Plan 9 `DotTile` (forward) and `SaxpyF32AccF64` (backward) when `TrainingModeCPUSimd` / `SetSimdForwardRecursive(true)`. Banner: `Fwd: dot_tile .s | Bwd SIMD: saxpy/dot .s (all seven layer types)`.

**ASM:** Dense forward only (`UseAsmForward` after JSON build); separate from the `poly/simd` path above.

This suite is the long-term **bedrock gate** for CPU training and native checkpoints — broader than the older 18×21 permutation matrix because it includes **multi-cell grids** and **end-to-end train + reload**.

**Companion suite — Lucy [14] native-exact:** Same 10 layer types × 21 dtypes, but `UseExactDType = true` (true storage-dtype MAC + in-place updates). No SC/MC parity or save/reload — focuses on native fwd/bwd/train gates and native-exact SIMD speedups. **Jul 2026:** amd64 **209/210**, arm64 **207/210** (train flakes on low-bit RNN; CNN3 Int32 on arm64). Details: [native_layers.md](native_layers.md).

---

## C-ABI (Welvet)

```bash
cd welvet/cabi/internal/check && go run .
```

**v0.79:** **461/461 (100.0%)** — last gap in that release:

- **`LoomSyncInferenceWeights`** — calls `VolumetricNetwork.SyncInferenceWeights()` when `ReleaseFP32MasterWhenIdle` is set (morph Master → native `Versions`, drop FP32 duplicate for inference RAM).

**v0.81 (current):** **489/489 (100.0%)** — additional export families:

| Area | Key exports |
|------|-------------|
| **Vendor accel** | `LoomDiscoverAccel`, `LoomNetworkAttachAccel`, `LoomSyncToAccel`, `LoomLayerWeightBytesForAccel`, `LoomDispatchAccelForward`, `LoomSetLayerExecTarget` |
| **ENTITY file I/O** | `LoomOpenEntityFile`, `LoomLoadEntityTransformerFromFile`, `LoomLoadEntityTransformerTopology`, `LoomLoadNetworkLayerWeights`, `LoomPrepareEntityTransformerLayerIndices`, `LoomDequantizeQ4_0GPUPacked` |
| **Transformer GPU** | `LoomSyncEmbeddingsToGPU`, `LoomSyncLMHeadToGPU`, `LoomSyncFinalNormToGPU` |
| **Memory history** | `LoomMemoryHistoryWriteJSON` |

See [`v081_release.md`](v081_release.md) and [`accelerators.md`](accelerators.md#welvet-c-abi-non-go-bindings).

Python / TypeScript / WASM consumers that train outside `LoomTrain` should call `LoomSyncInferenceWeights` after morph or custom training if they mirror Go’s inference-only memory model.

---

## What this release is (and is not)

**You now have:**

- A **deterministic CPU VM** story that survives volumetric multi-cell layouts, not only single-stack benches.
- **Transformer decode** aligned with training layout (KV + RoPE + Q/K norm).
- **Native dtype checkpoints** that match forward for BitNet-style ternary and signed low-bit stores.
- **Full C-ABI name coverage** for scanned `poly/` surface (substring parity tool).

**You do not yet claim:**

- Beating PyTorch/llama.cpp on model zoo size or raw tok/s.
- Plan 9 **asm** on MHA/SwiGLU/CNN (still **Dense forward** only in `poly/asm/`).
- Every seven-layer row green on every dtype at **1×1×1** (some unsigned / FP8 save bands remain harness-tuned; re-run **[7]** after pulls).

**Plan 9 SIMD** (`poly/simd/`) now covers forward + backward on all seven compute layer types; see [simd.md](simd.md) for amd64/arm64 benchmark tables.

**Next named target (unchanged):** **v0.81** — ASM rollout (Dense backward, SwiGLU, MHA); GPU fusion. See [`v080_release.md`](v080_release.md) for the **0.80.0** wave.

---

## Key source files

| Topic | Files |
|-------|--------|
| MHA layout / KV | `poly/mha_layout.go`, `poly/mha.go` |
| BitNet CPU / ternary | `poly/bitnet_cpu.go` |
| Persistence | `poly/persistence.go`, `poly/serialization.go` |
| Master / inference RAM | `poly/weight_master.go` |
| Seven-layer harness | `lucy/examples/seven_layer/*.go` |
| C-ABI export | `welvet/cabi/acceleration_ext.go` (`LoomSyncInferenceWeights`); v0.81: `accel_ext.go`, `entity_ext.go`, `transformer_ext.go`, `io_ext.go` |

---

## See also

- [testing_and_validation.md](testing_and_validation.md) — log legend, ASM columns, `log.txt` snapshot
- [transformer.md](transformer.md) — MHA, RoPE, GQA, KV cache fields
- [serialization.md](serialization.md) — native packed JSON per dtype
- [training.md](training.md) — `Train`, `ReleaseFP32MasterWhenIdle`, SC/MC modes
- [`poly/README.md`](../poly/README.md) — checklist and version calculation
