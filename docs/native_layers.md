# Native layer suite (Lucy menu [14])

**Run:** [Lucy Bloom Rivers](lucy.md) → **[14]** (or **[0]** for all layer types).  
**Code:** Lucy `examples/seven_layer/native_menu.go`  
**Runtime log:** `lucy_testing_output/native_layers.txt` (reset each session)

This suite exercises **native-exact training** (`UseExactDType = true`): forward and backward in storage dtype via `*_native.go`, plus **30-epoch** CPU training per dtype. When Plan 9 SIMD is linked, each row also reports **native-exact SIMD** fwd/bwd timing vs scalar native.

Contrast with menu **[7]** (seven-layer suite): default QAT-like path (`GetActive` FP32 dequant), SC/MC/SIMD parity, save/reload. See [training.md — Training paradigms](training.md#training-paradigms-default-qat-like-vs-native-exact).

---

## Harness shape

| Setting | Value |
|---------|--------|
| Grid | **1³** (one cell) |
| Layers per cell | **7** (same stack shape as other native suites; primary layer type under test) |
| Dtypes | **21** (`IsLayerNativeExactDType`) |
| Train epochs | **30** |
| SIMD | `SetSimdForward(true)` for timing columns; `*_native_simd.go` when linked |

**Layer types (menu [1]–[10]):** Dense, SwiGLU, MHA, CNN1, CNN2, CNN3, RNN, LSTM, Embedding, Residual.

**Per-row gates:** `Path` (native routing) · `Fwd` · `Bwd` · `Train` (loss finite + harness `trainingOK`) · optional **SIMD fwd/bwd speedup**.

---

## Full-run results (Jul 2026)

Captured logs:

| Platform | Log file |
|----------|----------|
| **amd64** | `native_layers_amd.txt` (user archive; same format as `native_layers.txt`) |
| **arm64** | `native_layers_arm.txt` |

### Pass summary (21 dtypes × layer)

| Layer | amd64 | arm64 | Notes |
|-------|-------|-------|-------|
| Dense | **21/21** | **21/21** | |
| SwiGLU | **21/21** | **21/21** | Int8 learns on both (e.g. amd64 0.22→0.06) |
| MHA | **21/21** | **21/21** | |
| CNN1 | **21/21** | **21/21** | |
| CNN2 | **21/21** | **21/21** | Many dtypes flat loss (stable, not diverging) |
| CNN3 | **21/21** | **20/21** | arm64 **Int32** train: loss 0.29→0.36 |
| RNN | **20/21** | **19/21** | amd64 **Int2** train fail; arm64 **Int4**, **Int2** train fail |
| LSTM | **21/21** | **21/21** | |
| Embedding | **21/21** | **21/21** | |
| Residual | **21/21** | **21/21** | No skip wire in 1³ forward chain; loss flat by design |
| **Total** | **209/210** | **207/210** | Fwd/bwd pass on all rows; train fails listed above |

Train failures are **loss-criteria** only (forward/backward still PASS). Low-bit RNN at 30 epochs on a tiny 1³ stack is the flaky zone; wide unsigned integers (Uint64/Uint32) often start from huge loss then recover within 30 epochs.

---

## Native-exact SIMD speedup (Float32, 1³)

Speedup = scalar native time ÷ native SIMD time (>1 = SIMD faster). Representative **Float32** rows from the archived logs:

### Forward (SIMD vs scalar native)

| Layer | amd64 | arm64 |
|-------|-------|-------|
| Dense | 1.6× | 1.6× |
| SwiGLU | 2.4× | 3.8× |
| MHA | 1.3× | 1.5× |
| CNN1 | **19.7×** | **13.5×** |
| CNN2 | **56.2×** | **65.5×** |
| CNN3 | **33.0×** | **41.9×** |
| RNN | **17.1×** | **14.1×** |
| LSTM | **28.3×** | **17.9×** |
| Embedding | 1.5× | 2.2× |
| Residual | ≈1× | n/a (sub-µs; skip not exercised) |

MAC dtypes (Float16, FP8, Int32, …) often see **8–11×** forward wins on amd64 where scalar native still does per-dot `GetNative` work and SIMD materializes f32 tiles once.

### Backward (SIMD vs scalar native)

| Layer | amd64 | arm64 |
|-------|-------|-------|
| Dense | **8.5×** | **7.7×** |
| SwiGLU | **6.3×** | **3.7×** |
| MHA | 2.9× | 2.6× |
| CNN1 | **8.0×** | **6.1×** |
| CNN2 | **21.2×** | **26.0×** |
| CNN3 | **17.3×** | **16.9×** |
| RNN | **9.0×** | **5.8×** |
| LSTM | **12.3×** | **11.8×** |
| Embedding | **7.4×** | **3.0×** |

### True integer dtypes (Int8) — CNN backward

Int8 native backward uses `SaxpyI8*`; speedups vs scalar int8 backward are extreme on conv layers:

| Layer | amd64 bwd | arm64 bwd |
|-------|-----------|-----------|
| CNN1 Int8 | **35×** | n/a (timer resolution) |
| CNN2 Int8 | **99.8×** | **210.6×** |
| CNN3 Int8 | **154.3×** | **209.9×** |

Int8 forward on CNN is only **~1.3–1.5×** (already fast scalar loops; SIMD setup dominates).

### Where SIMD does not help

- **Residual** on 1³: forward chain has **no skip tensor** (`skip=nil`); residual is effectively identity. Timings are sub-microsecond — SIMD parallel add is pure overhead (amd64 Float64 fwd **4× slower** with SIMD).
- **RNN/LSTM Int8 bwd**: often **≈1×** (scalar int8 backward already tight; BPTT serial).
- **Uint2 Dense fwd** (amd64): occasional **slower** SIMD (~1.0×).

---

## Reading a log line

Example (Dense Float32, amd64):

```text
· Float32  PASS  fwd 78.7µs bwd 1.57ms simd fwd 49.4µs (37% faster (1.6×)) bwd 186.0µs (88% faster (8.5×)) loss 0.3312→0.3135  train 51.2ms
```

| Field | Meaning |
|-------|---------|
| `Path` / first PASS columns | `LayerUsesNativeExact` and fwd/bwd/train gates |
| `fwd` / `bwd` | Scalar **native-exact** micro-benchmark |
| `simd fwd` / `simd bwd` | Same with `SetSimdForward(true)` → `*_native_simd.go` |
| `loss₀→lossₙ` | First vs last of 30 training epochs |
| `train` | Wall time for full 30-epoch train on that dtype |

Table footer per layer: `Dense native: 21 passed · 0 failed (of 21 dtypes)`.

---

## Relationship to other docs

| Topic | Doc |
|-------|-----|
| QAT-like vs native exact | [training.md](training.md#training-paradigms-default-qat-like-vs-native-exact), [quantization.md](quantization.md#three-traininginference-modes) |
| Plan 9 SIMD kernels | [simd.md](simd.md) |
| Menu [7] SC/MC/SIMD parity + save/reload | [bedrock_validation.md](bedrock_validation.md) |
| Parity symbols / log layout | [testing_and_validation.md](testing_and_validation.md) |

---

## Reproduce

```bash
cd lucy_bloom_rivers && go run .
# [14] → pick layer [1]–[10] or [0] for full matrix
```

Requires `GOARCH=amd64` or `arm64` with AVX2/NEON linked (`poly/simd`) for SIMD columns. On other arches, native scalar paths still run; SIMD timing columns are omitted when `Plan9SimdForwardForLayer` is false.
