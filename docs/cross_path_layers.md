# Cross-path CPU suite (Lucy menu [15])

**Run:** `cd lucy && go run .` → **[15]** → pick grid (default **2³**) → layer type (or **[0]** for all).  
**Code:** `lucy/examples/seven_layer/cross_path_menu.go`  
**Log:** `lucy/lucy_testing_output/cross_path_layers.txt`

Unifies **[7]** (tiled SC/MC/SIMD) and **[14]** (native exact + native SIMD) in one side-by-side matrix per layer × dtype.

---

## What it compares

| Path | Config | Math |
|------|--------|------|
| **SC** | `EnableMultiCoreTiling=false`, SIMD off | Tiled FP32-dequant (`GetActive`) |
| **MC** | `EnableMultiCoreTiling=true`, SIMD off | Tiled FP32-dequant, parallel tiles |
| **SIMD** | MC + `SetSimdForwardRecursive(true)` | Tiled + Plan 9 `DotTile` / saxpy |
| **Native** | `UseExactDType=true`, SIMD off | `*_native.go` storage-dtype MAC |
| **Native SIMD** | `UseExactDType=true`, SIMD on | `*_native_simd.go` |

**Grid:** selectable **1³ / 2³ / 3³** (default **2³**) · **[5] 3³ SIMD duel** (QAT-SIMD vs Nat-SIMD only) · **7 layers/cell** · **21 dtypes** · train epochs scale by grid (50 / 12 / 6)

**Layer types:** Dense, SwiGLU, MHA, CNN1, CNN2, CNN3, RNN, LSTM, Embedding, Residual

---

## Per-dtype output

1. **Raw timing — forward / backward** — SC/MC/SIMD/Nat/NatS wall times
2. **Comparison — forward / backward** — QAT SC→SIMD, Nat→NatS, best fwd/bwd (QAT vs Nat)
3. **Raw timing — training** — QAT-SC, QAT-MC, QAT-SIMD, Nat, Nat-SIMD (30 epochs)
4. **Train comparisons** — QAT SC/MC→SIMD, Nat→NatS, QAT SIMD vs Nat, QAT SIMD vs NatS, best train
5. **Parity table** — tiled SC↔MC, SC↔SIMD (gated); native↔native-SIMD and SC↔native (informational)
6. **Train loss table** — final loss per path + PASS/FAIL gates
7. **Test tally** — gated checks per category + session manifest

### Gated tests (per dtype, SIMD layers)

| Category | Count |
|----------|-------|
| tiled fwd/bwd finite (SC, MC, SIMD) | 6 |
| tiled parity (SC↔MC, SC↔SIMD fwd/bwd) | 4 |
| native path + fwd/bwd + native-SIMD finite | 5 |
| train SC, MC, SIMD, native, native-SIMD | 5 |
| **Total** | **20 × 21 = 420** per SIMD layer |

Non-SIMD layers omit SIMD columns (fewer checks).

Native↔native-SIMD parity is **reported but not gated** — MAC dtypes can legitimately differ from tiled SIMD tolerance bands.

---

## SIMD duel mode (grid **[5]**)

**3³ only** · **189-layer stack** · **6 train epochs** · compares **only**:

| Path | What runs |
|------|-----------|
| **QAT-SIMD** | Tiled `GetActive` FP32 + Plan 9 SIMD |
| **Nat-SIMD** | `UseExactDType` + `*_native_simd.go` |

Skips SC, MC, native scalar, and parity-vs-SC tables. Per dtype the log prints:

1. **One-line summary** — PASS/FAIL, 7/7 checks, fwd/bwd/train winner + speedup
2. **Raw timing (fwd / bwd)** — `QAT SIMD-f`, `NatS-f`, `QAT SIMD-b`, `NatS-b` wall times
3. **QAT-SIMD vs Nat-SIMD** — pairwise comparison and per-phase winner per dtype
4. **Raw timing (train)** — 6-epoch wall time per path
5. **Train comparisons** — QAT-SIMD vs Nat-SIMD train speedup
6. **Dtype spread** — slowest → fastest dtype per phase (among winning SIMD path per dtype)
7. **Train loss table** — `Loss₀` and final loss per path + PASS/FAIL gates
8. **Test tally** — 7 gated checks × 21 dtypes = **147** per layer

| Gated check (per dtype) | What it verifies |
|-------------------------|------------------|
| `native.path` | Native-exact routing available |
| `tiled.fwd.simd` | QAT-SIMD forward finite |
| `tiled.bwd.simd` | QAT-SIMD backward finite |
| `native.fwd.simd` | Nat-SIMD forward finite |
| `native.bwd.simd` | Nat-SIMD backward finite |
| `train.simd` | QAT-SIMD 6-epoch train OK |
| `train.native.simd` | Nat-SIMD 6-epoch train OK |

Use this when you want **apples-to-apples fastest SIMD** at the largest practical grid without noise from non-SIMD paths.

---

## Archived SIMD duel results (Jul 2026)

Full **[15] → grid [5] → [0] all layers** runs, captured off-machine:

| Platform | Archive path |
|----------|----------------|
| **amd64** (AVX2) | `~/Documents/loom/simd/cross_path_layers_amd.txt` |
| **arm64** (NEON) | `~/Documents/loom/simd/cross_path_layers_arm.txt` |

Runtime log during a session: `lucy/lucy_testing_output/cross_path_layers.txt` (reset each run).

### Pass summary (21 dtypes × layer)

| Layer | amd64 | arm64 | Notes |
|-------|-------|-------|-------|
| Dense | **21/21** | **20/21** | arm64 **Float64** QAT-SIMD train: loss explodes (~1×10²⁷) |
| SwiGLU | **21/21** | **21/21** | |
| MHA | **21/21** | **21/21** | |
| CNN1 | **21/21** | **21/21** | |
| CNN2 | **21/21** | **21/21** | |
| CNN3 | **21/21** | **20/21** | arm64 **BFloat16** QAT-SIMD train: final loss **0** (degenerate) |
| RNN | **20/21** | **20/21** | **Int8** Nat-SIMD train diverges on both (loss ~2.5 vs ~0.33) |
| LSTM | **21/21** | **21/21** | |
| Embedding | **21/21** | **21/21** | |
| Residual | **21/21** | **21/21** | |
| **Total dtype-rows** | **208/210** | **206/210** | **1469/1470** and **1467/1470** gated checks |

Failures are **train-criteria** only on the listed rows; forward/backward finiteness still passes. RNN Int8 is a known low-bit + BPTT flake zone at 6 epochs on 3³.

### What the duel answers

The duel isolates one design question: **for a given layer and dtype, which SIMD stack is faster — QAT-like (`GetActive` FP32 dequant + `DotTile`) or native-exact (`UseExactDType` + `*_native_simd.go`)?**

- **QAT-SIMD** still pays dequant/materialization cost on MAC dtypes; SIMD helps float paths most.
- **Nat-SIMD** avoids FP32 staging on integer/FP8 paths; often wins forward on MAC dtypes even when both use `DotTile`-class kernels.

Training at 6 epochs is mainly a **sanity gate** (finite loss, harness `trainingOK`); fwd/bwd tables carry the performance signal.

### Performance themes (Dense @ 3³, Float32)

| Metric | amd64 | arm64 |
|--------|-------|-------|
| QAT-SIMD fwd | 492 µs | **206 µs** |
| Nat-SIMD fwd | 557 µs | 309 µs |
| **Fwd winner** | QAT ~**1.1×** | QAT ~**1.5×** |
| Train (6 ep) | ~parity (~1.0×) | ~parity (~1.0×) |

ARM absolute fwd times are ~2× faster than AMD on this Dense stack; relative QAT-vs-Nat winner pattern is similar.

### Performance themes (MAC dtypes — Dense forward)

| Dtype | amd64 fwd winner | arm64 fwd winner |
|-------|------------------|------------------|
| FP8-E4M3 | Nat **2.6×** | QAT **1.5×** (Nat slower on this run) |
| FP8-E5M2 | Nat **3.0×** | Nat **2.8×** |
| Int64 | Nat **1.6×** | Nat **3.7×** |
| Uint8 | Nat **1.8×** | Nat **1.3×** |
| Uint16 | Nat **1.5×** | Nat **9.7×** |

On **amd64**, Nat-SIMD wins most MAC dtype forwards (QAT-SIMD still dequants through `GetActive`). On **arm64**, Float32/BFloat16/FP8-E4M3 forwards can still favor QAT-SIMD; integer paths strongly favor Nat-SIMD (Uint16 up to **9.7×**).

### Dtype spread tables

Each layer ends with a **dtype spread** block: among the *faster* SIMD path per dtype, which dtype is slowest vs fastest for forward, backward, and train.

Example (Dense, amd64):

```text
│ forward    │ Ternary NatS-f 811.6µs→Int64 NatS-f 439.8µs 1.8×  46%
│ backward   │ Uint8 QAT SIMD-b 3.83ms→Int64 NatS-b 2.64ms 1.4×  31%
│ train      │ FP4 NatS 68.6ms→Uint4 QAT SIMD 26.1ms      2.6×  62%
```

The **×** column is slow÷fast; **gap** is approximate percent spread. Use this to see whether perf is dtype-limited (e.g. FP4 train slowest) vs path-limited (same dtype, different QAT/Nat winner in the per-dtype rows above).

On arm64, Dense forward spread is much wider (**8.2×**, Float16 slowest → Int64 fastest) because some Nat-SIMD forwards time as **0** in the log (sub-timer resolution — treat as “very fast”, not a hard zero).

### Reading a one-line summary

```text
· Float32    PASS  7/7  fwd QAT SIMD-f 1.1×  bwd QAT SIMD-b 1.0×  train QAT SIMD 1.0×
```

| Field | Meaning |
|-------|---------|
| `7/7` | All gated checks passed for this dtype |
| `fwd QAT SIMD-f 1.1×` | QAT-SIMD forward beat Nat-SIMD by 1.1× |
| `bwd Nat NatS-b 1.2×` | Nat-SIMD backward won |
| `train QAT SIMD 1.0×` | Training wall-time parity (winner still named) |

`NatS-f` / `NatS-b` = native-exact SIMD; `SIMD-f` / `SIMD-b` = QAT tiled SIMD.

### Train loss table

```text
│ DType      │    Loss₀ QAT-SIMD Nat-SIMD │ QAT    NatS
│ Float64    │   0.3223   0.3223   0.3223 │ PASS   PASS
```

`Loss₀` is shared initial loss; columns are final loss after 6 epochs. **QAT** / **NatS** columns are independent PASS/FAIL — one path can fail while the other passes (arm64 Dense Float64: QAT FAIL, NatS PASS).

---

## Session manifest

After **[0]** or a single layer, the log ends with:

```text
╔══════════════════════════════════════════════════════════════════════╗
║  [15] Cross-path global manifest                                      ║
╚══════════════════════════════════════════════════════════════════════╝
  Dense         dtypes  21/ 21  tests   420/  420  PASS
  ...
  Session dtypes: N passed · M failed
  Session tests:  X passed · Y failed (of Z checks)
```

---

## Reproduce archived run

```bash
cd lucy && go run .
# [15] → grid [5] 3³ SIMD duel → layer [0] all types
# Copy lucy_testing_output/cross_path_layers.txt to ~/Documents/loom/simd/ for archiving
```

Requires `GOARCH=amd64` or `arm64` with Plan 9 SIMD linked. Full all-layer 3³ duel is ~10–20 minutes per platform.

---

## Related docs

| Topic | Doc |
|-------|-----|
| Tiled SC/MC/SIMD + save/reload | [bedrock_validation.md](bedrock_validation.md) (menu [7]) |
| Native exact only | [native_layers.md](native_layers.md) (menu [14]) |
| Log index + archive layout | [testing_and_validation.md](testing_and_validation.md) |
| Plan 9 SIMD kernels | [simd.md](simd.md) |
| Training paradigms | [training.md](training.md#training-paradigms-default-qat-like-vs-native-exact) |
