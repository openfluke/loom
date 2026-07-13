# Cross-path CPU suite (Lucy menu [15])

**Run:** `cd lucy && go run .` → **[15]** (or **[0]** for all layer types).  
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

**Grid:** 1³ · **7 layers/cell** · **21 dtypes** · **30 train epochs**

**Layer types:** Dense, SwiGLU, MHA, CNN1, CNN2, CNN3, RNN, LSTM, Embedding, Residual

---

## Per-dtype output

1. **Raw timing table** — wall times per path (SC/MC/SIMD/Nat/NatS + train SC/MC/SIMD/Nat)
2. **Comparison table** — explicit same-paradigm pairs only:
   - **QAT** SC→SIMD and MC→SIMD (fwd, bwd, train)
   - **Native** Nat-f→NatS-f and Nat-b→NatS-b
   - **best fwd/bwd/train** — fastest QAT path vs fastest Native path (who wins, ×, % faster)
2. **Parity table** — tiled SC↔MC, SC↔SIMD (gated); native↔native-SIMD and SC↔native (informational)
3. **Train table** — loss after SC, MC, SIMD, native train
4. **Test tally** — gated checks per category + session manifest

### Gated tests (per dtype, SIMD layers)

| Category | Count |
|----------|-------|
| tiled fwd/bwd finite (SC, MC, SIMD) | 6 |
| tiled parity (SC↔MC, SC↔SIMD fwd/bwd) | 4 |
| native path + fwd/bwd + native-SIMD finite | 5 |
| train SC, MC, SIMD, native | 4 |
| **Total** | **19 × 21 = 399** per SIMD layer |

Non-SIMD layers omit SIMD columns (fewer checks).

Native↔native-SIMD parity is **reported but not gated** — MAC dtypes can legitimately differ from tiled SIMD tolerance bands.

---

## Session manifest

After **[0]** or a single layer, the log ends with:

```text
╔══════════════════════════════════════════════════════════════════════╗
║  [15] Cross-path global manifest                                      ║
╚══════════════════════════════════════════════════════════════════════╝
  Dense         dtypes  21/ 21  tests   399/  399  PASS
  ...
  Session dtypes: N passed · M failed
  Session tests:  X passed · Y failed (of Z checks)
```

---

## Related docs

| Topic | Doc |
|-------|-----|
| Tiled SC/MC/SIMD + save/reload | [bedrock_validation.md](bedrock_validation.md) (menu [7]) |
| Native exact only | [native_layers.md](native_layers.md) (menu [14]) |
| Training paradigms | [training.md](training.md#training-paradigms-default-qat-like-vs-native-exact) |
