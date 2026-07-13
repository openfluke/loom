# Lucy Bloom Rivers (integration harness)

**Repo:** [github.com/openfluke/lucy_bloom_rivers](https://github.com/openfluke/lucy_bloom_rivers)  
**Role:** Loom’s interactive test shell — layer suites, HF chat, NPU bridges, archived benchmark logs.  
**Engine:** depends on [`github.com/openfluke/loom`](https://github.com/openfluke/loom) (`poly/`).

Lucy lived under `loom/lucy/` through v0.83; from **v0.84** it is a **separate repository**. Loom docs describe *what the suites mean*; Lucy owns *how to run them*.

---

## Quick start

Clone **next to** `loom` (recommended):

```bash
git clone https://github.com/openfluke/loom.git
git clone https://github.com/openfluke/lucy_bloom_rivers.git
cd lucy_bloom_rivers
# go.mod: replace github.com/openfluke/loom => ../loom
go run .
```

**Local monorepo layout** (nested inside `loom/`, gitignored by Loom): `loom/lucy_bloom_rivers/` with `replace github.com/openfluke/loom => ../` — same commands from that directory.

Requires **Go 1.26.2+**. GPU: see Lucy `README.md` for `VK_ICD_FILENAMES` / `WGPU_ADAPTER_NAME`.

---

## Where logs land

Runtime transcripts (reset each session):

| Log | Menu | Doc |
|-----|------|-----|
| `lucy_testing_output/seven_layer.txt` | **[7]** | [bedrock_validation.md](bedrock_validation.md) |
| `lucy_testing_output/native_layers.txt` | **[14]** | [native_layers.md](native_layers.md) |
| `lucy_testing_output/cross_path_layers.txt` | **[15]** | [cross_path_layers.md](cross_path_layers.md) |
| `lucy_testing_output/tween_native_layers.txt` | **[16]** | — |
| `lucy_testing_output/adaptation_suite.txt` | **[17]** | — |
| `lucy_testing_output/nine_layer.txt` | **[9]** | [accelerators.md](accelerators.md) |
| `lucy_testing_output/snapdragon.txt` | **[12]** | [snapdragon_npu.md](snapdragon_npu.md) |
| `lucy_testing_output/apple.txt` | **[13]** | [apple_metal.md](apple_metal.md) |
| `lucy_testing_output/log.txt` | **[3]** layer matrices | [testing_and_validation.md](testing_and_validation.md) |

Paths are relative to the **Lucy repo root** (`lucy_bloom_rivers/`). Per-dtype checkpoints: `lucy_testing_output/tag_DType.json` and `tag_DType.entity`.

**Archived cross-path SIMD duel logs** (off-machine): `~/Documents/loom/simd/cross_path_layers_amd.txt`, `cross_path_layers_arm.txt` — see [cross_path_layers.md](cross_path_layers.md#archived-simd-duel-results-jul-2026).

---

## Harness code map

| Suite | Path in Lucy repo |
|-------|-------------------|
| Seven-layer CPU | `examples/seven_layer/` |
| Native exact [14] | `examples/seven_layer/native_menu.go` |
| Cross-path [15] | `examples/seven_layer/cross_path_menu.go` |
| Tween native [16] | `examples/seven_layer/tween_native_menu.go` |
| Adaptation [17] | `examples/adaptation_suite/`, `examples/seven_layer/adaptation_menu.go` |
| Intel NPU [9] | `examples/nine_layer/` |
| Snapdragon [12] | `examples/snapdragon/` |
| Apple Metal [13] | `examples/apple/` |
| ENTITY Talk [8] | `hf_entity.go` |
| Poly Talk [1] | `poly_talk_session.go`, `lucy.go` |

Go module path (current): `github.com/openfluke/loom/lucy` — imports use that prefix until a module rename.

---

## Related Loom docs

| Topic | Doc |
|-------|-----|
| Parity tables & log legend | [testing_and_validation.md](testing_and_validation.md) |
| SC/MC/SIMD bedrock | [bedrock_validation.md](bedrock_validation.md) |
| Plan 9 SIMD | [simd.md](simd.md) |
| Training paradigms | [training.md](training.md#training-paradigms-default-qat-like-vs-native-exact) |
| ENTITY checkpoints | [entity.md](entity.md) |
