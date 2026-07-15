# Why `poly/` looks the way it does

This note explains the **layout** of `loom/poly` — not layer math (see [layers.md](layers.md)) or dispatch (see [dispatch.md](dispatch.md)). If the tree feels “one giant folder,” that is mostly intentional.

---

## Short version

| Question | Answer |
|:---------|:-------|
| Why ~180 `.go` files in one package? | Layers, weights, GPU, SIMD, and ENTITY all share the same types. One package avoids import cycles. |
| Why a fat `VolumetricLayer`? | Every cell in the 3D grid is the same kind of object; `Type` picks behavior. Same pattern as large systems structs (`task_struct`, `sk_buff`, …). |
| Why not `poly/dense/`, `poly/mha/`, …? | In Go, **a folder is a package**. Moving Dense into `dense/` without a shared `core` package creates `poly` ↔ `dense` cycles. |
| How do you find Dense / MHA / …? | **File prefixes**: `dense_*.go`, `mha_*.go`, `swiglu_*.go`, `wgpu_*.go`, … |
| Where are real subpackages? | Leaves that do **not** need `VolumetricLayer`: `simd/`, `accel/`, and external tests in `tests/`. |

Public code (Lucy, Welvet, Planet Bridging, …) imports **`github.com/openfluke/loom/poly`**. That import path is the stable API.

---

## Mental model

```
                    ┌─────────────────────────────┐
                    │   package poly  (facade)    │
                    │   VolumetricNetwork / Layer │
                    │   WeightStore, Dispatch…    │
                    └──────────────┬──────────────┘
           ┌───────────────────────┼───────────────────────┐
           ▼                       ▼                       ▼
    dense_*.go              mha_*.go / swiglu_*      wgpu_*.go / entity_*
    cnn_* / rnn_*           transformer / hf_*       training / seed_*
    (same package)          (same package)           (same package)
           │                       │                       │
           └───────────────────────┴───────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
              poly/simd                     poly/accel
           (tile / Q4 / BitNet            (vendor plugins;
            Plan 9 kernels)                no god-layer types)
```

Dense, MHA, SwiGLU, CNNs, etc. stay in **`package poly`** so they can take `*VolumetricLayer` and call `WeightStore` / tiling helpers without crossing a package boundary.

---

## `VolumetricLayer` (the “god” struct)

`VolumetricNetwork` is a **Depth × Rows × Cols × LayersPerCell** grid. Each cell is a `VolumetricLayer`.

That struct holds:

- identity (`Type`, `DType`, coordinates, `WeightStore`)
- geometry / CNN / MHA / embedding / softmax knobs
- nesting (`ParallelBranches`, `SequentialLayers`, remote links)
- exec (`UseGPU`, tile maps, accel binding, KV cache, metacognition)

Only some fields matter for a given `LayerType` — the rest are unused, same as optional fields on a kernel inode or socket buffer.

**Why not a `DenseLayer` type + interface?** Morphing dtype, Parallel/Sequential nesting, one grid allocator, and uniform GPU/`SyncTo*` paths would duplicate into every concrete type. The discriminator pattern (`switch layer.Type`) in `DispatchLayer` already is the polymorphism.

**Optional cleanup (not required):** group fields into embedded structs for readability (`Identity`, `Geom`, `Attention`, `Exec`, …) without changing the single-layer model.

---

## File naming = navigation

Within `package poly`, names cluster by concern:

| Prefix | Role |
|:-------|:-----|
| `poly.go` | Core enums/types, `Tensor`, network/layer structs, GPU sync entry points |
| `weights.go` / `weight_*.go` | `WeightStore`, morph / pack / master lifecycle |
| `forward.go` / `backward.go` | Central dispatch |
| `dense_*.go` | Dense forward/backward, native exact, SIMD |
| `mha_*.go` / `swiglu_*.go` | Attention / MLP blocks |
| `cnn*_*.go` / `rnn_*.go` / `lstm_*.go` / `embedding_*.go` | Other layer families |
| `q4_*.go` / `lm_head_q4.go` / `quantization.go` / `bitnet_*.go` | Packed / ternary CPU·GPU paths |
| `entity*.go` / `hf_*.go` | `.entity` + HuggingFace bridge |
| `transformer_*.go` / `sampling.go` / `bpe.go` | Autoregressive generation |
| `seed_*.go` | Seed manifests / He-init (no weight blobs) |
| `wgpu_*.go` | WebGPU context, shaders, tiled kernels |
| `donate_compute_*.go` | LAN donate-compute protocol |
| `hardware_*.go` | Host audit (OS/CPU/RAM/GPU probes) |
| `memory_*.go` / `process_memory_*.go` | Footprint / RSS / charts |

A prefix search in the IDE (`dense_`, `wgpu_`) is the practical substitute for a `dense/` folder.

---

## Real subdirectories (separate packages)

Go only allows a new package when dependencies allow it.

| Path | Package | Why it can live alone |
|:-----|:--------|:----------------------|
| `poly/simd/` | `simd` | Pure kernels on slices (`DotTile`, Q4 fused dots, BitNet asm). No `VolumetricLayer`. |
| `poly/accel/` | `accel` | Vendor plugin registry / C ABI loader. Layer wires in via root `accel_*.go`. |
| `poly/tests/` | `poly_test` | Black-box tests importing `poly` (see below). |
| `poly/cmd/` | (tools) | Optional benches / helpers. |

**What does *not* (yet) live in a folder:** Dense, MHA, SwiGLU, CNN, WebGPU body, ENTITY, training. Those need the shared god types. A future `poly/core` (types + `WeightStore` only) could unlock `poly/dense`, but that is a large extract — not a rename.

---

## Tests

Layer/runtime tests live under **`poly/tests/`** as `package poly_test` (import `github.com/openfluke/loom/poly`).

That keeps the root package free of `*_test.go` and matches the public API surface Lucy and Welvet use. White-box tests that need unexported helpers either:

- go through exported helpers (e.g. `GemvQ4_0Packed`), or
- stay impossible across the package boundary (by design).

SIMD unit tests may also live next to kernels under `poly/simd/`.

---

## What “cleaning up” means here

Useful:

- Keep **prefixes** consistent when adding files
- Prefer **`poly/tests`** for new tests
- Extract **leaf** packages only when they do not take `*VolumetricLayer` (kernels, protocols, host probes)
- Document behavior in `loom/docs/*.md` (this tree)

Usually *not* useful as a first move:

- Creating empty `dense/` / `mha/` folders “for later”
- Splitting every layer into its own package without `core`
- Rewriting the god struct into many interface types just to look OOP

Complexity here is **local density** (many dtypes × CPU/GPU/SIMD × layer kinds), not Linux-scale breadth. The flat package is the Go-friendly way to keep a volumetric polymorphic runtime coherent.

---

## Related docs

- [overview.md](overview.md) — architecture picture
- [layers.md](layers.md) — what each `LayerType` does
- [dispatch.md](dispatch.md) — how `DispatchLayer` routes
- [simd.md](simd.md) / [gpu.md](gpu.md) / [entity.md](entity.md) — major backends and checkpoints
- [testing_and_validation.md](testing_and_validation.md) — how Lucy suites touch `poly/`
