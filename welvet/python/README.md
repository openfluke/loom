# welvet — Loom Python Bindings

[![PyPI version](https://img.shields.io/pypi/v/welvet.svg)](https://pypi.org/project/welvet/)
[![PyPI downloads](https://img.shields.io/pypi/dm/welvet.svg)](https://pypi.org/project/welvet/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/pypi/pyversions/welvet.svg)](https://pypi.org/project/welvet/)

**M-POLY-VTD AI Engine (Loom v0.80.0)** — Python bindings: 21 numerical types, volumetric grids, CPU/GPU training, DNA/NEAT, JSON wire + native `.entity` checkpoints.

`welvet` wraps the [Loom](https://github.com/openfluke/loom) C-ABI with zero runtime Python dependencies. The PyPI wheel ships **prebuilt native libraries for every supported OS/arch**; at import time only the matching binary is loaded (`linux_amd64/welvet.so`, `windows_amd64/welvet.dll`, etc.).

> **Bedrock validation (v0.80):** seven-layer CPU suite (10 layer types × 21 dtypes × train × JSON serialize × `.entity` roundtrip). See [`docs/bedrock_validation.md`](https://github.com/openfluke/loom/blob/main/docs/bedrock_validation.md).

---

## Install

```bash
pip install welvet
```

```python
import welvet
print(welvet.__version__)  # 0.80.0
```

Supported platforms (64-bit): **Linux** (x86-64, ARM64), **macOS** (x86-64, ARM64, universal fallback), **Windows** (x86-64, ARM64), **Android** (ARM64, x86-64), **iOS** (device / simulator / XCFramework when built into the wheel).

### Build from source (monorepo)

PyPI wheels ship prebuilt `.so` / `.dylib` / `.dll`. To run **latest `main`** against your checkout:

**Option A — build + copy in one step**

```bash
cd welvet/cabi/internal/build
./build_unix.sh linux amd64    # native Linux x86_64 (+ --test for CABI smoke)
# or: ./build_unix.sh all       # every platform you have cross-toolchains for
```

`build_unix.sh` already copies `dist/*` → `welvet/python/src/welvet/`.

**Option B — you already compiled into `dist/` (or copied builds there)**

```bash
cd welvet/cabi/internal/build
./copy_to_python.sh              # default: ./dist → python/src/welvet/
# or if your tree is elsewhere:
./copy_to_python.sh ../../dist   # e.g. welvet/cabi/dist from build_macos.sh
```

One-liner from the Python folder:

```bash
cd welvet/python
./copy_from_cabi.sh              # copy + pip install -e .
```

**Then install / verify**

```bash
cd welvet/python
pip install -e .                 # editable install picks up src/welvet/*.so
python3 -m welvet.cabi_verify    # C-ABI symbols + smoke
python3 examples/run_all.py      # README examples
python3 benchmark_seven_layer.py --layer Dense
```

Artifacts land as `python/src/welvet/linux_amd64/welvet.so` (etc.). Without that, `import welvet` fails with “native library not found”.

**Publish to PyPI** (maintainers)

```bash
pip install build twine
cd welvet/cabi/internal/build && ./copy_to_python.sh   # all platforms → src/welvet/
cd ../../../python
pip install -e . && python3 examples/run_all.py        # smoke before release
./publish.sh                                           # python3 -m build + twine upload
```

The wheel is **multi-platform**: it contains every `*/welvet.{so,dylib,dll}` you copied; each machine only loads its own folder.

---

## Examples (runnable)

Scripts in [`examples/`](examples/) mirror the snippets below. Run one file or verify all:

```bash
cd welvet/python
pip install -e .
python3 examples/01_dense_forward.py
python3 examples/run_all.py          # runs 01–05
```

| Script | What it shows |
|--------|----------------|
| [`01_dense_forward.py`](examples/01_dense_forward.py) | Volumetric JSON → `forward_polymorphic` + `forward` |
| [`02_morph_and_train.py`](examples/02_morph_and_train.py) | `morph(INT8)`, CPU MC `train()` with shapes |
| [`03_save_reload.py`](examples/03_save_reload.py) | JSON wire + `.entity` checkpoint roundtrip |
| [`04_mha_forward.py`](examples/04_mha_forward.py) | MHA with `[batch, seq, d_model]` |
| [`05_dna_compare.py`](examples/05_dna_compare.py) | `dna()` + `compare_dna()` |

---

## Quick start

Layers live on a 3D grid `(z, y, x, l)` — see [`docs/overview.md`](https://github.com/openfluke/loom/blob/main/docs/overview.md).

### 1. Build and forward

```python
from welvet import Network

net = Network({
    "id": "demo",
    "depth": 1, "rows": 1, "cols": 1, "layers_per_cell": 2,
    "layers": [
        {"z": 0, "y": 0, "x": 0, "l": 0, "type": "dense",
         "dtype": "float32", "input_height": 16, "output_height": 8, "activation": "relu"},
        {"z": 0, "y": 0, "x": 0, "l": 1, "type": "dense",
         "dtype": "float32", "input_height": 8, "output_height": 4, "activation": "linear"},
    ],
})

inp = [0.1] * 16
out = net.forward_polymorphic(inp, [1, 16])   # preferred: explicit shape
print(out)                                    # length 4

net.free()
```

### 2. Morph precision (21 types)

```python
from welvet import DType, Network

# ... same net as above ...
net.morph(0, DType.INT8)          # layer index 0 only
out_q = net.forward_polymorphic(inp, [1, 16])
```

Use `net.morph_all(DType.INT8)` to morph every layer that has weights (skips Residual / Softmax).

### 3. Training (CPU, shape-aware)

```python
from welvet import Network, train

net = Network({...})  # single dense 16→8
inp, tgt = [0.1] * 16, [0.5] * 8
in_shape, out_shape = [1, 16], [1, 8]

net.set_training_mode(2)   # 1 = CPU SC, 2 = CPU MC (multicore on native C-ABI)
losses = train(
    net, [[inp]], [[tgt]],   # [[batch of input rows]], [[batch of target rows]]
    epochs=10, learning_rate=0.05, mode=2,
    input_shape=in_shape, target_shape=out_shape,
)
print(losses[-1])
net.free()
```

### 4. Save / reload (JSON wire + `.entity`)

**JSON wire** (human-readable checkpoint, same as v0.79):

```python
wire = net.serialize()
copy = Network.deserialize(wire)
# ... forward on copy, then copy.free()
```

**Native `.entity` blob** (compact binary checkpoint, v0.80+):

```python
blob = net.serialize_entity()
copy = Network.deserialize_entity(blob)
copy.sync_inference_weights()   # after training reload, before inference
# ... forward on copy, then copy.free()
```

`layer_persistence_from_entity(blob, layer_index)` inspects per-layer weight blobs in a checkpoint without loading a full network.

Full scripts: [`examples/03_save_reload.py`](examples/03_save_reload.py).

---

## Core Concepts

### Supported Layer Types

| Type | Description |
|------|-------------|
| `dense` | Fully connected / linear layer |
| `mha` | Multi-Head Attention (with RoPE, GQA/MQA, Causal Masking) |
| `swiglu` | SwiGLU gated MLP (LLaMA-style) |
| `rmsnorm` | Root Mean Square Normalization |
| `layernorm` | Layer Normalization |
| `cnn1` / `cnn2` / `cnn3` | 1D / 2D / 3D Convolution |
| `convtransposed1d` / `2d` / `3d` | Transposed Convolution |
| `rnn` / `lstm` | Recurrent layers |
| `embedding` | Token embedding lookup |
| `kmeans` | Differentiable K-Means clustering |
| `softmax` | 10 softmax variants (Standard, Gumbel, Masked, Entmax, ...) |
| `parallel` | MoE / ensemble branching |
| `sequential` | Nested sequential sub-graph |
| `residual` | Residual / skip connection |

### 21 Numerical Types

`float64`, `float32`, `float16`, `bfloat16`, `int64/32/16/8`, `uint64/32/16/8`, `fp8_e4m3`, `fp8_e5m2`, `int4`, `uint4`, `fp4_e2m1`, `int2`, `uint2`, `ternary`, `binary`

Morph a layer's precision at runtime (zero realloc when cached):

```python
from welvet import DType, morph_layer

morph_layer(net.handle, layer_index=0, target_dtype=DType.INT8)
# or: net.morph(0, DType.INT8)
```

---

## WebGPU Acceleration

```python
import welvet
from welvet import Network

net = Network({...})

# Upload weights to GPU
welvet.init_wgpu(net._handle)
welvet.sync_to_gpu(net._handle)

# GPU forward pass
output = welvet.forward_wgpu(net._handle, inputs)

net.free()
```

### Numerical tiling (SC vs MC)

v0.80+ uses specialized tiling profiles to maximize throughput:
- **SC (Single-Core)**: Optimized for Edge/WASM/Small NPUs.
- **MC (Multi-Core)**: Optimized for high-bandwidth L1/L2 caches (Ryzen, RTX, M4).

GPU backward training is live for Dense, RMSNorm, CNN 1D/2D/3D — **17x–65x** speedup over CPU on real workloads.

---

## DNA & Network Comparison

Extract a topological fingerprint and compare networks:

```python
from welvet import Network, compare_dna

dna_a = net_a.dna()
dna_b = net_b.dna()
result = compare_dna(dna_a, dna_b)
print(f"Overlap: {result['OverallOverlap']:.4f}")
print(f"Logic shifts: {len(result.get('LogicShifts', []))}")
```

---

## NEAT Evolution

Genetically evolve a population of networks:

```python
from welvet import (
    default_neat_config, neat_mutate,
    new_neat_population, neat_population_size,
    neat_population_get_network, neat_population_evolve,
    neat_population_best, neat_population_best_fitness,
    neat_population_summary, free_neat_population,
    build_network, free_network,
)

# Create a seed network
seed = Network({
    "id": "seed", "depth": 1, "rows": 1, "cols": 1, "layers_per_cell": 2,
    "layers": [
        {"z": 0, "y": 0, "x": 0, "l": 0, "type": "dense", 
         "input_height": 32, "output_height": 32},
        {"z": 0, "y": 0, "x": 0, "l": 1, "type": "dense", 
         "input_height": 32, "output_height": 1},
    ]
})

cfg = default_neat_config(32)
pop = seed.create_population(size=16, config=cfg)

for gen in range(5):
    fitnesses = [0.5 + 0.1 * i for i in range(pop.size())]
    pop.evolve(fitnesses)
    print(pop.summary(gen))

best = pop.best()
print(f"Best fitness: {pop.best_fitness():.6f}")
best.free()
pop.free()
seed.free()
```

---

## DNA Splice / Genetic Crossover

Combine two parent networks into a child:

```python
from welvet import default_splice_config, splice_dna, splice_dna_with_report

cfg = default_splice_config()
cfg["CrossoverMode"] = "blend"   # "blend" | "point" | "uniform"
cfg["FitnessA"] = 0.8
cfg["FitnessB"] = 0.5

child_handle = splice_dna(parent_a._handle, parent_b._handle, cfg)

# Or get a full diagnostic report
report = splice_dna_with_report(parent_a._handle, parent_b._handle, cfg)
print(f"Layers blended: {report['blended_count']}")
child_handle = report["child_handle"]
```

---

## Step mesh (online learning)

The volumetric 3D grid supports clock-cycle accurate propagation with spatial feedback loops:

```python
state = welvet.create_step_state(net._handle)
welvet.set_input(state, inputs)
welvet.mesh_step(net._handle, state)
output = welvet.get_output(state, layer_idx=-1)
welvet.free_step_state(state)
```

---

## Training

- **`train(net, batches, …)`** — poly `LoomTrain`, shape-aware (used in seven-layer suite). See [`examples/02_morph_and_train.py`](examples/02_morph_and_train.py).
- **`train_network(net, inputs, targets, …)`** — step-mesh clock-cycle path.

GPU backward dispatch and benchmarks: [`benchmark_training.py`](benchmark_training.py), [`benchmark_seven_layer.py`](benchmark_seven_layer.py).

---

## Tween (neural target propagation)

An alternative to backpropagation using localized Hebbian gap-based learning. We call this **tween** in APIs; papers often say *target propagation*.

```python
from welvet import (
    create_tween_state,
    get_default_tween_config,
    tween_forward,
    tween_backward,
)

_ = get_default_tween_config()
handle = create_tween_state(net.handle)
tween_forward(net.handle, handle, inputs)
tween_backward(net.handle, handle, targets)
net.free()
```

---

## LLM Inference

Load a SafeTensors model and run token generation:

```python
from welvet import Network, Tokenizer, sequential_forward

net = Network.from_file("path/to/model.safetensors")
tok = Tokenizer("path/to/tokenizer.json")

ids = tok.encode("Hello, world!")
output = net.forward([float(i) for i in ids])
tok.free()
net.free()
```

---

## Seven-layer validation (Python → CABI)

Same bedrock gate as Lucy and `@openfluke/welvet` (WASM). Logic in `seven_layer_spec.py`; engine work stays in the `.so`.

Each layer × dtype run: forward → train → **JSON** save/reload → **`.entity`** save/reload → `sync_inference_weights()` → native persistence check.

```bash
cd welvet/python
pip install -e .
python3 benchmark_seven_layer.py --layer Dense
python3 benchmark_seven_layer.py --layer Embedding
python3 benchmark_seven_layer.py --layer Residual
# full suite (slow): python3 benchmark_seven_layer.py
```

---

## Platform Support

| Platform | Architecture | Binary |
|----------|-------------|--------|
| Windows  | x86-64      | `welvet.dll` |
| Windows  | ARM64       | `welvet.dll` |
| Linux    | x86-64      | `welvet.so` |
| Linux    | ARM64       | `welvet.so` |
| Linux    | ARM (v7)    | `welvet.so` |
| macOS    | ARM64 (M-series) | `welvet.dylib` |
| macOS    | x86-64      | `welvet.dylib` |
| macOS    | Universal   | `welvet.dylib` |
| Android  | ARM64       | `welvet.so` |
| Android  | x86-64      | `welvet.so` |
| iOS      | ARM64 (device) | `welvet.dylib` |
| iOS      | Simulator (x86-64) | `welvet.dylib` |
| iOS      | Simulator (ARM64) | `welvet.dylib` |
| iOS      | XCFramework (all slices) | `.xcframework` |

At runtime, `import welvet` resolves `welvet/<platform>_<arch>/welvet.{so,dylib,dll}` for the current machine (see `src/welvet/utils.py`).

---

## Version alignment

| Component | Version |
|-----------|---------|
| **Loom engine (C-ABI / poly)** | **0.80.0** — Bedrock Validation + `.entity` |
| **PyPI `welvet`** | **0.80.0** |
| **npm `@openfluke/welvet`** | **0.80.0** |

---

## Links

- **GitHub**: [github.com/openfluke/loom](https://github.com/openfluke/loom)
- **Python package**: [welvet/python](https://github.com/openfluke/loom/tree/main/welvet/python)
- **Engine docs**: [poly/README.md](https://github.com/openfluke/loom/blob/main/poly/README.md) · [docs index](https://github.com/openfluke/loom/blob/main/docs/index.md)
- **TypeScript / WASM**: [@openfluke/welvet](https://www.npmjs.com/package/@openfluke/welvet)
- **Issues**: [github.com/openfluke/loom/issues](https://github.com/openfluke/loom/issues)

---

## License

Apache 2.0 — see [LICENSE](https://github.com/openfluke/loom/blob/main/LICENSE).

*Loom: Universal precision. Volumetric freedom. Bedrock performance.*
