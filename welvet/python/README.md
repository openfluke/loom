# welvet — Loom Python Bindings

[![PyPI version](https://img.shields.io/pypi/v/welvet.svg)](https://pypi.org/project/welvet/)
[![PyPI downloads](https://img.shields.io/pypi/dm/welvet.svg)](https://pypi.org/project/welvet/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/pypi/pyversions/welvet.svg)](https://pypi.org/project/welvet/)

**Python bindings for the M-POLY-VTD neural engine — 21 numerical types, WebGPU acceleration, NEAT evolution, and 100% determinism.**

`welvet` wraps the [Loom](https://github.com/openfluke/loom) C-ABI with zero Python dependencies. It ships precompiled native libraries for all major platforms.

---

## Install

```bash
pip install welvet
```

Supported platforms: **Windows** (x86-64, ARM64), **Linux** (x86-64, ARM64, ARM, x86), **macOS** (x86-64, ARM64, Universal), **Android** (ARM64, ARM).

### Build from source (monorepo)

PyPI wheels ship prebuilt `.so` / `.dylib` / `.dll`. To run **latest `main`** against your checkout:

```bash
# From repo root — builds C-ABI and copies into welvet/python/src/welvet/
cd welvet/cabi/internal/build
./build_unix.sh linux amd64    # or: darwin arm64, windows amd64, etc.

cd ../../../python
pip install -e .
python3 -m welvet.cabi_verify   # 328/328 C-ABI symbols + smoke
python3 consumer_smoke.py       # forward · morph · train · serialize
python3 examples/run_all.py   # README examples (5 scripts)
python3 benchmark_seven_layer.py --layer Dense
```

`build_unix.sh` mirrors `dist/*` → `python/src/welvet/linux_amd64/welvet.so` (and headers). Without that step, `import welvet` fails with “native library not found”.

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
| [`03_save_reload.py`](examples/03_save_reload.py) | `serialize()` / `Network.deserialize()` |
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

### 4. Save / reload

```python
wire = net.serialize()
copy = Network.deserialize(wire)
# ... forward on copy, then copy.free()
```

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

v0.79+ uses specialized tiling profiles to maximize throughput:
- **SC (Single-Core)**: Optimized for Edge/WASM/Small NPUs.
- **MC (Multi-Core)**: Optimized for high-bandwidth L1/L2 caches (Ryzen, RTX, M4).

GPU backward training is live for Dense, RMSNorm, CNN 1D/2D/3D — **17x–65x** speedup over CPU on real workloads.

---

## DNA & Network Comparison

Extract a topological fingerprint and compare networks:

```python
from welvet import extract_dna, compare_dna

dna_a = extract_dna(net_a._handle)
dna_b = extract_dna(net_b._handle)

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
from welvet import load_network, load_tokenizer, tokenize, detokenize, sequential_forward

net = load_network("path/to/model.safetensors")
tok = load_tokenizer("path/to/tokenizer.json")

ids = tokenize(tok, "Hello, world!")
output = sequential_forward(net, [float(i) for i in ids])
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

---

## Links

- **GitHub**: [github.com/openfluke/loom](https://github.com/openfluke/loom)
- **Engine docs**: [poly/README.md](https://github.com/openfluke/loom/blob/main/poly/README.md)
- **TypeScript bindings**: [@openfluke/welvet](https://www.npmjs.com/package/@openfluke/welvet)
- **Issues**: [github.com/openfluke/loom/issues](https://github.com/openfluke/loom/issues)

---

## License

Apache 2.0 — see [LICENSE](https://github.com/openfluke/loom/blob/main/LICENSE).

*Loom: Universal precision. Volumetric freedom. Bedrock performance.*
