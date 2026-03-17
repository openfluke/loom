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

---

## Quick Start

```python
import welvet
from welvet import Network, LayerType, DType

# Build a 3-layer dense MLP
net = Network({
    "id": "my_net",
    "depth": 1, "rows": 1, "cols": 1, "layers_per_cell": 3,
    "layers": [
        {"type": "dense", "input_height": 128, "output_height": 256,
         "activation": "relu", "dtype": "float32"},
        {"type": "dense", "input_height": 256, "output_height": 256,
         "activation": "relu", "dtype": "float32"},
        {"type": "dense", "input_height": 256, "output_height": 10,
         "activation": "sigmoid", "dtype": "float32"},
    ]
})

# Forward pass
output = net.forward([0.5] * 128)
print(output[:5])

net.free()
```

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

Morph a layer's precision at runtime with no reallocation:

```python
welvet.morph_layer(net._handle, layer_index=0, new_dtype="int8")
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

seed = build_network({
    "id": "seed", "depth": 1, "rows": 1, "cols": 1, "layers_per_cell": 2,
    "layers": [
        {"type": "dense", "input_height": 32, "output_height": 32, "dtype": "float32"},
        {"type": "dense", "input_height": 32, "output_height": 1,  "dtype": "float32"},
    ]
})

cfg = default_neat_config(32)
pop = new_neat_population(seed, size=16, config=cfg)

for gen in range(20):
    n = neat_population_size(pop)
    fitnesses = []
    for i in range(n):
        h = neat_population_get_network(pop, i)
        fitnesses.append(my_fitness_fn(h))
        free_network(h)
    neat_population_evolve(pop, fitnesses)
    print(neat_population_summary(pop, gen))

best = neat_population_best(pop)
print(f"Best fitness: {neat_population_best_fitness(pop):.6f}")
free_network(best)
free_neat_population(pop)
free_network(seed)
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

## Systolic Grid (Online Learning)

The volumetric 3D grid supports clock-cycle accurate propagation with spatial feedback loops:

```python
state = welvet.create_systolic_state(net._handle)
welvet.set_input(state, inputs)
welvet.systolic_step(state)
output = welvet.get_output(state, layer_idx=-1)
welvet.free_systolic_state(state)
```

---

## Training

```python
from welvet import train_network

# High-level training helper
train_network(net._handle, inputs, targets, epochs=100, learning_rate=0.001)
```

Or use full GPU backward dispatch for maximum performance — see the [benchmark scripts](https://github.com/openfluke/loom/tree/main/welvet/python).

---

## Target Propagation

An alternative to backpropagation using localized Hebbian gap-based learning:

```python
from welvet import (
    create_target_prop_state, get_default_target_prop_config,
    target_prop_forward, target_prop_backward,
)

cfg = get_default_target_prop_config()
state = create_target_prop_state(net._handle, cfg)
target_prop_forward(state, inputs)
target_prop_backward(state, targets)
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
