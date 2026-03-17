# Loom / poly Documentation Index

This directory contains comprehensive documentation for the `poly/` package — the **M-POLY-VTD** (Multi-numerical POLYmorphic Volumetric Tiled-tensor Dispatcher) engine that powers the Loom neural framework.

---

## Documents

| File | Description |
|:-----|:------------|
| [overview.md](overview.md) | Big-picture architecture: the 3D grid, six design pillars, key types |
| [deployment.md](deployment.md) | **Polyglot Ecosystem**: NPM deployment, TypeScript SDK, WASM bridge, and Browser/Node usage |
| [numerical_types.md](numerical_types.md) | All 21 DTypes, the `Numeric` generic constraint, `WeightStore` lifecycle, `SimulatePrecision`, Q4_0, and compression ratios |
| [layers.md](layers.md) | Every layer type (Dense, CNN, RNN, MHA, SwiGLU, RMSNorm, Residual, Softmax, Parallel, Sequential, and more) with ASCII data-flow diagrams |
| [dispatch.md](dispatch.md) | `DispatchLayer` routing, the 3D grid traversal, tiled parallel execution, `IsRemoteLink` spatial hopping, and the GPU dispatch path |
| [training.md](training.md) | CPU and GPU training pipelines, loss functions, gradient flow, `TargetProp` (chain-rule and gap-based modes), link budgets |
| [gpu.md](gpu.md) | `WGPUContext`, `InitWGPU`, `BeginFrame`/`FlushFrame`, buffer management, bind group cache, GPU support matrix, WGSL shader overview |
| [systolic.md](systolic.md) | The systolic grid engine: `SystolicState`, one-clock-cycle forward, spatial feedback via remote links, BPTT, online learning |
| [dna.md](dna.md) | Topological network fingerprinting: `ExtractDNA`, `CosineSimilarity`, `CompareNetworks`, `LogicShift` detection, recursive extraction for all 19 layer types |
| [evolution.md](evolution.md) | DNA Splice / Genetic Crossover and NEAT-style Topology Evolution: `SpliceDNA`, `NEATMutate`, `NEATPopulation`, all 3 crossover modes, all 6 mutation types |
| [softmax.md](softmax.md) | All 10 softmax variants: Standard, Temperature, Gumbel, Masked, Sparse, Entmax, Grid, Hierarchical, Adaptive, Mixture |
| [serialization.md](serialization.md) | Full save/load (`SerializeNetwork`/`DeserializeNetwork`), bit-packing formats, idempotency guarantee, SafeTensors support |
| [parallel_sequential.md](parallel_sequential.md) | `LayerParallel` (5 combine modes, activation tree), `LayerSequential` (step containers, skip gradients), nesting patterns |
| [quantization.md](quantization.md) | PTQ pipeline, `WeightStore` versioning, `Morph`/`Unpack`, `Q4_0Block` block quantization, calibration, accuracy trade-offs |
| [transformer.md](transformer.md) | MHA with RoPE, GQA/MQA, KV cache, SwiGLU, RMSNorm, full transformer block assembly, `Transformer[T]` generation type |
| [quick_reference.md](quick_reference.md) | Concise copy-paste snippets for all common operations |

---

## Where to Start

**New to the codebase?** Read [overview.md](overview.md) first for the architecture picture, then [layers.md](layers.md) to see what layer types are available.
**Deploying to Web or JS?** Read [deployment.md](deployment.md).

**Want to train a model?** Read [training.md](training.md) and [dispatch.md](dispatch.md).

**Using the GPU?** Read [gpu.md](gpu.md).

**Loading a HuggingFace model?** Read [transformer.md](transformer.md) and [serialization.md](serialization.md).

**Changing precision / quantizing?** Read [numerical_types.md](numerical_types.md) and [quantization.md](quantization.md).

**Evolving or merging trained networks?** Read [dna.md](dna.md) and [evolution.md](evolution.md).

**Building parallel/sequential sub-networks?** Read [parallel_sequential.md](parallel_sequential.md).

**Just need a code snippet?** Go straight to [quick_reference.md](quick_reference.md).

---

## Package Layout

```
poly/
├── poly.go              Core types: LayerType, DType, Tensor[T], VolumetricNetwork
├── weights.go           WeightStore, Morph, Unpack, ApplyGradients
├── forward.go           DispatchLayer, ForwardPolymorphic
├── backward.go          DispatchLayerBackward, BackwardPolymorphic
├── training.go          Train, TrainingConfig, CalculateLoss, ComputeLossGradient
├── dense.go             DenseForwardPolymorphic, tiled fast-paths
├── rnn.go               RNNForwardPolymorphic
├── mha.go               MHAForwardPolymorphic, RoPE, GQA, KV cache
├── softmax.go           All 10 softmax variants
├── parallel.go          ParallelForwardPolymorphic, 5 combine modes
├── sequential.go        SequentialForwardPolymorphic, step containers
├── target_prop.go       TargetPropState, TargetPropBackward, ApplyTargetPropGaps
├── systolic.go          SystolicState, SystolicForward, SystolicBackward
├── dna.go               ExtractDNA, CompareNetworks, LogicShift, recursive all-19-type extraction
├── evolution.go         SpliceDNA, SpliceDNAWithReport, NEATMutate, NEATPopulation
├── quantization.go      Q4_0Block, QuantizeQ4_0, DequantizeQ4_0
├── serialization.go     BuildNetworkFromJSON, ParseLayerType/DType/Activation
├── persistence.go       SerializeNetwork, DeserializeNetwork, bit-packing
├── transformer.go       Transformer[T], NewTransformer, Generate
├── wgpu_context.go      WGPUContext, InitWGPU, BeginFrame, FlushFrame
├── wgpu_forward.go      GPU forward dispatch, ForwardTokenIDsWGPU
├── wgpu_backward_shaders.go  WGSL shader strings for dense backward
├── safetensors.go       SafeTensors format reader
├── prefix_safetensor.go Weight name prefix stripping
└── universal_loader.go  Auto-detecting checkpoint loader
```
