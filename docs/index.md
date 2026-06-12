# Loom / poly Documentation Index (v0.80.0)

This directory contains comprehensive documentation for the `poly/` package — the **M-POLY-VTD** (Multi-numerical POLYmorphic Volumetric Tiled-tensor Dispatcher) engine that powers the Loom neural framework. For the live checklist and completion ratio, see [`poly/README.md`](../poly/README.md#-true-version-calculation).

---

## Documents

| File | Description |
|:-----|:------------|
| [overview.md](overview.md) | Big-picture architecture: the 3D grid, six design pillars, key types |
| [deployment.md](deployment.md) | **Polyglot Ecosystem**: NPM deployment, TypeScript SDK, WASM bridge, and Browser/Node usage |
| [donate_compute.md](donate_compute.md) | **Donate compute**: LAN TCP protocol (`donate_compute_*.go`), framed JSON, model push vs local LM, client/server API |
| [tanhi.md](tanhi.md) | **TANHI**: UDP JSON-line layer telemetry (`poly/tanhi.go`), SoulGlitch HUD, env/C-ABI, wire format |
| [numerical_types.md](numerical_types.md) | All 21 DTypes, the `Numeric` generic constraint, `WeightStore` lifecycle, `MorphToFloat32ForGPU` PTQ simulation, Q4_0, and compression ratios |
| [layers.md](layers.md) | Every layer type (Dense, CNN, RNN, MHA, SwiGLU, RMSNorm, Residual, Softmax, Parallel, Sequential, and more) with ASCII data-flow diagrams |
| [dispatch.md](dispatch.md) | `DispatchLayer` routing, the 3D grid traversal, tiled parallel execution, `IsRemoteLink` spatial hopping, and the GPU dispatch path |
| [training.md](training.md) | CPU and GPU training pipelines, loss functions, gradient flow, tween / neural target propagation (chain-rule and gap-based modes), link budgets |
| [gpu.md](gpu.md) | `WGPUContext`, `InitWGPU`, `BeginFrame`/`FlushFrame`, buffer management, bind group cache, GPU support matrix, WGSL shader overview |
| [windows_arm64.md](windows_arm64.md) | **Windows on ARM**: index → [`README_WINDOWS_ARM64.md`](../welvet/cabi/internal/build/README_WINDOWS_ARM64.md) (recovery script + `build_unix.sh windows arm64`) |
| [step.md](step.md) | The step mesh engine: `StepState`, one-clock-cycle forward, spatial feedback via remote links, BPTT, online learning |
| [dna.md](dna.md) | Topological network fingerprinting: `ExtractDNA`, `CosineSimilarity`, `CompareNetworks`, `LogicShift` detection, recursive extraction for all 19 layer types |
| [evolution.md](evolution.md) | DNA Splice / Genetic Crossover and NEAT-style Topology Evolution: `SpliceDNA`, `NEATMutate`, `NEATPopulation`, all 3 crossover modes, all 6 mutation types |
| [softmax.md](softmax.md) | All 10 softmax variants: Standard, Temperature, Gumbel, Masked, Sparse, Entmax, Grid, Hierarchical, Adaptive, Mixture |
| [serialization.md](serialization.md) | JSON + ENTITY save/load, bit-packing, idempotency, SafeTensors import |
| [entity.md](entity.md) | **ENTITY** (`.entity`) — native binary checkpoint; topology + weights in one file; HF→native bridge (Lucy [8]), Q4 LLM bake, experimental 3D unlock |
| [parallel_sequential.md](parallel_sequential.md) | `LayerParallel` (5 combine modes, activation tree), `LayerSequential` (step containers, skip gradients), nesting patterns |
| [quantization.md](quantization.md) | PTQ pipeline, `WeightStore` versioning, `Morph`/`Unpack`, `Q4_0Block` block quantization, calibration, accuracy trade-offs |
| [transformer.md](transformer.md) | MHA with RoPE, GQA/MQA, KV cache, SwiGLU, RMSNorm, Qwen-style expanded-query + Q/K norm support, `Transformer[T]` generation type; CPU vs GPU tiling behavior |
| [quick_reference.md](quick_reference.md) | Concise copy-paste snippets for all common operations |
| [testing_and_validation.md](testing_and_validation.md) | **Lucy logs**, parity table legend, how to read `lucy_testing_output/log.txt`, Dense **Go÷ASM** benchmarks, and a compact map of `poly/` files the suites hit |
| [bedrock_validation.md](bedrock_validation.md) | **v0.79.0** — seven-layer CPU suite, MHA/KV/persistence fixes, C-ABI 100%, what shipped vs roadmap |
| [v080_release.md](v080_release.md) | **v0.80.0** — ENTITY native checkpoints, WebGPU v1.0.4, cross-platform GPU, Planet Bridging POC |
| [`../poly/asm/README.md`](../poly/asm/README.md) | **Plan 9 CPU kernels**: `UseAsmForward`, dense forward routing, dot/matmul layout, Lucy speedup interpretation |

---

## Where to Start

**New to the codebase?** Read [overview.md](overview.md) first for the architecture picture, then [layers.md](layers.md) to see what layer types are available.
**Deploying to Web or JS?** Read [deployment.md](deployment.md).

**Sharing inference over LAN (donor node / TCP)?** Read [donate_compute.md](donate_compute.md).

**Visualizing layer-by-layer execution (UDP → SoulGlitch TANHI)?** Read [tanhi.md](tanhi.md).

**Want to train a model?** Read [training.md](training.md) and [dispatch.md](dispatch.md).

**Using the GPU?** Read [gpu.md](gpu.md).

**Loading a HuggingFace model?** Read [transformer.md](transformer.md) and [serialization.md](serialization.md).

**Saving a native Loom checkpoint (not HF)?** Read [entity.md](entity.md) — includes Lucy **[8]** ENTITY Talk (HF → `.entity` → chat) and what the format unlocks for grafting / 3D experiments.

**Changing precision / quantizing?** Read [numerical_types.md](numerical_types.md) and [quantization.md](quantization.md).

**Evolving or merging trained networks?** Read [dna.md](dna.md) and [evolution.md](evolution.md).

**Building parallel/sequential sub-networks?** Read [parallel_sequential.md](parallel_sequential.md).

**Just need a code snippet?** Go straight to [quick_reference.md](quick_reference.md).

**Reading Lucy / Glitch test transcripts or parity tables?** See [testing_and_validation.md](testing_and_validation.md).

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
├── tween.go       TweenState, TweenBackward, ApplyTweenGaps
├── step.go              StepState, StepForward, StepBackward
├── dna.go               ExtractDNA, CompareNetworks, LogicShift, recursive all-19-type extraction
├── evolution.go         SpliceDNA, SpliceDNAWithReport, NEATMutate, NEATPopulation
├── quantization.go      Q4_0Block, QuantizeQ4_0, DequantizeQ4_0
├── serialization.go     BuildNetworkFromJSON, ParseLayerType/DType/Activation
├── persistence.go       SerializeNetwork, DeserializeNetwork, bit-packing, EncodeNativeWeightsRaw
├── entity.go            SerializeEntity, LoadEntity, DeserializeEntity — native `.entity` checkpoints
├── transformer.go       Transformer[T], NewTransformer, Generate
├── wgpu_context.go      WGPUContext, InitWGPU, BeginFrame, FlushFrame
├── wgpu_forward.go      GPU forward dispatch, ForwardTokenIDsWGPU
├── wgpu_backward_shaders.go  WGSL shader strings for dense backward
├── safetensors.go       SafeTensors format reader
├── prefix_safetensor.go Weight name prefix stripping
├── donate_compute_*.go  LAN TCP donate-compute protocol (see donate_compute.md)
├── tanhi.go             UDP TANHI telemetry (see tanhi.md)
├── native_layer_matrix.go   Dtype × mode benchmark matrix (hooks + reports)
├── native_matrix_builtin_hooks.go  Default hooks for `RunNativeLayerMatrixBuiltin`
└── universal_loader.go  Auto-detecting checkpoint loader
```
