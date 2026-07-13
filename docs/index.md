# Loom / poly Documentation Index (v0.84.0)

This directory contains comprehensive documentation for the `poly/` package — the **M-POLY-VTD** (Multi-numerical POLYmorphic Volumetric Tiled-tensor Dispatcher) engine that powers the Loom neural framework. For the live checklist and completion ratio, see [`poly/README.md`](../poly/README.md#-true-version-calculation).

---

## Documents

| File | Description |
|:-----|:------------|
| [overview.md](overview.md) | Big-picture architecture: the 3D grid, six design pillars, key types |
| [deployment.md](deployment.md) | **Polyglot Ecosystem**: NPM deployment, TypeScript SDK, WASM bridge, and Browser/Node usage |
| [flutter.md](flutter.md) | **Flutter / Dart**: `welvet` on pub.dev, FFI quick start, runnable examples, loom-flutter-quickstart showcase |
| [donate_compute.md](donate_compute.md) | **Donate compute**: LAN TCP protocol (`donate_compute_*.go`), framed JSON, model push vs local LM, client/server API |
| [tanhi.md](tanhi.md) | **TANHI**: UDP JSON-line layer telemetry (`poly/tanhi.go`), SoulGlitch HUD, env/C-ABI, wire format |
| [numerical_types.md](numerical_types.md) | All 21 DTypes, the `Numeric` generic constraint, `WeightStore` lifecycle, `MorphToFloat32ForGPU` PTQ simulation, Q4_0, and compression ratios |
| [layers.md](layers.md) | Every layer type (Dense, CNN, RNN, MHA, SwiGLU, RMSNorm, Residual, Softmax, Parallel, Sequential, and more) with ASCII data-flow diagrams |
| [dispatch.md](dispatch.md) | `DispatchLayer` routing, the 3D grid traversal, tiled parallel execution, `IsRemoteLink` spatial hopping, and the GPU dispatch path |
| [training.md](training.md) | CPU and GPU training pipelines; **default (QAT-like) vs native exact** (`UseExactDType`); loss, tween, link budgets; Lucy menu [14] |
| [native_layers.md](native_layers.md) | **Lucy [14]** native-exact layer suite — amd64/arm64 benchmark results, SIMD speedups, known train flakes |
| [cross_path_layers.md](cross_path_layers.md) | **Lucy [15]** cross-path CPU suite — SC/MC/SIMD vs native vs native-SIMD side-by-side |
| [gpu.md](gpu.md) | `WGPUContext`, `InitWGPU`, `BeginFrame`/`FlushFrame`, buffer management, bind group cache, GPU support matrix, WGSL shader overview |
| [memory_history.md](memory_history.md) | **Memory history**: GPU load chart/diagnosis; block-wise HF→`.entity` import **and** block-wise encode (`ImportHFSaveEntityTransformerBlockwise`); GPU upload + sequential global release |
| [accelerators.md](accelerators.md) | **Vendor NPU/TPU/GPU** — `poly/accel`, Intel OpenVINO CPU+NPU (Lucy [9]) + Qualcomm/Hexagon QNN (Lucy [12]) + Apple Metal/MPSGraph (Lucy [13]); all experimental; Google TPU planned; `SyncToAccel` |
| [snapdragon_npu.md](snapdragon_npu.md) | **Snapdragon (Hexagon) NPU** — QNN AI Engine Direct plugin (`loom_accel_qualcomm.dll`), Windows/ARM64, Lucy [12], experimental; achievements + honest gaps from `snapdragon.txt` |
| [apple_metal.md](apple_metal.md) | **Apple Metal GPU** — MPSGraph plugin (`libloom_accel_apple.dylib`), macOS Apple silicon, Lucy [13], experimental; GPU MatMul/MHA + CPU reference, BF16, results + honest gaps from `apple.txt` |
| [windows_arm64.md](windows_arm64.md) | **Windows on ARM**: index → [`README_WINDOWS_ARM64.md`](../welvet/cabi/internal/build/README_WINDOWS_ARM64.md) (recovery script + `build_unix.sh windows arm64`) |
| [`../welvet/cabi/internal/build/build_linux.sh`](../welvet/cabi/internal/build/build_linux.sh) | **Linux C-ABI build** — `dist/linux_amd64/` or `linux_arm64/` (`welvet.so` + `welvet.h`); wrapper over `build_unix.sh` |
| [step.md](step.md) | The step mesh engine: `StepState`, one-clock-cycle forward, spatial feedback via remote links, BPTT, online learning |
| [dna.md](dna.md) | Topological network fingerprinting: `ExtractDNA`, `CosineSimilarity`, `CompareNetworks`, `LogicShift` detection, recursive extraction for all 19 layer types |
| [evolution.md](evolution.md) | DNA Splice / Genetic Crossover and NEAT-style Topology Evolution: `SpliceDNA`, `NEATMutate`, `NEATPopulation`, all 3 crossover modes, all 6 mutation types |
| [softmax.md](softmax.md) | All 10 softmax variants: Standard, Temperature, Gumbel, Masked, Sparse, Entmax, Grid, Hierarchical, Adaptive, Mixture |
| [serialization.md](serialization.md) | JSON + ENTITY save/load, bit-packing, idempotency, SafeTensors import |
| [seed_manifests.md](seed_manifests.md) | **Seed manifests** — topology + `layer_seed` → He-init weights; no weight blobs; Lucy [18–20] |
| [entity.md](entity.md) | **ENTITY** (`.entity`) — native binary checkpoint; topology + weights in one file; HF→native bridge (Lucy [8]), Q4 LLM bake, experimental 3D unlock |
| [planetbridging.md](planetbridging.md) | **Planet Bridging** — PyPI `planetbridging` package; live PyTorch/TF/JAX → `loom-stream` → `.entity`; welvet reload; roadmap (Loom → export v1.0) |
| [parallel_sequential.md](parallel_sequential.md) | `LayerParallel` (5 combine modes, activation tree), `LayerSequential` (step containers, skip gradients), nesting patterns |
| [quantization.md](quantization.md) | **Three modes**: default QAT-like train, PTQ inference, native exact train; `Morph`/`Unpack`, `Q4_0Block`, calibration |
| [transformer.md](transformer.md) | MHA with RoPE, GQA/MQA, KV cache, SwiGLU, RMSNorm, Qwen-style expanded-query + Q/K norm support, `Transformer[T]` generation type; CPU vs GPU tiling behavior |
| [quick_reference.md](quick_reference.md) | Concise copy-paste snippets for all common operations |
| [lucy.md](lucy.md) | **Lucy Bloom Rivers** — separate harness repo; menus [1]–[20], log paths, clone layout |
| [testing_and_validation.md](testing_and_validation.md) | **Lucy logs**, parity table legend, how to read `lucy_testing_output/log.txt`, Dense **Go÷ASM** benchmarks, and a compact map of `poly/` files the suites hit |
| [bedrock_validation.md](bedrock_validation.md) | **v0.79.0** — seven-layer CPU suite, MHA/KV/persistence fixes; C-ABI **489/489** (v0.81 accel + entity exports) |
| [v080_release.md](v080_release.md) | **v0.80.0** — ENTITY native checkpoints, WebGPU v1.0.4, cross-platform GPU, Planet Bridging POC |
| [v081_release.md](v081_release.md) | **v0.81.0** — Intel NPU bridge (`poly/accel`), Lucy [9], vendor plugin model, Qualcomm/Google TPU roadmap |
| [v082_release.md](v082_release.md) | **v0.82.0** — SIMD CPU fast-path (AVX2/NEON) + Snapdragon/Hexagon NPU bridge (QNN, Windows ARM64), Lucy [12] |
| [v083_release.md](v083_release.md) | **v0.83.0** — Apple GPU / Metal (MPSGraph) bridge (macOS Apple silicon), Lucy [13], + BF16 wire dtype for the shared accel bridge |
| [simd.md](simd.md) | **Plan 9 SIMD** (forward + backward): AVX2/NEON `DotTile` + `SaxpyF32AccF64`; default vs `*_native_simd.go`; Lucy [7] and [14] |
| [`../poly/asm/README.md`](../poly/asm/README.md) | **Plan 9 CPU kernels**: `UseAsmForward`, dense forward routing, dot/matmul layout, Lucy speedup interpretation |
| [asm-and-volumetric-exploration.md](asm-and-volumetric-exploration.md) | **Archive (Jun 2026)**: BitNet W8A8 ASM, I2_S scaffolding, volumetric executor v1, Lucy `[7]` findings — exploratory work removed from tree |

---

## Where to Start

**New to the codebase?** Read [overview.md](overview.md) first for the architecture picture, then [layers.md](layers.md) to see what layer types are available.
**Deploying to Web or JS?** Read [deployment.md](deployment.md).

**Building a Flutter or Dart app?** Read [flutter.md](flutter.md) and clone [loom-flutter-quickstart](https://github.com/openfluke/loom-flutter-quickstart).

**Sharing inference over LAN (donor node / TCP)?** Read [donate_compute.md](donate_compute.md).

**Visualizing layer-by-layer execution (UDP → SoulGlitch TANHI)?** Read [tanhi.md](tanhi.md).

**Want to train a model?** Read [training.md](training.md) and [dispatch.md](dispatch.md).

**Training in storage dtype (not FP32 surrogate)?** Read [training.md — Training paradigms](training.md#training-paradigms-default-qat-like-vs-native-exact) and [quantization.md — Three modes](quantization.md#three-traininginference-modes). Run Lucy **[14]** ([native_layers.md](native_layers.md)) — see [lucy.md](lucy.md).

**Using the GPU?** Read [gpu.md](gpu.md).

**Offloading to Intel NPU (experimental)?** Read [accelerators.md](accelerators.md) — build `accel/intel`, `SyncToAccel`, Lucy **[9]** or `accel/intel/example`. C/FFI: build Welvet with [`build_linux.sh`](../welvet/cabi/internal/build/build_linux.sh).

**Offloading to Snapdragon / Hexagon NPU (experimental, Windows/ARM64)?** Read [snapdragon_npu.md](snapdragon_npu.md) — build `accel/qualcomm`, `DiscoverQualcommAccel`, Lucy **[12]**.

**Offloading to Apple GPU / Metal (experimental, macOS Apple silicon)?** Read [apple_metal.md](apple_metal.md) — build `accel/apple`, `DiscoverAppleAccel`, Lucy **[13]**.

**Debugging GPU load RAM spikes (Lucy ENTITY/Poly Talk)?** Read [memory_history.md](memory_history.md).

**Converting HF safetensors to `.entity` on mobile (SoulGlitch)?** See [entity.md — convert memory](entity.md#hf--entity-convert-memory) and [memory_history.md — low-RAM lane](memory_history.md#hf--entity-convert-import--encode-memory) (`ImportHFSaveEntityTransformerBlockwise`).

**Converting HF safetensors to `.entity` on Mac (Lucy [8])?** Same docs — Lucy uses the **standard** lane (`ImportHFCheckpointDir` + `SaveEntityTransformer`).

**Loading a HuggingFace model?** Read [transformer.md](transformer.md) and [serialization.md](serialization.md).

**Saving a native Loom checkpoint (not HF)?** Read [entity.md](entity.md) — includes Lucy **[8]** ENTITY Talk (HF → `.entity` → chat) and what the format unlocks for grafting / 3D experiments.

**Saving or reloading by seeds only (no weight blob)?** Read [seed_manifests.md](seed_manifests.md) — topology + per-layer `layer_seed`, He-init weights, Lucy **[19]** round trip and **[20]** train→save→reload proof.

**Streaming live PyTorch / TensorFlow / JAX weights into Loom (no HTTP)?** Read [planetbridging.md](planetbridging.md) — `pip install planetbridging`, bundled `loom-stream`, 13 layer bedrocks, welvet reload.

**Changing precision / quantizing?** Read [numerical_types.md](numerical_types.md) and [quantization.md](quantization.md) — distinguish **default QAT-like train**, **PTQ deploy**, and **native exact train**.

**Evolving or merging trained networks?** Read [dna.md](dna.md) and [evolution.md](evolution.md).

**Building parallel/sequential sub-networks?** Read [parallel_sequential.md](parallel_sequential.md).

**Just need a code snippet?** Go straight to [quick_reference.md](quick_reference.md).

**Reading Lucy / Glitch test transcripts or parity tables?** See [lucy.md](lucy.md) and [testing_and_validation.md](testing_and_validation.md).

**Speeding up CPU inference and training (SIMD)?** Read [simd.md](simd.md) — `TrainingModeCPUSimd`, AVX2/NEON `DotTile` + `SaxpyF32AccF64`, Lucy **[7]** seven-layer SC/MC/SIMD parity and benchmarks. Native-exact SIMD timings: [native_layers.md](native_layers.md).

---

## Package Layout

```
poly/
├── poly.go              Core types: LayerType, DType, Tensor[T], VolumetricNetwork
├── weights.go           WeightStore, Morph, Unpack, ApplyGradients
├── forward.go           DispatchLayer, ForwardPolymorphic (+ vendor accel hook)
├── accel/               Vendor plugin loader (Intel dlopen; Qualcomm LoadLibrary; Apple dlopen; C ABI)
├── accel_intel.go       Vendor-neutral SyncToAccel, DispatchAccelForward, dtype bytes
├── accel_qualcomm.go    Qualcomm/Hexagon plugin discovery (QNN)
├── accel_apple.go       Apple Metal/MPSGraph plugin discovery
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
├── entity_convert_io.go Block-wise ENTITY encode: streaming payload, Q4 bake helpers, writeEntityWireStreaming
├── hf_entity_convert.go ImportHFSaveEntityTransformerBlockwise(Progress) — mobile-safe HF→`.entity`
├── hf_import.go         ImportHFCheckpointDir, ImportHFToEntity, ImportHFBitNetCheckpointDir
├── seed_core.go         SeedFrom, DeriveLayerSeed, He-init from layer_seed
├── seed_dense.go        Dense weight manifests (seeds only, no blobs)
├── seed_*.go            Per-layer seed manifests (SwiGLU, MHA, CNN, …)
├── seed_manifest.go     Entity .wseed tiny manifests
├── transformer.go       Transformer[T], NewTransformer, Generate, SyncGlobalWeightsToGPUSequential
├── memory_history.go    Load-path MemoryHistory, terminal chart, diagnosis
├── memory_history_chart.go  Braille/sparkline renderers for memory timeline
├── process_memory_unix.go   Process RSS sampling (getrusage)
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
