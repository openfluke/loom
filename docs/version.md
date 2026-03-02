# Loom Version Roadmap (roughly)

## Current: v0.0.8 — The GPU Era
GPU inference working. 30 t/s on 135M models in pure Go. WebGPU pipeline stable for MHA, SwiGLU, RMSNorm. FP32 path solid. Cross-language bindings across 5 ecosystems. 2,298 passing tests.

---

## v0.1.0 — The Integration Update *(next release)*

Merging the pieces that already exist into a unified engine. No new capabilities — rough edges from v0.0.7/v0.0.8 get resolved.

- [ ] Integrate geometry prober (`universal_safetensor_loading`) into `nn/load_transformer.go` — replace prefix maps with shape-based detection
- [ ] Validate against all 40 HuggingFace test models
- [ ] Merge `GenericTweenState[T]` into the generic `[T Numeric]` forward/backward paths
- [ ] Merge `ForwardCPU` / `BackwardCPU` into `GenericForwardPass[T]` / `GenericBackwardPass[T]`
- [ ] Collapse `parallelForwardCPU` / `ParallelForward[T]` dual path into one
- [ ] GPU layers feed into the same `LayerConfig` dispatch as CPU via unified backend
- [ ] Complete and pass `arcagitesting/test43a`
- [ ] Minor: remove `_ = gateUpSize` / `_ = downSize`, fix deprecated `rand.Seed`, mark approximate backward as experimental

**Release test:** Any conforming safetensors model loads and runs through a single unified forward path, CPU or GPU, any supported dtype, zero architecture-specific code.

---

## v0.1.0 — API Stabilisation

- [ ] Public API locked — no planned breaking changes from here
- [ ] Complete `nn` package godoc coverage
- [ ] `LayerConfig.Validate() error` to catch misconfiguration early
- [ ] Unified serialization: single `serialization.go` handles all dtypes (replace split float32 / multiprecision files)
- [ ] TVA green across all layer types × all 15 supported dtypes

---

## v0.2.0 — Multi-Dtype GPU

- [ ] fp16 weights packed on upload, GPU shaders operate natively in f16
- [ ] Weight management refactored: no duplicate copies in RAM during GPU mount
- [ ] int8 inference path on GPU (quantised weights, dequant in shader)
- [ ] VRAM budget estimation before mounting — graceful fallback to CPU if OOM
- [ ] Benchmark suite: fp32 vs fp16 vs int8 t/s and quality tradeoffs documented

---

## v0.3.0 — NAS (Neural Architecture Search)

- [ ] Architecture search over the 2D grid structure — layer composition, branch topology, combine modes
- [ ] Fitness function interface: plug in any evaluation metric
- [ ] Evolutionary / random search over `LayerConfig` space
- [ ] Results serialise to standard `Network` for immediate use
- [ ] Integration with `tween.go` for rapid fitness evaluation without full training

---

## v0.4.0 — Mobile & UMA

- [ ] Validated inference on Snapdragon X (Adreno) via native ARM64 + WebGPU
- [ ] UMA-aware memory model: CPU and GPU share physical memory, eliminate redundant copies
- [ ] 1B parameter model running on-device on phone
- [ ] iOS and Android inference demo via C-ABI
- [ ] Power/thermal profiling for edge deployment

---

## v0.5.0 — Training Maturity

- [ ] Accurate MHA backward pass (replace simplified/approximate shader)
- [ ] Full GPU training verified for all layer types including MHA and SwiGLU
- [ ] Gradient checkpointing for large model training under memory pressure
- [ ] Mixed precision training (fp16 forward, fp32 accumulation)
- [ ] Stable fine-tuning of HuggingFace models through loom

---

## v0.6.0 — Primecraft & Spatial AI Integration

- [ ] Loom inference callable from Primecraft 3D engine
- [ ] 3D object spawning driven by LLM output (structured generation → object spec)
- [ ] On-device models powering planet/world simulation AI
- [ ] Agent loop: perception → loom inference → action in 3D space
- [ ] Demo: language model controlling entity behaviour on a planet

---

## v0.7.0 — Performance

- [ ] Flash Attention implementation in WGSL (O(N) memory instead of O(N²))
- [ ] Fused kernels: RMSNorm + residual add in single GPU shader
- [ ] Speculative decoding support (draft model + verifier)
- [ ] Multi-batch inference on GPU
- [ ] Target: 100+ t/s on 135M model, 30+ t/s on 1B model

---

## v0.8.0 — Ecosystem & Tooling

- [ ] Loom model hub: publish/pull trained networks similarly to HuggingFace
- [ ] Network visualiser: render the 2D grid structure with live activation telemetry
- [ ] ONNX export path (loom network → ONNX for interop)
- [ ] Python `welvet` catches up: full parity with Go on all new capabilities
- [ ] Browser inference demo updated: 1B model in WASM

---

## v0.9.0 — Pre-Production

- [ ] Everything from v0.1–v0.8 stable and passing TVA
- [ ] Security audit of C-ABI and WASM surface
- [ ] Versioned API documentation site
- [ ] Loom paper: architecture, design decisions, benchmarks — publishable
- [ ] Cross-platform binary release for all 13 platforms verified at this capability level

**v0.9.0 is the "if nothing breaks after this, we call it 1.0" release.**

---

## Summary Table

| Version | Theme | Key Unlock |
|---|---|---|
| v0.0.8 | GPU Era | 30 t/s, WebGPU, cross-language ✅ |
| v0.1.0 | Integration | Unified path, geometry loader |
| v0.1.0 | API Stability | Locked public API |
| v0.2.0 | Multi-Dtype GPU | fp16/int8 on GPU |
| v0.3.0 | NAS | Architecture search over grid |
| v0.4.0 | Mobile/UMA | 1B model on phone |
| v0.5.0 | Training | Accurate backward, fine-tuning |
| v0.6.0 | Primecraft | 3D spatial AI integration |
| v0.7.0 | Performance | Flash attention, 100+ t/s |
| v0.8.0 | Ecosystem | Tooling, hub, visualiser |
| v0.9.0 | Pre-Production | Stable, documented, publishable |
| **v1.0.0** | **Release** | **Everything works. Ship it.** |



"rough estimate ehhh good enough  lol"