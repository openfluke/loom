# v0.80.0 — Native Ship (ENTITY + Modern GPU)

**Release:** **0.79.0 "Bedrock Validation"** → **0.80.0 "Native Ship"**  
**Checklist:** **111 / 142** (78.2%) → **114 / 142** (80.3%)

This wave ships **native Loom checkpoints** (ENTITY), moves production GPU to **openfluke/webgpu v1.0.4** (wgpu-native **v29**), and validates real LLM inference on **Metal, Vulkan (Intel + NVIDIA), and Windows ARM64**. The **Planet Bridging** POC in [`../planetbridging/`](../planetbridging/) completes the “planets → Loom” half of the hub; it **releases as its own repo/version after Loom 0.80** — Loom must land first.

---

## What shipped

### ENTITY — native `.entity` checkpoints

| Item | Detail |
|------|--------|
| **Format** | [`entity.md`](entity.md) — magic `ENTITY`, JSON topology header + native-packed weight blobs |
| **Semantics** | Same as JSON persistence: 21 dtypes, volumetric `(Z,Y,X,L)`, parallel/sequential trees, per-layer `Scale` |
| **Lucy [7]** | Seven-layer CPU suite: JSON **and** `.entity` save/reload PASS on all trained rows |
| **Lucy [8]** | **ENTITY Talk** — HF cache → `ImportHFToEntity` → optional Q4 bake → GPU chat without safetensors at runtime |
| **Size** | ~25% smaller than JSON checkpoints (Base64 removed); quant dtype still dominates absolute size |
| **Unlock** | Real LLM weights become `.entity` citizens — same container as volumetric experiments (graft, remote links, per-layer dtype) |

Import lane unchanged: HuggingFace **`.safetensors`** for download. Ship lane: **`.entity`** for trained or converted brains.

### WebGPU v29 — `github.com/openfluke/webgpu@v1.0.4`

| Item | Detail |
|------|--------|
| **Module** | Standalone [openfluke/webgpu](https://github.com/openfluke/webgpu) (no longer a cogentcore fork) |
| **Native stack** | wgpu-native **v29** C API — futures, `WGPUStringView`, Go-side validation error scopes |
| **Loom dependency** | `require github.com/openfluke/webgpu v1.0.4` in root and `lucy/go.mod` |
| **Binaries** | Prebuilt `libwgpu_native.a` per platform under the module; `ios/amd64` (Intel simulator) dropped to satisfy Go module size limits |

See webgpu README for platform table and version history.

### Cross-platform GPU validation (Lucy Poly Talk / ENTITY Talk)

Same SmolLM2-135M-Instruct, Q4, block-wise GPU upload — **webgpu v1.0.4 + poly WGSL**:

| Platform | GPU | Backend | Decode (approx.) | Notes |
|----------|-----|---------|------------------|-------|
| macOS arm64 | Apple M5 | Metal | ✅ parity with prior v29 work | Adapter → device → buffer → forward |
| Windows arm64 | Snapdragon | Vulkan | ✅ validated | Previously broken on old bindings |
| Linux | Intel Iris Xe | Vulkan (Mesa i915) | ~19 tok/s decode | Headless; tier fallback OK |
| Linux | RTX 3050 Mobile | Vulkan (NVIDIA) | ~69 tok/s decode, ~492 tok/s prefill | Requires healthy `nvidia-smi` + `VK_ICD_FILENAMES` |

Not a llama.cpp/Ollama tok/s contest yet — custom WGSL through wgpu-native — but **~3.5× decode vs iGPU** on the same box confirms the v29 stack is production-real on NVIDIA Linux.

### Planet Bridging POC (in monorepo — separate release)

[`../planetbridging/`](../planetbridging/) reached **v0.5.0** internally:

- **Direction:** planets → Loom (**complete** for standard volumetric layer types)
- **13 compare tabs:** Dense, CNN1/2/3, MHA, LSTM, RNN, LayerNorm, Embedding, RMSNorm, SwiGLU, Residual, Mixer v1/v2
- **Planets:** PyTorch, TensorFlow, JAX (+ sklearn on Dense)
- **Mechanism:** live weight stream → `.stream.entity` → Loom infer → PASS vs native (fp32 tolerance)
- **Mixer v2:** 16-layer stack, all 12 types chained (~5e-5 max diff POC)

**Release order:** **Loom 0.80 first** → then **Planet Bridging 0.5.0** as its own published hub (v1.0 = Loom → ONNX/Safetensors/GGUF export).

---

## What this release is (and is not)

**You now have:**

- A **shippable native checkpoint** (`.entity`) beside JSON debug persistence and HF import
- **HF LLMs as Loom citizens** via Lucy [8] — not just flat safetensor guests each run
- **Modern GPU bindings** decoupled from upstream fork politics
- **Multi-vendor GPU proof** on one engine (Metal, Qualcomm, Intel, NVIDIA Vulkan)
- A **complete planet→Loom POC** waiting on Loom’s release tag

**You do not yet claim:**

- Planet Bridging **published** (repo/version ships after Loom)
- Loom → export hub formats (ONNX/GGUF out) — Planet Bridging **v1.0**
- Ollama-class decode on every GPU (WGSL matmul path still has headroom)
- ENTITY v2 binary topology (header still JSON; see [entity.md — future](entity.md#future-smaller-files-with-full-topology))

**Next named targets:**

- **v0.81** — ASM rollout (Dense backward, SwiGLU, MHA); GPU kernel fusion
- **Planet Bridging v0.5.0** — publish after Loom 0.80 tag
- **Planet Bridging v1.0** — Loom → hub formats → any inference engine

---

## How to verify

```bash
# Lucy ENTITY + GPU (from repo root)
cd lucy && go get github.com/openfluke/webgpu@v1.0.4 && go mod tidy
go run .   # [7] seven-layer (entity save/reload), [8] ENTITY Talk, [1] Poly Talk GPU

# ENTITY round-trip tests
cd ../poly/tests && go test -run Entity -v

# Planet Bridging compare host (POC — not part of Loom release artifact yet)
cd ../planetbridging && go run .
```

Linux NVIDIA:

```bash
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json
export WGPU_ADAPTER_NAME=NVIDIA
```

---

## Key source files

| Area | Files |
|------|-------|
| ENTITY | `poly/entity.go`, `poly/entity_q4.go`, `poly/hf_import.go` |
| Lucy [8] | `lucy/hf_entity.go` |
| WebGPU init | `poly/wgpu_context_native.go` |
| Docs | [`entity.md`](entity.md), [`gpu.md`](gpu.md), [`transformer.md`](transformer.md) |
| Planet Bridging | `planetbridging/README.md`, `planetbridging/PROGRESS.md` |

---

## See also

- [bedrock_validation.md](bedrock_validation.md) — v0.79 CPU/MHA/C-ABI wave
- [entity.md](entity.md) — format spec and Lucy [7]/[8] validation
- [../planetbridging/README.md](../planetbridging/README.md) — bridging POC (release after Loom)
