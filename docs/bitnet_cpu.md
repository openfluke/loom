# BitNet CPU Ternary Path

`poly` has an explicit CPU path for BitNet b1.58-style ternary weights.
The target dtype is `DTypeTernary` (`{-1, 0, +1}`), not `DTypeBinary`
(`{-1, +1}`).

## What Is Supported

- `WeightStore.MorphBitNetTernary()` converts FP32 master weights using the
  BitNet b1.58 absmean scale used by HF `utils_quant.py`:

  ```text
  scale = mean(abs(weights))
  q = round(clamp(weight / scale, -1, +1))
  ```

- `MorphLayerBitNetTernary()` and `MorphNetworkBitNetTernary()` provide public
  conversion helpers. The network helper leaves normalization layers in their
  existing dtype.

- `MorphLayerBitNetNativeTernary()` and `MorphNetworkBitNetNativeTernary()` are
  for BitNet-trained checkpoints. They replace projection weights with raw
  `{-1, 0, +1}` execution weights so the packed CPU path does not apply a PTQ
  dequant scale.

- When `VolumetricNetwork.UseExactDType` is true and the layer dtype is
  `DTypeTernary`, CPU inference uses packed 2-bit ternary matrix-vector kernels
  for:
  - Dense layers
  - MHA Q/K/V/O projections
  - SwiGLU gate/up/down projections
  - Transformer `lm_head` when it is a separate output head

If `lm_head` is tied to the embedding table, the output head stays FP32. This
matches common decoder layouts where token embeddings are not BitLinear weights.

The packed kernel stores 16 ternary weights per `uint32` and computes dot
products with add/subtract/skip logic. Inputs are quantized per token to int8:

```text
activation_scale = 127 / max(abs(input))
xq = clamp(round(input * activation_scale), -128, 127)
out = dot(xq, wq) * weight_absmean / activation_scale
```

For BitNet-style transformer blocks, the CPU path also applies the model's
learned inner RMSNorm after attention and after the SwiGLU gate/up product,
matching the HF `modeling_bitnet.py` layout.

The `1bitLLM/bitnet_b1_58-*` checkpoints are base models, not instruction-tuned
assistants. Lucy uses the tokenizer-native LLaMA-style `[INST] ... [/INST]`
wrapper for these models, but the output can still look like web-text
completion rather than a reliable chat answer.

Lucy also exposes ordinary FP32-to-ternary PTQ for non-BitNet CPU models as an
explicit experimental option. This is technically possible, but it is not
equivalent to BitNet training and may produce low-quality or broken text.

For CPU speed, packed ternary projections quantize each activation row once and
reuse the int8 row for sibling projections such as Q/K/V and gate/up. The tied
FP32 LM head remains exact but is parallelized across vocabulary rows.

The hot CPU kernel is row-aligned and word tiled: each row stores
`ceil(cols / 16)` packed `uint32` words, then the dot loop consumes one word
at a time with an unrolled, branchless 16-weight ternary decode. Large matrices
are split across output-row ranges using `GOMAXPROCS`.

### AVX2 SIMD ternary MAD (opt-in)

When Plan 9 SIMD forward is enabled (`net.SetSimdForward(true)` /
`SetSimdForwardRecursive`), the packed ternary matvec switches from the scalar
word-decode to an AVX2 **MAD kernel** mirroring Microsoft's `ggml-bitnet-mad.cpp`:

- Weights are unpacked **once** into a cached byte-per-weight code buffer
  (`{0,1,2}`, row-padded to a multiple of 32) — no on-the-fly unpack per token.
- The inner loop uses `VPMADDUBSW` (unsigned code × signed int8 activation) →
  `VPMADDWD` → `VPADDD`, 32 weights per instruction.
- Using `weight = code − 1`, the row dot is `Σ(code·act) − Σ(act)`; the `Σ(act)`
  correction is computed once per activation vector.

Output is **bit-identical** to the scalar path (both do exact integer
accumulation). Measured BitNet b1.58-2B CPU decode: ~2.6× single-core vs scalar.
The cached code buffer costs ~1 byte/weight of extra RAM. See
[`simd.md`](simd.md) for the full SIMD system and coverage.

Lucy loads BitNet checkpoints block-by-block for CPU inference: it decodes only
global tensors first, then decodes one transformer block, packs Dense/MHA/SwiGLU
BitLinear projections, releases that block's FP32 tensors, and moves to the next
block. Embeddings, tied `lm_head`, final norm, and learned inner norm scales
remain FP32 because the HF checkpoint uses them that way.

## Important Limits

This is a fast CPU storage/execution path, not a guarantee that any arbitrary
FP32 model will remain good after 1.58-bit post-training quantization. The
Microsoft BitNet b1.58 quality results assume BitNet-style trained checkpoints,
8-bit activations, and specialized CPU kernels. Plain FP32-to-ternary conversion
is useful for experiments, but it should be treated as lossy.

The scalar path is pure Go (the correctness and integration baseline). An x86
**AVX2** ternary MAD kernel is available as an opt-in SIMD path (see above and
[`simd.md`](simd.md)); ARM NEON and AVX512 ternary kernels are not yet written
(arm64 falls back to the scalar Go dot).

## Benchmark

Run the focused packed dense benchmark with:

```bash
go test ./poly -run '^$' -bench BenchmarkPackedTernaryDenseForward -benchmem
```

Run correctness coverage with:

```bash
go test ./poly -run 'BitNet|PackedTernary|TernaryNative'
```
