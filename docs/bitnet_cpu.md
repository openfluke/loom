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

The current implementation is pure Go. It is intended as the correctness and
integration layer before adding architecture-specific kernels such as ARM NEON
or x86 AVX2/AVX512.

## Benchmark

Run the focused packed dense benchmark with:

```bash
go test ./poly -run '^$' -bench BenchmarkPackedTernaryDenseForward -benchmem
```

Run correctness coverage with:

```bash
go test ./poly -run 'BitNet|PackedTernary|TernaryNative'
```
