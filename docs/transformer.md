# Transformer Architecture: MHA, RoPE, GQA, and Full Block Assembly

This document covers `LayerMultiHeadAttention` (MHA), how RoPE positional encoding is applied, Grouped-Query Attention (GQA) and Multi-Query Attention (MQA), the KV cache, SwiGLU and RMSNorm layers, full transformer block assembly inside `VolumetricNetwork`, and the `Transformer[T]` high-level generation type.

---

## LayerMultiHeadAttention

`LayerMultiHeadAttention` (type index 16) implements scaled dot-product attention with optional RoPE, optional GQA/MQA, and an incremental KV cache.

### Key Fields on VolumetricLayer

```go
layer.Type       = poly.LayerMultiHeadAttention
layer.DModel     = 512   // model dimension (embedding size)
layer.NumHeads   = 8     // query heads
layer.NumKVHeads = 8     // key/value heads (set < NumHeads for GQA/MQA)
layer.HeadDim    = 64    // dimensions per head (DModel / NumHeads)
layer.MaxSeqLen  = 2048  // maximum sequence length (KV cache size)
layer.RoPEFreqBase = 10000.0  // RoPE theta; 0 = no positional encoding
```

### Weight Layout

All four projection matrices and their bias vectors are stored contiguously in `WeightStore.Master`:

```
Offset 0                  dModel × dModel         Q weight matrix
Offset dModel²            dModel × kvDim          K weight matrix
Offset dModel² + dModel×kvDim    dModel × kvDim   V weight matrix
Offset dModel² + 2×dModel×kvDim  dModel × dModel  O weight matrix

After all weight matrices:
  + dModel   bytes  Q bias vector
  + kvDim    bytes  K bias vector
  + kvDim    bytes  V bias vector
  + dModel   bytes  O bias vector

Total: dModel² + 2×dModel×kvDim + dModel² + dModel + 2×kvDim + dModel
     = 2×dModel² + 2×dModel×kvDim + 2×dModel + 2×kvDim weights
```

Where `kvDim = NumKVHeads × HeadDim`.

For standard MHA (`NumKVHeads == NumHeads`):
```
kvDim = dModel
Total = 4 × dModel² + 4 × dModel weights (including biases)
```

---

## Forward Pass: Step by Step

### 1. Linear Projections

Input shape: `[seqLen, dModel]`

```
For each token position s:
  Q[s, i] = bias_Q[i] + Σⱼ input[s, j] × W_Q[i, j]
  K[s, i] = bias_K[i] + Σⱼ input[s, j] × W_K[i, j]
  V[s, i] = bias_V[i] + Σⱼ input[s, j] × W_V[i, j]

Q shape: [seqLen, dModel]     (numHeads × headDim)
K shape: [seqLen, kvDim]      (numKVHeads × headDim)
V shape: [seqLen, kvDim]
```

### 2. RoPE: Rotary Positional Encoding

If `layer.RoPEFreqBase > 0`, RoPE is applied to Q and K after projection.

RoPE encodes position by rotating adjacent pairs of values in the head dimension:

```
For each token at position pos, head h, dimension pair (d, d + headDim/2):

  freq  = 1 / (RoPEFreqBase ^ (2d / headDim))
  angle = freq × pos
  cos_a, sin_a = cos(angle), sin(angle)

  Q[pos, h×headDim + d]              = Q0 × cos_a - Q1 × sin_a
  Q[pos, h×headDim + d + headDim/2]  = Q0 × sin_a + Q1 × cos_a

  (same for K, using the KV head index)
```

RoPE gives the attention mechanism a way to learn relative positions without adding learned positional embeddings. Positions encode directly into the dot-product scores.

```
┌──────────────────────────────────────────────────────────────────┐
│  RoPE effect on attention scores                                 │
│                                                                  │
│  Token at pos 0:  angle = 0 → cos=1, sin=0 → no rotation        │
│  Token at pos 1:  angle = freq → slight rotation                 │
│  Token at pos N:  angle = N×freq → large rotation for low d      │
│                                                                  │
│  Relative distance (pos_q - pos_k) is captured in the dot       │
│  product because cos(angle_q - angle_k) = cos(Δangle).          │
└──────────────────────────────────────────────────────────────────┘
```

### 3. KV Cache (Float32 Path Only)

The Float32 forward path maintains an incremental KV cache:

```go
// Lazy initialization on first forward call
if layer.KVCacheK == nil {
    layer.KVCacheK = NewTensor[float32](MaxSeqLen, kvDim)
    layer.KVCacheV = NewTensor[float32](MaxSeqLen, kvDim)
    layer.KVOffset = 0
}

// Write current position into the ring buffer
pos := layer.KVOffset + s
kRow := KVCacheK.Data[(pos % MaxSeqLen) * kvDim : ...]
// compute K for this token and write into kRow
layer.KVOffset += seqLen  // advance after full sequence
```

The cache is a ring buffer of size `MaxSeqLen`. On each call, new K and V values are written at positions `[KVOffset, KVOffset + seqLen)`. The attention score computation then looks back over all `currentTotalPos + 1` cached positions, giving the model memory of the full context up to `MaxSeqLen` tokens.

To clear the KV cache between independent prompts:

```go
transformer.Reset()  // sets KVOffset = 0 for all layers
```

### 4. Grouped-Query Attention (GQA / MQA)

GQA reduces memory bandwidth by sharing KV heads across multiple query heads:

```
headsPerKV = NumHeads / NumKVHeads

For query head h:
  kvHead = h / headsPerKV  ← all query heads in a group share one KV head
```

```
┌──────────────────────────────────────────────────────────────────────┐
│  Standard MHA: NumHeads = NumKVHeads = 8                            │
│  Each head has its own K and V.                                     │
│                                                                      │
│  Q0──K0/V0   Q1──K1/V1   Q2──K2/V2  ...  Q7──K7/V7                │
│                                                                      │
│  GQA: NumHeads = 8, NumKVHeads = 2                                  │
│  4 query heads share each KV head.                                  │
│                                                                      │
│  Q0, Q1, Q2, Q3 ──K0/V0                                            │
│  Q4, Q5, Q6, Q7 ──K1/V1                                            │
│                                                                      │
│  MQA: NumHeads = 8, NumKVHeads = 1                                  │
│  All query heads share one KV head.                                 │
│                                                                      │
│  Q0...Q7 ─────────K0/V0                                             │
└──────────────────────────────────────────────────────────────────────┘
```

GQA is the default in modern LLMs like Llama 3 because it reduces KV cache memory by `NumHeads / NumKVHeads`× without measurable quality loss.

### 5. Causal Attention

Causality is enforced by the score computation loop:

```go
// For query at position qPos, only attend to positions <= qPos
for kPos := 0; kPos <= qPos; kPos++ {
    dot = Q[qPos] · K[kPos]
    scores[kPos] = dot / sqrt(headDim)
}
// positions > qPos are never included — no explicit mask needed
```

This is equivalent to a causal mask but avoids allocating a mask tensor.

### 6. Output Projection

After attention-weighted value aggregation, the output is projected back to `dModel`:

```
O[s, i] = bias_O[i] + Σⱼ attnOut[s, j] × W_O[i, j]
```

---

## MHAForwardTiled

When `layer.UseTiling = true`, `MHAForwardPolymorphic` delegates to `MHAForwardTiled`, which partitions the sequence dimension into tiles and uses goroutines for parallel attention computation. The tile size is set via `layer.TileSize`.

`CalculateOptimalTileSize(headDim)` returns a tile size based on the head dimension — larger heads benefit from smaller tiles (more goroutines), smaller heads run faster without parallelism overhead.

---

## RMSNorm

`LayerRMSNorm` (type 8) implements Root Mean Square Layer Normalization:

```
rms = sqrt( (1/n) × Σᵢ xᵢ² + ε )
output[i] = (x[i] / rms) × weight[i]
```

Unlike LayerNorm, RMSNorm does not subtract the mean. This makes it faster (fewer operations) while preserving the same stabilizing effect on gradient flow.

Key fields:
```go
layer.Type        = poly.LayerRMSNorm
layer.InputHeight = 512   // must match OutputHeight
layer.OutputHeight = 512
layer.Epsilon     = 1e-6  // default; stored in layer config
```

Weight storage: one scale weight per hidden dimension (`len(Master) == OutputHeight`).

---

## SwiGLU

`LayerSwiGLU` (type 12) implements the gated linear unit variant used in modern transformers:

```
Given input x of shape [seqLen, inputHeight]:

  gate   = x × W_gate   (shape [seqLen, outputHeight])
  up     = x × W_up     (shape [seqLen, outputHeight])
  hidden = SiLU(gate) × up
  output = hidden × W_down  (shape [seqLen, inputHeight])

SiLU(x) = x × sigmoid(x) = x / (1 + exp(-x))
```

```
┌────────────────────────────────────────────────────────────────────┐
│  SwiGLU Data Flow                                                  │
│                                                                    │
│  Input [seqLen, 512]                                               │
│       │                                                            │
│       ├──▶ W_gate [512, 1364] ──▶ gate [seqLen, 1364]             │
│       │                               │                            │
│       └──▶ W_up   [512, 1364] ──▶ up [seqLen, 1364]              │
│                                       │                            │
│                               SiLU(gate) × up                      │
│                                       │                            │
│                          W_down [1364, 512]                        │
│                                       │                            │
│                               Output [seqLen, 512]                 │
└────────────────────────────────────────────────────────────────────┘
```

The hidden dimension (~2.67× the model dimension) is the intermediate expansion factor. For `dModel=512`, the typical hidden size is 1364.

Key fields:
```go
layer.Type         = poly.LayerSwiGLU
layer.InputHeight  = 512
layer.OutputHeight = 1364  // hidden dimension (intermediate expansion)
```

Weight storage: `W_gate` (inputHeight × outputHeight) + `W_up` (inputHeight × outputHeight) + `W_down` (outputHeight × inputHeight), stored contiguously in `Master`.

---

## Full Transformer Block Assembly

A standard decoder-only transformer block (pre-norm style) is assembled as a `LayerSequential` containing four sub-layers:

```go
block := poly.VolumetricLayer{
    Type: poly.LayerSequential,
    SequentialLayers: []poly.VolumetricLayer{
        // Sub-layer 0: Attention norm
        {
            Type:         poly.LayerRMSNorm,
            InputHeight:  512,
            OutputHeight: 512,
        },
        // Sub-layer 1: Multi-head attention
        {
            Type:          poly.LayerMultiHeadAttention,
            DModel:        512,
            NumHeads:      8,
            NumKVHeads:    8,
            HeadDim:       64,
            MaxSeqLen:     2048,
            RoPEFreqBase:  10000.0,
        },
        // Sub-layer 2: FFN norm
        {
            Type:         poly.LayerRMSNorm,
            InputHeight:  512,
            OutputHeight: 512,
        },
        // Sub-layer 3: Feed-forward (SwiGLU)
        {
            Type:         poly.LayerSwiGLU,
            InputHeight:  512,
            OutputHeight: 1364,
        },
    },
}
```

This entire block is a single `VolumetricLayer` entry in the 3D grid. Multiple blocks are placed at coordinates `(0, blockIdx, 0, 0)` in a `VolumetricNetwork`.

### Residual Connections

Residual connections are handled by `LayerResidual` (type 14). In the sequential backward pass, residuals produce skip gradients that are accumulated via `skipGradients` (see `parallel_sequential.md`). For transformer blocks, the typical pattern using `LayerSequential` with `LayerResidual` as a sub-layer:

```go
block := poly.VolumetricLayer{
    Type: poly.LayerSequential,
    SequentialLayers: []poly.VolumetricLayer{
        {Type: poly.LayerRMSNorm, ...},
        {Type: poly.LayerMultiHeadAttention, ...},
        {Type: poly.LayerResidual, ...},  // adds input to output
        {Type: poly.LayerRMSNorm, ...},
        {Type: poly.LayerSwiGLU, ...},
        {Type: poly.LayerResidual, ...},  // adds pre-FFN to FFN output
    },
}
```

---

## The Transformer[T] Type

`Transformer[T]` is a high-level wrapper around `VolumetricNetwork` for autoregressive language model inference. It holds the components that live outside the main layer grid:

```go
type Transformer[T Numeric] struct {
    Network    *VolumetricNetwork
    Embeddings []float32  // token embedding table: [vocabSize × hiddenSize]
    LMHead     []float32  // output projection: [hiddenSize × vocabSize]
    FinalNorm  []float32  // final RMSNorm weights (one per hidden dim)
    HiddenSize int
    VocabSize  int
    Template   Template   // prompt formatting (chat template)
}
```

### NewTransformer

```go
func NewTransformer[T Numeric](
    network     *VolumetricNetwork,
    embeddings  []float32,
    lmHead      []float32,
    finalNorm   []float32,
    template    Template,
) *Transformer[T]
```

Creates the wrapper and infers `HiddenSize` from the first network layer's `DModel` or `InputHeight`. `VocabSize` is inferred as `len(Embeddings) / HiddenSize`.

If `finalNorm` is non-nil, a synthetic `VolumetricLayer` of type `LayerRMSNorm` is created internally to hold the final normalization weights. This layer is not part of the main grid — it runs separately after the last transformer block.

### Tied Weights Detection

When `LMHead` and `Embeddings` point to the same backing array (common in weight-tied models), `SyncToGPU` detects this and reuses the same GPU buffer for both:

```go
if &t.LMHead[0] == &t.Embeddings[0] {
    t.Network.GPULMHead = t.Network.GPUEmbeddings  // no second upload
}
```

### Tiling

```go
func (t *Transformer[T]) EnableTiling(tileSize int)
```

Enables `UseTiling` and sets `TileSize` on all layers, including the final norm layer. If `tileSize <= 0`, `CalculateOptimalTileSize(headDim)` auto-selects the tile size.

### Generate

```go
func (t *Transformer[T]) Generate(
    encode func(text string) []uint32,
    decode func(tokens []uint32) string,
    turns []Turn,
    systemPrompt, userMsg string,
    opts GenOptions,
) string
```

Full autoregressive text generation pipeline:

```
┌──────────────────────────────────────────────────────────────────────┐
│  GENERATE FLOW                                                       │
│                                                                      │
│  1. Template.BuildPrompt(turns, systemPrompt, userMsg)               │
│     → apply chat template (e.g., <|im_start|>user\n...)             │
│                                                                      │
│  2. encode(prompt) → inputIDs []uint32                               │
│                                                                      │
│  3. Reset() → clear KV cache                                         │
│                                                                      │
│  4. Prefill (process all input tokens at once):                      │
│     a. tokensToTensor(inputIDs) → embed all tokens                  │
│     b. ForwardPolymorphic or ForwardTokenIDsWGPU (GPU)               │
│     c. applyLMHead(lastHiddenState) → logits over vocabulary         │
│                                                                      │
│  5. Decode loop (one token at a time):                               │
│     a. applyRepetitionPenalty(logits, generatedTokens)               │
│     b. SampleTopK(logits, TopK, Temperature, Deterministic)          │
│     c. stream.Push(tokens) → streaming decode callback               │
│     d. Forward single new token (incremental):                       │
│        getEmbedding(nextToken) → forwardOne(input)                  │
│        (KVOffset advances by 1 each step)                            │
│     e. check EOS condition or max tokens                             │
│                                                                      │
│  6. Return accumulated decoded string                                │
└──────────────────────────────────────────────────────────────────────┘
```

### GenOptions

```go
type GenOptions struct {
    MaxTokens    int
    Temperature  float64
    TopK         int
    Deterministic bool
    UseKVCache   bool
    EOSTokens    []int
}
```

`Deterministic = true` with `Temperature = 0` produces greedy decoding. `TopK` limits sampling to the top K logits before applying temperature.

---

## GPU Transformer Inference

When `network.UseGPU = true` and `SyncToGPU()` has been called, `Generate` uses `ForwardTokenIDsWGPU` for both prefill and incremental decode:

```go
logitTensor, err := t.ForwardTokenIDsWGPU(tokens, nil, true, true)
```

This dispatches into `wgpu_forward.go`'s GPU transformer block execution path, which runs matrix multiplications and attention as WebGPU compute shader invocations. All intermediate activations stay on VRAM; only the final logit tensor is read back to CPU for sampling.

The GPU path uses the `BeginFrame` / `FlushFrame` pattern (see `gpu.md`) — one GPU command buffer encodes the entire forward pass across all transformer layers, then flushes in a single submit. This minimizes CPU–GPU synchronization overhead.

---

## Loading from SafeTensors / HuggingFace

`universal_loader.go` auto-detects the checkpoint format. For HuggingFace models:

1. `safetensors.go` reads the weight tensor map (key → `[]float32`)
2. `prefix_safetensor.go` strips model-specific prefix patterns (e.g., `model.layers.0.self_attn.q_proj.weight`)
3. Weight slices are copied into the correct `VolumetricLayer.WeightStore.Master` at the computed offsets

The key-to-layer mapping follows the weight layout described earlier:
```
model.layers.{N}.self_attn.q_proj.weight  → layer N's Q weight sub-slice
model.layers.{N}.self_attn.k_proj.weight  → layer N's K weight sub-slice
...
```

After loading, call `poly.MorphLayer(network, targetDtype)` to convert to your desired inference precision.

---

## Practical: Building a 7-layer Transformer Network

```go
hiddenSize := 512
numHeads   := 8
numLayers  := 7
seqLen     := 2048

network := poly.NewVolumetricNetwork("llm-7l", 1, numLayers, 1, 1)

for i := 0; i < numLayers; i++ {
    l := network.GetLayer(0, i, 0, 0)
    l.Type         = poly.LayerSequential
    l.SequentialLayers = []poly.VolumetricLayer{
        {Type: poly.LayerRMSNorm, InputHeight: hiddenSize, OutputHeight: hiddenSize},
        {
            Type:         poly.LayerMultiHeadAttention,
            DModel:       hiddenSize,
            NumHeads:     numHeads,
            NumKVHeads:   2,         // GQA: 4 query heads share each KV head
            HeadDim:      hiddenSize / numHeads,
            MaxSeqLen:    seqLen,
            RoPEFreqBase: 10000.0,
        },
        {Type: poly.LayerRMSNorm,  InputHeight: hiddenSize, OutputHeight: hiddenSize},
        {Type: poly.LayerSwiGLU,   InputHeight: hiddenSize, OutputHeight: hiddenSize * 8 / 3},
    }
    poly.InitializeLayerWeights(l)
}

transformer := poly.NewTransformer[float32](
    network,
    embeddings,
    lmHead,
    finalNormWeights,
    chatTemplate,
)
transformer.EnableTiling(0)  // auto-detect tile size
```
