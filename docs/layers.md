# Layer Reference

This document describes every `LayerType` in `poly/`. For each layer: what it computes, which fields of `VolumetricLayer` configure it, weight layout in the `WeightStore`, and an ASCII data-flow diagram.

---

## LayerType Constants

```go
const (
    LayerDense              LayerType = 0
    LayerMultiHeadAttention LayerType = 1
    LayerSwiGLU             LayerType = 2
    LayerRMSNorm            LayerType = 3
    LayerCNN1               LayerType = 4
    LayerCNN2               LayerType = 5
    LayerCNN3               LayerType = 6
    LayerRNN                LayerType = 7
    LayerLSTM               LayerType = 8
    LayerLayerNorm          LayerType = 9
    LayerConvTransposed1D   LayerType = 10
    LayerConvTransposed2D   LayerType = 11
    LayerConvTransposed3D   LayerType = 12
    LayerEmbedding          LayerType = 13
    LayerKMeans             LayerType = 14
    LayerSoftmax            LayerType = 15
    LayerParallel           LayerType = 16
    LayerSequential         LayerType = 17
    LayerResidual           LayerType = 18
)
```

> [!NOTE]
> There is no explicit `LayerGRU` constant; GRU is implemented in `rnn.go` as a variant of the RNN pattern referenced through the same dispatcher slot.

---

## Dense (LayerDense = 0)

**What it does:** Fully-connected linear transformation: `output = input × W^T + b`, followed by an activation function. Every input connects to every output.

**Key fields:**

| Field | Meaning |
|:------|:--------|
| `InputHeight` | Number of input features |
| `OutputHeight` | Number of output features |
| `Activation` | One of ReLU, SiLU, GELU, Tanh, Sigmoid, Linear |
| `DType` | Active numerical type |
| `UseTiling` / `TileSize` | Enable register-tiled matmul |

**Weight layout:** `WeightStore.Master` is a flat `[OutputHeight × InputHeight]` row-major matrix. No bias is stored in the Master by default (the polymorphic engine absorbs bias via zero-biased initialization).

```
Input [batch, inputSize]
      │
      ▼
┌─────────────────────────────────────────────┐
│  preAct[b, o] = Σᵢ  input[b, i] × W[o, i]  │
│                                             │
│  W shape: [OutputHeight, InputHeight]       │
└─────────────────────────────────────────────┘
      │
      ▼
   Activation(preAct)
      │
      ▼
Output [batch, outputSize]
```

The tiled variant (`DenseForwardTiled`) loads input tiles into a local buffer and unrolls the dot product 4× to help the compiler auto-vectorize. The INT8 and Binary tiled paths each have their own hot loops in `denseForwardTiledInt8` and `denseForwardTiledBinary`.

---

## CNN1 / CNN2 / CNN3 (LayerCNN1–3 = 4–6)

**What they do:** Convolutional layers across 1D sequences, 2D images, and 3D volumes respectively. A learnable kernel is slid across the spatial dimensions and a dot product is computed at each position.

**Key fields:**

| Field | Meaning |
|:------|:--------|
| `InputChannels` | Channels in the input |
| `Filters` | Number of output channels (kernels) |
| `KernelSize` | Spatial size (k for CNN1, k×k for CNN2, k×k×k for CNN3) |
| `Stride` | Step between kernel positions |
| `Padding` | Zero-padding added on each side |
| `InputHeight` / `InputWidth` / `InputDepth` | Input spatial dimensions |
| `OutputHeight` / `OutputWidth` / `OutputDepth` | Output spatial dimensions |

**Weight layout:** `Filters × InputChannels × KernelSize^N`

```
CNN2 Data Flow:

Input [batch, inChannels, H, W]
      │
      ▼  slide kernel [f, c, kH, kW] over H, W
┌─────────────────────────────────────────────────────────────┐
│  for each filter f:                                         │
│    for each (oh, ow):                                       │
│      out[b,f,oh,ow] = Σ_c Σ_kh Σ_kw  in[b,c,oh+kh,ow+kw]  │
│                                       × W[f,c,kh,kw]       │
└─────────────────────────────────────────────────────────────┘
      │
      ▼  Activation
Output [batch, Filters, outH, outW]
```

Output size formula (same for each spatial dimension):

```
outDim = (inDim + 2*Padding - KernelSize) / Stride + 1
```

> [!TIP]
> CNN3 on GPU achieves over 7600x speedup versus CPU tiling because the 3D spatial loop maps perfectly to 3D WebGPU workgroups. Always prefer GPU for CNN3.

---

## ConvTransposed1D / 2D / 3D (LayerConvTransposed1D–3D = 10–12)

**What they do:** Transposed convolution (also called "deconvolution"). It inverts the spatial compression of a regular convolution — used in decoder networks and generative models to upsample feature maps.

**Key fields:** Same as CNN variants plus `OutputPadding` for controlling output dimensions.

**Weight layout:** `InputChannels × Filters × KernelSize^N`

```
ConvTransposed2D conceptual reverse:

  CNN2:  [H, W] ──kernel──▶  [H', W']     (downsample)
  ConvT: [H', W'] ──kernel──▶ [H, W]      (upsample)

  Internal mechanism: insert (Stride-1) zeros between input elements,
  then apply regular convolution with kernel flipped.
```

---

## RNN (LayerRNN = 7)

**What it does:** Vanilla recurrent network. Processes a sequence step-by-step, feeding the hidden state forward through time.

```
h_t = tanh(x_t × W_ih^T + h_{t-1} × W_hh^T + b_h)
```

**Key fields:**

| Field | Meaning |
|:------|:--------|
| `InputHeight` | Input feature size |
| `OutputHeight` | Hidden state size |
| `SeqLength` | Number of time steps |

**Weight layout in Master:**

```
[  W_ih  |  W_hh  |  b_h  ]
   ihSize   hhSize   hSize
```

Where `ihSize = hiddenSize × inputSize`, `hhSize = hiddenSize × hiddenSize`, `hSize = hiddenSize`.

```
Step 0:          Step 1:          Step t:
 x₀   h₋₁=0     x₁    h₀         xₜ    h_{t-1}
  │      │        │     │           │      │
  └──┬───┘        └──┬──┘           └──┬───┘
     ▼               ▼                 ▼
   [RNN cell]     [RNN cell]        [RNN cell]
     │               │                 │
     ▼               ▼                 ▼
    h₀              h₁               hₜ
```

---

## LSTM (LayerLSTM = 8)

**What it does:** Long Short-Term Memory. Adds a cell state `c_t` and three gating mechanisms (forget, input, output) to control information flow through time. Solves the vanishing gradient problem for long sequences.

**Gate equations:**

```
i_t = σ(x_t × W_i^T + h_{t-1} × U_i^T + b_i)   ← input gate
f_t = σ(x_t × W_f^T + h_{t-1} × U_f^T + b_f)   ← forget gate
g_t = tanh(x_t × W_g^T + h_{t-1} × U_g^T + b_g) ← cell gate
o_t = σ(x_t × W_o^T + h_{t-1} × U_o^T + b_o)   ← output gate
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
h_t = o_t ⊙ tanh(c_t)
```

**Weight layout:** Four gate blocks concatenated:

```
[ W_i | U_i | b_i | W_f | U_f | b_f | W_g | U_g | b_g | W_o | U_o | b_o ]
  ←── gate i ──────────▶ ←── gate f ──────────▶ ...
  gateWeightCount = ihSize + hhSize + hiddenSize
  Total = 4 × gateWeightCount
```

```
                   ┌─────────────────────────────────────┐
    c_{t-1} ──────▶│                                     │──▶ c_t
                   │   Forget ×  +  Input × Cell         │
    h_{t-1} ──────▶│                                     │──▶ h_t
                   │       Output gate × tanh(c_t)       │
    x_t     ──────▶│                                     │
                   └─────────────────────────────────────┘
```

---

## GRU

GRU (Gated Recurrent Unit) is implemented in `rnn.go` alongside the vanilla RNN. It uses two gates (reset and update) and eliminates the separate cell state.

```
z_t = σ(x_t × W_z + h_{t-1} × U_z + b_z)   ← update gate
r_t = σ(x_t × W_r + h_{t-1} × U_r + b_r)   ← reset gate
n_t = tanh(x_t × W_n + (r_t ⊙ h_{t-1}) × U_n + b_n)
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ n_t
```

---

## MultiHeadAttention (LayerMultiHeadAttention = 1)

**What it does:** Standard multi-head scaled dot-product attention with optional RoPE positional encoding, Grouped Query Attention (GQA), and a KV cache for autoregressive decoding.

**Key fields:**

| Field | Meaning |
|:------|:--------|
| `DModel` | Model dimension (total embedding size) |
| `NumHeads` | Number of query heads |
| `NumKVHeads` | Number of key/value heads (< NumHeads for GQA/MQA) |
| `HeadDim` | Dimension per head (usually DModel / NumHeads) |
| `SeqLength` | Current sequence length |
| `RoPEFreqBase` | RoPE frequency base (default 10000.0) |
| `MaxSeqLen` | KV cache capacity |
| `KVCacheK` / `KVCacheV` | CPU-side KV cache tensors |
| `KVOffset` | Current filled position in the KV cache |

**Weight layout:**

```
Master = [ Q_W | K_W | V_W | O_W | Q_b | K_b | V_b | O_b ]

  Q_W: [DModel × DModel]
  K_W: [DModel × kvDim]        (kvDim = NumKVHeads × HeadDim)
  V_W: [DModel × kvDim]
  O_W: [DModel × DModel]
  biases follow
```

**Attention computation:**

```
Q = input × Q_W^T + Q_b     [seqLen, DModel]
K = input × K_W^T + K_b     [seqLen, kvDim]
V = input × V_W^T + V_b     [seqLen, kvDim]

Apply RoPE to Q, K (rotate pairs by position-dependent angle)

For each head h:
  q_h = Q[:, h*headDim:(h+1)*headDim]     [seqLen, headDim]
  k_h = K[:, kv_head_idx*headDim:...]     [seqLen, headDim]
  v_h = V[:, kv_head_idx*headDim:...]

  scores = q_h × k_h^T / sqrt(headDim)   [seqLen, seqLen]
  weights = softmax(scores, causal_mask)
  out_h = weights × v_h                   [seqLen, headDim]

output = concat(out_0..out_{numHeads-1}) × O_W^T
```

---

## SwiGLU (LayerSwiGLU = 2)

**What it does:** Gated feedforward block used in modern LLMs. Two parallel linear projections, one acting as a gate through SiLU activation, combined element-wise before a down projection.

```
gate = SiLU(x × W_gate^T + b_gate)
up   = x × W_up^T + b_up
hidden = gate ⊙ up
output = hidden × W_down^T + b_down
```

**Key fields:** `InputHeight` (in), `OutputHeight` (intermediate/hidden size). The actual output to the next layer is back to `InputHeight` via the down projection.

**Weight layout:**

```
Master = [ W_gate | W_up | W_down | b_gate | b_up | b_down ]
          in×int    in×int  int×in    int      int    in
```

Where `int = OutputHeight` (intermediate size).

```
Input [seqLen, in]
  │
  ├──────────────────────────────────┐
  │                                  │
  ▼                                  ▼
W_gate (in → int)              W_up (in → int)
  │                                  │
SiLU                                 │
  │                                  │
  └──────────── ⊙ (element multiply) ┘
                    │
                    ▼
               W_down (int → in)
                    │
                    ▼
              Output [seqLen, in]
```

---

## RMSNorm (LayerRMSNorm = 3)

**What it does:** Root Mean Square normalization. Divides each element by the RMS of the vector, then scales by a learned gamma parameter.

```
rms = sqrt( mean(x²) + ε )
output = (x / rms) × γ
```

**Key fields:** `InputHeight` (size), `DType`. **Always kept in FP32 on GPU** — the `SyncToGPU` code explicitly refuses to quantize RMSNorm weights.

**Weight layout:** `Master` is a flat `[InputHeight]` gamma vector (no beta/bias term).

---

## LayerNorm (LayerLayerNorm = 9)

**What it does:** Layer normalization. Computes mean and variance across the feature dimension, normalizes, then applies learnable gamma and beta.

```
μ = mean(x),  σ² = var(x)
x_hat = (x - μ) / sqrt(σ² + ε)
output = γ ⊙ x_hat + β
```

**Weight layout:** `Master` is `[2 × InputHeight]`: first half is gamma, second half is beta.

---

## Embedding (LayerEmbedding = 13)

**What it does:** Token lookup table. Given a vector of integer token IDs, returns the corresponding rows from the embedding matrix.

**Key fields:** `VocabSize`, `EmbeddingDim`.

**Weight layout:** `[VocabSize × EmbeddingDim]` row-major matrix.

```
Token IDs: [42, 7, 115]
                │
                ▼  lookup rows 42, 7, 115
┌──────────────────────────────────────────────────┐
│  Embedding Table [VocabSize × EmbeddingDim]      │
│                                                  │
│  Row 7:   [0.12, -0.33, 0.87, ...]              │
│  Row 42:  [0.55,  0.11, -0.22, ...]             │
│  Row 115: [-0.01, 0.77, 0.44, ...]              │
└──────────────────────────────────────────────────┘
                │
                ▼
Output [3, EmbeddingDim]  (gradient only applied to used rows)
```

---

## KMeans (LayerKMeans = 14)

**What it does:** Differentiable clustering. Computes soft assignment probabilities (or raw feature distances) between the input and a set of learnable cluster centroids.

**Key fields:**

| Field | Meaning |
|:------|:--------|
| `NumClusters` | K — number of cluster centers |
| `InputHeight` | Feature vector size |
| `KMeansTemperature` | Controls sharpness of soft assignment |
| `KMeansOutputMode` | `"probabilities"` or `"features"` |

**Weight layout:** `[NumClusters × InputHeight]` centroid matrix.

```
Input [batch, featureDim]
      │
      ▼  compute squared distance to each centroid
  dist[b, k] = ||input[b] - centroid[k]||²
      │
      ▼  temperature-scaled negative softmax
  p[b, k] = softmax(-dist / temperature)
      │
      ▼
Output [batch, NumClusters]  (if mode="probabilities")
    or [batch, featureDim]   (if mode="features")
```

---

## Softmax (LayerSoftmax = 15)

**What it does:** Normalizes a vector (or matrix rows) into a probability distribution. Has 10 variants controlled by `SoftmaxType`. See [softmax.md](./softmax.md) for the full variant reference.

**Key fields:** `SoftmaxType`, `Temperature`, `SoftmaxRows`, `SoftmaxCols`, `HierarchyLevels`, `EntmaxAlpha`, `Mask`, `GumbelNoise`.

No weights — `WeightStore` is nil for Softmax layers.

---

## Parallel (LayerParallel = 16)

**What it does:** Fans the input to N sub-layers simultaneously and combines their outputs. Supports five combination modes.

**Key fields:**

| Field | Meaning |
|:------|:--------|
| `ParallelBranches` | `[]VolumetricLayer` — the sub-layer definitions |
| `CombineMode` | `"add"`, `"avg"`, `"concat"`, `"filter"`, `"grid_scatter"` |
| `FilterGateConfig` | Optional gate network for MoE routing (filter mode) |

```
            Input
              │
   ┌──────────┼──────────┐
   ▼          ▼          ▼
Branch 0   Branch 1   Branch 2
   │          │          │
   └──────────┼──────────┘
              │
        CombineMode:
        ┌─────────────────────────────────────────┐
        │ "add"         element-wise sum          │
        │ "avg"         element-wise average      │
        │ "concat"      [b0, b1, b2] concatenated │
        │ "filter"      gate × b0 + gate × b1 ... │
        │ "grid_scatter" same as concat            │
        └─────────────────────────────────────────┘
              │
           Output
```

The `preAct` tensor returned by `ParallelForwardPolymorphic` stores the branch preActs in `preAct.Nested`, enabling correct recursive backpropagation. See [parallel_sequential.md](./parallel_sequential.md).

---

## Sequential (LayerSequential = 17)

**What it does:** Chains N sub-layers in series. Each sub-layer receives the output of the previous one. The sub-layers can be of any type — this enables composing mini-architectures inside a single grid cell.

**Key fields:** `SequentialLayers []VolumetricLayer`

```
Input
  │
  ▼
Sub-layer 0 ──▶ Sub-layer 1 ──▶ Sub-layer 2
                                      │
                                   Output
```

Each step container stores `[bPre, bInput, bSkip]` in the nested tensor for accurate backward computation through skip connections within the sequence.

---

## Residual (LayerResidual = 18)

**What it does:** Skip connection — adds the input directly to the output of its sub-network.

```
        Input
          │
     ┌────┴────┐
     │         │ skip
     ▼         │
 Sub-layers    │
     │         │
     ▼         │
   ┌───┐       │
   │ + │◀──────┘
   └─┬─┘
     │
   Output = SubLayers(Input) + Input
```

The skip tensor is passed as the second argument to `DispatchLayer` and is added inside `ResidualForwardPolymorphic`. Gradients flow back both through the sub-layers and directly through the skip branch.

---

## Activation Functions

All layers that produce a `preAct` / `postAct` pair apply an activation via `Activate[T](v T, act ActivationType)`:

| Constant | Formula |
|:---------|:--------|
| `ActivationReLU` (0) | `max(0, x)` |
| `ActivationSilu` (1) | `x × σ(x)` |
| `ActivationGELU` (2) | `0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))` |
| `ActivationTanh` (3) | `tanh(x)` |
| `ActivationSigmoid` (4) | `1/(1+e^−x)` |
| `ActivationLinear` (-1) | `x` (identity — no nonlinearity) |

`ActivateDerivative[T]` returns the analytic derivative for backpropagation.

---

## Layer Summary Table

| Layer | Parameters | GPU Forward | GPU Backward |
|:------|:-----------|:-----------|:------------|
| Dense | in×out | EXACT | EXACT |
| CNN1 | f×c×k | EXACT | EXACT |
| CNN2 | f×c×k² | EXACT | EXACT |
| CNN3 | f×c×k³ | EXACT | EXACT |
| RNN | ih+hh+b | EXACT | — |
| LSTM | 4×(ih+hh+b) | EXACT | — |
| MHA | 4×d² + biases | BROKEN (dets) | pending |
| SwiGLU | 3×in×int | BROKEN (dets) | not wired |
| RMSNorm | hidden | EXACT | EXACT |
| LayerNorm | 2×hidden | — | — |
| Embedding | vocab×dim | EXACT (DW) | — |
| KMeans | k×dim | — | — |
| Softmax | none | — | — |
| Parallel | per-branch | — | — |
| Sequential | per-layer | — | — |
| Residual | per-sub | — | — |
