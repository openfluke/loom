# Transformer Inference in Loom

Loom supports full transformer-based LLM inference from HuggingFace safetensors models, running in pure Go with optional WebGPU acceleration. This covers architecture loading, the prefill/decode loop, KV caching, and GPU mounting.

---

## Architecture Overview

A loom transformer inference session has four components:

```
┌────────────────────────────────────────────────────────────┐
│                     tokenizer package                      │
│                                                            │
│  Tokenizer (BPE)  →  LLMEngine  →  Transformer            │
│       │                                │                   │
│   Encode/Decode                 Generate loop              │
└────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────┐
│                       nn package                           │
│                                                            │
│  Network (Layers: RMSNorm → MHA → RMSNorm → SwiGLU ...)   │
│       │                                                    │
│  ForwardTransformer(embedding, pos)                        │
│       │                                                    │
│  KV Cache (per MHA layer: K/V buffers, CachePos tracker)  │
└────────────────────────────────────────────────────────────┘
```

| Component | Role |
|---|---|
| `Tokenizer` | BPE encode/decode, HuggingFace `tokenizer.json` compatible |
| `LLMEngine` | Top-level generate API, wraps `Transformer` |
| `Transformer` | Prefill/decode loop, embedding lookup, LM head, sampling |
| `Network` | Layer-by-layer forward pass (MHA + SwiGLU + RMSNorm) |

---

## Loading a Model

```go
import (
    "github.com/openfluke/loom/nn"
    "github.com/openfluke/loom/tokenizer"
)

// 1. Load network architecture from safetensors dir
network, err := nn.LoadTransformerFromSafetensors(snapshotDir)

// 2. Load weights
tensors := make(map[string][]float32)
for _, f := range safetensorFiles {
    t, _ := nn.LoadSafetensors(f)
    for k, v := range t { tensors[k] = v }
}

// 3. Map weights to embeddings / lm_head / final norm
mapper := tokenizer.NewWeightMapper()
embeddings, lmHead, finalNorm, _ := mapper.MapWeights(tensors)

// 4. Load BPE tokenizer
tk, _ := tokenizer.LoadFromFile(tokenizerPath)

// 5. Create the engine
engine := tokenizer.NewLLMEngine(network, embeddings, lmHead, finalNorm, tokenizer.ChatML)
```

---

## GPU Mounting

Before inference, mount weights to GPU memory. Loom handles all WebGPU buffer allocation and shader compilation automatically.

```go
network.BatchSize = 1
for i := range network.Layers {
    network.Layers[i].SeqLength = maxSeqLen  // e.g. 512
}
network.GPU = true
network.GPUInferenceOnly = true    // Skip backward/gradient buffers
network.EnableGPUResiduals = true  // Enable skip-connections on GPU

if err := network.WeightsToGPU(); err != nil {
    // Graceful fallback
    network.GPU = false
}
```

### Adapter Preference

On systems with multiple GPUs or software adapters (e.g. WARP on Windows), prefer the discrete GPU:

```go
gpu.SetAdapterPreference("nvidia")
```

---

## The Prefill / Decode Loop

Generation runs in two phases:

### Phase 1: Prefill (prompt processing)

The entire prompt is embedded and forwarded in one shot. Only the last hidden state is retained for generation.

```
Prompt tokens: [tok₀, tok₁, ..., tokₙ]
      │
      ▼ Embed each token
allEmbeds = [embed₀ | embed₁ | ... | embedₙ]  (flat, seqLen × hiddenSize)
      │
      ▼ ForwardTransformer(allEmbeds, pos=0)
      │   MHA processes full sequence, fills KV cache positions 0..n
      │
hidden = last hiddenSize floats of output
```

### Phase 2: Decode (autoregressive generation)

Each step takes one token's embedding and returns the next hidden state, using the KV cache to avoid recomputing past context.

```
For each new token:
  embed = getEmbedding(lastToken)       // hiddenSize floats
  hidden, _ = ForwardTransformer(embed, len(tokens)-1)
             │   MHA reads KV cache positions 0..pos-1
             │   Computes attention only for current token
             │   Writes K/V for position pos to cache
  logits = applyLMHead(hidden)          // vocab-size dot product
  nextToken = SampleTopK(logits, topK, temperature, deterministic)
```

Without KV cache, the full context is recomputed every step — much slower but numerically identical output:

```go
opts := tokenizer.GenOptions{
    UseKVCache: false,  // Recompute full context each step
}
```

---

## Generation Options

```go
opts := tokenizer.GenOptions{
    MaxTokens:         200,
    Temperature:       0.9,     // 0 = deterministic greedy, >0 = sampling
    TopK:              40,      // Top-K candidates for sampling
    Deterministic:     true,    // Force TopK=1, Temperature=0
    UseKVCache:        true,    // Enable KV cache (recommended for GPU)
    RepetitionPenalty: 1.15,    // Penalise recently seen tokens
    RepetitionWindow:  64,      // How many past tokens to penalise
    EOSTokens:         []int{2, 0},  // Stop on these token IDs
}

reply := engine.Generate(tk, chatTurns, systemPrompt, userMsg, opts)
```

| Option | Effect |
|---|---|
| `Temperature=0` / `Deterministic=true` | Greedy decoding — reproducible output |
| `Temperature=0.9, TopK=40` | Creative sampling with top-40 candidates |
| `RepetitionPenalty=1.15` | Divides/multiplies logits of recently seen tokens |
| `UseKVCache=true` | ~10–30x faster decode on GPU |

---

## Chat Templates

The `tokenizer` package supports prompt templates. Currently implemented:

| Template | Models |
|---|---|
| `tokenizer.ChatML` | Qwen, SmolLM, many instruction-tuned models |

```go
// ChatML wraps messages as:
// <|im_start|>system\n{systemPrompt}<|im_end|>
// <|im_start|>user\n{userMsg}<|im_end|>
// <|im_start|>assistant\n

engine := tokenizer.NewLLMEngine(network, embeddings, lmHead, finalNorm, tokenizer.ChatML)
```

Multi-turn conversation is managed by passing accumulated `[]tokenizer.Turn`:

```go
chatTurns = append(chatTurns, tokenizer.Turn{
    User:      userMsg,
    Assistant: reply,
})
```

---

## KV Cache Internals

Each `MHALayer` on GPU maintains two persistent buffers:

```
KCacheBuffer: [maxSeq × D_KV × 4 bytes]  — Key vectors per position
VCacheBuffer: [maxSeq × D_KV × 4 bytes]  — Value vectors per position
CachePos:     int                         — Next write position
```

During the QKV shader:
1. Projects input → Q, K, V
2. Applies RoPE to Q and K (position-dependent rotation)
3. Writes K and V at `cache_pos + seq` into the cache buffers

During the attention shader:
- Reads K and V from `k_cache` / `v_cache` for all positions `0..total_len`
- Applies causal mask: position `seq_j` is ignored if `seq_j > cache_pos + seq_i`
- Supports Grouped Query Attention (GQA): multiple Q heads share KV heads

```go
// GQA mapping in shader:
// heads_per_kv = NUM_HEADS / NUM_KV_HEADS
// kv_head = query_head / heads_per_kv
```

---

## Performance

Benchmarks on SmolLM2-135M (FP32, GPU):

| Configuration | Tokens/s |
|---|---|
| CPU only | ~2–5 |
| GPU, no KV cache | ~10 |
| GPU + KV cache | **~30** |

The KV cache eliminates redundant attention computation over the full context on every decode step, which is the dominant cost at longer sequence lengths.

---

## Supported Model Architectures

Models successfully loaded and run through loom's transformer path:

| Model | Architecture | Notes |
|---|---|---|
| SmolLM2-135M / 360M / 1.7B | Llama-style (GQA, SwiGLU, RMSNorm) | ✅ |
| Qwen2.5-0.5B / 1.5B | Qwen2 (GQA, SwiGLU, RMSNorm) | ✅ |
| TinyLlama-1.1B | Llama-style | ✅ |
| SmolLM3-3B | Llama-style | ✅ |

Architecture is loaded by parsing `config.json` from the HuggingFace snapshot directory. Supported `model_type` values: `qwen2`, `llama`, and compatible variants.
