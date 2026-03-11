# Tokenizer Package

The `tokenizer` package provides pure-Go BPE tokenization, chat prompt templating, weight mapping, and the top-level generation engine that ties a loom `Network` to an LLM inference loop.

```
github.com/openfluke/loom/tokenizer
```

---

## Files

| File | Purpose |
|---|---|
| `bpe.go` | BPE tokenizer — encode/decode, `tokenizer.json` loading |
| `transformer.go` | `Transformer`, `Streamer`, `SampleTopK`, generation loop |
| `engine.go` | `LLMEngine` — top-level API wrapping `Transformer` |
| `mapping.go` | `WeightMapper` — maps safetensors keys to embeddings/lm_head/norm |
| `templates.go` | Chat prompt templates (ChatML etc.) |

---

## Tokenizer

### Loading

```go
tk, err := tokenizer.LoadFromFile("/path/to/tokenizer.json")
```

Loads a HuggingFace `tokenizer.json` (BPE format). Compatible with Qwen, SmolLM, LLaMA, GPT-2, Mistral tokenizers.

### Encode / Decode

```go
// Encode text to token IDs
ids := tk.Encode("Hello, world!", false)  // []uint32

// Decode token IDs back to text
text := tk.Decode(ids, false)             // string

// Vocab size
size := tk.VocabSize()  // e.g. 151643 for Qwen
```

### EOS Tokens

Load from `config.json` — handles both single int and array formats:

```go
// Automatic loading via loadEOSTokens(configPath)
// Supports: "eos_token_id": 2  OR  "eos_token_id": [2, 151643]
// Falls back to [2, 0] if not found
```

---

## Weight Mapper

Maps raw safetensors weight names to the three components needed for inference:

```go
mapper := tokenizer.NewWeightMapper()
embeddings, lmHead, finalNorm, err := mapper.MapWeights(tensors)
```

| Output | What it is | Typical name in safetensors |
|---|---|---|
| `embeddings` | Token embedding table `[vocab × hidden]` | `model.embed_tokens.weight` |
| `lmHead` | Language model head `[vocab × hidden]` | `lm_head.weight` |
| `finalNorm` | Final RMSNorm gamma `[hidden]` | `model.norm.weight` |

---

## LLMEngine

The main entry point for inference. Wraps a `Transformer` with a clean `Generate` API.

```go
engine := tokenizer.NewLLMEngine(network, embeddings, lmHead, finalNorm, tokenizer.ChatML)

reply := engine.Generate(tk, chatTurns, systemPrompt, userMsg, opts)
```

### GenOptions

```go
opts := tokenizer.GenOptions{
    MaxTokens:         200,
    Temperature:       0.9,
    TopK:              40,
    Deterministic:     false,
    UseKVCache:        true,
    RepetitionPenalty: 1.15,
    RepetitionWindow:  64,
    EOSTokens:         []int{151643, 2},
}
```

| Field | Type | Description |
|---|---|---|
| `MaxTokens` | `int` | Maximum tokens to generate |
| `Temperature` | `float32` | Sampling temperature. `0` = greedy |
| `TopK` | `int` | Top-K candidates. `1` = greedy |
| `Deterministic` | `bool` | Forces `Temperature=0, TopK=1` |
| `UseKVCache` | `bool` | Enable GPU KV cache for fast decode |
| `RepetitionPenalty` | `float32` | Penalty multiplier for repeated tokens (e.g. `1.15`) |
| `RepetitionWindow` | `int` | How many past tokens to check (e.g. `64`) |
| `EOSTokens` | `[]int` | Stop generation on these token IDs |

---

## Sampling

### SampleTopK

```go
nextToken := tokenizer.SampleTopK(logits, topK, temperature, deterministic)
```

Algorithm:
1. If `topK==1` or `temperature<=0`: return argmax (greedy)
2. Otherwise: divide all logits by temperature, sort descending, keep top-K
3. Apply softmax with numerical stability (subtract max before exp)
4. Sample from the resulting distribution

### Repetition Penalty

Applied to logits before sampling:
- For each token seen in the last `RepetitionWindow` tokens:
  - If logit > 0: divide by penalty (reduce probability)
  - If logit < 0: multiply by penalty (push further negative)

---

## Streaming

The `Streamer` handles real-time output — printing tokens as they are generated, not waiting for the full reply.

```go
stream := tokenizer.NewStreamer(tk, promptTokens)

// In the generate loop:
stream.Push(allTokensGenerated)  // Prints new tokens to stdout as they arrive

// Get full reply string at end:
reply := stream.String()
```

The streamer also detects if the model starts generating a new user turn (hallucinated `<|im_start|>user`) and signals early stop.

---

## Chat Templates

### ChatML (default)

Used by Qwen, SmolLM2, and many instruction-tuned models:

```
<|im_start|>system
{systemPrompt}
<|im_end|>
<|im_start|>user
{userMessage}
<|im_end|>
<|im_start|>assistant
```

### Multi-turn

```go
var chatTurns []tokenizer.Turn

// After each exchange:
chatTurns = append(chatTurns, tokenizer.Turn{
    User:      userMsg,
    Assistant: reply,
})

// Next call automatically includes history
reply = engine.Generate(tk, chatTurns, systemPrompt, nextUserMsg, opts)
```

---

## Complete Example

```go
package main

import (
    "fmt"
    "github.com/openfluke/loom/gpu"
    "github.com/openfluke/loom/nn"
    "github.com/openfluke/loom/tokenizer"
)

func main() {
    snapshotDir := "/path/to/model/snapshot"

    // Load network structure
    network, _ := nn.LoadTransformerFromSafetensors(snapshotDir)

    // Load weights
    tensors := map[string][]float32{}
    // ... load from safetensors files ...

    // Map weight components
    mapper := tokenizer.NewWeightMapper()
    embeddings, lmHead, finalNorm, _ := mapper.MapWeights(tensors)

    // Load tokenizer
    tk, _ := tokenizer.LoadFromFile(snapshotDir + "/tokenizer.json")

    // Mount to GPU
    gpu.SetAdapterPreference("nvidia")
    network.BatchSize = 1
    for i := range network.Layers {
        network.Layers[i].SeqLength = 512
    }
    network.GPU = true
    network.GPUInferenceOnly = true
    network.EnableGPUResiduals = true
    network.WeightsToGPU()

    // Create engine
    engine := tokenizer.NewLLMEngine(network, embeddings, lmHead, finalNorm, tokenizer.ChatML)

    // Generate
    opts := tokenizer.GenOptions{
        MaxTokens:         200,
        Temperature:       0,
        TopK:              1,
        Deterministic:     true,
        UseKVCache:        true,
        RepetitionPenalty: 1.15,
        RepetitionWindow:  64,
        EOSTokens:         []int{2},
    }

    fmt.Print("Bot: ")
    reply := engine.Generate(tk, nil, "You are a helpful assistant.", "Hello!", opts)
    fmt.Println()
    fmt.Println("Reply:", reply)
}
```
