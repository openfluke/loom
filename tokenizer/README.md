# Loom Tokenizer & Universal LLM Framework

The `tokenizer` package is the core intelligence layer of the Loom framework. It provides a modular, extensible, and high-performance system for Large Language Model (LLM) inference, encompassing everything from raw byte-level tokenization to complex autoregressive generation with Key-Value (KV) caching.

## Core Architecture

The package is designed with a "pluggable" philosophy, allowing different model architectures (Llama, Qwen, SmolLM, etc.) to reuse the same inference engine by providing their own layer handlers and templates.

---

## 1. BPE Tokenization (`bpe.go`)
At the foundation is a sophisticated **Byte-Pair Encoding (BPE)** implementation.

- **Byte-Fallibility**: Handles UTF-8 characters by falling back to byte-level tokens when a direct mapping isn't found, ensuring no text is ever "un-encodable".
- **Special Token Management**: Supports a variety of control tokens (e.g., `<|im_start|>`, `<|endoftext|>`) which are essential for structural prompting.
- **Performance**: Uses optimized lookup tables for the `Vocab` and `Merges` to ensure tokenization is never a bottleneck during inference.

---

## 2. Universal LLM Engine (`engine.go`)
The `LLMEngine` is the orchestrator that brings neural networks and tokenization together.

### Generation Logic
- **`Generate()`**: The main entry point. It takes a conversation history (turns), formats it via a template, encodes it, and runs the autoregressive loop.
- **Autoregressive Loop**: Continually predicts the next token, appends it to the sequence, and feeds it back into the model until an EOS (End Of Sentence) token or a user-defined stop sequence is encountered.
- **Inference Paths**:
    - **Standard Path**: Re-evaluates the entire prompt for every new token. Useful for testing or when memory is extremely constrained.
    - **KV-Cached Path**: Uses the `KVCacheState` to store previously computed Key and Value tensors, reducing computation from $O(N^2)$ to $O(N)$ for token generation.

### Generation Options (`GenOptions`)
- **Temperature**: Controls randomness (lower = more focused, higher = more creative).
- **Top-K**: Limits the sampling pool to the top $K$ most likely tokens.
- **Repetition Penalty**: penalizes tokens that have appeared recently to prevent loops.
- **Deterministic**: A hard toggle that forces greedy sampling (always picking the highest probability token), ensuring bit-exact reproducibility.

---

## 3. Dynamic KV Cache Registry (`kvcache.go`)
This is where Loom achieves "Universality".

### `LayerInference` Interface
Instead of hardcoding how a "Transformer Layer" works, Loom uses an interface:
```go
type LayerInference interface {
    ForwardKV(input []float32, cfg *nn.LayerConfig, pos int, state *KVCacheLayer) ([]float32, error)
}
```
### Global Registry
Handlers are registered for specific `nn.LayerType`s.
- **Default Multi-Head Attention (MHA)**: Handles Query/Key/Value projections, **Rotary Position Embeddings (RoPE)**, and scaled dot-product attention with KV cache updates.
- **Default SwiGLU**: Handles the gated feed-forward activation logic.
Users can register custom handlers to support MoE (Mixture of Experts) or other novel architectures without modifying the core package.

---

## 4. Dynamic Weight Mapper (`mapping.go`)
Loading weights from a `.safetensors` file is often difficult because every model uses different naming conventions (e.g., `model.embed_tokens` vs `transformer.wte`).

- **`WeightMapper`**: Uses a pattern-based search. You define a "Role" (like `embeddings`) and a list of patterns to look for.
- **Matching Priority**: It tries exact matches first, followed by suffix matching (e.g., if a model names its head `layers.31.lm_head.weight`, a suffix pattern `lm_head.weight` will find it).
- **Tied Weights**: Intelligently handles models that share weights between the embedding layer and the output head.

---

## 5. Flexible Chat Templates (`templates.go`)
Modern models are highly sensitive to how prompts are formatted.

- **`Template` Struct**: Defines the exact markers used to denote different roles (`system`, `user`, `assistant`).
- **`BuildPrompt()`**: Automatically handles the insertion of "Turn" markers, system instructions, and the final generation trigger.
- **Presets**:
    - **`ChatML`**: Standard `<|im_start|>role\ncontent<|im_end|>\n` used by Qwen and SmolLM.
    - **`Llama3`**: Uses the `<|start_header_id|>` and `<|eot_id|>` markers specifically tuned for Meta's models.

---

## 6. Determinism & Sampling
Loom prioritizes predictability.

- **Greedy Path**: When `Deterministic` is set to true, the engine ignores temperature and Top-K settings, selecting the absolute maximum logit. This is critical for debugging and production stability.
- **PRNG Isolation**: While the global `rand` seed is used for stochastic paths, the core logic is designed to be side-effect free, allowing for multiple independent engines to run in parallel without interfering with each other's randomness.

---

## Technical Flow Diagram

1. **User Input** -> `Template.BuildPrompt()` -> **Raw String**
2. **Raw String** -> `Tokenizer.Encode()` -> **Token IDs**
3. **Token IDs** -> `LLMEngine.Prefill()` -> **Initial Hidden State**
4. **Hidden State** -> `LLMEngine.SampleNextToken()` -> **Predicted ID**
5. **Predicted ID** -> `Streamer` -> **Standard Out**
6. **(Loop)** Predicted ID -> `ForwardTokenKV()` (using Cache) -> **New Hidden State** -> **(Back to Step 4)**
