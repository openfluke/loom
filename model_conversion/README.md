# LOOM Model Conversion & Inference

This directory contains tools for running LOOM transformer models with a **pure Go stack** (no Python dependencies).

## Quick Start

### 1. Download a model

Using HuggingFace CLI:

```bash
pip install -U huggingface_hub
huggingface-cli download Qwen/Qwen2.5-0.5B
```

Or using Python:

```bash
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B')"
```

The model will be cached in `~/.cache/huggingface/hub/`

### 2. Start the backend server

**Terminal 1:**

```bash
cd model_conversion
go run serve_model_bytes.go -model Qwen/Qwen2.5-0.5B -port 8080
```

### 3. Start the web interface

**Terminal 2:**

```bash
cd model_conversion
go run web_interface.go -model Qwen/Qwen2.5-0.5B -backend http://localhost:8080 -port 5000
```

### 4. Open your browser

Navigate to: **http://localhost:5000**

## Model Servers

### serve_model_bytes.go ⭐ RECOMMENDED

**Pure Go inference server** - Loads entire model into memory using `nn.LoadTransformerFromBytes()`

- Reads config.json and model.safetensors into byte slices
- Uses the nn framework's native transformer loading
- Automatic HuggingFace cache path resolution
- Cleaner code, easier to maintain
- HTTP API with streaming support

```bash
go run serve_model_bytes.go -model Qwen/Qwen2.5-0.5B -port 8080
```

### web_interface.go ⭐ RECOMMENDED

**Pure Go web frontend** - Replaces Python Flask with native Go HTTP server

- Pure Go BPE tokenizer (no Python dependencies)
- Embedded HTML templates
- Real-time streaming generation
- Connects to any backend server

```bash
go run web_interface.go -model Qwen/Qwen2.5-0.5B -backend http://localhost:8080 -port 5000
```

### serve_model_auto.go (Legacy)

Automatically loads models with dynamic EOS token detection

- Reads safetensors directly from disk
- Flexible tensor key matching
- Good for quick testing
- **Note:** Consider using `serve_model_bytes.go` instead

### serve_model_manual.go (Legacy)

Manual model construction with explicit layer-by-layer loading

- Most control over model structure
- Useful for debugging
- **Note:** Consider using `serve_model_bytes.go` instead

### web_interface.py (Deprecated)

Python Flask web interface - replaced by `web_interface.go`

- **Deprecated:** Use pure Go `web_interface.go` instead
- Requires Python + transformers library (2GB+ dependencies)
- Can be removed from your system

## Architecture

### Pure Go Stack 🎯

The recommended setup uses **zero Python dependencies**:

```
┌─────────────────────────────────────────────┐
│         Browser (http://localhost:5000)     │
└─────────────────┬───────────────────────────┘
                  │
         ┌────────▼──────────┐
         │  web_interface.go  │ ← Pure Go BPE Tokenizer
         │   (Port 5000)      │   Embedded HTML templates
         └────────┬───────────┘
                  │ HTTP
         ┌────────▼───────────┐
         │ serve_model_bytes  │ ← nn.LoadTransformerFromBytes
         │   (Port 8080)      │   ForwardCPU inference
         └────────────────────┘
```

**Benefits:**

- ✅ No Python runtime required
- ✅ No transformers library (saves 2GB+)
- ✅ Single binary deployment
- ✅ Consistent performance
- ✅ Easy to cross-compile

## How It Works

1. **Backend Server** (`serve_model_bytes.go`):

   - Resolves model name to HuggingFace cache path
   - Loads config.json and model.safetensors into memory
   - Uses `nn.LoadTransformerFromBytes()` for initialization
   - Provides HTTP API on port 8080
   - Supports streaming generation (Server-Sent Events)
   - Uses `Network.ForwardCPU` for inference

2. **Web Interface** (`web_interface.go`):

   - Pure Go HTTP server on port 5000
   - Loads tokenizer.json into memory
   - Uses pure Go BPE tokenizer (supports GPT-2, LLaMA, Qwen, Mistral, T5, etc.)
   - Tokenizes input prompts
   - Streams tokens in real-time from backend
   - Serves embedded HTML templates

3. **Streaming Flow**:
   ```
   Browser → Go Web → Go Backend → Go Web → Browser
   (EventSource)   (HTTP+SSE)         (SSE)
   ```

## API Endpoints

### Backend Server (Port 8080)

**POST /generate** - Generate text

Request:

```json
{
  "input_ids": [12522, 5193, 264, 882],
  "max_new_tokens": 50,
  "temperature": 0.7,
  "stream": true // Enable streaming
}
```

Response (streaming):

```
data: {"token": 151643, "done": false}
data: {"token": 8256, "done": false}
data: {"done": true}
```

Response (non-streaming):

```json
{
  "output_ids": [12522, 5193, 264, 882, 151643, 8256, ...]
}
```

**GET /health** - Health check

Response:

```json
{
  "status": "ok",
  "model": "Qwen/Qwen2.5-0.5B",
  "vocab_size": 151936,
  "hidden_size": 896
}
```

### Web Interface (Port 5000)

**GET /** - Web UI (HTML page)

**POST /generate** - Generate text (non-streaming)

Request:

```json
{
  "prompt": "Once upon a time",
  "max_tokens": 50
}
```

Response:

```json
{
  "generated_text": "Once upon a time, there was a...",
  "num_tokens": 15
}
```

**POST /generate_stream** - Stream generation (Server-Sent Events)

**GET /health** - Health check

Response:

```json
{
  "web_interface": "ok",
  "backend": "ok",
  "model": "Qwen/Qwen2.5-0.5B",
  "tokenizer": "Pure Go BPE (vocab: 151936)"
}
```

## Requirements

### Pure Go Stack (Recommended)

**No Python required!** Just Go 1.21+:

```bash
go version  # Should be 1.21 or higher
```

### Legacy Python Stack (Deprecated)

If you still want to use `web_interface.py`:

```bash
pip install flask transformers requests
```

**Note:** The Python stack adds ~2GB of dependencies and is no longer maintained.

## Troubleshooting

**Backend won't start:**

- Verify model is downloaded: `ls ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/`
- Check for errors in terminal output
- Ensure port 8080 is available

**Web interface can't connect to backend:**

- Verify backend is running: `curl http://localhost:8080/health`
- Check backend URL matches: `-backend http://localhost:8080`
- Ensure firewall allows local connections

**Model not found:**

- Download the model first (see Quick Start step 1)
- Or specify full path: `-model ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/xxx`

**Tokenizer errors:**

- Ensure model has `tokenizer.json` file
- Check model is compatible (GPT-2 style BPE, not WordPiece/Unigram)

**Slow generation:**

- Normal for CPU inference (~1-3 tokens/sec on small models)
- Larger models (7B+) are very slow on CPU
- GPU acceleration coming soon

**Garbled text output:**

- Should now be fixed! Text is cleaned automatically
- Supports both `Ġ` (GPT-2, LLaMA, Qwen) and `▁` (T5, SentencePiece) space encoding

## Technical Details

### Pure Go BPE Tokenizer

Located in `/tokenizer/bpe.go`, this is a from-scratch implementation supporting:

- **HuggingFace tokenizer.json format**
- **BPE merging algorithm** with proper pair ranking
- **Pre-tokenization** using regex patterns
- **Byte fallback** for unknown characters
- **Special tokens** handling
- **Universal space encoding:**
  - `Ġ` (U+0120) → space (GPT-2, GPT-3, GPT-4, LLaMA, Qwen, Mistral, Phi)
  - `▁` (U+2581) → space (T5, XLNet, SentencePiece)
  - `<0xHH>` → byte tokens

**Supports models:**

- ✅ GPT-2, GPT-3, GPT-4
- ✅ LLaMA, LLaMA 2, LLaMA 3
- ✅ Qwen, Qwen 2, Qwen 2.5
- ✅ Mistral, Mixtral
- ✅ Phi, Phi-2, Phi-3
- ✅ CodeLlama, StarCoder
- ✅ T5, mT5 (with ▁ encoding)

### Bytes-Based Loading Pattern

All loaders follow a consistent pattern for flexibility:

```go
// Tokenizer
tokenizer.LoadFromBytes(data []byte)
tokenizer.LoadFromFile(path string)  // wrapper

// Safetensors
nn.LoadSafetensorsFromBytes(data []byte)
nn.LoadSafetensors(path string)  // wrapper

// Transformer
nn.LoadTransformerFromBytes(configData, weightsData []byte)
nn.LoadTransformerFromSafetensors(modelDir string)  // wrapper
```

**Benefits:**

- Enables embedding models in binaries
- Supports streaming from network
- Easier testing with in-memory data
- Clean separation of I/O and parsing

### Fixed SwiGLU Bug

The original implementation had incorrect weight indexing for transposed weight matrices. The fix changes from row-major to column-major indexing:

**Before (wrong):**

```go
gateWeights[i*inputSize+j]  // Row-major for [intermediate][input]
```

**After (correct):**

```go
gateWeights[j*intermediateSize+i]  // Column-major for transposed [input][intermediate]
```

This affects Gate, Up, and Down projections in the SwiGLU MLP layers.

### Model Path Resolution

Short model names are automatically resolved to HuggingFace cache:

```go
"Qwen/Qwen2.5-0.5B" → "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/xxx/"
```

Full paths also work:

```go
"~/models/Qwen2.5-0.5B" → "/home/user/models/Qwen2.5-0.5B/"
```

### Validation

Use `trace_all_layers_loom.go` to validate layer outputs against PyTorch:

```bash
cd model_conversion
go run trace_all_layers_loom.go
```

Should show all layers matching within 1e-5 tolerance.
