# LOOM Model Conversion & Inference

This directory contains tools for running LOOM transformer models with a web interface.

## Model Servers

### serve_model_bytes.go ⭐ NEW

**Best for production use** - Loads entire model into memory using `nn.LoadTransformerFromBytes()`

- Reads config.json and model.safetensors into byte slices
- Uses the nn framework's native transformer loading
- Cleaner code, easier to maintain
- HTTP API compatible with other servers

```bash
go build -o serve_model_bytes serve_model_bytes.go
./serve_model_bytes -model ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/xxx -port 8080
```

### serve_model_auto.go

Automatically loads models with dynamic EOS token detection

- Reads safetensors directly from disk
- Flexible tensor key matching
- Good for quick testing

### serve_model_manual.go

Manual model construction with explicit layer-by-layer loading

- Most control over model structure
- Useful for debugging

## Quick Start

### 1. Download the model:

```bash
huggingface-cli download Qwen/Qwen2.5-0.5B
```

### 2. Start the servers:

**Terminal 1 - Start backend:**

```bash
cd model_conversion
go run serve_model_auto.go -model Qwen/Qwen2.5-0.5B
```

**Terminal 2 - Start web interface:**

```bash
cd model_conversion
python3 web_interface.py
```

Then open: **http://localhost:5000**

## Files

### Production Files (keep these):

- **`serve_model_auto.go`** - Production inference server using Network.ForwardCPU (with fixed SwiGLU)
- **`serve_model_manual.go`** - Reference implementation with manual layer processing
- **`trace_all_layers_loom.go`** - Layer-by-layer validation tool
- **`web_interface.py`** - Flask web server with streaming support
- **`templates/index.html`** - Web UI with real-time token streaming

### How It Works

1. **Backend Server** (`serve_model_auto.go`):

   - Loads model from HuggingFace safetensors
   - Provides HTTP API on port 8080
   - Supports streaming generation (Server-Sent Events)
   - Uses fixed `Network.ForwardCPU` with corrected SwiGLU weight indexing

2. **Web Interface** (`web_interface.py` + `templates/index.html`):

   - Flask server on port 5000
   - Tokenizes input using HuggingFace transformers
   - Streams tokens in real-time from backend
   - Beautiful gradient UI with live stats

3. **Streaming Flow**:
   ```
   Browser → Flask → Backend (Go) → Flask → Browser
   (EventSource)   (HTTP+SSE)         (SSE)
   ```

## API Endpoints

### Backend (Port 8080)

**POST /generate** - Generate text

```json
{
  "input_ids": [12522, 5193, 264, 882],
  "max_new_tokens": 50,
  "stream": true // Enable streaming
}
```

**GET /health** - Health check

### Web Interface (Port 5000)

**GET /** - Web UI  
**POST /generate_stream** - Stream generation (used by web UI)  
**GET /health** - Health check

## Requirements

**Python packages:**

```bash
pip install flask transformers requests
```

## Troubleshooting

**Backend won't start:**

- Check `/tmp/serve_auto.log` for errors
- Ensure model exists in HuggingFace cache

**Streaming not working:**

- Check browser console for errors
- Verify backend is running: `curl http://localhost:8080/health`

**Slow generation:**

- Normal for CPU inference (~1-3 tokens/sec on small models)
- GPU acceleration coming soon

## Technical Details

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

### Validation

Use `trace_all_layers_loom.go` to validate layer outputs against PyTorch:

```bash
cd model_conversion
go run trace_all_layers_loom.go
```

Should show all layers matching within 1e-5 tolerance.
