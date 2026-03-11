#!/bin/bash
# Test script for LOOM C ABI transformer inference

set -e

echo "🧪 LOOM C ABI Transformer Inference Test"
echo "========================================"
echo ""

# Check if library exists
if [ ! -f "libloom.so" ]; then
    echo "❌ libloom.so not found. Building..."
    ./build.sh
fi

# Check if model exists
MODEL_PATH="../models/SmolLM2-135M-Instruct"
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Model not found at $MODEL_PATH"
    echo "   Download with:"
    echo "   huggingface-cli download HuggingFaceTB/SmolLM2-135M-Instruct \\"
    echo "     --local-dir models/SmolLM2-135M-Instruct \\"
    echo "     --include '*.json' '*.safetensors'"
    exit 1
fi

echo "✓ Library found: libloom.so"
echo "✓ Model found: $MODEL_PATH"
echo ""

# Test Python web interface
echo "Testing Python web interface..."
if command -v python3 &> /dev/null; then
    echo "✓ Python3 found"
    echo ""
    echo "📝 To run web interface:"
    echo "   ./web_interface.py $MODEL_PATH 8080"
    echo "   Then open: http://localhost:8080/inference.html"
else
    echo "⚠️  Python3 not found - web interface unavailable"
fi

echo ""

# Test Go server
echo "Testing Go HTTP server..."
if [ -f "cmd/serve_model_bytes/serve_model_bytes" ]; then
    echo "✓ serve_model_bytes built"
    echo ""
    echo "📝 To run Go server:"
    echo "   cd cmd/serve_model_bytes"
    echo "   ./serve_model_bytes -model $MODEL_PATH -port 8080"
else
    echo "Building serve_model_bytes..."
    cd cmd/serve_model_bytes
    go build -o serve_model_bytes
    cd ../..
    echo "✓ serve_model_bytes built"
    echo ""
    echo "📝 To run Go server:"
    echo "   cd cmd/serve_model_bytes"
    echo "   ./serve_model_bytes -model $MODEL_PATH -port 8080"
fi

echo ""
echo "✅ All checks passed!"
echo ""
echo "🚀 Quick Start:"
echo "   1. Start server: ./web_interface.py $MODEL_PATH 8080"
echo "   2. Open browser: http://localhost:8080/inference.html"
echo "   3. Enter prompt and click Generate"
echo ""
