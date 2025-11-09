#!/bin/bash
# serve_wasm.sh - Serve LOOM WASM inference with CORS enabled

PORT=8888

echo "🚀 LOOM WASM Inference Server"
echo "=============================="
echo ""
echo "Building standard 32-bit WASM module with optimizations..."
GOOS=js GOARCH=wasm go build -ldflags="-s -w" -o loom.wasm
echo "✓ Standard WASM built: loom.wasm (4GB limit)"
MEMORY_MODE="Standard (4GB limit, all browsers)"
echo ""
echo "Copying wasm_exec.js..."
cp "$(go env GOROOT)/misc/wasm/wasm_exec.js" .
echo "✓ wasm_exec.js copied"
echo ""
echo "Starting HTTP server with CORS on port $PORT..."
echo "Open: http://localhost:$PORT/inference.html"
echo ""
echo "📊 Build mode: $MEMORY_MODE"
echo ""
echo "📦 Loading models from local files:"
echo "  ✅ SmolLM2-135M-Instruct (default, ~260MB)"
echo ""
echo "⚠️  Standard WASM (4GB limit) - Recommended models:"
echo "  ✅ SmolLM2-135M: ~1GB (BEST for WASM)"
echo "  ⚠️  SmolLM2-360M: ~2GB (may OOM)"
echo ""
echo "Note: Models loaded from ../models/ directory"
echo ""

# Start Python HTTP server with CORS
python3 -c "
import http.server
import socketserver

class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()

PORT = $PORT
with socketserver.TCPServer(('', PORT), CORSHTTPRequestHandler) as httpd:
    print(f'Server running at http://localhost:{PORT}/')
    httpd.serve_forever()
"
