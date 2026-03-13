#!/bin/bash
# Simple HTTP server with CORS for WASM demo

PORT=${1:-8080}

echo "=========================================="
echo "  LOOM WASM Server (CORS Enabled)"
echo "=========================================="
echo ""
echo "Starting HTTP server on port $PORT..."
echo ""
echo "Available demos:"
echo "  http://localhost:$PORT/example.html"
echo "  http://localhost:$PORT/all_layers_test.html"
echo ""
echo "Press Ctrl+C to stop"
echo "=========================================="
echo ""

# Use Python with CORS support
if command -v python3 &> /dev/null; then
    python3 -c "
import http.server
import socketserver

class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

PORT = $PORT
with socketserver.TCPServer(('', PORT), CORSHTTPRequestHandler) as httpd:
    print(f'Serving HTTP on 0.0.0.0 port {PORT} (http://0.0.0.0:{PORT}/) ...')
    httpd.serve_forever()
"
else
    echo "Error: Python 3 not found. Please install Python 3 to use this server."
    exit 1
fi
