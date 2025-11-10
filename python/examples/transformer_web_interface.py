#!/usr/bin/env python3
"""
LOOM Python Transformer Web Interface
Similar to cabi/web_interface.py but uses the welvet pip package
"""

from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse
import json
import os
import sys
import welvet

# Global state
model_loaded = False
model_path = None


class CORSHTTPRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()
    
    def do_GET(self):
        parsed = urlparse(self.path)
        
        if parsed.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {
                'status': 'ok',
                'model': os.path.basename(model_path) if model_path else 'none',
                'model_loaded': model_loaded,
                'backend': 'welvet-python'
            }
            self.wfile.write(json.dumps(response).encode())
        elif parsed.path == '/' or parsed.path == '/inference.html':
            # Serve the HTML file
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            with open('inference.html', 'r') as f:
                self.wfile.write(f.read().encode())
        else:
            super().do_GET()
    
    def do_POST(self):
        if self.path == '/generate':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data)
                prompt = data.get('prompt', '')
                max_tokens = data.get('max_tokens', 50)
                temperature = data.get('temperature', 0.7)
                
                if not model_loaded:
                    raise Exception("Model not loaded")
                
                # Set up SSE (Server-Sent Events) for streaming
                self.send_response(200)
                self.send_header('Content-type', 'text/event-stream')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('Connection', 'keep-alive')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                # Stream tokens
                token_count = 0
                for token in welvet.generate_stream(prompt, max_tokens, temperature):
                    event_data = {
                        'token': token,
                        'index': token_count,
                    }
                    self.wfile.write(f"data: {json.dumps(event_data)}\n\n".encode('utf-8'))
                    self.wfile.flush()
                    token_count += 1
                
                # Send completion event
                self.wfile.write(f"data: {json.dumps({'done': True})}\n\n".encode('utf-8'))
                self.wfile.flush()
                
            except Exception as e:
                error_msg = f"data: {json.dumps({'error': str(e)})}\n\n"
                self.wfile.write(error_msg.encode('utf-8'))


def load_model(model_dir):
    """Load transformer model and tokenizer."""
    global model_loaded, model_path
    
    model_path = model_dir
    config_path = os.path.join(model_dir, 'config.json')
    tokenizer_path = os.path.join(model_dir, 'tokenizer.json')
    weights_path = os.path.join(model_dir, 'model.safetensors')
    
    print(f"Loading model from: {model_dir}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    with open(tokenizer_path, 'rb') as f:
        tok_data = f.read()
    
    result = welvet.load_tokenizer_from_bytes(tok_data)
    print(f"✓ Tokenizer loaded (vocab: {result['vocab_size']})")
    
    # Load transformer
    print("Loading transformer model...")
    with open(config_path, 'rb') as f:
        config_data = f.read()
    with open(weights_path, 'rb') as f:
        weights_data = f.read()
    
    print(f"  Config: {len(config_data)} bytes")
    print(f"  Weights: {len(weights_data) / (1024*1024):.2f} MB")
    
    result = welvet.load_transformer_from_bytes(config_data, weights_data)
    print(f"✓ Transformer loaded")
    print(f"  Vocab: {result['vocab_size']}")
    print(f"  Hidden: {result['hidden_size']}")
    print(f"  Layers: {result['num_layers']}")
    
    model_loaded = True


def main():
    if len(sys.argv) < 2:
        print("Usage: ./transformer_web_interface.py <model_path> [port]")
        print("Example: ./transformer_web_interface.py ../../models/SmolLM2-135M-Instruct 8080")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080
    
    # Load model
    try:
        load_model(model_dir)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)
    
    # Start server
    server_address = ('', port)
    httpd = HTTPServer(server_address, CORSHTTPRequestHandler)
    
    print(f"\n🚀 LOOM Python Transformer Web Interface")
    print(f"   Model: {os.path.basename(model_dir)}")
    print(f"   Server: http://localhost:{port}")
    print(f"   Backend: welvet (pip package)")
    print()
    print("Press Ctrl+C to stop")
    print()
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
        httpd.shutdown()


if __name__ == '__main__':
    main()
