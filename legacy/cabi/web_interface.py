#!/usr/bin/env python3
"""
LOOM C ABI Web Interface - Transformer Inference
Serves model files via HTTP and provides a web UI for text generation
"""

from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import json
import os
import sys
import ctypes
from pathlib import Path

# Load LOOM C ABI
lib_path = Path(__file__).parent / "libloom.so"
if not lib_path.exists():
    print(f"Error: {lib_path} not found. Run ./build.sh first!")
    sys.exit(1)

loom = ctypes.CDLL(str(lib_path))

# Define function signatures
# Use c_void_p for return types to avoid Python's automatic string conversion
# which would invalidate the pointer before we can free it
loom.LoadTokenizerFromBytes.argtypes = [ctypes.c_char_p, ctypes.c_int]
loom.LoadTokenizerFromBytes.restype = ctypes.c_void_p

loom.LoadTransformerFromBytes.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
loom.LoadTransformerFromBytes.restype = ctypes.c_void_p

loom.GenerateText.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_float]
loom.GenerateText.restype = ctypes.c_void_p

loom.GenerateNextToken.argtypes = [ctypes.c_char_p, ctypes.c_float]
loom.GenerateNextToken.restype = ctypes.c_void_p

loom.EncodeText.argtypes = [ctypes.c_char_p, ctypes.c_bool]
loom.EncodeText.restype = ctypes.c_void_p

loom.DecodeTokens.argtypes = [ctypes.c_char_p, ctypes.c_bool]
loom.DecodeTokens.restype = ctypes.c_void_p

loom.Loom_FreeCString.argtypes = [ctypes.c_void_p]
loom.Loom_FreeCString.restype = None

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
                'backend': 'loom-cabi-python'
            }
            self.wfile.write(json.dumps(response).encode())
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
                
                # Encode the prompt
                encode_ptr = loom.EncodeText(prompt.encode('utf-8'), True)
                encode_json = ctypes.string_at(encode_ptr).decode('utf-8')
                loom.Loom_FreeCString(encode_ptr)
                encode_result = json.loads(encode_json)
                
                if not encode_result.get('success'):
                    error_msg = f"data: {json.dumps({'error': encode_result.get('error', 'Encoding failed')})}\n\n"
                    self.wfile.write(error_msg.encode('utf-8'))
                    return
                
                tokens = encode_result['ids']
                generated_tokens = []
                
                # Generate tokens one at a time
                for i in range(max_tokens):
                    # Generate next token
                    token_json = json.dumps(tokens)
                    gen_ptr = loom.GenerateNextToken(token_json.encode('utf-8'), ctypes.c_float(temperature))
                    gen_json = ctypes.string_at(gen_ptr).decode('utf-8')
                    loom.Loom_FreeCString(gen_ptr)
                    gen_result = json.loads(gen_json)
                    
                    if not gen_result.get('success'):
                        error_msg = f"data: {json.dumps({'error': gen_result.get('error', 'Generation failed')})}\n\n"
                        self.wfile.write(error_msg.encode('utf-8'))
                        break
                    
                    next_token = gen_result['token']  # Fixed: was 'next_token', should be 'token'
                    tokens.append(next_token)
                    generated_tokens.append(next_token)
                    
                    # Decode just the new token to get the text
                    decode_ptr = loom.DecodeTokens(json.dumps([next_token]).encode('utf-8'), True)
                    decode_json = ctypes.string_at(decode_ptr).decode('utf-8')
                    loom.Loom_FreeCString(decode_ptr)
                    decode_result = json.loads(decode_json)
                    
                    if decode_result.get('success'):
                        token_text = decode_result['text']
                        
                        # Send the token as SSE
                        event_data = {
                            'token': token_text,
                            'token_id': next_token,
                            'index': i,
                            'is_eos': gen_result.get('is_eos', False)
                        }
                        self.wfile.write(f"data: {json.dumps(event_data)}\n\n".encode('utf-8'))
                        self.wfile.flush()
                        
                        # Check for EOS
                        if gen_result.get('is_eos', False):
                            break
                
                # Send completion event
                self.wfile.write(f"data: {json.dumps({'done': True})}\n\n".encode('utf-8'))
                self.wfile.flush()
                
            except Exception as e:
                error_msg = f"data: {json.dumps({'error': str(e)})}\n\n"
                self.wfile.write(error_msg.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

def load_model(model_dir):
    """Load model files into C ABI"""
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
    
    result_ptr = loom.LoadTokenizerFromBytes(tok_data, len(tok_data))
    result_json = ctypes.string_at(result_ptr).decode('utf-8')
    loom.Loom_FreeCString(result_ptr)
    result = json.loads(result_json)
    
    if not result.get('success'):
        raise Exception(f"Failed to load tokenizer: {result.get('error')}")
    print(f"✓ Tokenizer loaded (vocab: {result['vocab_size']})")
    
    # Load transformer
    print("Loading transformer model...")
    with open(config_path, 'rb') as f:
        config_data = f.read()
    with open(weights_path, 'rb') as f:
        weights_data = f.read()
    
    print(f"  Config: {len(config_data)} bytes")
    print(f"  Weights: {len(weights_data) / (1024*1024):.2f} MB")
    
    result_ptr = loom.LoadTransformerFromBytes(
        config_data, len(config_data),
        weights_data, len(weights_data)
    )
    result_json = ctypes.string_at(result_ptr).decode('utf-8')
    loom.Loom_FreeCString(result_ptr)
    result = json.loads(result_json)
    
    if not result.get('success'):
        raise Exception(f"Failed to load transformer: {result.get('error')}")
    
    print(f"✓ Transformer loaded")
    print(f"  Vocab: {result['vocab_size']}")
    print(f"  Hidden: {result['hidden_size']}")
    print(f"  Layers: {result['num_layers']}")
    
    model_loaded = True

def main():
    if len(sys.argv) < 2:
        print("Usage: ./web_interface.py <model_path> [port]")
        print("Example: ./web_interface.py ../models/SmolLM2-135M-Instruct 8080")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080
    
    # Load model
    try:
        load_model(model_dir)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Start server
    print(f"\n🚀 LOOM C ABI Web Interface")
    print(f"   Model: {os.path.basename(model_dir)}")
    print(f"   Server: http://localhost:{port}")
    print(f"\nPress Ctrl+C to stop\n")
    
    server = HTTPServer(('', port), CORSHTTPRequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()

if __name__ == '__main__':
    main()
