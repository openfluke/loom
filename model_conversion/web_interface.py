#!/usr/bin/env python3
"""
Web Interface for LOOM Transformer Auto Server
Provides a web UI for text generation using the fixed serve_model_auto backend
"""

from flask import Flask, render_template, request, jsonify, Response
from transformers import AutoTokenizer
import requests
import json
import os

app = Flask(__name__)

# Initialize tokenizer
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Backend server URL
BACKEND_URL = "http://localhost:8080"

@app.route('/')
def index():
    """Serve the main web interface"""
    return render_template('index.html')

@app.route('/generate_stream', methods=['POST'])
def generate_stream():
    """Stream text generation token by token"""
    # Get request data BEFORE creating generator (inside request context)
    data = request.json
    prompt = data.get('prompt', '')
    max_tokens = data.get('max_tokens', 50)
    
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    
    # Tokenize the prompt (inside request context)
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    
    def generate():
        try:
            
            # Call backend with streaming enabled
            response = requests.post(
                f"{BACKEND_URL}/generate",
                json={
                    'input_ids': input_ids,
                    'max_new_tokens': max_tokens,
                    'stream': True
                },
                stream=True,
                timeout=120
            )
            
            if response.status_code != 200:
                yield f"data: {json.dumps({'error': f'Backend error: {response.status_code}'})}\n\n"
                return
            
            # Stream tokens from backend
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        backend_data = json.loads(line[6:])
                        
                        if backend_data.get('done'):
                            yield f"data: {json.dumps({'done': True})}\n\n"
                            break
                        
                        token_id = backend_data.get('token')
                        if token_id is not None:
                            # Decode single token
                            token_text = tokenizer.decode([token_id], skip_special_tokens=False)
                            yield f"data: {json.dumps({'token': token_text, 'token_id': token_id})}\n\n"
            
        except requests.exceptions.ConnectionError:
            yield f"data: {json.dumps({'error': 'Backend server not available'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/generate', methods=['POST'])
def generate():
    """Generate text using the auto server backend (non-streaming)"""
    try:
        data = request.json
        prompt = data.get('prompt', '')
        max_tokens = data.get('max_tokens', 50)
        
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        # Tokenize the prompt
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        
        # Call backend server
        response = requests.post(
            f"{BACKEND_URL}/generate",
            json={
                'input_ids': input_ids,
                'max_new_tokens': max_tokens,
                'stream': False
            },
            timeout=60
        )
        
        if response.status_code != 200:
            return jsonify({'error': f'Backend error: {response.status_code}'}), 500
        
        # Decode output tokens
        output_data = response.json()
        output_ids = output_data.get('output_ids', [])
        
        # Full sequence is input + output
        full_ids = input_ids + output_ids
        generated_text = tokenizer.decode(full_ids, skip_special_tokens=True)
        
        return jsonify({
            'generated_text': generated_text,
            'input_ids': input_ids,
            'output_ids': output_ids,
            'num_tokens': len(output_ids)
        })
        
    except requests.exceptions.ConnectionError:
        return jsonify({'error': 'Backend server not available. Is serve_model_auto running?'}), 503
    except requests.exceptions.Timeout:
        return jsonify({'error': 'Generation timeout'}), 504
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Check health of both web interface and backend"""
    try:
        backend_health = requests.get(f"{BACKEND_URL}/health", timeout=2)
        return jsonify({
            'web_interface': 'ok',
            'backend': 'ok' if backend_health.status_code == 200 else 'error',
            'model': MODEL_NAME
        })
    except:
        return jsonify({
            'web_interface': 'ok',
            'backend': 'unavailable',
            'model': MODEL_NAME
        }), 503

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Check if backend is available
    try:
        requests.get(f"{BACKEND_URL}/health", timeout=2)
        print("✅ Backend server detected")
    except:
        print("⚠️  Warning: Backend server not detected at", BACKEND_URL)
        print("   Make sure serve_model_auto is running!")
    
    print(f"🚀 Starting web interface on http://localhost:5000")
    print(f"   Model: {MODEL_NAME}")
    app.run(host='0.0.0.0', port=5000, debug=True)
