#!/usr/bin/env python3
"""
Simple test script for LOOM transformer inference via welvet
"""

import welvet
import sys
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: ./test_transformer.py <model_path>")
        print("Example: ./test_transformer.py ../../models/SmolLM2-135M-Instruct")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    
    print("=== LOOM Python Transformer Test ===\n")
    
    # Load tokenizer
    print("1. Loading tokenizer...")
    tokenizer_path = os.path.join(model_dir, 'tokenizer.json')
    with open(tokenizer_path, 'rb') as f:
        tok_data = f.read()
    
    result = welvet.load_tokenizer_from_bytes(tok_data)
    print(f"   ✓ Vocab size: {result['vocab_size']}\n")
    
    # Load transformer
    print("2. Loading transformer model...")
    config_path = os.path.join(model_dir, 'config.json')
    weights_path = os.path.join(model_dir, 'model.safetensors')
    
    with open(config_path, 'rb') as f:
        config_data = f.read()
    with open(weights_path, 'rb') as f:
        weights_data = f.read()
    
    result = welvet.load_transformer_from_bytes(config_data, weights_data)
    print(f"   ✓ Layers: {result['num_layers']}")
    print(f"   ✓ Hidden size: {result['hidden_size']}")
    print(f"   ✓ Vocab size: {result['vocab_size']}\n")
    
    # Test encoding
    print("3. Testing encoding...")
    prompt = "Once upon a time"
    ids = welvet.encode_text(prompt, add_special_tokens=True)
    print(f"   Input: \"{prompt}\"")
    print(f"   Token IDs: {ids}\n")
    
    # Test decoding
    print("4. Testing decoding...")
    decoded = welvet.decode_tokens(ids, skip_special_tokens=True)
    print(f"   Decoded: \"{decoded}\"\n")
    
    # Test generation (non-streaming)
    print("5. Testing text generation (non-streaming)...")
    generated = welvet.generate_text(prompt, max_tokens=30, temperature=0.7)
    print(f"   Generated: {generated}\n")
    
    # Test streaming generation
    print("6. Testing streaming generation...")
    print(f"   Prompt: \"{prompt}\"")
    print("   Output: ", end='', flush=True)
    
    for token in welvet.generate_stream(prompt, max_tokens=50, temperature=0.7):
        print(token, end='', flush=True)
    
    print("\n\n✅ All tests passed!")

if __name__ == '__main__':
    main()
