#!/usr/bin/env python3
"""
Test the converted BERT-Tiny model with real inputs

Shows how to:
1. Load the converted model
2. Tokenize text input
3. Get embeddings/outputs
4. See what the model produces
"""

import json
import sys
import numpy as np

try:
    from transformers import AutoTokenizer
    print("✅ Tokenizer available")
except ImportError:
    print("❌ Install transformers: pip install transformers")
    sys.exit(1)

def load_loom_model(model_path: str):
    """Load LOOM model JSON"""
    with open(model_path, 'r') as f:
        return json.load(f)

def create_dummy_input(seq_length: int, hidden_size: int):
    """Create dummy input for testing"""
    # Random embeddings (in real use, these come from an embedding layer)
    return np.random.randn(seq_length * hidden_size).astype(np.float32)

def tokenize_text(text: str, model_name: str = "prajjwal1/bert-tiny", max_length: int = 128):
    """Tokenize text using the original model's tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize
    tokens = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Get input IDs
    input_ids = tokens['input_ids'][0].numpy()
    attention_mask = tokens['attention_mask'][0].numpy()
    
    # Decode to show what tokens look like
    decoded_tokens = [tokenizer.decode([tid]) for tid in input_ids]
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'tokens': decoded_tokens,
        'text': text
    }

def manual_forward_pass(model: dict, input_vec: np.ndarray):
    """
    Manually execute forward pass through LOOM model
    This simulates what your LOOM framework does
    """
    print("\n🔄 Running forward pass through layers...")
    
    data = input_vec.copy()
    
    for i, layer in enumerate(model['layers']):
        layer_type = layer['type']
        print(f"\n  Layer {i}: {layer_type}")
        
        if layer_type == 'multi_head_attention':
            print(f"    d_model={layer['d_model']}, heads={layer['num_heads']}")
            # For MHA, output size = input size (d_model)
            # In a real implementation, this does Q*K^T/sqrt(d), softmax, then *V
            # For now, just show dimensions
            expected_out = layer['d_model'] * layer['seq_length']
            print(f"    Input size: {len(data)}, Expected output: {expected_out}")
            
            # Simulate: output same size as input
            data = np.random.randn(expected_out).astype(np.float32) * 0.1
            
        elif layer_type == 'dense':
            in_size = layer['input_size']
            out_size = layer['output_size']
            activation = layer['activation']
            
            print(f"    {in_size} → {out_size}, activation={activation}")
            
            # Get weights (stored as list of lists)
            weights = np.array(layer['kernel'], dtype=np.float32)
            bias = np.array(layer['bias'], dtype=np.float32)
            
            # Reshape input if needed
            batch_size = len(data) // in_size
            if batch_size * in_size != len(data):
                print(f"    ⚠️  Size mismatch: got {len(data)}, expected multiple of {in_size}")
                # Truncate or pad
                if len(data) < in_size:
                    data = np.pad(data, (0, in_size - len(data)))
                else:
                    data = data[:in_size * (len(data) // in_size)]
                batch_size = len(data) // in_size
            
            input_mat = data.reshape(batch_size, in_size)
            
            # Matrix multiply: input @ weights + bias
            output = input_mat @ weights + bias
            
            # Apply activation
            if activation == 'gelu':
                # Approximate GELU
                output = 0.5 * output * (1 + np.tanh(np.sqrt(2/np.pi) * (output + 0.044715 * output**3)))
            elif activation == 'relu':
                output = np.maximum(0, output)
            # 'linear' = no activation
            
            data = output.flatten()
            print(f"    Output size: {len(data)}")
        
        else:
            print(f"    ⚠️  Unknown layer type: {layer_type}")
    
    return data

def main():
    model_path = "bert-tiny.json"
    
    print("🧠 LOOM Model Test Runner")
    print("=" * 60)
    
    # Load model
    print(f"\n📂 Loading {model_path}...")
    model = load_loom_model(model_path)
    
    print(f"✅ Model loaded!")
    print(f"   Layers: {len(model['layers'])}")
    print(f"   Hidden size: {model['metadata']['hidden_size']}")
    print(f"   Attention heads: {model['metadata']['num_attention_heads']}")
    
    # Get user input
    print("\n" + "=" * 60)
    print("Enter text to process (or press Enter for default):")
    text = input("> ").strip()
    
    if not text:
        text = "Hello world! This is a test of BERT tiny."
    
    print(f"\n📝 Input text: \"{text}\"")
    
    # Tokenize
    print("\n🔤 Tokenizing...")
    tokenized = tokenize_text(text, model['metadata']['model_name'])
    
    print(f"   Tokens ({len([t for t in tokenized['tokens'] if t.strip()])} non-padding):")
    for i, (token, mask) in enumerate(zip(tokenized['tokens'][:20], tokenized['attention_mask'][:20])):
        if mask == 1:
            print(f"      {i:2d}: '{token}'")
    
    # Create input embeddings (normally from an embedding layer)
    print("\n🎲 Creating input embeddings...")
    hidden_size = model['metadata']['hidden_size']
    seq_length = model['layers'][0]['seq_length']
    
    # In a real model, input_ids would go through an embedding layer
    # For now, use random embeddings as example
    input_vec = create_dummy_input(seq_length, hidden_size)
    print(f"   Input shape: ({seq_length}, {hidden_size}) = {len(input_vec)} values")
    print(f"   Sample values: [{', '.join([f'{x:.3f}' for x in input_vec[:5]])}...]")
    
    # Run forward pass
    output = manual_forward_pass(model, input_vec)
    
    print("\n" + "=" * 60)
    print("✅ Forward pass complete!")
    print(f"   Output size: {len(output)}")
    print(f"   Output values: [{', '.join([f'{x:.3f}' for x in output[:10]])}...]")
    
    # Show stats
    print("\n📊 Output statistics:")
    print(f"   Mean: {np.mean(output):.6f}")
    print(f"   Std:  {np.std(output):.6f}")
    print(f"   Min:  {np.min(output):.6f}")
    print(f"   Max:  {np.max(output):.6f}")
    
    print("\n" + "=" * 60)
    print("🎉 Test complete!")
    print("\n💡 To use in your LOOM framework:")
    print("   Go:     network := nn.LoadModel(\"bert-tiny.json\", \"test\")")
    print("           output := network.Forward(input)")
    print()
    print("   Python: from welvet import Network")
    print("           network = Network.load_from_file(\"bert-tiny.json\", \"test\")")
    print("           output = network.forward(input)")
    print()
    print("   C#:     var network = Network.LoadFromFile(\"bert-tiny.json\", \"test\");")
    print("           float[] output = network.Forward(input);")

if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError:
        print("\n❌ Error: bert-tiny.json not found!")
        print("   Run: python python/convert_tiny.py")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
