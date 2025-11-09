#!/usr/bin/env python3
"""
Quick model converter for tiny models that can run on phones

Downloads and converts small pre-trained models to LOOM format.
Optimized for mobile/edge deployment.
"""

import json
import sys

try:
    import torch
    from transformers import AutoModel, AutoConfig, AutoTokenizer
    print("✅ Dependencies available")
except ImportError as e:
    print("❌ Missing dependencies. Install with:")
    print("   pip install torch transformers")
    sys.exit(1)

def convert_tiny_model(model_name: str, output_path: str, max_layers: int = 2):
    """
    Convert a tiny transformer model to LOOM format
    
    Args:
        model_name: HuggingFace model name
        output_path: Where to save the JSON
        max_layers: Maximum number of transformer layers to extract (for size)
    """
    print(f"📥 Downloading {model_name}...")
    
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    
    print(f"✅ Model loaded!")
    print(f"   Hidden size: {config.hidden_size}")
    print(f"   Attention heads: {config.num_attention_heads}")
    print(f"   Total layers: {config.num_hidden_layers}")
    print(f"   Converting first {max_layers} layers for mobile...")
    
    # Find transformer layers
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        transformer_layers = model.encoder.layer  # BERT
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        transformer_layers = model.transformer.h  # GPT-2
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        transformer_layers = model.model.layers  # Llama
    else:
        print(f"❌ Unsupported architecture: {type(model)}")
        sys.exit(1)
    
    layers = []
    
    # Convert only first N layers (for mobile)
    num_layers = min(max_layers, len(transformer_layers))
    
    for i in range(num_layers):
        layer = transformer_layers[i]
        print(f"\n🔄 Layer {i+1}/{num_layers}...")
        
        # Extract attention
        if hasattr(layer, 'attention') and hasattr(layer.attention, 'self'):
            # BERT-style
            attn = layer.attention.self
            
            mha_layer = {
                "type": "multi_head_attention",
                "d_model": config.hidden_size,
                "num_heads": config.num_attention_heads,
                "head_dim": config.hidden_size // config.num_attention_heads,
                "seq_length": 128,
                "activation": "gelu",
                "q_weights": attn.query.weight.detach().cpu().numpy().T.tolist(),
                "q_bias": attn.query.bias.detach().cpu().numpy().tolist(),
                "k_weights": attn.key.weight.detach().cpu().numpy().T.tolist(),
                "k_bias": attn.key.bias.detach().cpu().numpy().tolist(),
                "v_weights": attn.value.weight.detach().cpu().numpy().T.tolist(),
                "v_bias": attn.value.bias.detach().cpu().numpy().tolist(),
                "output_weight": layer.attention.output.dense.weight.detach().cpu().numpy().T.tolist(),
                "output_bias": layer.attention.output.dense.bias.detach().cpu().numpy().tolist()
            }
            layers.append(mha_layer)
            
            # Add LayerNorm after attention (if exists)
            if hasattr(layer.attention.output, 'LayerNorm'):
                ln = layer.attention.output.LayerNorm
                layernorm = {
                    "type": "layer_norm",
                    "norm_size": config.hidden_size,
                    "gamma": ln.weight.detach().cpu().numpy().tolist(),
                    "beta": ln.bias.detach().cpu().numpy().tolist(),
                    "epsilon": ln.eps if hasattr(ln, 'eps') else 1e-12
                }
                layers.append(layernorm)
            
            # Add feedforward
            if hasattr(layer, 'intermediate') and hasattr(layer, 'output'):
                ff1 = {
                    "type": "dense",
                    "input_size": config.hidden_size,
                    "output_size": config.intermediate_size,
                    "activation": "gelu",
                    "kernel": layer.intermediate.dense.weight.detach().cpu().numpy().T.tolist(),
                    "bias": layer.intermediate.dense.bias.detach().cpu().numpy().tolist()
                }
                layers.append(ff1)
                
                ff2 = {
                    "type": "dense",
                    "input_size": config.intermediate_size,
                    "output_size": config.hidden_size,
                    "activation": "linear",
                    "kernel": layer.output.dense.weight.detach().cpu().numpy().T.tolist(),
                    "bias": layer.output.dense.bias.detach().cpu().numpy().tolist()
                }
                layers.append(ff2)
                
                # Add LayerNorm after FFN (if exists)
                if hasattr(layer.output, 'LayerNorm'):
                    ln = layer.output.LayerNorm
                    layernorm = {
                        "type": "layer_norm",
                        "norm_size": config.hidden_size,
                        "gamma": ln.weight.detach().cpu().numpy().tolist(),
                        "beta": ln.bias.detach().cpu().numpy().tolist(),
                        "epsilon": ln.eps if hasattr(ln, 'eps') else 1e-12
                    }
                    layers.append(layernorm)
        
        elif hasattr(layer, 'attn'):
            # GPT-2 style
            attn = layer.attn
            
            # GPT-2 uses combined QKV weights, need to split
            c_attn_weight = attn.c_attn.weight.detach().cpu().numpy()
            c_attn_bias = attn.c_attn.bias.detach().cpu().numpy()
            
            # Split into Q, K, V
            hidden = config.hidden_size
            q_w = c_attn_weight[:, :hidden].T.tolist()
            k_w = c_attn_weight[:, hidden:2*hidden].T.tolist()
            v_w = c_attn_weight[:, 2*hidden:].T.tolist()
            
            q_b = c_attn_bias[:hidden].tolist()
            k_b = c_attn_bias[hidden:2*hidden].tolist()
            v_b = c_attn_bias[2*hidden:].tolist()
            
            mha_layer = {
                "type": "multi_head_attention",
                "d_model": config.hidden_size,
                "num_heads": config.num_attention_heads,
                "head_dim": config.hidden_size // config.num_attention_heads,
                "seq_length": 128,
                "activation": "gelu",
                "q_weights": q_w,
                "q_bias": q_b,
                "k_weights": k_w,
                "k_bias": k_b,
                "v_weights": v_w,
                "v_bias": v_b,
                "output_weight": attn.c_proj.weight.detach().cpu().numpy().T.tolist(),
                "output_bias": attn.c_proj.bias.detach().cpu().numpy().tolist()
            }
            layers.append(mha_layer)
            
            # Add MLP
            if hasattr(layer, 'mlp'):
                mlp = layer.mlp
                ff1 = {
                    "type": "dense",
                    "input_size": config.hidden_size,
                    "output_size": config.hidden_size * 4,
                    "activation": "gelu",
                    "kernel": mlp.c_fc.weight.detach().cpu().numpy().T.tolist(),
                    "bias": mlp.c_fc.bias.detach().cpu().numpy().tolist()
                }
                layers.append(ff1)
                
                ff2 = {
                    "type": "dense",
                    "input_size": config.hidden_size * 4,
                    "output_size": config.hidden_size,
                    "activation": "linear",
                    "kernel": mlp.c_proj.weight.detach().cpu().numpy().T.tolist(),
                    "bias": mlp.c_proj.bias.detach().cpu().numpy().tolist()
                }
                layers.append(ff2)
        
        print(f"   ✅ Extracted {len(layers)} layers so far")
    
    # Create LOOM model
    seq_length = 128  # Fixed sequence length for mobile models
    loom_model = {
        "input_size": config.hidden_size,
        "grid_rows": 1,
        "grid_cols": len(layers),
        "layers_per_cell": 1,
        "batch_size": seq_length,  # Set to sequence length for proper dense layer sizing
        "use_gpu": False,
        "layers": layers,
        "metadata": {
            "source": "huggingface",
            "model_name": model_name,
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "num_attention_heads": config.num_attention_heads,
            "layers_extracted": num_layers,
            "total_layers": config.num_hidden_layers,
            "tokenizer_info": {
                "pad_token": tokenizer.pad_token,
                "eos_token": tokenizer.eos_token,
                "bos_token": tokenizer.bos_token
            }
        }
    }
    
    # Save
    with open(output_path, 'w') as f:
        json.dump(loom_model, f, indent=2)
    
    # Stats
    size_mb = len(json.dumps(loom_model)) / 1024 / 1024
    
    print(f"\n✅ Saved to: {output_path}")
    print(f"   File size: {size_mb:.2f} MB")
    print(f"   Layers: {len(layers)}")
    print(f"   Hidden size: {config.hidden_size}")
    print(f"   Attention heads: {config.num_attention_heads}")
    
    if size_mb > 50:
        print(f"\n⚠️  Warning: {size_mb:.0f}MB may be large for mobile")
        print(f"   Try reducing --max-layers")
    else:
        print(f"\n✅ Size OK for mobile deployment!")
    
    return loom_model


if __name__ == '__main__':
    print("🤖 LOOM Tiny Model Converter")
    print("=" * 50)
    
    # Best models for mobile (ordered by size)
    models = {
        "1": ("prajjwal1/bert-tiny", "bert-tiny.json", 2, "BERT Tiny (4MB) - 2 layers, 128 hidden"),
        "2": ("distilgpt2", "gpt2-tiny.json", 2, "GPT-2 Distilled (8MB) - 2 layers, 768 hidden"),
        "3": ("google/electra-small-discriminator", "electra-small.json", 2, "ELECTRA Small (12MB) - 2 layers, 256 hidden"),
        "4": ("prajjwal1/bert-mini", "bert-mini.json", 2, "BERT Mini (6MB) - 2 layers, 256 hidden"),
        "5": ("custom", None, None, "Enter custom model name")
    }
    
    print("\nAvailable tiny models:")
    for key, (name, output, layers, desc) in models.items():
        print(f"  {key}. {desc}")
        print(f"     → {name}")
    
    choice = input("\nSelect model (1-5): ").strip()
    
    if choice not in models:
        print("❌ Invalid choice")
        sys.exit(1)
    
    model_name, output_path, max_layers, _ = models[choice]
    
    if choice == "5":
        model_name = input("Enter HuggingFace model name: ").strip()
        output_path = input("Output filename (e.g., model.json): ").strip()
        max_layers = int(input("Max layers to extract: ").strip())
    
    print(f"\n🚀 Converting {model_name}...")
    print(f"   Output: {output_path}")
    print(f"   Max layers: {max_layers}")
    print()
    
    try:
        convert_tiny_model(model_name, output_path, max_layers)
        
        print("\n🎉 SUCCESS!")
        print("\nLoad in Go:")
        print(f'  network := nn.LoadModel("{output_path}", "mobile_model")')
        print("\nLoad in Python:")
        print(f'  from welvet import Network')
        print(f'  network = Network.load_from_file("{output_path}", "mobile_model")')
        print("\nLoad in C#:")
        print(f'  var network = Network.LoadFromFile("{output_path}", "mobile_model");')
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
