#!/usr/bin/env python3
"""
LOOM Model Converter - Import Pre-trained PyTorch/TensorFlow Models

Converts models with Multi-Head Attention layers to LOOM format.
Supports: Transformers, BERT, GPT-2, Vision Transformers, etc.

Usage:
    python convert_model.py --model bert-base-uncased --output model.json
    python convert_model.py --pytorch my_model.pt --config config.yaml --output model.json
"""

import argparse
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
import sys

try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("⚠️  PyTorch not available - install with: pip install torch")

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️  TensorFlow not available - install with: pip install tensorflow")

try:
    from transformers import AutoModel, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Transformers not available - install with: pip install transformers")


class LOOMModelConverter:
    """Converts pre-trained models to LOOM format"""
    
    def __init__(self):
        self.layers = []
        self.metadata = {}
    
    def convert_pytorch_mha(
        self,
        mha_layer: nn.Module,
        d_model: int,
        num_heads: int,
        seq_length: int = 128,
        activation: str = "gelu"
    ) -> Dict:
        """
        Convert PyTorch MultiheadAttention to LOOM format
        
        Args:
            mha_layer: torch.nn.MultiheadAttention or similar
            d_model: Model dimension
            num_heads: Number of attention heads
            seq_length: Sequence length
            activation: Activation function name
        
        Returns:
            LOOM layer config dict
        """
        head_dim = d_model // num_heads
        
        # Extract weights from PyTorch layer
        state_dict = mha_layer.state_dict()
        
        # PyTorch MultiheadAttention stores QKV as single matrices
        # We need to split them
        if 'in_proj_weight' in state_dict:
            # Combined QKV projection
            in_proj = state_dict['in_proj_weight'].cpu().numpy()
            q_weight = in_proj[:d_model, :].T.tolist()  # Transpose for LOOM
            k_weight = in_proj[d_model:2*d_model, :].T.tolist()
            v_weight = in_proj[2*d_model:, :].T.tolist()
            
            if 'in_proj_bias' in state_dict:
                in_proj_bias = state_dict['in_proj_bias'].cpu().numpy()
                q_bias = in_proj_bias[:d_model].tolist()
                k_bias = in_proj_bias[d_model:2*d_model].tolist()
                v_bias = in_proj_bias[2*d_model:].tolist()
            else:
                q_bias = [0.0] * d_model
                k_bias = [0.0] * d_model
                v_bias = [0.0] * d_model
        else:
            # Separate Q, K, V projections
            q_weight = state_dict['q_proj_weight'].cpu().numpy().T.tolist()
            k_weight = state_dict['k_proj_weight'].cpu().numpy().T.tolist()
            v_weight = state_dict['v_proj_weight'].cpu().numpy().T.tolist()
            
            q_bias = state_dict.get('q_proj_bias', torch.zeros(d_model)).cpu().numpy().tolist()
            k_bias = state_dict.get('k_proj_bias', torch.zeros(d_model)).cpu().numpy().tolist()
            v_bias = state_dict.get('v_proj_bias', torch.zeros(d_model)).cpu().numpy().tolist()
        
        # Output projection
        out_weight = state_dict['out_proj.weight'].cpu().numpy().T.tolist()
        out_bias = state_dict.get('out_proj.bias', torch.zeros(d_model)).cpu().numpy().tolist()
        
        return {
            "type": "multi_head_attention",
            "d_model": d_model,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "seq_length": seq_length,
            "activation": activation,
            "q_weights": q_weight,
            "q_bias": q_bias,
            "k_weights": k_weight,
            "k_bias": k_bias,
            "v_weights": v_weight,
            "v_bias": v_bias,
            "output_weight": out_weight,
            "output_bias": out_bias
        }
    
    def convert_pytorch_linear(self, linear_layer: nn.Module) -> Dict:
        """Convert PyTorch Linear layer to LOOM Dense layer"""
        weight = linear_layer.weight.cpu().numpy().T.tolist()  # Transpose
        bias = linear_layer.bias.cpu().numpy().tolist() if linear_layer.bias is not None else None
        
        return {
            "type": "dense",
            "input_size": linear_layer.in_features,
            "output_size": linear_layer.out_features,
            "activation": "linear",
            "kernel": weight,
            "bias": bias
        }
    
    def convert_huggingface_model(
        self,
        model_name: str,
        seq_length: int = 128,
        extract_layer: Optional[int] = None
    ) -> Dict:
        """
        Convert HuggingFace Transformers model to LOOM
        
        Args:
            model_name: HuggingFace model name (e.g., 'bert-base-uncased')
            seq_length: Maximum sequence length
            extract_layer: If set, only extract this layer (0-indexed)
        
        Returns:
            LOOM model dict
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required: pip install transformers")
        
        print(f"📥 Loading HuggingFace model: {model_name}")
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        
        print(f"✅ Loaded {model_name}")
        print(f"   Hidden size: {config.hidden_size}")
        print(f"   Num attention heads: {config.num_attention_heads}")
        print(f"   Num layers: {config.num_hidden_layers}")
        
        layers = []
        
        # Determine model architecture
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
            # BERT-style architecture
            transformer_layers = model.encoder.layer
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            # GPT-style architecture
            transformer_layers = model.transformer.h
        elif hasattr(model, 'layers'):
            # Generic transformer
            transformer_layers = model.layers
        else:
            raise ValueError(f"Unsupported model architecture: {type(model)}")
        
        # Extract layers
        layer_indices = [extract_layer] if extract_layer is not None else range(len(transformer_layers))
        
        for i in layer_indices:
            if i >= len(transformer_layers):
                print(f"⚠️  Layer {i} out of range (max: {len(transformer_layers)-1})")
                continue
            
            layer = transformer_layers[i]
            print(f"\n📊 Converting layer {i}...")
            
            # Extract attention layer
            if hasattr(layer, 'attention') and hasattr(layer.attention, 'self'):
                # BERT-style
                attn = layer.attention.self
                d_model = config.hidden_size
                num_heads = config.num_attention_heads
                
                # Convert attention
                loom_layer = {
                    "type": "multi_head_attention",
                    "d_model": d_model,
                    "num_heads": num_heads,
                    "head_dim": d_model // num_heads,
                    "seq_length": seq_length,
                    "activation": "gelu",
                    "q_weights": attn.query.weight.cpu().numpy().T.tolist(),
                    "q_bias": attn.query.bias.cpu().numpy().tolist(),
                    "k_weights": attn.key.weight.cpu().numpy().T.tolist(),
                    "k_bias": attn.key.bias.cpu().numpy().tolist(),
                    "v_weights": attn.value.weight.cpu().numpy().T.tolist(),
                    "v_bias": attn.value.bias.cpu().numpy().tolist(),
                    "output_weight": layer.attention.output.dense.weight.cpu().numpy().T.tolist(),
                    "output_bias": layer.attention.output.dense.bias.cpu().numpy().tolist()
                }
                
                layers.append(loom_layer)
                print(f"   ✅ Attention layer extracted")
                
                # Add feedforward layers
                if hasattr(layer, 'intermediate'):
                    ff_intermediate = {
                        "type": "dense",
                        "input_size": d_model,
                        "output_size": config.intermediate_size,
                        "activation": "gelu",
                        "kernel": layer.intermediate.dense.weight.cpu().numpy().T.tolist(),
                        "bias": layer.intermediate.dense.bias.cpu().numpy().tolist()
                    }
                    layers.append(ff_intermediate)
                    print(f"   ✅ Feedforward intermediate layer extracted")
                
                if hasattr(layer, 'output'):
                    ff_output = {
                        "type": "dense",
                        "input_size": config.intermediate_size,
                        "output_size": d_model,
                        "activation": "linear",
                        "kernel": layer.output.dense.weight.cpu().numpy().T.tolist(),
                        "bias": layer.output.dense.bias.cpu().numpy().tolist()
                    }
                    layers.append(ff_output)
                    print(f"   ✅ Feedforward output layer extracted")
        
        # Create LOOM model structure
        loom_model = {
            "input_size": config.hidden_size,
            "grid_rows": 1,
            "grid_cols": len(layers),
            "layers_per_cell": 1,
            "batch_size": 1,
            "use_gpu": False,
            "layers": layers,
            "metadata": {
                "source": "huggingface",
                "model_name": model_name,
                "original_config": {
                    "hidden_size": config.hidden_size,
                    "num_attention_heads": config.num_attention_heads,
                    "num_hidden_layers": config.num_hidden_layers,
                    "seq_length": seq_length
                }
            }
        }
        
        return loom_model
    
    def save_loom_model(self, model_dict: Dict, output_path: str):
        """Save LOOM model to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(model_dict, f, indent=2)
        
        print(f"\n✅ Model saved to: {output_path}")
        
        # Print statistics
        total_params = 0
        for layer in model_dict['layers']:
            if 'kernel' in layer and layer['kernel']:
                total_params += len(layer['kernel']) * len(layer['kernel'][0])
            if 'q_weights' in layer and layer['q_weights']:
                total_params += len(layer['q_weights']) * len(layer['q_weights'][0]) * 4  # Q, K, V, Out
        
        print(f"   Total parameters: {total_params:,}")
        print(f"   Total layers: {len(model_dict['layers'])}")
        print(f"   File size: {len(json.dumps(model_dict)) / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description='Convert pre-trained models to LOOM format')
    parser.add_argument('--model', type=str, help='HuggingFace model name (e.g., bert-base-uncased)')
    parser.add_argument('--pytorch', type=str, help='Path to PyTorch model file (.pt, .pth)')
    parser.add_argument('--output', type=str, required=True, help='Output LOOM JSON file')
    parser.add_argument('--seq-length', type=int, default=128, help='Sequence length (default: 128)')
    parser.add_argument('--layer', type=int, help='Extract only this layer (0-indexed)')
    
    args = parser.parse_args()
    
    if not args.model and not args.pytorch:
        parser.error("Must specify either --model or --pytorch")
    
    converter = LOOMModelConverter()
    
    try:
        if args.model:
            # Convert HuggingFace model
            loom_model = converter.convert_huggingface_model(
                args.model,
                seq_length=args.seq_length,
                extract_layer=args.layer
            )
        elif args.pytorch:
            # Convert custom PyTorch model
            print("⚠️  Custom PyTorch conversion requires manual configuration")
            print("    Use convert_huggingface_model() as a template")
            sys.exit(1)
        
        # Save to file
        converter.save_loom_model(loom_model, args.output)
        
        print("\n🎉 Conversion complete!")
        print(f"\nLoad in your code with:")
        print(f'    network := nn.LoadModel("{args.output}", "converted_model")')
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
