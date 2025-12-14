
import os
import sys
import json
import torch
from transformers import AutoModelForSeq2SeqLM, AutoConfig
from huggingface_hub import snapshot_download

def convert_model(model_name_or_path):
    print(f"📦 Loading model: {model_name_or_path}")
    
    try:
        # Resolve the cache path using huggingface_hub
        # This returns the path to the most recent snapshot in the cache
        output_dir = snapshot_download(repo_id=model_name_or_path, allow_patterns=["*.json", "*.bin", "*.safetensors"])
        print(f"📍 Found cache directory: {output_dir}")

        # Load model (this handles downloading and pytorch_model.bin loading)
        # Try loading config first
        try:
            config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        except Exception as e:
            print(f"⚠️  Could not load config from {model_name_or_path}: {e}")
            print("Trying fallback to base model: google/flan-t5-small")
            config = AutoConfig.from_pretrained("google/flan-t5-small", trust_remote_code=True)
            
            # Patch vocab size to match checkpoint (32100 vs 32128)
            if "HRM-LLM" in model_name_or_path:
                print("⚠️  Patching vocab_size to 32100 to match checkpoint")
                config.vocab_size = 32100

        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, config=config, trust_remote_code=True)
        
        # Patch config to match LOOM expectation (Llama-style keys)
        # LOOM expects: hidden_size, num_hidden_layers, num_attention_heads, intermediate_size
        # T5 uses: d_model, num_layers, num_heads, d_ff
        # We will add the aliases to the config object before saving
        if not hasattr(config, "hidden_size") and hasattr(config, "d_model"):
            config.hidden_size = config.d_model
        if not hasattr(config, "num_hidden_layers") and hasattr(config, "num_layers"):
            config.num_hidden_layers = config.num_layers
        if not hasattr(config, "num_attention_heads") and hasattr(config, "num_heads"):
            config.num_attention_heads = config.num_heads
        if not hasattr(config, "intermediate_size") and hasattr(config, "d_ff"):
            config.intermediate_size = config.d_ff
        if not hasattr(config, "num_key_value_heads"):
            config.num_key_value_heads = config.num_heads # T5 uses MHA by default
        if not hasattr(config, "rms_norm_eps") and hasattr(config, "layer_norm_epsilon"):
            config.rms_norm_eps = config.layer_norm_epsilon

        # Save config
        config.save_pretrained(output_dir)
        print(f"✓ Config saved to {output_dir}/config.json")
        
        # Save as safetensors
        # Handle shared tensors by clone or save_model
        try:
            from safetensors.torch import save_model
            save_model(model, os.path.join(output_dir, "model.safetensors"))
        except ImportError:
            # Fallback if save_model not available (older versions)
            state_dict = model.state_dict()
            # Remove shared weights to avoid error
            if "shared.weight" in state_dict and "encoder.embed_tokens.weight" in state_dict:
                 # In T5, shared.weight is usually the source
                 pass 
            save_file(state_dict, os.path.join(output_dir, "model.safetensors"))
            
        print(f"✓ Weights saved to {output_dir}/model.safetensors")
        
        print("\n🎉 Conversion complete!")
        print(f"Files saved directly to cache: {output_dir}")
        print("You can now run 'go run quick_talk.go' immediately.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_hf_model.py <model_name_or_path>")
        sys.exit(1)
        
    convert_model(sys.argv[1])
