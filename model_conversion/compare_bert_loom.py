#!/usr/bin/env python3
"""
Compare LOOM vs Real BERT outputs
This loads embeddings from real BERT and passes them through both models
"""

import json
import numpy as np
import sys
import subprocess

try:
    from transformers import BertTokenizer, BertModel
    import torch
except ImportError:
    print("❌ Missing dependencies. Install with:")
    print("   pip install torch transformers")
    sys.exit(1)


def get_bert_embeddings_and_outputs(text, model_name="prajjwal1/bert-tiny"):
    """Get embeddings and outputs from real BERT"""
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()
    
    # Tokenize
    encoding = tokenizer(text, return_tensors="pt", padding="max_length", 
                        max_length=128, truncation=True)
    
    input_ids = encoding["input_ids"][0]
    attention_mask = encoding["attention_mask"][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    # Get embeddings (word + position)
    with torch.no_grad():
        word_emb = model.embeddings.word_embeddings(input_ids).numpy()[0]
        pos_emb = model.embeddings.position_embeddings(
            torch.arange(128).unsqueeze(0)
        ).numpy()[0]
        
        full_embeddings = word_emb + pos_emb
        
        # Get real BERT output
        outputs = model(**encoding)
        bert_output = outputs.last_hidden_state[0].numpy()
    
    return {
        "tokens": tokens,
        "attention_mask": attention_mask.tolist(),
        "embeddings": full_embeddings,
        "bert_output": bert_output
    }


def run_loom_model(embeddings_flat):
    """Run LOOM model via Go"""
    # Save embeddings to temp file
    with open("/tmp/loom_input.json", "w") as f:
        json.dump({"embeddings": embeddings_flat}, f)
    
    # Run Go program to process through LOOM
    # (This would need a Go program that reads the embeddings and runs forward pass)
    # For now, we'll show what would be compared
    return None


def compare_outputs(text):
    """Compare LOOM vs real BERT"""
    print(f"\n{'='*60}")
    print(f"🔍 Comparing: \"{text}\"")
    print('='*60)
    
    # Get real BERT results
    print("\n📥 Loading real BERT...")
    bert_data = get_bert_embeddings_and_outputs(text)
    
    tokens = bert_data["tokens"]
    attention_mask = bert_data["attention_mask"]
    embeddings = bert_data["embeddings"]
    bert_output = bert_data["bert_output"]
    
    # Show tokens
    actual_tokens = [t for t, m in zip(tokens, attention_mask) if m == 1]
    print(f"   Tokens ({len(actual_tokens)}): {actual_tokens}")
    
    # Show embedding stats
    print(f"\n📊 Input Embeddings:")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Mean: {embeddings.mean():.6f}")
    print(f"   Std:  {embeddings.std():.6f}")
    
    # Show BERT output stats
    print(f"\n📊 Real BERT Output:")
    print(f"   Shape: {bert_output.shape}")
    print(f"   Mean: {bert_output.mean():.6f}")
    print(f"   Std:  {bert_output.std():.6f}")
    
    # Show per-token outputs
    print(f"\n📈 Per-Token Real BERT Outputs:")
    print("   " + "-" * 58)
    for i, (token, mask) in enumerate(zip(tokens[:10], attention_mask[:10])):
        if mask == 0:
            break
        vec = bert_output[i]
        print(f"   {i:2d} {token:15s} | "
              f"mean: {vec.mean():7.4f} | "
              f"std: {vec.std():6.4f} | "
              f"min: {vec.min():7.4f} | "
              f"max: {vec.max():7.4f}")
    
    # Save for Go to load
    output_file = "bert_comparison.json"
    embeddings_flat = embeddings.flatten().tolist()
    
    with open(output_file, "w") as f:
        json.dump({
            "text": text,
            "tokens": actual_tokens,
            "embeddings_flat": embeddings_flat,
            "embeddings_shape": list(embeddings.shape),
            "bert_output": bert_output.tolist(),
            "attention_mask": attention_mask
        }, f, indent=2)
    
    print(f"\n💾 Saved to: {output_file}")
    print(f"\n💡 Next: Run Go program to load embeddings and compare outputs")
    print(f"   The embeddings are ready for LOOM to process")
    
    return embeddings_flat, bert_output


def main():
    test_sentences = [
        "Hello world",
        "The cat sat on the mat",
    ]
    
    print("🧠 BERT vs LOOM Comparison")
    print("="*60)
    print("\nThis tool:")
    print("  1. Gets real BERT embeddings for text")
    print("  2. Saves them for LOOM to process")
    print("  3. Compares outputs")
    print()
    
    for sentence in test_sentences:
        embeddings, bert_output = compare_outputs(sentence)
    
    print(f"\n{'='*60}")
    print("✅ Comparison data generated!")
    print("\n🔧 To complete comparison:")
    print("   1. Load bert_comparison.json in Go")
    print("   2. Pass embeddings through LOOM BERT")
    print("   3. Compare LOOM output vs bert_output")
    print("\n📝 Expected differences:")
    print("   • LOOM has only 2 BERT layers (vs full model)")
    print("   • Missing layer normalization")
    print("   • Missing residual connections")
    print("   • Outputs will differ but should be similar patterns")


if __name__ == "__main__":
    main()
