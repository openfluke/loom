#!/usr/bin/env python3
"""
BERT Text Processor - Proper tokenization and embeddings
This shows real text → BERT → output with actual learned embeddings
"""

import json
import numpy as np
import sys

try:
    from transformers import BertTokenizer, BertModel
    import torch
except ImportError:
    print("❌ Missing dependencies. Install with:")
    print("   pip install torch transformers")
    sys.exit(1)


def process_text(text, model_name="prajjwal1/bert-tiny"):
    """Process text through BERT and return embeddings + outputs"""
    
    print(f"🔤 Processing: \"{text}\"")
    print("=" * 60)
    
    # Load tokenizer and model
    print("\n📥 Loading BERT tokenizer and embeddings...")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()
    
    # Tokenize
    print("\n🔤 Tokenizing...")
    encoding = tokenizer(text, return_tensors="pt", padding="max_length", 
                        max_length=128, truncation=True)
    
    input_ids = encoding["input_ids"][0]
    attention_mask = encoding["attention_mask"][0]
    
    # Get tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    # Show tokens
    actual_tokens = [t for t, m in zip(tokens, attention_mask) if m == 1]
    print(f"   Tokens ({len(actual_tokens)}): {actual_tokens}")
    
    # Get embeddings (just the embedding layer, before BERT layers)
    with torch.no_grad():
        embeddings = model.embeddings.word_embeddings(input_ids).numpy()
        position_embeddings = model.embeddings.position_embeddings(
            torch.arange(128).unsqueeze(0)
        ).numpy()
        
        # Combine word + position embeddings (simplified - real BERT also adds token type)
        full_embeddings = embeddings + position_embeddings
        
    print(f"   Embedding shape: {full_embeddings.shape}")
    print(f"   Embedding stats: mean={full_embeddings.mean():.4f}, "
          f"std={full_embeddings.std():.4f}")
    
    # Pass through BERT
    print("\n▶️  Running through BERT...")
    with torch.no_grad():
        outputs = model(**encoding)
        last_hidden_state = outputs.last_hidden_state[0].numpy()
    
    print(f"   Output shape: {last_hidden_state.shape}")
    print(f"   Output stats: mean={last_hidden_state.mean():.4f}, "
          f"std={last_hidden_state.std():.4f}")
    
    # Analyze per token
    print("\n📊 Per-Token Analysis:")
    print("   " + "-" * 58)
    for i, (token, mask) in enumerate(zip(tokens[:15], attention_mask[:15])):
        if mask == 0:
            continue
        
        vec = last_hidden_state[i]
        print(f"   Token {i:2d} {token:15s} | "
              f"mean: {vec.mean():7.4f} | "
              f"std: {vec.std():7.4f} | "
              f"min: {vec.min():7.4f} | "
              f"max: {vec.max():7.4f}")
    
    print("\n" + "=" * 60)
    
    return {
        "text": text,
        "tokens": actual_tokens,
        "input_embeddings": full_embeddings[0].tolist(),  # First in batch
        "output": last_hidden_state.tolist(),
        "attention_mask": attention_mask.tolist()
    }


def main():
    sentences = [
        "Hello world",
        "The cat sat on the mat",
        "Machine learning is amazing",
        "BERT processes text very well",
    ]
    
    print("🧠 BERT Text Processing Demo (Real Tokenization)")
    print("=" * 60)
    print()
    
    for i, sentence in enumerate(sentences):
        print(f"\n{'=' * 60}")
        print(f"Sentence {i+1}/{len(sentences)}")
        print('=' * 60)
        
        result = process_text(sentence)
        
        # Save to file for potential loading in Go
        output_file = f"bert_text_output_{i+1}.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Saved to: {output_file}")
    
    print("\n" + "=" * 60)
    print("✅ Done!")
    print()
    print("💡 These show REAL BERT outputs with:")
    print("   • WordPiece tokenization (BERT vocab)")
    print("   • Learned token embeddings")
    print("   • Position embeddings")
    print("   • Full BERT transformer layers")
    print()
    print("🔧 To use in LOOM:")
    print("   1. Extract embedding layer weights from BERT")
    print("   2. Add embedding lookup to LOOM")
    print("   3. Pass embeddings to converted BERT layers")


if __name__ == "__main__":
    main()
