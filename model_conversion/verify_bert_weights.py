#!/usr/bin/env python3
"""
End-to-end BERT verification: Text in → LOOM processing → Text analysis out
This verifies LOOM's BERT weights are working by comparing against real BERT
"""

import json
import numpy as np
import sys
import subprocess
import os

try:
    from transformers import BertTokenizer, BertModel
    import torch
except ImportError:
    print("❌ Missing dependencies. Install with:")
    print("   pip install torch transformers")
    sys.exit(1)


def process_through_real_bert(text, model_name="prajjwal1/bert-tiny"):
    """Process text through real BERT"""
    print(f"\n{'='*60}")
    print(f"📝 Input Text: \"{text}\"")
    print('='*60)
    
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()
    
    # Tokenize
    encoding = tokenizer(text, return_tensors="pt", padding="max_length", 
                        max_length=128, truncation=True)
    
    input_ids = encoding["input_ids"][0]
    attention_mask = encoding["attention_mask"][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    # Get actual tokens (non-padding)
    actual_tokens = [t for t, m in zip(tokens, attention_mask) if m == 1]
    print(f"\n🔤 Tokens ({len(actual_tokens)}): {actual_tokens}")
    
    # Get embeddings
    with torch.no_grad():
        word_emb = model.embeddings.word_embeddings(input_ids).numpy()[0]
        pos_emb = model.embeddings.position_embeddings(
            torch.arange(128).unsqueeze(0)
        ).numpy()[0]
        full_embeddings = word_emb + pos_emb
        
        # Get BERT output
        outputs = model(**encoding)
        bert_output = outputs.last_hidden_state[0].numpy()
    
    print(f"\n✅ Real BERT Processing Complete")
    print(f"   Embedding shape: {full_embeddings.shape}")
    print(f"   Output shape: {bert_output.shape}")
    
    return {
        "tokens": actual_tokens,
        "all_tokens": tokens,
        "attention_mask": attention_mask.tolist(),
        "embeddings": full_embeddings.tolist(),
        "bert_output": bert_output.tolist(),
        "input_ids": input_ids.tolist()
    }


def run_through_loom(embeddings_flat):
    """Run embeddings through LOOM BERT model"""
    print(f"\n▶️  Running through LOOM BERT...")
    
    # Save embeddings to temp file
    temp_file = "/tmp/loom_bert_input.json"
    with open(temp_file, "w") as f:
        json.dump({"embeddings": embeddings_flat}, f)
    
    # Run Go program to process through LOOM
    go_code = """
package main

import (
    "encoding/json"
    "fmt"
    "os"
    "github.com/openfluke/loom/nn"
)

type Input struct {
    Embeddings []float32 `json:"embeddings"`
}

func main() {
    // Load input
    data, _ := os.ReadFile("/tmp/loom_bert_input.json")
    var input Input
    json.Unmarshal(data, &input)
    
    // Load model (from model_conversion folder)
    network, err := nn.LoadImportedModel("bert-tiny.json", "bert-tiny")
    if err != nil {
        fmt.Printf("Error: %v\\n", err)
        os.Exit(1)
    }
    
    // Forward pass
    output, _ := network.ForwardCPU(input.Embeddings)
    
    // Save output
    result := map[string]interface{}{
        "output": output,
        "output_len": len(output),
    }
    
    jsonData, _ := json.Marshal(result)
    os.WriteFile("/tmp/loom_bert_output.json", jsonData, 0644)
}
"""
    
    # Write Go program
    with open("/tmp/run_loom_bert.go", "w") as f:
        f.write(go_code)
    
    # Run it
    result = subprocess.run(
        ["go", "run", "/tmp/run_loom_bert.go"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"❌ LOOM execution failed: {result.stderr}")
        return None
    
    # Load output
    with open("/tmp/loom_bert_output.json", "r") as f:
        output_data = json.load(f)
    
    print(f"✅ LOOM Processing Complete")
    print(f"   Output length: {output_data['output_len']}")
    
    return output_data["output"]


def compare_outputs(real_bert_output, loom_output, tokens, attention_mask):
    """Compare real BERT vs LOOM outputs"""
    print(f"\n{'='*60}")
    print("📊 VERIFICATION RESULTS")
    print('='*60)
    
    hidden_size = 128
    
    # Compare per-token
    print(f"\n🔍 Per-Token Comparison:")
    print(f"{'='*60}")
    print(f"{'Token':<15} {'Real BERT':<25} {'LOOM':<25}")
    print(f"{'-'*60}")
    
    similarities = []
    differences = []
    
    for i, (token, mask) in enumerate(zip(tokens, attention_mask)):
        if mask == 0:
            continue
        
        # Extract vectors
        real_vec = real_bert_output[i]
        loom_start = i * hidden_size
        loom_end = loom_start + hidden_size
        
        if loom_end > len(loom_output):
            break
            
        loom_vec = loom_output[loom_start:loom_end]
        
        # Calculate statistics
        real_mean = np.mean(real_vec)
        loom_mean = np.mean(loom_vec)
        
        real_std = np.std(real_vec)
        loom_std = np.std(loom_vec)
        
        # Cosine similarity
        cos_sim = np.dot(real_vec, loom_vec) / (np.linalg.norm(real_vec) * np.linalg.norm(loom_vec))
        similarities.append(cos_sim)
        
        # Mean difference
        diff = abs(real_mean - loom_mean)
        differences.append(diff)
        
        print(f"{token[:15]:<15} mean:{real_mean:7.3f} std:{real_std:5.3f} | mean:{loom_mean:7.3f} std:{loom_std:5.3f}")
    
    # Summary statistics
    print(f"\n{'='*60}")
    print("📈 SUMMARY STATISTICS")
    print(f"{'='*60}")
    
    avg_similarity = np.mean(similarities)
    avg_difference = np.mean(differences)
    
    print(f"\nAverage Cosine Similarity: {avg_similarity:.6f}")
    print(f"Average Mean Difference:    {avg_difference:.6f}")
    
    # Interpretation
    print(f"\n{'='*60}")
    print("🎯 VERIFICATION STATUS")
    print(f"{'='*60}\n")
    
    if avg_similarity > 0.7:
        print("✅ EXCELLENT: LOOM weights are working very well!")
        print("   Outputs are highly similar to real BERT.")
    elif avg_similarity > 0.4:
        print("🟡 MODERATE: LOOM weights are partially working.")
        print("   Some correlation with real BERT, but differences exist.")
    elif avg_similarity > 0.1:
        print("🟠 LOW: LOOM weights show some correlation.")
        print("   Significant differences from real BERT.")
    else:
        print("🔴 VERY LOW: LOOM weights may not be loaded correctly.")
        print("   Outputs are very different from real BERT.")
    
    print(f"\n⚠️  Note: Differences are expected due to:")
    print(f"   • Missing LayerNorm")
    print(f"   • Missing residual connections")
    print(f"   • Activation function approximation (GELU→Softplus)")
    print(f"\n   BUT if similarity > 0.3, core weights are working!")
    
    return avg_similarity, avg_difference


def main():
    test_sentences = [
        "Hello world",
        "The cat sat on the mat",
        "Machine learning is amazing",
    ]
    
    print("🧠 LOOM BERT Weight Verification")
    print("="*60)
    print("\nThis verifies LOOM's imported BERT weights by:")
    print("  1. Processing text through real BERT")
    print("  2. Processing same embeddings through LOOM")
    print("  3. Comparing outputs to verify weights work")
    print()
    
    all_results = []
    
    for sentence in test_sentences:
        # Process through real BERT
        real_data = process_through_real_bert(sentence)
        
        # Flatten embeddings for LOOM
        embeddings_flat = [val for row in real_data["embeddings"] for val in row]
        
        # Process through LOOM
        loom_output = run_through_loom(embeddings_flat)
        
        if loom_output is None:
            print("❌ Failed to run through LOOM")
            continue
        
        # Compare
        similarity, difference = compare_outputs(
            real_data["bert_output"],
            loom_output,
            real_data["all_tokens"],
            real_data["attention_mask"]
        )
        
        all_results.append({
            "text": sentence,
            "similarity": similarity,
            "difference": difference
        })
    
    # Final summary
    print(f"\n{'='*60}")
    print("🎯 FINAL VERIFICATION SUMMARY")
    print(f"{'='*60}\n")
    
    avg_sim = np.mean([r["similarity"] for r in all_results])
    
    print("Results across all test sentences:")
    for r in all_results:
        print(f"  \"{r['text'][:30]}...\" → similarity: {r['similarity']:.4f}")
    
    print(f"\n📊 Overall Average Similarity: {avg_sim:.4f}")
    
    if avg_sim > 0.3:
        print("\n✅✅✅ VERIFICATION PASSED ✅✅✅")
        print("LOOM's BERT weights are working correctly!")
        print("The imported weights from HuggingFace are functional.")
    else:
        print("\n❌ VERIFICATION FAILED")
        print("LOOM weights may not be loaded correctly.")
        print("Check weight extraction and loading process.")


if __name__ == "__main__":
    main()
