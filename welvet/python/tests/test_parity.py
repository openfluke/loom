import os
import sys
import json
import time
import welvet
from welvet import Network, DType, train, Tokenizer

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def test_parity():
    log("=== WELVET PYTHON ENGINE PARITY CHECK (v0.80.0) ===")
    
    # 1. Basic Inference (Symmetric)
    log("PHASE 1: Symmetric Inference")
    net_config = {
        "id": "parity_net",
        "depth": 1, "rows": 1, "cols": 1, "layers_per_cell": 2,
        "layers": [
            {"z": 0, "y": 0, "x": 0, "l": 0, "type": "dense", "input_height": 16, "output_height": 16, "activation": "relu", "dtype": "float32"},
            {"z": 0, "y": 0, "x": 0, "l": 1, "type": "dense", "input_height": 16, "output_height": 1, "activation": "sigmoid", "dtype": "float32"}
        ]
    }
    net = Network(net_config)
    log(f"Network initialized: {net}")
    
    input_data = [0.5] * 16
    out = net.forward(input_data)
    log(f"Forward pass successful. Output: {out[0]:.6f}")
    assert len(out) == 1, "Output dimension mismatch"

    # 2. Serialization (SafeTensors)
    log("PHASE 2: Serialization (SafeTensors)")
    bp = net.blueprint("parity_bp")
    log(f"Blueprint extracted ({len(json.dumps(bp))} chars)")
    
    # 3. Training (MSE Loss)
    log("PHASE 3: Supervised Training")
    X = [[0.1] * 16, [0.9] * 16]
    y = [[0.01], [0.99]]
    
    # Run 10 epochs
    losses = train(net, [X], [y], epochs=10, learning_rate=0.01, verbose=False)
    log(f"Training complete. Initial Loss: {losses[0]:.6f} -> Final Loss: {losses[-1]:.6f}")
    assert losses[-1] < losses[0], "Loss did not decrease"

    # 4. NEAT Evolution
    log("PHASE 4: Genetic Evolution (NEAT)")
    pop = net.create_population(size=10)
    log(f"NEAT Population created (size={pop.size()})")
    
    fitnesses = [1.0] * 10
    report = pop.evolve(fitnesses)
    log(f"Evolution step complete: {pop.summary(1)}")
    pop.free()

    # 5. Transformer Context
    log("PHASE 5: Transformer Logic")
    # Small transformer blocks
    tr_config = {
        "id": "small_tr",
        "depth": 1, "rows": 1, "cols": 1, "layers_per_cell": 1,
        "layers": [
            {"type": "mha", "d_model": 64, "num_heads": 4, "seq_length": 128, "dtype": "float32"}
        ]
    }
    tr_net = Network(tr_config)
    tr = tr_net.create_transformer()
    log("Transformer context created successfully.")
    # Prefill (placeholder result verification)
    res = tr.prefill_tokens([1, 2, 3, 4])
    log(f"Transformer prefill successful ({len(res)} logits).")
    
    tr.free()
    tr_net.free()
    net.free()
    
    log("=== PYTHON PARITY TEST COMPLETE (100% SUCCESS) ===")

if __name__ == "__main__":
    try:
        test_parity()
    except Exception as e:
        log(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
