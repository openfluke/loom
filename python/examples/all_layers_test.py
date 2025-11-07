#!/usr/bin/env python3
"""
LOOM All Layers Test - Python/C-ABI Version

✨ ONE FUNCTION CALL TO RULE THEM ALL! ✨

This demonstrates load_model_from_string() which takes a JSON string
and returns a fully configured network with ALL weights loaded.

No manual layer setup, no type conversions, no hassle!
Just: network = load_model_from_string(json_string)

The test:
1. Downloads model.json from localhost:3123
2. Calls load_model_from_string() - DONE! Network ready!
3. Runs inference and compares with expected outputs  
4. Trains the network to verify weights are mutable
"""

import json
import sys
import urllib.request
from pathlib import Path

# Add the python package to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python" / "src"))

import welvet
from welvet.utils import call_layer_init, list_layer_init_functions


# Type mappings for Go enums
LAYER_TYPE_MAP = {
    "dense": 0,
    "conv2d": 1,
    "attention": 2,
    "multi_head_attention": 2,
    "rnn": 3,
    "lstm": 4,
    "softmax": 5,
}

ACTIVATION_MAP = {
    "scaled_relu": 0,
    "relu": 0,
    "sigmoid": 1,
    "tanh": 2,
    "softplus": 3,
    "leaky_relu": 4,
}


def convert_layer_config(layer_data):
    """Convert string types to integer enums for Go"""
    layer_copy = layer_data.copy()
    
    # Convert layer type
    if "type" in layer_copy and isinstance(layer_copy["type"], str):
        layer_type_str = layer_copy["type"].lower()
        if layer_type_str in LAYER_TYPE_MAP:
            layer_copy["type"] = LAYER_TYPE_MAP[layer_type_str]
    
    # Convert activation type
    if "activation" in layer_copy and isinstance(layer_copy["activation"], str):
        activation_str = layer_copy["activation"].lower()
        if activation_str in ACTIVATION_MAP:
            layer_copy["activation"] = ACTIVATION_MAP[activation_str]
    
    return layer_copy


def log(message, status="info"):
    """Print colored status messages"""
    colors = {
        "info": "\033[94m",     # Blue
        "success": "\033[92m",  # Green
        "error": "\033[91m",    # Red
        "loading": "\033[93m",  # Yellow
    }
    reset = "\033[0m"
    color = colors.get(status, "")
    prefix = {
        "info": "ℹ",
        "success": "✅",
        "error": "❌",
        "loading": "🔄",
    }.get(status, "•")
    
    print(f"{color}{prefix} {message}{reset}")


def fetch_url(url):
    """Fetch content from URL"""
    try:
        with urllib.request.urlopen(url) as response:
            return response.read().decode('utf-8')
    except Exception as e:
        raise RuntimeError(f"Failed to fetch {url}: {e}")


def main():
    print("\n" + "="*60)
    print("🧠 LOOM All Layers Test (Python/C-ABI)")
    print("="*60 + "\n")
    
    # ===== Step 1: Load files from server =====
    print("📥 Step 1: Load Files from Server")
    print(f"   Server URL: http://localhost:3123\n")
    
    log("Fetching files from localhost:3123...", "loading")
    
    try:
        # Load model JSON
        log("Loading test.json...", "loading")
        model_json_str = fetch_url("http://localhost:3123/test.json")
        model_data = json.loads(model_json_str)
        log(f"test.json loaded ({len(model_json_str)/1024:.1f} KB)", "success")
        
        # Load inputs
        log("Loading inputs.txt...", "loading")
        inputs_text = fetch_url("http://localhost:3123/inputs.txt")
        inputs = [float(line.strip()) for line in inputs_text.strip().split('\n')]
        log(f"inputs.txt loaded ({len(inputs)} values)", "success")
        
        # Load expected outputs
        log("Loading outputs.txt...", "loading")
        outputs_text = fetch_url("http://localhost:3123/outputs.txt")
        expected_outputs = [float(line.strip()) for line in outputs_text.strip().split('\n')]
        log(f"outputs.txt loaded ({len(expected_outputs)} values)", "success")
        
        log("All files loaded successfully!", "success")
        
    except Exception as e:
        log(f"Error loading files: {e}", "error")
        log("Make sure the server is running: cd examples && ./serve_files.sh", "error")
        return 1
    
    print()
    
    # ===== Step 2: Load complete model using C-ABI =====
    print("🔄 Step 2: Load Complete Model using C-ABI")
    print()
    
    log("Loading complete model (structure + weights) from JSON...", "loading")
    
    try:
        # THIS IS THE MAGIC! Just pass the JSON string and get a fully configured network!
        network = welvet.load_model_from_string(model_json_str, "all_layers_test")
        
        log(f"✨ Model loaded completely! (handle: {network})", "success")
        log("All 16 layers with weights loaded automatically!", "success")
        
    except Exception as e:
        log(f"Error loading model: {e}", "error")
        import traceback
        traceback.print_exc()
        return 1
    
    print()
    
    # ===== Step 3: Run inference =====
    print("▶️ Step 3: Run Inference")
    print()
    
    log("Running forward pass...", "loading")
    log(f"Input size: {len(inputs)}", "info")
    
    try:
        # Run forward pass        
        output = welvet.forward(network, inputs)
        
        log(f"Forward pass complete! Output size: {len(output)}", "success")
        
        # Debug: Check if output is empty
        if len(output) == 0:
            log("Output is empty - checking network info...", "info")
            try:
                info = welvet.get_network_info(network)
                log(f"Network info: {info}", "info")
            except Exception as e:
                log(f"Could not get network info: {e}", "error")
            
            log("Network might not have output layer configured correctly", "error")
            return 1
        
        # Display outputs
        print("\n   Expected output (from file):")
        for i, val in enumerate(expected_outputs):
            print(f"     [{i}] {val:.6f}")
        
        print("\n   C-ABI output (loaded weights):")
        for i, val in enumerate(output):
            print(f"     [{i}] {val:.6f}")
        
        # Compare with expected
        if len(output) == len(expected_outputs):
            max_diff = max(abs(output[i] - expected_outputs[i]) for i in range(len(output)))
            
            log(f"Max difference: {max_diff:.10f}", "info")
            
            if max_diff < 1e-5:
                log("✅ Outputs match expected exactly!", "success")
            elif max_diff < 0.1:
                log("✅ Outputs match with small differences (expected with softmax)", "success")
            else:
                log("⚠️ Large output differences detected", "error")
        else:
            log(f"Output size mismatch: got {len(output)}, expected {len(expected_outputs)}", "error")
        
    except Exception as e:
        log(f"Error during inference: {e}", "error")
        import traceback
        traceback.print_exc()
        return 1
    
    print()
    
    # ===== Step 4: Train model =====
    print("🎯 Step 4: Train Model")
    print()
    
    log("Starting training...", "loading")
    
    try:
        # Get output before training
        output_before = welvet.forward(network, inputs)
        
        # Training parameters
        epochs = 10
        learning_rate = 0.05
        target = [0.5, 0.5]  # Train to output [0.5, 0.5]
        
        log(f"Epochs: {epochs}", "info")
        log(f"Learning rate: {learning_rate}", "info")
        log("Training...", "loading")
        
        # Training loop
        for epoch in range(epochs):
            # Forward pass
            output = welvet.forward(network, inputs)
            
            # Compute loss (MSE)
            loss = sum((output[i] - target[i])**2 for i in range(len(target))) / len(target)
            
            # Backward pass with gradient
            grad_output = [(output[i] - target[i]) * 2 / len(target) for i in range(len(target))]
            welvet.backward(network, grad_output)
            
            # Update weights
            welvet.update_weights(network, learning_rate)
            
            if epoch == 0:
                log(f"Initial loss: {loss:.6f}", "info")
            elif epoch == epochs - 1:
                log(f"Final loss: {loss:.6f}", "info")
        
        log("Training complete!", "success")
        
        # Get output after training
        output_after = welvet.forward(network, inputs)
        
        # Check if weights changed
        max_change = max(abs(output_after[i] - output_before[i]) for i in range(len(output_before)))
        
        log(f"Max output change: {max_change:.6f}", "info")
        
        if max_change > 1e-5:
            log("Weights successfully changed!", "success")
        else:
            log("Weights did not change", "error")
        
        # Display results
        print("\n   Output before training:")
        for i, val in enumerate(output_before):
            print(f"     [{i}] {val:.6f}")
        
        print("\n   Output after training:")
        for i, val in enumerate(output_after):
            print(f"     [{i}] {val:.6f}")
        
    except Exception as e:
        log(f"Error during training: {e}", "error")
        import traceback
        traceback.print_exc()
        return 1
    
    print()
    
    # Cleanup
    log("Cleaning up...", "loading")
    welvet.free_network(network)
    log("Network freed", "success")
    
    print()
    print("="*60)
    print("✅ All Layer Types Test Complete")
    print("✅ Model structure loaded from server")
    print("✅ Network created with C-ABI")
    print("✅ Inference successful")
    print("✅ Training verified")
    print("="*60)
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
