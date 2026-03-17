# benchmark_tiling.py
import time
import sys
import os
from typing import List

# Ensure we can import welvet from the current directory or src
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import welvet
from welvet import Network, DType, LayerType, Activation, build_network, free_network, \
                   init_wgpu, sync_to_gpu, LayerType, Network, DType

def run_bench(l_type: int, name: str):
    iterations = 10
    
    # Setup network config for a single layer
    config = {
        "id": "bench_net",
        "depth": 1, "rows": 1, "cols": 1, "layers_per_cell": 1,
        "layers": []
    }
    
    # Match the Go benchmark settings
    layer_spec = {
        "type": LayerType.name(l_type).lower().replace("_", ""),
        "input_channels": 128,
        "input_height": 128,
        "input_width": 64,
        "input_depth": 64,
        "filters": 128,
        "output_height": 128,
        "output_width": 64,
        "output_depth": 64,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "seq_length": 1,
        "num_heads": 4,
        "num_kv_heads": 4,
        "head_dim": 32,
        "d_model": 128,
        "max_seq_len": 512,
        "dtype": "float32"
    }

    # Fine-tune layer specs to match Go benchmark_tiling.go
    if l_type == LayerType.DENSE:
        layer_spec["input_height"] = 1024
        layer_spec["output_height"] = 1024
    elif l_type == LayerType.RNN:
        layer_spec["input_height"] = 512
        layer_spec["output_height"] = 512
    elif l_type == LayerType.LSTM:
        layer_spec["input_height"] = 512
        layer_spec["output_height"] = 512
    elif l_type == LayerType.CNN1:
        layer_spec["input_channels"] = 64
        layer_spec["input_height"] = 512
        layer_spec["filters"] = 64
    elif l_type == LayerType.CNN2:
        layer_spec["input_channels"] = 32
        layer_spec["input_height"] = 64
        layer_spec["input_width"] = 64
        layer_spec["filters"] = 32
    elif l_type == LayerType.CNN3:
        layer_spec["input_channels"] = 16
        layer_spec["input_depth"] = 16
        layer_spec["input_height"] = 16
        layer_spec["input_width"] = 16
        layer_spec["filters"] = 16
    elif l_type == LayerType.EMBEDDING:
        layer_spec["vocab_size"] = 2048
        layer_spec["embedding_dim"] = 128
    elif l_type == LayerType.RMS_NORM:
        layer_spec["input_height"] = 1024
        layer_spec["output_height"] = 1024
    elif l_type == LayerType.MHA:
        layer_spec["d_model"] = 128
        layer_spec["num_heads"] = 4
        layer_spec["num_kv_heads"] = 4
        layer_spec["head_dim"] = 32
    elif l_type == LayerType.SWIGLU:
        layer_spec["input_height"] = 512
        layer_spec["output_height"] = 1024
    elif l_type == LayerType.RESIDUAL:
        layer_spec["input_height"] = 1024
        layer_spec["output_height"] = 1024

    config["layers"].append(layer_spec)
    
    net = Network(config)
    try:
        # Prepare inputs
        if l_type == LayerType.CNN3:
            input_len = 1 * layer_spec["input_channels"] * layer_spec["input_depth"] * layer_spec["input_height"] * layer_spec["input_width"]
        elif l_type == LayerType.CNN2:
            input_len = 1 * layer_spec["input_channels"] * layer_spec["input_height"] * layer_spec["input_width"]
        elif l_type == LayerType.CNN1:
            input_len = 1 * layer_spec["input_channels"] * layer_spec["input_height"]
        elif l_type == LayerType.EMBEDDING:
            input_len = 64
        else:
            input_len = 1 * layer_spec["input_height"]

        if l_type == LayerType.EMBEDDING:
            input_data = [float(i % layer_spec["vocab_size"]) for i in range(input_len)]
        else:
            input_data = [0.5] * input_len

        # 1. CPU Simple
        cpu_out = []
        start = time.perf_counter()
        for _ in range(iterations):
            with net.create_state(DType.FLOAT32) as state:
                state.set_input(input_data)
                state.step()
                cpu_out = state.output(0)
        simple_time = (time.perf_counter() - start) / iterations
        
        # 2. GPU
        gpu_time = 0.0
        max_diff = -1.0
        sanity = False
        gpu_sample = []
        
        try:
            welvet.init_wgpu(net._handle)
            welvet.sync_to_gpu(net._handle)
            
            out_size = len(cpu_out) if cpu_out else 1024
            in_buf = welvet.create_gpu_buffer(net._handle, len(input_data) * 4)
            out_buf = welvet.create_gpu_buffer(net._handle, out_size * 4)
            
            welvet.write_gpu_buffer(net._handle, in_buf, input_data)
            
            start = time.perf_counter()
            for _ in range(iterations):
                welvet.dispatch_forward_layer(net._handle, 0, 1, in_buf, out_buf)
            gpu_time = (time.perf_counter() - start) / iterations
            
            gpu_res = welvet.read_gpu_buffer(net._handle, out_buf)
            
            welvet.free_gpu_buffer(in_buf)
            welvet.free_gpu_buffer(out_buf)
            
            # Parity
            if gpu_res and cpu_out:
                max_diff = max(abs(g - c) for g, c in zip(gpu_res[:100], cpu_out[:100]))
                sanity = any(abs(v) > 1e-6 for v in gpu_res[:100])
                gpu_sample = gpu_res[:3]
        except Exception as e:
            print(f"  [!] GPU skip for {name}: {e}")
            pass

        return simple_time, gpu_time, max_diff, sanity, cpu_out[:3], gpu_sample

    finally:
        net.free()

def main():
    print("=== M-POLY-VTD Performance Showdown: CPU vs GPU Acceleration (Python) ===")
    
    layers = [
        (LayerType.DENSE, "Dense (Linear)"),
        (LayerType.RNN, "RNN Cell"),
        (LayerType.LSTM, "LSTM Cell"),
        (LayerType.CNN1, "CNN 1D"),
        (LayerType.CNN2, "CNN 2D"),
        (LayerType.CNN3, "CNN 3D"),
        (LayerType.EMBEDDING, "Embedding"),
        (LayerType.RMS_NORM, "RMSNorm"),
        (LayerType.MHA, "MHA (Attn)"),
        (LayerType.SWIGLU, "SwiGLU (MLP)"),
        (LayerType.RESIDUAL, "Residual Add"),
    ]

    print(f"| {'Layer type':<15} | {'CPU (Simple)':<12} | {'GPU (WebGPU)':<12} | {'Speedup':<12} | {'Parity':<15} | {'Sanity':<10} |")
    print(f"|{'-'*17}|{'-'*14}|{'-'*14}|{'-'*14}|{'-'*17}|{'-'*12}|")

    samples = []

    for l_type, l_name in layers:
        simple, gpu, diff, sanity, c_samp, g_samp = run_bench(l_type, l_name)
        
        samples.append((l_name, c_samp, g_samp))
        
        gpu_str = f"{gpu*1000:.2f}ms" if gpu > 0 else "N/A"
        simple_str = f"{simple*1000:.2f}ms"
        speedup = f"{simple/gpu:.2f}x" if gpu > 0 else "N/A"
        
        parity = "EXACT [OK]" if diff < 1e-7 and diff >= 0 else ("GOOD [OK]" if diff < 1e-4 else "ERROR [!!]")
        if diff < 0: parity = "N/A"
        
        san_str = "REAL [OK]" if sanity else "ZERO [!!]"
        
        print(f"| {l_name:<15} | {simple_str:<12} | {gpu_str:<12} | {speedup:<12} | {parity:<15} | {san_str:<10} |")

    print("\n=== Sample Comparison (First 3) ===")
    for name, c, g in samples:
        c_str = ", ".join(f"{v:.4f}" for v in c) if c else "N/A"
        g_str = ", ".join(f"{v:.4f}" for v in g) if g else "N/A"
        print(f"{name:<15} CPU: [{c_str}]")
        print(f"{'':<15} GPU: [{g_str}]")

if __name__ == "__main__":
    main()
