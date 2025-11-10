#!/usr/bin/env python3
"""
Generate Conv2D test cases from PyTorch for validation against LOOM.
This script creates JSON files with inputs, weights, and expected outputs.
"""

import torch
import torch.nn as nn
import json
import argparse

def generate_test_case(batch_size, in_channels, height, width, out_channels, 
                       kernel_size, stride, padding, seed=42):
    """Generate a PyTorch Conv2D test case"""
    torch.manual_seed(seed)
    
    # Create Conv2D layer (no activation for direct comparison)
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, 
                     padding=padding, bias=True)
    conv.eval()  # Set to evaluation mode
    
    # Generate random input [batch, channels, height, width]
    input_data = torch.randn(batch_size, in_channels, height, width)
    
    # Forward pass
    with torch.no_grad():
        output = conv(input_data)
    
    # Extract weights and biases
    weight = conv.weight.detach().numpy()  # [out_channels, in_channels, kH, kW]
    bias = conv.bias.detach().numpy()      # [out_channels]
    
    # Create test case dictionary
    test_case = {
        "batch_size": batch_size,
        "input_channels": in_channels,
        "input_height": height,
        "input_width": width,
        "output_channels": out_channels,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "input": input_data.numpy().tolist(),      # [batch, in_ch, h, w]
        "weight": weight.tolist(),                 # [out_ch, in_ch, kh, kw]
        "bias": bias.tolist(),                     # [out_ch]
        "expected_out": output.numpy().tolist()    # [batch, out_ch, out_h, out_w]
    }
    
    return test_case

def main():
    parser = argparse.ArgumentParser(description='Generate PyTorch Conv2D test cases')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
    parser.add_argument('--in-channels', type=int, default=3, help='Input channels')
    parser.add_argument('--height', type=int, default=8, help='Input height')
    parser.add_argument('--width', type=int, default=8, help='Input width')
    parser.add_argument('--out-channels', type=int, default=4, help='Output channels (filters)')
    parser.add_argument('--kernel-size', type=int, default=3, help='Kernel size')
    parser.add_argument('--stride', type=int, default=1, help='Stride')
    parser.add_argument('--padding', type=int, default=1, help='Padding')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='pytorch_conv2d_test.json', 
                        help='Output JSON file')
    
    args = parser.parse_args()
    
    print(f"Generating Conv2D test case:")
    print(f"  Input: [{args.batch_size}, {args.in_channels}, {args.height}, {args.width}]")
    print(f"  Filters: {args.out_channels}")
    print(f"  Kernel: {args.kernel_size}x{args.kernel_size}")
    print(f"  Stride: {args.stride}, Padding: {args.padding}")
    print(f"  Seed: {args.seed}")
    
    test_case = generate_test_case(
        args.batch_size,
        args.in_channels,
        args.height,
        args.width,
        args.out_channels,
        args.kernel_size,
        args.stride,
        args.padding,
        args.seed
    )
    
    # Calculate output dimensions
    out_h = (args.height + 2*args.padding - args.kernel_size) // args.stride + 1
    out_w = (args.width + 2*args.padding - args.kernel_size) // args.stride + 1
    
    with open(args.output, 'w') as f:
        json.dump(test_case, f, indent=2)
    
    print(f"\n✅ Test case saved to: {args.output}")
    print(f"  Output shape: [{args.batch_size}, {args.out_channels}, {out_h}, {out_w}]")
    print(f"\nTo test with LOOM:")
    print(f"  cd examples/cnn_validation && go run cnn_pytorch_validation.go {args.output}")
    
    # Print sample values
    print(f"\nSample values:")
    print(f"  First input pixel: {test_case['input'][0][0][0][0]:.6f}")
    print(f"  First output pixel: {test_case['expected_out'][0][0][0][0]:.6f}")
    print(f"  First weight: {test_case['weight'][0][0][0][0]:.6f}")
    print(f"  First bias: {test_case['bias'][0]:.6f}")

if __name__ == '__main__':
    main()
