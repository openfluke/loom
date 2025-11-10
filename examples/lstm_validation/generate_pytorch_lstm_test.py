#!/usr/bin/env python3
"""
Generate LSTM test cases from PyTorch for validation against LOOM.
This script creates JSON files with inputs, weights, and expected outputs.
"""

import torch
import torch.nn as nn
import json
import argparse

def generate_test_case(input_size, hidden_size, batch_size, seq_length, seed=42):
    """Generate a PyTorch LSTM test case"""
    torch.manual_seed(seed)
    
    # Create LSTM layer
    lstm = nn.LSTM(input_size, hidden_size, batch_first=False, bias=True)
    lstm.eval()  # Set to evaluation mode
    
    # Generate random input [seq_len, batch, input_size]
    input_data = torch.randn(seq_length, batch_size, input_size)
    
    # Initialize hidden and cell states to zeros
    h0 = torch.zeros(1, batch_size, hidden_size)
    c0 = torch.zeros(1, batch_size, hidden_size)
    
    # Forward pass
    with torch.no_grad():
        output, (hn, cn) = lstm(input_data, (h0, c0))
    
    # Extract weights and biases
    weight_ih = lstm.weight_ih_l0.detach().numpy()  # [4*hidden_size, input_size]
    weight_hh = lstm.weight_hh_l0.detach().numpy()  # [4*hidden_size, hidden_size]
    bias_ih = lstm.bias_ih_l0.detach().numpy()      # [4*hidden_size]
    bias_hh = lstm.bias_hh_l0.detach().numpy()      # [4*hidden_size]
    
    # Create test case dictionary
    test_case = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "batch_size": batch_size,
        "seq_length": seq_length,
        "input": input_data.numpy().tolist(),           # [seq, batch, input]
        "initial_h": h0.squeeze(0).numpy().tolist(),    # [batch, hidden]
        "initial_c": c0.squeeze(0).numpy().tolist(),    # [batch, hidden]
        "weight_ih": weight_ih.tolist(),                # [4*hidden, input]
        "weight_hh": weight_hh.tolist(),                # [4*hidden, hidden]
        "bias_ih": bias_ih.tolist(),                    # [4*hidden]
        "bias_hh": bias_hh.tolist(),                    # [4*hidden]
        "expected_h": hn.squeeze(0).numpy().tolist(),   # [batch, hidden]
        "expected_c": cn.squeeze(0).numpy().tolist(),   # [batch, hidden]
        "expected_out": output.numpy().tolist()         # [seq, batch, hidden]
    }
    
    return test_case

def main():
    parser = argparse.ArgumentParser(description='Generate PyTorch LSTM test cases')
    parser.add_argument('--input-size', type=int, default=5, help='Input feature size')
    parser.add_argument('--hidden-size', type=int, default=8, help='Hidden state size')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
    parser.add_argument('--seq-length', type=int, default=3, help='Sequence length')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='pytorch_lstm_test.json', 
                        help='Output JSON file')
    
    args = parser.parse_args()
    
    print(f"Generating LSTM test case:")
    print(f"  Input size: {args.input_size}")
    print(f"  Hidden size: {args.hidden_size}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Sequence length: {args.seq_length}")
    print(f"  Seed: {args.seed}")
    
    test_case = generate_test_case(
        args.input_size, 
        args.hidden_size, 
        args.batch_size, 
        args.seq_length,
        args.seed
    )
    
    with open(args.output, 'w') as f:
        json.dump(test_case, f, indent=2)
    
    print(f"\n✅ Test case saved to: {args.output}")
    print(f"\nTo test with LOOM:")
    print(f"  cd examples && go run lstm_pytorch_validation.go {args.output}")
    
    # Print sample values
    print(f"\nSample values:")
    print(f"  First input: {test_case['input'][0][0][:3]}...")
    print(f"  First output: {test_case['expected_out'][0][0][:3]}...")
    print(f"  Final hidden: {test_case['expected_h'][0][:3]}...")

if __name__ == '__main__':
    main()
