#!/usr/bin/env python3
"""Example 1: Build a dense stack and run forward (sequential + shape-aware)."""

import math

from welvet import DType, Network


def main() -> None:
    net = Network({
        "id": "example-dense",
        "depth": 1,
        "rows": 1,
        "cols": 1,
        "layers_per_cell": 2,
        "layers": [
            {
                "z": 0, "y": 0, "x": 0, "l": 0,
                "type": "dense",
                "dtype": "float32",
                "input_height": 16,
                "output_height": 8,
                "activation": "relu",
            },
            {
                "z": 0, "y": 0, "x": 0, "l": 1,
                "type": "dense",
                "dtype": "float32",
                "input_height": 8,
                "output_height": 4,
                "activation": "linear",
            },
        ],
    })

    inp = [0.2 * math.sin(i * 0.2) for i in range(16)]
    in_shape = [1, 16]

    # Shape-aware forward (preferred for training / MHA / CNN)
    out = net.forward_polymorphic(inp, in_shape)
    assert len(out) == 4, f"expected 4 outputs, got {len(out)}"

    # One-shot sequential forward (flat [batch×features] when layout matches)
    out_seq = net.forward(inp)
    assert len(out_seq) == 4

    info = net.info()
    assert info.get("total_layers") == 2

    net.free()
    print("01_dense_forward OK — out[0]=%.4f layers=%d" % (out[0], info.get("total_layers", 0)))


if __name__ == "__main__":
    main()
