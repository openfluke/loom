#!/usr/bin/env python3
"""Example 4: Multi-head attention with [batch, seq, d_model] layout."""

import math

from welvet import Network


def main() -> None:
    batch, seq, d_model, heads = 2, 4, 32, 4
    in_shape = [batch, seq, d_model]

    net = Network({
        "id": "example-mha",
        "depth": 1,
        "rows": 1,
        "cols": 1,
        "layers_per_cell": 1,
        "layers": [{
            "z": 0, "y": 0, "x": 0, "l": 0,
            "type": "mha",
            "dtype": "float32",
            "d_model": d_model,
            "num_heads": heads,
            "seq_length": seq,
            "activation": "relu",
        }],
    })

    n = batch * seq * d_model
    inp = [0.15 * math.sin(i * 0.09) for i in range(n)]
    out = net.forward_polymorphic(inp, in_shape)

    assert len(out) == n, f"expected {n} outputs, got {len(out)}"
    assert all(math.isfinite(x) for x in out[:8])

    net.free()
    print("04_mha_forward OK — shape=%s out[0]=%.4f" % (in_shape, out[0]))


if __name__ == "__main__":
    main()
