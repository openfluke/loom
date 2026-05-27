#!/usr/bin/env python3
"""Example 3: Serialize network wire and reload (native Loom checkpoint format)."""

import math

from welvet import Network


def main() -> None:
    in_shape = [1, 16]
    inp = [0.1 * math.sin(i * 0.11) for i in range(16)]

    net = Network({
        "id": "example-save",
        "depth": 1,
        "rows": 1,
        "cols": 1,
        "layers_per_cell": 1,
        "layers": [{
            "z": 0, "y": 0, "x": 0, "l": 0,
            "type": "dense",
            "dtype": "float32",
            "input_height": 16,
            "output_height": 8,
            "activation": "relu",
        }],
    })

    before = net.forward_polymorphic(inp, in_shape)
    wire = net.serialize()
    assert isinstance(wire, str) and len(wire) > 32

    reloaded = Network.deserialize(wire)
    after = reloaded.forward_polymorphic(inp, in_shape)

    drift = max(abs(a - b) for a, b in zip(before, after))
    reloaded.free()
    net.free()

    assert drift < 1e-5, f"reload drift too large: {drift}"
    print("03_save_reload OK — max|Δ|=%.2e wire_bytes=%d" % (drift, len(wire)))


if __name__ == "__main__":
    main()
