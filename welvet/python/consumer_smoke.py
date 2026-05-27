#!/usr/bin/env python3
"""Quick smoke test: import welvet, forward, morph, train, serialize (matches TS consumer_demo)."""

import math
import sys

from welvet import DType, Network, __version__, morph_layer, train


def main() -> int:
    print(f"consumer_smoke — welvet {__version__}\n")

    in_shape = [1, 16]
    out_shape = [1, 8]
    inp = [0.2 * math.sin(i * 0.3) for i in range(16)]
    tgt = [0.5 + 0.3 * math.sin(i * 0.17) for i in range(8)]

    net = Network({
        "id": "consumer-smoke",
        "depth": 1,
        "rows": 1,
        "cols": 1,
        "layers_per_cell": 1,
        "layers": [{
            "z": 0, "y": 0, "x": 0, "l": 0,
            "type": "dense",
            "input_height": 16,
            "output_height": 8,
            "activation": "relu",
            "dtype": "float32",
        }],
    })

    net.set_training_mode(2)  # CPU MC
    out0 = net.forward_polymorphic(inp, in_shape)
    assert len(out0) == 8, f"expected len 8, got {len(out0)}"

    morph_layer(net._handle, 0, DType.INT8)
    out1 = net.forward_polymorphic(inp, in_shape)

    hist = train(
        net,
        [[inp]],
        [[tgt]],
        epochs=5,
        learning_rate=0.05,
        mode=2,
        use_gpu=False,
        verbose=False,
        input_shape=in_shape,
        target_shape=out_shape,
    )
    assert hist, "empty loss history"
    assert math.isfinite(hist[-1]), "non-finite final loss"

    wire = net.serialize()
    reloaded = Network.deserialize(wire)
    out2 = reloaded.forward_polymorphic(inp, in_shape)
    reloaded.free()

    drift = max(abs(a - b) for a, b in zip(out1, out2))
    net.free()
    assert drift < 0.25, f"reload drift too large: {drift}"

    print("✅ consumer_smoke passed\n")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)
