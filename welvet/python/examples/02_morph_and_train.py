#!/usr/bin/env python3
"""Example 2: Morph dtype, CPU train with explicit tensor shapes."""

import math

from welvet import DType, Network, train

TRAINING_MODE_CPU_MC = 2


def main() -> None:
    in_shape = [1, 16]
    out_shape = [1, 8]
    inp = [0.2 * math.sin(i * 0.3) for i in range(16)]
    tgt = [0.5 + 0.3 * math.sin(i * 0.17) for i in range(8)]

    net = Network({
        "id": "example-train",
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

    net.set_training_mode(TRAINING_MODE_CPU_MC)
    out_fp32 = net.forward_polymorphic(inp, in_shape)

    net.morph(0, DType.INT8)
    out_int8 = net.forward_polymorphic(inp, in_shape)

    hist = train(
        net,
        [[inp]],
        [[tgt]],
        epochs=8,
        learning_rate=0.05,
        mode=TRAINING_MODE_CPU_MC,
        use_gpu=False,
        verbose=False,
        input_shape=in_shape,
        target_shape=out_shape,
    )
    assert hist and all(math.isfinite(x) for x in hist), "bad loss history"

    net.free()
    print(
        "02_morph_and_train OK — fp32[0]=%.4f int8[0]=%.4f loss %.4e→%.4e"
        % (out_fp32[0], out_int8[0], hist[0], hist[-1])
    )


if __name__ == "__main__":
    main()
