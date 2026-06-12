#!/usr/bin/env python3
"""Example 3: JSON wire and native .entity checkpoint roundtrip."""

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
    from_wire = Network.deserialize(wire)
    drift_wire = max(abs(a - b) for a, b in zip(before, from_wire.forward_polymorphic(inp, in_shape)))
    from_wire.free()

    blob = net.serialize_entity()
    assert isinstance(blob, (bytes, bytearray)) and len(blob) > 32
    from_entity = Network.deserialize_entity(blob)
    from_entity.sync_inference_weights()
    drift_entity = max(abs(a - b) for a, b in zip(before, from_entity.forward_polymorphic(inp, in_shape)))
    from_entity.free()
    net.free()

    assert drift_wire < 1e-5, f"JSON reload drift too large: {drift_wire}"
    assert drift_entity < 1e-5, f"entity reload drift too large: {drift_entity}"
    print(
        "03_save_reload OK — max|Δ| wire=%.2e entity=%.2e wire_bytes=%d entity_bytes=%d"
        % (drift_wire, drift_entity, len(wire), len(blob))
    )


if __name__ == "__main__":
    main()
