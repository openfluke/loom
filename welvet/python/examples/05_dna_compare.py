#!/usr/bin/env python3
"""Example 5: Extract DNA fingerprints and compare two networks."""

from welvet import Network, compare_dna


def _tiny_net(net_id: str) -> Network:
    return Network({
        "id": net_id,
        "depth": 1,
        "rows": 1,
        "cols": 1,
        "layers_per_cell": 1,
        "layers": [{
            "z": 0, "y": 0, "x": 0, "l": 0,
            "type": "dense",
            "dtype": "float32",
            "input_height": 8,
            "output_height": 4,
            "activation": "relu",
        }],
    })


def main() -> None:
    a = _tiny_net("dna-a")
    b = _tiny_net("dna-b")

    dna_a = a.dna()
    dna_b = b.dna()
    same = compare_dna(dna_a, dna_a)
    diff = compare_dna(dna_a, dna_b)

    a.free()
    b.free()

    overlap_same = float(same.get("OverallOverlap", 0))
    overlap_diff = float(diff.get("OverallOverlap", 0))
    assert overlap_same > 0.99, f"identical DNA should match, got {overlap_same}"
    assert overlap_diff < overlap_same, "distinct nets should overlap less than self"

    print(
        "05_dna_compare OK — self=%.3f cross=%.3f"
        % (overlap_same, overlap_diff)
    )


if __name__ == "__main__":
    main()
