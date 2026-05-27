#!/usr/bin/env python3
"""Run every script in examples/ (used by README and CI smoke)."""

import subprocess
import sys
from pathlib import Path

EXAMPLES = [
    "01_dense_forward.py",
    "02_morph_and_train.py",
    "03_save_reload.py",
    "04_mha_forward.py",
    "05_dna_compare.py",
]


def main() -> int:
    root = Path(__file__).resolve().parent
    py = sys.executable
    failed = []

    print("welvet examples — running %d scripts\n" % len(EXAMPLES))
    for name in EXAMPLES:
        path = root / name
        print("→ %s" % name)
        r = subprocess.run([py, str(path)], cwd=root.parent)
        if r.returncode != 0:
            failed.append(name)
            print("  ❌ failed (exit %d)\n" % r.returncode)
        else:
            print("  ✓\n")

    if failed:
        print("FAILED: %s" % ", ".join(failed))
        return 1
    print("✅ all examples passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
