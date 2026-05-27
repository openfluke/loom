#!/usr/bin/env python3
"""Redirects to benchmark_seven_layer.py (lucy [7] suite via Loom CABI)."""
import subprocess
import sys
import os

if __name__ == "__main__":
    script = os.path.join(os.path.dirname(__file__), "benchmark_seven_layer.py")
    raise SystemExit(subprocess.call([sys.executable, script] + sys.argv[1:]))
