import sys
import os
import ctypes
import json
from pathlib import Path

# Add the package source to sys.path
sys.path.append(str(Path(__file__).parent))

try:
    from welvet import utils
    from welvet.utils import _LIB
except ImportError as e:
    print(f"Error importing welvet: {e}")
    sys.exit(1)

def get_expected_symbols():
    """Extract expected Loom symbols from welvet.h"""
    # Use the header in the same directory as the dll
    header_path = Path(utils._lib_path()).parent / "welvet.h"
    if not header_path.exists():
        # Try fallback to windows_amd64 source
        header_path = Path(__file__).parent / "windows_amd64" / "welvet.h"
    
    if not header_path.exists():
        print(f"Warning: welvet.h not found at {header_path}")
        return []

    symbols = []
    with open(header_path, "r") as f:
        for line in f:
            if line.startswith("extern "):
                parts = line.split("(")
                if len(parts) > 0:
                    decl = parts[0].strip()
                    name = decl.split()[-1].strip("*")
                    if name.startswith("Loom") or name == "FreeLoomString":
                        symbols.append(name)
    return symbols

def main():
    print(f"=== Welvet Python C-ABI Verification ===")
    print(f"[*] Loaded library: {utils._lib_path()}")

    expected = get_expected_symbols()
    if not expected:
        print("[!] Could not retrieve expected symbols from header.")
        sys.exit(1)

    print(f"[*] Found {len(expected)} expected symbols in header.")

    missing = []
    found_count = 0
    for sym in expected:
        if hasattr(_LIB, sym):
            found_count += 1
        else:
            missing.append(sym)

    print(f"[+] Found {found_count} / {len(expected)} symbols.")

    if missing:
        print(f"[!] Missing symbols ({len(missing)}):")
        for m in missing:
            print(f"  - {m}")
    else:
        print("[+] All expected symbols are present in the shared library.")

    # Functional test
    print("\n[*] Running basic functional test (Sequential Forward)...")
    try:
        config = {
            "id": "verify_net",
            "depth": 1, "rows": 1, "cols": 1, "layers_per_cell": 1,
            "layers": [
                {
                    "z": 0, "y": 0, "x": 0, "l": 0,
                    "type": "dense", 
                    "input_height": 4, 
                    "output_height": 2, 
                    "activation": "silu",
                    "dtype": "float32"
                }
            ]
        }
        net = utils.Network(config)
        print("[+] Created network.")
        
        inp = [1.0, 0.5, -0.2, 0.8]
        out = net.forward(inp)
        print(f"[+] Forward pass output: {out}")
        
        net.free()
        print("[+] Freed network.")
        print("\n=== VERIFICATION SUCCESSFUL ===")
    except Exception as e:
        print(f"\n[!] Functional test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
