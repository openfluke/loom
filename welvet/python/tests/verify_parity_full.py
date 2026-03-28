import os
import json
import ctypes
import welvet
from welvet.utils import _LIB  # Import the loaded CDLL instance

def log(msg):
    print(msg)

def verify_cabi_parity():
    # 1. Load the expected API
    json_path = "../wasm/expected_api.json"
    if not os.path.exists(json_path):
        log(f"Error: {json_path} not found")
        return

    with open(json_path, 'r') as f:
        expected = json.load(f)

    log("=== LOOM C-ABI PARITY AUDIT (345+ FEATURES) ===")
    
    total_expected = 0
    total_found = 0
    missing = []
    
    for category, items in expected.items():
        cat_found = 0
        cat_total = len(items)
        total_expected += cat_total
        
        for item in items:
            itype = item.get('type', 'Func')
            if itype == 'Struct':
                # Structs are in headers, not symbols in DLL
                total_expected -= 1
                continue

            name = item['name']
            c_name = f"Loom{name}"
            
            # Check if symbol exists in DLL
            try:
                getattr(_LIB, c_name)
                total_found += 1
                cat_found += 1
            except AttributeError:
                # Check without Loom prefix just in case
                try:
                    getattr(_LIB, name)
                    total_found += 1
                    cat_found += 1
                except AttributeError:
                    missing.append(f"{category}: {name}")
        
    log("-" * 40)
    log(f"ADJUSTED PARITY SCORE: {total_found} / {total_expected}")
    log("-" * 40)

    if missing:
        log("MISSING SYMBOLS:")
        for m in missing[:10]: # Just show first 10
            log(f"  - {m}")
        if len(missing) > 10:
            log(f"  ... and {len(missing)-10} more")
    else:
        log("100% C-ABI PARITY ACHIEVED!")

    log("\nVERIFYING NEW v0.75.0 FEATURES:")
    new_features = [
        "CreateTransformer",
        "TokensToTensor",
        "ForwardFull",
        "FreeTensor",
        "NewNEATPopulation",
        "NEATPopulationEvolveWithFitnesses"
    ]
    for feat in new_features:
        try:
            getattr(_LIB, f"Loom{feat}")
            log(f"  [OK] Loom{feat}")
        except AttributeError:
            log(f"  [FAIL] Loom{feat} MISSING")

if __name__ == "__main__":
    verify_cabi_parity()
