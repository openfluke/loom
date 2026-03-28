import os
import re
import glob

# Configuration matching welvet/cabi/internal/check/check.go
POLY_PATH = "../../poly"
CABI_PATH = "../cabi"

CATEGORIES = {
    "CORE MECHANICS": ["VolumetricNetwork", "Forward", "Backward", "Systolic", "Layer", "Tensor"],
    "ACCELERATION":   ["WGPU", "GPU", "Sync", "Dispatch", "Shader"],
    "LEARNING/DNA":   ["TargetProp", "DNA", "Compare", "Refit", "Gradient"],
    "IO/UTIL":        ["JSON", "Safetensors", "Load", "Extract", "Tokenizer"],
}

class APIItem:
    def __init__(self, name, itype):
        self.name = name
        self.itype = itype # "Struct", "Func", "Method"

def scan_poly(path):
    items = []
    # Regex for exported structs and funcs/methods
    struct_re = re.compile(r"type\s+([A-Z][a-zA-Z0-9_]*)\s+struct")
    func_re = re.compile(r"func\s+(?:\([^)]+\)\s+)?([A-Z][a-zA-Z0-9_]*)\(")
    
    for filename in glob.glob(os.path.join(path, "*.go")):
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            # Structs
            for match in struct_re.finditer(content):
                items.append(APIItem(match.group(1), "Struct"))
            # Funcs/Methods
            for match in func_re.finditer(content):
                name = match.group(1)
                itype = "Method" if "func (" in match.group(0) else "Func"
                items.append(APIItem(name, itype))
    return items

def scan_cabi(path):
    identifiers = set()
    # Deep Inspection logic: extract all identifiers from all CABI go files
    id_re = re.compile(r"[a-zA-Z0-9_]+")
    
    for filename in glob.glob(os.path.join(path, "*.go")):
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            # Find all words
            for match in id_re.finditer(content):
                identifiers.add(match.group(0))
    return identifiers

def run_report():
    print("====================================================")
    print(" LOOM C-ABI Coverage Analysis Tool (Python Port)")
    print("====================================================")

    print(f"\n[1] Scanning poly/ core at {POLY_PATH}...")
    core_api = scan_poly(POLY_PATH)
    print(f"    Found {len(core_api)} public API items in poly/")

    print(f"\n[2] Scanning welvet/cabi/ exports at {CABI_PATH}...")
    cabi_identifiers = scan_cabi(CABI_PATH)
    print(f"    Deep inspection found {len(cabi_identifiers)} identifiers in C-ABI")

    print(f"\n[3] Functional Parity Report (Categorized)")
    print("-" * 70)

    global_total = 0
    global_covered = 0

    for cat, keywords in CATEGORIES.items():
        print(f"\n>>> {cat}")
        cat_total = 0
        cat_covered = 0
        
        # Track items to avoid duplicates in category report
        seen_in_cat = set()

        for item in core_api:
            # Filter by category keywords
            match = False
            for kw in keywords:
                if kw.lower() in item.name.lower():
                    match = True
                    break
            if not match:
                continue

            if item.name in seen_in_cat:
                continue
            seen_in_cat.add(item.name)

            cat_total += 1
            covered = False
            # Deep match: is this item name mentioned in any CABI file?
            for ident in cabi_identifiers:
                if item.name.lower() in ident.lower():
                    covered = True
                    break

            status = "[X]" if covered else "[ ]"
            if covered:
                cat_covered += 1
            
            # Print status (first 20 chars of name for table)
            print(f"  {status} {item.itype:7} {item.name}")

        if cat_total > 0:
            coverage = (cat_covered / cat_total) * 100
            print(f"  CATEGORY COVERAGE: {cat_covered}/{cat_total} ({coverage:.1f}%)")
            global_total += cat_total
            global_covered += cat_covered

    print("\n" + "-" * 70)
    print(f"Global Functional Overlap: High-level mappings preserved in C-ABI")
    if global_total > 0:
        total_coverage = (global_covered / global_total) * 100
        print(f"TOTAL API COVERAGE: {global_covered}/{global_total} ({total_coverage:.1f}%)")
    print("====================================================")

if __name__ == "__main__":
    run_report()
