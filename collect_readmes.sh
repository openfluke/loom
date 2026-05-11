#!/bin/bash

# Script to collect all README files into a single docs folder

set -e

OUTPUT_DIR="docs/readmes"

echo "📚 Collecting all README files..."
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Copy main README
if [ -f "README.md" ]; then
    cp README.md "$OUTPUT_DIR/main-readme.md"
    echo "✅ Copied: README.md -> main-readme.md"
fi

# Copy subdirectory READMEs with descriptive names
declare -A readme_map=(
    ["nn/README.md"]="nn-readme.md"
    ["python/README.md"]="python-readme.md"
    ["csharp/README.md"]="csharp-readme.md"
    ["typescript/README.md"]="typescript-readme.md"
    ["cabi/README.md"]="cabi-readme.md"
    ["detector/README.md"]="detector-readme.md"
    ["wasm/README.md"]="wasm-readme.md"
    ["examples/README.md"]="examples-readme.md"
    ["tokenizer/README.md"]="tokenizer-readme.md"
    ["model_conversion/README.md"]="model-conversion-readme.md"
)

for source in "${!readme_map[@]}"; do
    dest="${readme_map[$source]}"
    if [ -f "$source" ]; then
        cp "$source" "$OUTPUT_DIR/$dest"
        echo "✅ Copied: $source -> $dest"
    else
        echo "⚠️  Not found: $source"
    fi
done

echo ""
echo "📁 All READMEs collected in: $OUTPUT_DIR/"
ls -lh "$OUTPUT_DIR/"
echo ""
echo "✅ Done!"
