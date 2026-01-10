#!/bin/bash
# Compile all nn documentation into a single DOCX file
# Requires: pandoc (install with: sudo dnf install pandoc)

set -e

# Configuration
OUTPUT_FILE="Loom_NN_Documentation_$(date +%Y-%m-%d).docx"
TITLE="Loom Neural Network Package Documentation"
DATE=$(date +"%B %d, %Y")
AUTHOR="OpenFluke"

# Output directory (same as script location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║           Compiling Loom NN Documentation to DOCX               ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Check for pandoc
if ! command -v pandoc &> /dev/null; then
    echo "❌ Error: pandoc is not installed."
    echo "   Install with: sudo dnf install pandoc"
    echo "   Or on Ubuntu: sudo apt install pandoc"
    exit 1
fi

# Define file order (logical reading order)
FILES=(
    "overview.md"
    "layers.md"
    "gpu_layers.md"
    "training.md"
    "optimizers.md"
    "tween.md"
    "parallel.md"
    "serialization.md"
    "introspection.md"
    "evaluation.md"
    "telemetry.md"
    "examples.md"
    "quick_reference.md"
)

# Check which files exist
EXISTING_FILES=()
for file in "${FILES[@]}"; do
    if [[ -f "$file" ]]; then
        EXISTING_FILES+=("$file")
        echo "✓ Found: $file"
    else
        echo "⚠ Missing: $file (skipping)"
    fi
done

echo ""
echo "📄 Found ${#EXISTING_FILES[@]} documentation files"
echo ""

# Create title page markdown
TITLE_PAGE=$(mktemp)
cat > "$TITLE_PAGE" << EOF
---
title: "$TITLE"
author: "$AUTHOR"
date: "$DATE"
---

# $TITLE

**Generated:** $DATE

**Version:** Based on Loom v0.0.7+

---

## Table of Contents

EOF

# Add TOC entries
for file in "${EXISTING_FILES[@]}"; do
    # Extract first H1 heading from file
    heading=$(grep -m1 "^# " "$file" | sed 's/^# //')
    if [[ -n "$heading" ]]; then
        echo "- $heading" >> "$TITLE_PAGE"
    else
        # Fallback to filename
        name=$(basename "$file" .md)
        echo "- ${name^}" >> "$TITLE_PAGE"
    fi
done

echo "" >> "$TITLE_PAGE"
echo "---" >> "$TITLE_PAGE"
echo "" >> "$TITLE_PAGE"
echo "\\newpage" >> "$TITLE_PAGE"
echo "" >> "$TITLE_PAGE"

# Combine all files with page breaks
COMBINED=$(mktemp)
cat "$TITLE_PAGE" > "$COMBINED"

for file in "${EXISTING_FILES[@]}"; do
    echo "" >> "$COMBINED"
    cat "$file" >> "$COMBINED"
    echo "" >> "$COMBINED"
    echo "\\newpage" >> "$COMBINED"
    echo "" >> "$COMBINED"
done

# Convert to DOCX
echo "🔄 Converting to DOCX..."
pandoc "$COMBINED" \
    -o "$OUTPUT_FILE" \
    --from markdown \
    --to docx \
    --toc \
    --toc-depth=3 \
    --highlight-style=tango \
    --metadata title="$TITLE" \
    --metadata author="$AUTHOR" \
    --metadata date="$DATE"

# Cleanup temp files
rm -f "$TITLE_PAGE" "$COMBINED"

# Report success
if [[ -f "$OUTPUT_FILE" ]]; then
    SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║                        ✅ SUCCESS                                ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "📁 Output: $SCRIPT_DIR/$OUTPUT_FILE"
    echo "📊 Size: $SIZE"
    echo ""
else
    echo "❌ Error: Failed to create DOCX file"
    exit 1
fi
