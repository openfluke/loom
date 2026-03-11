#!/usr/bin/env python3
"""
Compile all nn documentation into a single DOCX file.
Requires: pandoc (install with: sudo dnf install pandoc or choco install pandoc)
"""

import os
import subprocess
import tempfile
import datetime
import re
import sys

# Configuration
DATE = datetime.datetime.now()
OUTPUT_FILE = f"Loom_NN_Documentation_{DATE.strftime('%Y-%m-%d')}.docx"
TITLE = "Loom Neural Network Package Documentation"
DATE_STR = DATE.strftime("%B %d, %Y")
AUTHOR = "OpenFluke"

# Output directory (same as script location)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

print("╔══════════════════════════════════════════════════════════════════╗")
print("║           Compiling Loom NN Documentation to DOCX               ║")
print("╚══════════════════════════════════════════════════════════════════╝\n")

# Check for pandoc
try:
    subprocess.run(["pandoc", "--version"], capture_output=True, check=True)
except (subprocess.CalledProcessError, FileNotFoundError):
    print("❌ Error: pandoc is not installed or not in PATH.")
    print("   Install with: choco install pandoc (Windows) or sudo apt install pandoc (Linux)")
    sys.exit(1)

# Define file order (logical reading order)
FILES = [
    "overview.md",
    "layers.md",
    "gpu_layers.md",
    "fp4_quantization.md",
    "training.md",
    "optimizers.md",
    "tween.md",
    "step_tween_chain.md",
    "parallel.md",
    "serialization.md",
    "introspection.md",
    "evaluation.md",
    "telemetry.md",
    "examples.md",
    "quick_reference.md",
    "architecture.md",
    "clustering.md",
    "grafting.md",
    "kmeans.md",
    "stepping.md",
    "tokenizer.md",
    "transformer.md",
    "type_conversion.md",
]

# Check which files exist
existing_files = []
for file in FILES:
    if os.path.isfile(file):
        existing_files.append(file)
        print(f"✓ Found: {file}")
    else:
        print(f"⚠ Missing: {file} (skipping)")

print(f"\n📄 Found {len(existing_files)} documentation files\n")

# Extract headings for TOC
toc_entries = []
for file in existing_files:
    heading = ""
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("# "):
                heading = line[2:].strip()
                break
    
    if heading:
         toc_entries.append(heading)
    else:
         toc_entries.append(os.path.splitext(os.path.basename(file))[0].title())

# Create combined markdown content
temp_md_fd, temp_md_path = tempfile.mkstemp(suffix=".md")
with os.fdopen(temp_md_fd, 'w', encoding='utf-8') as f:
    # Write Title Page
    f.write("---\n")
    f.write(f"title: \"{TITLE}\"\n")
    f.write(f"author: \"{AUTHOR}\"\n")
    f.write(f"date: \"{DATE_STR}\"\n")
    f.write("---\n\n")
    
    f.write(f"# {TITLE}\n\n")
    f.write(f"**Generated:** {DATE_STR}\n\n")
    f.write("**Version:** Based on Loom v0.0.7+\n\n")
    f.write("---\n\n")
    f.write("## Table of Contents\n\n")
    
    for entry in toc_entries:
        f.write(f"- {entry}\n")
        
    f.write("\n---\n\n\\newpage\n\n")
    
    # Append files
    for file in existing_files:
        f.write("\n")
        with open(file, 'r', encoding='utf-8') as infile:
            content = infile.read()
            # Fix up relative links to be absolute or internal if possible, 
            # but for now just appending it as the bash script did.
            f.write(content)
        f.write("\n\n\\newpage\n\n")

# Convert to DOCX
print("🔄 Converting to DOCX...")

cmd = [
    "pandoc", temp_md_path,
    "-o", OUTPUT_FILE,
    "--from", "markdown",
    "--to", "docx",
    "--toc",
    "--toc-depth=3",
    "--highlight-style=tango",
    f"--metadata=title:{TITLE}",
    f"--metadata=author:{AUTHOR}",
    f"--metadata=date:{DATE_STR}"
]

try:
    subprocess.run(cmd, check=True)
    size_bytes = os.path.getsize(OUTPUT_FILE)
    size_mb = size_bytes / (1024 * 1024)
    
    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║                        ✅ SUCCESS                                ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")
    print(f"📁 Output: {os.path.join(SCRIPT_DIR, OUTPUT_FILE)}")
    print(f"📊 Size: {size_mb:.2f} MB\n")
    
except subprocess.CalledProcessError as e:
    print(f"❌ Error: Failed to create DOCX file. Pandoc returned {e.returncode}")
    sys.exit(1)
finally:
    # Cleanup temp file
    os.remove(temp_md_path)
