#!/bin/bash

set -e

# 1. Build loom.wasm in wasm/
echo "[1/3] Building loom.wasm in wasm/ ..."
cd "$(dirname "$0")/wasm"
./build.sh

# 2. Copy loom.wasm to typescript/assets/
echo "[2/3] Copying loom.wasm to typescript/assets/ ..."
cp loom.wasm ../typescript/assets/loom.wasm

# 3. Delete cabi/compiled and run build_all.sh
echo "[3/3] Cleaning and rebuilding C ABI targets ..."
cd ../cabi
rm -rf compiled
./build_all.sh

echo "All builds complete."
