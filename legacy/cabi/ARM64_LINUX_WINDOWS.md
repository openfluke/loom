# Cross-Compiling Windows ARM64 from Linux ARM64 (Fedora)

## ✅ Working (Dec 2025)

These steps successfully build wgpu-native and LOOM for Windows ARM64 from an ARM64 Linux host.

---

## 1. Install llvm-mingw

```bash
cd /tmp
curl -LO https://github.com/mstorsjo/llvm-mingw/releases/download/20241217/llvm-mingw-20241217-ucrt-ubuntu-20.04-aarch64.tar.xz
tar xf llvm-mingw-20241217-ucrt-ubuntu-20.04-aarch64.tar.xz
sudo mv llvm-mingw-20241217-ucrt-ubuntu-20.04-aarch64 /opt/llvm-mingw

# Add to PATH permanently
echo 'export PATH="/opt/llvm-mingw/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

## 2. Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
rustup target add aarch64-pc-windows-gnullvm
```

## 3. Configure Rust linker

```bash
cat > ~/.cargo/config.toml << 'EOF'
[target.aarch64-pc-windows-gnullvm]
linker = "/opt/llvm-mingw/bin/aarch64-w64-mingw32-gcc"
ar = "/opt/llvm-mingw/bin/aarch64-w64-mingw32-ar"

[target.aarch64-pc-windows-gnu]
linker = "/opt/llvm-mingw/bin/aarch64-w64-mingw32-gcc"
ar = "/opt/llvm-mingw/bin/aarch64-w64-mingw32-ar"
EOF
```

## 4. Install build dependencies

```bash
sudo dnf install clang-devel
```

## 5. Build wgpu-native (use stable tag!)

```bash
cd /tmp
git clone https://github.com/gfx-rs/wgpu-native.git
cd wgpu-native

# IMPORTANT: Use a stable tag, not main (main has header mismatches)
git fetch --tags
git checkout v22.1.0.5
git submodule update --init --recursive

# Set environment and build
export BINDGEN_EXTRA_CLANG_ARGS="--target=aarch64-pc-windows-gnu"
export CC_aarch64_pc_windows_gnullvm="/opt/llvm-mingw/bin/aarch64-w64-mingw32-gcc"
export AR_aarch64_pc_windows_gnullvm="/opt/llvm-mingw/bin/aarch64-w64-mingw32-ar"

cargo build --release --target aarch64-pc-windows-gnullvm
```

Output: `target/aarch64-pc-windows-gnullvm/release/libwgpu_native.a`

## 6. Copy library to webgpu package

```bash
cp target/aarch64-pc-windows-gnullvm/release/libwgpu_native.a \
   ~/go/pkg/mod/github.com/openfluke/webgpu@v0.0.1/wgpu/lib/windows/arm64/
```

## 7. Build LOOM

```bash
cd /path/to/loom/cabi
./build_windows_arm64.sh
```

---

## Notes

- `aarch64-pc-windows-gnu` target doesn't exist in stable Rust on ARM64 Linux
- Use `aarch64-pc-windows-gnullvm` instead (works with llvm-mingw)
- **Must use a stable wgpu-native tag** (e.g., v22.1.0.5) - main branch has header mismatches
- Fedora's `mingw64-gcc` is for x86_64 Windows only, not ARM64
