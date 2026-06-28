# Welvet C-ABI Bridge

This directory contains the **C-ABI (Application Binary Interface)** for the Loom engine. It acts as the universal bridge between the Go-based **M-POLY-VTD** core and all other programming languages.

## 🛠️ The Build System
The build system uses a **"Polyglot Builder"** architecture designed for cross-platform reliability.

### How it Works
1.  **`internal/build/builder.go`**: The core build manager. It orchestrates the Go compiler, manages environment variables (`GOOS`, `GOARCH`), and handles the output directory structure.
2.  **Platform Wrappers** (in `internal/build/`):
    - `build_unix.sh` — Linux / macOS / cross-targets
    - `build_linux.sh` — Linux amd64 + arm64 shortcut
    - `build_windows.bat` — Windows amd64
    - `build_windows_arm64.sh` — Windows ARM64 pipeline
3.  **`build_macos.sh`** (this directory) — quick local macOS dylib only
4.  **Output Path**: Binaries land in `internal/build/dist/<os>_<arch>/` (`welvet.so`, `welvet.h`, `lucy`, `cabi_verify`)

### Why this Architecture?
- **Source of Truth**: The `main.go` file is the absolute authority. The `.h` header and shared libraries are **derived artifacts**. Generating them on-demand ensures they never go out of sync with the logic.
- **Cross-Platform Consistency**: By using a Go script (`builder.go`) instead of complex Makefiles or dozens of shell scripts, the build logic remains consistent regardless of the host OS.
- **Decentralized 'Dist'**: Organized subdirectories in `dist/` make it trivial to package the engine for multiple targets (e.g., Python `ctypes`, Unity, or Mobile) without accidentally overwriting previous builds.

## 🚀 Commands

```bash
# Linux (recommended — full pipeline: welvet.so, cabi_verify, lucy, Python mirror)
cd internal/build && ./build_linux.sh          # native arch
cd internal/build && ./build_linux.sh all      # amd64 + arm64

# Any Unix target via builder
cd internal/build && ./build_unix.sh linux amd64

# macOS quick build (this directory)
./build_macos.sh
```

- **Cross-compile everything**: `cd internal/build && go run builder.go -os all`
- **Clean builds**: add `--clean` to `build_linux.sh` / `build_unix.sh`

## Parity check (v0.81+)

Scan `poly/` public API vs `//export` names:

```bash
cd internal/check && go run .
```

Expect **489/489 (100.0%)** — includes vendor accel (`LoomSyncToAccel`, `LoomDispatchAccelForward`), `EntityFile` I/O, transformer GPU sync, and `LoomMemoryHistoryWriteJSON`. See [`docs/bedrock_validation.md`](../../docs/bedrock_validation.md) and [`docs/accelerators.md`](../../docs/accelerators.md).

---
*Loom C-ABI: Bit-perfect performance, exposed to the world.*
