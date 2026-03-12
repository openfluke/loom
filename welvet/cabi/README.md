# Welvet C-ABI Bridge

This directory contains the **C-ABI (Application Binary Interface)** for the Loom engine. It acts as the universal bridge beTargetProp the Go-based **M-POLY-VTD** core and all other programming languages.

## 🛠️ The Build System
The build system uses a **"Polyglot Builder"** architecture designed for cross-platform reliability.

### How it Works
1.  **`builder.go`**: The core build manager. It orchestrates the Go compiler, manages environment variables (`GOOS`, `GOARCH`), and handles the output directory structure.
2.  **Platform Wrappers**: 
    - `build_windows.bat`: Local Windows trigger.
    - `build_unix.sh`: Linux/macOS trigger.
3.  **Output Path**: All binaries are compiled into `dist/<os>_<arch>/` to prevent platform-specific name collisions.

### Why this Architecture?
- **Source of Truth**: The `main.go` file is the absolute authority. The `.h` header and shared libraries are **derived artifacts**. Generating them on-demand ensures they never go out of sync with the logic.
- **Cross-Platform Consistency**: By using a Go script (`builder.go`) instead of complex Makefiles or dozens of shell scripts, the build logic remains consistent regardless of the host OS.
- **Decentralized 'Dist'**: Organized subdirectories in `dist/` make it trivial to package the engine for multiple targets (e.g., Python `ctypes`, Unity, or Mobile) without accidentally overwriting previous builds.

## 🚀 Commands
- **Build for local OS**: Run the appropriate `.bat` or `.sh` script.
- **Cross-compile everything**: `go run builder.go -os all` (requires appropriate cross-compilers in PATH).
- **Clean builds**: Add the `-clean` flag to start from a fresh slate.

---
*Loom C-ABI: Bit-perfect performance, exposed to the world.*
