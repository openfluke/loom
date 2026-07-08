# Snapdragon (Hexagon NPU) — Loom C ABI plugin

Builds **`loom_accel_qualcomm.dll`**, the Qualcomm **QNN AI Engine Direct** bridge
Loom loads at runtime via `poly/accel`. This is the Snapdragon mirror of
[`accel/intel`](../intel) (which uses OpenVINO); both implement the identical
vendor-neutral C ABI in `include/loom_accel.h`.

## What you need (two pieces)

The C ABI plugin talks to the NPU through Qualcomm's QNN runtime DLLs. You need
**both** of these:

| Piece | What it is | Why you need it |
|---|---|---|
| **QAIRT runtime** | `QnnHtp.dll`, `QnnHtpPrepare.dll`, `QnnSystem.dll`, signed Hexagon v73 skel/cat | **Run** graphs on the Hexagon NPU |
| **QAIRT SDK** (full) | QNN C headers (`QnnInterface.h`, …), `QnnCpu.dll`, `QnnGpu.dll`, tools | **Build** `loom_accel_qualcomm.dll` and load the CPU reference backend |

- **Runtime only** → NPU might work but you cannot rebuild the plugin and `QnnCpu.dll` will not resolve.
- **SDK only** → you can build, but HTP execution still needs the runtime skel/cat beside the backend DLLs.

On our Snapdragon X dev box the full SDK lives at:

```text
accel/qualcomm/deps/qairt-sdk/qairt/2.40.0.251030/
```

## Install (Windows on Snapdragon X — Hexagon v73 HTP)

### 1. Download QAIRT pieces

```powershell
cd accel/qualcomm
pwsh -File .\install_qairt.ps1
```

This script:

1. Downloads the **free QAIRT HTP runtime** (v2.38, Hexagon v73) from Qualcomm's
   public GitHub release into `deps/qairt-runtime/`.
2. Auto-detects the **full QAIRT SDK** from, in order:
   - `$env:QNN_SDK_ROOT`
   - `C:\Qualcomm\AIStack\QAIRT\<version>` (Software Center installer default)
   - `deps/qairt-sdk/qairt/<version>` (vendored copy in this repo)
3. Writes `setup_env.ps1` with `QNN_SDK_ROOT`, `LOOM_QUALCOMM_RUNTIME`, and PATH.

If headers are missing, install the full SDK:

1. Create a free account at [Qualcomm Software Center](https://softwarecenter.qualcomm.com).
2. Download **Qualcomm AI Runtime SDK** (QAIRT / QNN).
3. Install (defaults to `C:\Qualcomm\AIStack\QAIRT\<version>`) **or** extract the
   SDK zip into `accel/qualcomm/deps/qairt-sdk/qairt/<version>/`.
4. Re-run `install_qairt.ps1`.

### 2. Persist env for all users (recommended)

The Go loader (`PrepareQualcommRuntime`) prepends
`%QNN_SDK_ROOT%\lib\aarch64-windows-msvc` to PATH at runtime, but the SDK root
itself must be visible to every shell / user:

```powershell
# elevated PowerShell — sets Machine scope for all users
pwsh -File .\install_qairt.ps1 -Persist
```

Or set manually (admin):

```powershell
$sdk = 'C:\git\chaosglue\loom\accel\qualcomm\deps\qairt-sdk\qairt\2.40.0.251030'
$rt  = 'C:\git\chaosglue\loom\accel\qualcomm\deps\qairt-runtime\aarch64-windows-msvc'
[Environment]::SetEnvironmentVariable('QNN_SDK_ROOT', $sdk, 'Machine')
[Environment]::SetEnvironmentVariable('LOOM_QUALCOMM_RUNTIME', $rt, 'Machine')
```

Open a **new** shell after persisting. Verify:

```powershell
echo $env:QNN_SDK_ROOT
Test-Path "$env:QNN_SDK_ROOT\lib\aarch64-windows-msvc\QnnCpu.dll"
Test-Path "$env:LOOM_QUALCOMM_RUNTIME\QnnHtp.dll"
```

### 3. Build the C ABI plugin

```powershell
cd accel/qualcomm
. .\setup_env.ps1
.\build_clang.ps1          # llvm-mingw on windows/arm64 (what we use today)
# or: .\build.ps1          # CMake + MSVC, if installed
```

Output: `build/loom_accel_qualcomm.dll`

**Toolchain note:** on windows/arm64 we build the plugin with **llvm-mingw**
(`C:\llvm-mingw\bin\aarch64-w64-mingw32-clang++.exe`). The QNN backend DLLs are
MSVC-ABI but are loaded at runtime via `LoadLibrary` and only called through C
entry points, so mingw/MSVC split is fine for the plugin shell.

### 4. Build & run Lucy

```powershell
cd lucy
go run .        # choose [12] Snapdragon NPU bridge
```

## Windows/arm64: `webgpu` `go.mod` replace

`go run` / `go build` for Lucy on **windows/arm64** usually fails to link the
upstream `github.com/openfluke/webgpu` module: the published static archive
(`libwgpu_native.a`) is built with a different C++ ABI than llvm-mingw expects
(`undefined symbol: const type_info::vftable`).

**Fix:** point both `go.mod` files at a **local** `webgpu` checkout built for
this toolchain. These `replace` lines must be present (paths relative to each module):

**`loom/go.mod`**

```go
require github.com/openfluke/webgpu v1.0.4

replace github.com/openfluke/webgpu => ../webgpu
```

**`lucy/go.mod`**

```go
require github.com/openfluke/webgpu v1.0.4

replace github.com/openfluke/webgpu => ../../webgpu
```

Clone/build `webgpu` next to the loom repo (`../webgpu` from `loom/`, or adjust
the replace path). This is only needed for windows/arm64 CGO builds; other
platforms can use the published module.

## Lucy menu [12]

```text
[12] Snapdragon NPU bridge — Loom ↔ loom_accel_qualcomm.dll (QNN)
```

Sub-menus mirror the Intel `[9]` suite: CABI matrix, DispatchLayer timing tables,
multi-hop demo, single-layer picker. Log: `lucy_testing_output/snapdragon.txt`.

**Quiet by default:** QNN/HTP is chatty. Menu `[12]` filters known noise from the
terminal. Set `LOOM_QNN_VERBOSE=1` to see DSP-transport warnings, graph-prepare
stage lines, etc.

## Layout

| Path | Role |
|---|---|
| `include/loom_accel.h` | Vendor-neutral C ABI (in sync with `poly/accel/include/`) |
| `src/qnn_wrapper.*` | QNN backend load + graph build/finalize/execute |
| `src/layer_models.*` | Layer → tensor-shape/op resolution (small/medium/large) |
| `src/loom_accel_qualcomm.cpp` | C ABI implementation (compile once, infer many) |
| `bench_manifest.json` | Layer × dtype × size matrix (Lucy menu [12]) |
| `install_qairt.ps1` | Download runtime + detect SDK + write `setup_env.ps1` |
| `build_clang.ps1` | Build plugin with llvm-mingw (no CMake required) |
| `build.ps1` | CMake build (MSVC toolchain) |
| `deps/qairt-runtime/` | Free HTP runtime DLLs + Hexagon skel (gitignored) |
| `deps/qairt-sdk/` | Full QAIRT SDK tree (gitignored, ~1.3 GB) |
| `build/loom_accel_qualcomm.dll` | Output (gitignored) |

## Accelerators & numerical types

Snapdragon exposes three AI cores; the Loom C ABI `device` string selects the QNN
backend (`"NPU"`→`QnnHtp`, `"GPU"`→`QnnGpu`, `"CPU"`→`QnnCpu`). Per
[Qualcomm's AI hardware cores docs](https://docs.qualcomm.com/nav/home/AI-hardware-cores-accelerators.html?product=1601111740009303):

| Core | Backend | Data types | Quantization | Notes |
|---|---|---|---|---|
| **HTP** (Hexagon Tensor Processor) | `QnnHtp` | INT4, INT8, INT16, FP16 | needed | Dedicated AI accelerator, hardware convolution engine. Low power, high throughput. |
| **GPU** (Adreno) | `QnnGpu` | INT8, INT16, FP16, FP32 | not needed | Unquantized FP32/FP16 at higher throughput than CPU; OpenCL UDOs. |
| **CPU** (Kryo) | `QnnCpu` | FP32 reference | — | Parity anchor; `QnnCpu` rejects FP16/INT* `graphAddNode` in our graphs today. |

Data type details:

| dtype | Meaning | HTP behaviour |
|---|---|---|
| **FP32** | 32-bit float | Not on HTP — falls back to `QnnCpu` (parity anchor); Adreno GPU also runs FP32 |
| **FP16** | 16-bit float | Native HTP (no quantization) |
| **INT16** | 8-bit weights + 16-bit activations | Native HTP fixed-point (highest-accuracy quant) |
| **INT8** | 8-bit weights + 8-bit activations | Native HTP fixed-point |
| **INT4** | 4-bit weights + 8-bit activations | Native HTP fixed-point (block quant not wired yet — compile may fail) |

**Weight-quantization model:** the bench uploads FP32 weights + FP32 activations
(matching `poly/accel_intel.go`'s handover contract); the plugin requantizes the
static weights to the target precision (INT4→4-bit, INT8/INT16→8-bit weights) with a
per-tensor symmetric scale.

## Environment

| Variable | Purpose |
|---|---|
| `QNN_SDK_ROOT` | Full QAIRT SDK install (headers + `QnnCpu.dll` + tools). **Required** for fresh shells / all users. |
| `LOOM_QUALCOMM_RUNTIME` | HTP runtime backend dir (`QnnHtp.dll` + Hexagon skel/cat) |
| `LOOM_ACCEL_QUALCOMM_DLL` | Plugin path (optional — auto-discovered under `accel/qualcomm/build/`) |
| `LOOM_ROOT` | Loom repo root when cwd is outside the tree |
| `LOOM_QNN_VERBOSE` | Set to `1` to show QNN/HTP warnings on the terminal (filtered by default in menu [12]) |

`PrepareQualcommRuntime()` (called from Lucy `[12]`) prepends
`%QNN_SDK_ROOT%\lib\aarch64-windows-msvc` and `%LOOM_QUALCOMM_RUNTIME%` to the
process `PATH` so `LoadLibrary("QnnHtp.dll")` / `QnnCpu.dll` resolve.

## Normal QNN warnings (not failures)

On Snapdragon X you may see these **once** at startup (hidden unless `LOOM_QNN_VERBOSE=1`):

```text
DspTransport.openSession qnn_open failed … Failed to load skel …
Traditional path not available. Switching to user driver path
HTP user driver is loaded. Switched to user driver path
```

This is expected on many Windows-on-Snapdragon builds: the signed skel path is
unavailable, QNN falls back to the **user driver**, and inference still works
(`Hexagon NPU: available`). Do **not** modify the signed `*Skel.so` / `*.cat` files.

**Full matrix bench (`[5]`):** compiles 15 layers × 5 dtypes × 3 sizes on CPU + HTP.
First HTP compile per graph can take hundreds of ms; the full run takes several
minutes. Avoid Ctrl+C mid-compile — QNN may fault (`exit status 0xc0000005`).

## Notes

- QNN op names/params (`Conv2d`, `MatMul`, `PoolAvg2d`, `Softmax`, …) target the
  `qti.aisw` op package. Validate against your installed QAIRT version if a
  `graphAddNode` error appears — op param encodings occasionally shift between SDK releases.
- Each released compiled layer resets the QNN context (QNN has no per-graph free API)
  so long benchmark runs do not leak graphs or slow down over time.
