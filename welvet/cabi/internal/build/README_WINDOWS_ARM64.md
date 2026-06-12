# Windows ARM64 native builds (WSL / Linux)

One script. Native `windows/arm64` for **lucy** + **welvet** (no Prism x64 emulation).

```bash
cd loom/welvet/cabi/internal/build
./build_windows_arm64.sh --skip-lib
```

Do **not** use `./build_unix.sh windows arm64` — it tells you to use this script instead.

---

## Repo layout

```text
endgame/          (or any parent folder)
  loom/           go.mod → webgpu@v1.0.4
  webgpu/         GNU libwgpu_native.a in wgpu/lib/windows/arm64/
```

No `replace` in go.mod. `build_windows_arm64.sh` copies the vendored lib into the Go module cache.

---

## Fresh machine

```bash
export LLVM_MINGW_HOME=/opt/llvm-mingw   # or /mnt/c/llvm-mingw

git clone <loom> loom
git clone <webgpu> webgpu

cd loom/welvet/cabi/internal/build
./build_windows_arm64.sh --skip-lib
```

First time without GNU lib in webgpu (or after wgpu-native bump):

```bash
./build_windows_arm64.sh              # builds lib (~10 min) + lucy + welvet
./build_windows_arm64.sh --rebuild-lib  # force lib rebuild
./build_windows_arm64.sh --lucy-only
./build_windows_arm64.sh --welvet-only
```

### Prereqs

- **llvm-mingw** with `aarch64-w64-mingw32-clang`
- **Go 1.26+**
- **rustup + clang** — only for `--rebuild-lib` / first lib build

### Outputs

| Artifact | Path |
|----------|------|
| lucy.exe | `loom/lucy/dist/windows_arm64/lucy.exe` |
| welvet.dll | `dist/windows_arm64/welvet.dll` |

Run on WoA: `loom\lucy\dist\windows_arm64\lucy.exe`

---

## Why GNU lib?

Official wgpu-native arm64 zip is **MSVC**. llvm-mingw cannot link it. We vendor a **GNU** `libwgpu_native.a` in webgpu — commit it so `--skip-lib` works on every clone.

---

## Troubleshooting

**`set: pipefail: invalid option`** — CRLF on `/mnt/c/`. Run `sed -i 's/\r$//' build_windows_arm64.sh` or clone/edit under `~/` in WSL.

**`type_info::vftable` link error** — module cache has MSVC lib. Re-run `./build_windows_arm64.sh --skip-lib`.

**`go env GOARCH=amd64` on Windows** — Prism x64. Use the WSL-built `lucy.exe` for native ARM64.

---

## What to commit

- **webgpu**: `wgpu/lib/windows/arm64/libwgpu_native.a` (GNU)
- **loom**: `build_windows_arm64.sh`, this README

Do not commit `dist/` or `wgpu-native/` temp clones.
