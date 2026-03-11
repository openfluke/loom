# Loom Wrappers

Language bindings and client-side wrappers for the [Loom](https://github.com/openfluke/loom) inference engine.

Each sub-folder is a self-contained SDK that communicates with Loom through the C ABI bridge (`cabi/`). The bridge itself lives in the repo root and is **not** included here.

---

## Sub-packages

| Folder | Language | Description |
|--------|----------|-------------|
| [`wasm/`](./wasm) | Go → WebAssembly | Compiles the Go runtime to WASM so Loom models can run directly in the browser. Includes HTML demos and a dev server. |
| [`typescript/`](./typescript) | TypeScript / JS | NPM package providing typed bindings for the Loom HTTP/WebSocket API. Works with Node.js and modern browsers. |
| [`csharp/`](./csharp) | C# / .NET | NuGet-ready wrapper (Welvet) that P/Invokes the native shared library produced by `cabi/`. |
| [`python/`](./python) | Python | pip-installable package that loads the native shared library via `ctypes`/`cffi`. Ships with full universal-test coverage. |
| [`dart/`](./dart) | Dart | `dart:ffi` package — calls the native library directly with no code generation. Works in Flutter and standalone Dart. |
| [`java/`](./java) | Java | Maven package using JNA (Java Native Access) — no JNI boilerplate. Works with any JVM language (Kotlin, Scala, etc.). |

---

## Quick start

Refer to each folder's own `README.md` for build instructions, examples, and publishing steps.

```
wrapper/
├── wasm/         # go build -tags wasm → wasm_exec.js + main.wasm
├── typescript/   # bun install && bun run build
├── csharp/       # dotnet build / pack.sh
└── python/       # pip install -e . / publish.sh
```

---

## Architecture

```
┌─────────────────────────────────────┐
│            Loom Core (Go)           │
└──────────────┬──────────────────────┘
               │ C ABI  (cabi/)
       ┌───────┴────────┐
       │  shared lib    │
       │  (.so / .dll)  │
       └──┬──────┬──────┘
          │      │          ┌──── wasm/  (in-process, no shared lib)
     csharp/   python/      └──── typescript/ (HTTP / WS)
```

The WASM and TypeScript wrappers communicate over HTTP/WebSocket rather than
linking the native library directly. The C# and Python wrappers load the
native shared library directly via P/Invoke and ctypes respectively.
