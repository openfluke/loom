/**
 * LOOM WASM Loader
 * Loads and initializes the LOOM WebAssembly module (Node.js only)
 */

// Node.js only loader - using dynamic imports to allow bundling for browser (where this file is not used but might be analyzed)

export async function loadLoomWASM(): Promise<void> {
  const fs = await import("fs");
  const url = await import("url");
  const path = await import("path");

  const __filename = url.fileURLToPath(import.meta.url);
  const __dirname = path.dirname(__filename);

  // __dirname points to:
  // - dist/   → in production
  // - src/    → when running via Bun, ts-node, or example files

  let root: string;

  if (__dirname.endsWith("dist")) {
    // Normal production layout
    root = __dirname;
  } else {
    // Running from src/ or example/
    // Point to project’s dist/ directory
    root = path.join(__dirname, "..", "dist");
  }

  // Load wasm_exec.js
  const wasmExecPath = path.join(root, "wasm_exec.js");
  const wasmExecCode = fs.readFileSync(wasmExecPath, "utf-8");

  // Execute wasm_exec.js to get the Go runtime
  eval(wasmExecCode);

  // Load main.wasm
  const wasmPath = path.join(root, "main.wasm");
  const wasmBuffer = fs.readFileSync(wasmPath);

  // @ts-ignore - Go runtime from wasm_exec.js
  const go = new Go();

  const { instance } = await WebAssembly.instantiate(
    wasmBuffer,
    go.importObject
  );

  go.run(instance);

  // Wait for WASM runtime to finish bootstrapping
  await new Promise((resolve) => setTimeout(resolve, 100));
}
