/**
 * welvet WASM Loader — Node.js
 * Loads and initializes the welvet WebAssembly module.
 */

export async function loadLoomWASM(): Promise<void> {
  const fs   = await import("fs");
  const url  = await import("url");
  const path = await import("path");

  const __filename = url.fileURLToPath(import.meta.url);
  const __dirname  = path.dirname(__filename);

  // Resolve root: dist/ in production, or one level up from src/
  const root = __dirname.endsWith("dist")
    ? __dirname
    : path.join(__dirname, "..", "dist");

  // Bootstrap Go runtime
  const wasmExecCode = fs.readFileSync(path.join(root, "wasm_exec.js"), "utf-8");
  eval(wasmExecCode);

  // Load and instantiate the WASM module
  const wasmBuffer = fs.readFileSync(path.join(root, "main.wasm"));

  // @ts-ignore — Go is injected by wasm_exec.js
  const go = new Go();

  const { instance } = await WebAssembly.instantiate(wasmBuffer, go.importObject);
  go.run(instance);

  // Allow Go goroutines to settle
  await new Promise<void>((resolve) => setTimeout(resolve, 100));
}
