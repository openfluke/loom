/**
 * LOOM WASM Loader
 * Loads and initializes the LOOM WebAssembly module (Node.js only)
 */

import { readFileSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

export async function loadLoomWASM(): Promise<void> {
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
    root = join(__dirname, "..", "dist");
  }

  // Load wasm_exec.js
  const wasmExecPath = join(root, "wasm_exec.js");
  const wasmExecCode = readFileSync(wasmExecPath, "utf-8");

  // Execute wasm_exec.js to get the Go runtime
  eval(wasmExecCode);

  // Load main.wasm
  const wasmPath = join(root, "main.wasm");
  const wasmBuffer = readFileSync(wasmPath);

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
