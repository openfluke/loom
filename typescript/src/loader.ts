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
  // Load wasm_exec.js
  const wasmExecPath = join(__dirname, "../assets/wasm_exec.js");
  const wasmExecCode = readFileSync(wasmExecPath, "utf-8");

  // Execute wasm_exec.js to get the Go runtime
  eval(wasmExecCode);

  // Load main.wasm
  const wasmPath = join(__dirname, "../assets/main.wasm");
  const wasmBuffer = readFileSync(wasmPath);

  // @ts-ignore - Go is defined by wasm_exec.js
  const go = new Go();

  const { instance } = await WebAssembly.instantiate(
    wasmBuffer,
    go.importObject
  );

  // Run the Go WASM module
  go.run(instance);

  // Wait a bit for initialization
  await new Promise((resolve) => setTimeout(resolve, 100));
}
