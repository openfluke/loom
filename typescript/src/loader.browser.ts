/**
 * LOOM WASM Browser Loader
 * Browser-only version without Node.js dependencies
 */

export async function loadLoomWASMBrowser(): Promise<void> {
  // For browser environments - load wasm_exec.js first if not already loaded
  if (typeof (globalThis as any).Go === "undefined") {
    // Load wasm_exec.js dynamically from /dist/
    const script = document.createElement("script");
    script.src = "/dist/wasm_exec.js";
    await new Promise<void>((resolve, reject) => {
      script.onload = () => resolve();
      script.onerror = () => reject(new Error("Failed to load wasm_exec.js"));
      document.head.appendChild(script);
    });
  }

  const response = await fetch("/dist/main.wasm");
  const wasmBuffer = await response.arrayBuffer();

  // @ts-ignore - Go is defined by wasm_exec.js
  const go = new Go();

  const { instance } = await WebAssembly.instantiate(
    wasmBuffer,
    go.importObject
  );

  go.run(instance);

  // Wait for initialization
  await new Promise((resolve) => setTimeout(resolve, 100));
}
