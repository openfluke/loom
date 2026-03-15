/**
 * welvet WASM Browser Loader
 * Loads the Go runtime and WASM module in a browser environment.
 */

export async function loadLoomWASMBrowser(wasmUrl?: string): Promise<void> {
  // Inject wasm_exec.js if the Go runtime is not yet available
  if (typeof (globalThis as Record<string, unknown>)["Go"] === "undefined") {
    const script = document.createElement("script");
    script.src = "/dist/wasm_exec.js";
    await new Promise<void>((resolve, reject) => {
      script.onload = () => resolve();
      script.onerror = () => reject(new Error("Failed to load wasm_exec.js"));
      document.head.appendChild(script);
    });
  }

  const response   = await fetch(wasmUrl ?? "/dist/main.wasm");
  const wasmBuffer = await response.arrayBuffer();

  // @ts-ignore — Go is injected by wasm_exec.js
  const go = new Go();

  const { instance } = await WebAssembly.instantiate(wasmBuffer, go.importObject);
  go.run(instance);

  // Allow Go goroutines to settle
  await new Promise<void>((resolve) => setTimeout(resolve, 100));
}
