import { isBrowser, isNode } from "./env";

let goRuntimeInjected = false;
let wasmExecTextBundled: string | undefined;
let wasmUrlBundled: string | undefined;

async function tryLoadBundlerAssets() {
  try {
    const raw = await import("./wasm_exec.js?raw");
    wasmExecTextBundled = (raw as any).default as string;
  } catch {}
  try {
    const url = await import("./loom.wasm?url");
    wasmUrlBundled = (url as any).default as string;
  } catch {}
}

export async function ensureGoRuntime(inject = true): Promise<void> {
  if (!inject || goRuntimeInjected) return;

  if (isBrowser) {
    if (!wasmExecTextBundled) await tryLoadBundlerAssets();
    if (wasmExecTextBundled) {
      new Function(wasmExecTextBundled)(); // defines global Go
      goRuntimeInjected = true;
      return;
    }
    const url = new URL("./wasm_exec.js", import.meta.url);
    const res = await fetch(url);
    if (!res.ok)
      throw new Error(
        `Failed to fetch wasm_exec.js (${res.status} ${res.statusText}) from ${url}`
      );
    const jsText = await res.text();
    new Function(jsText)();
    goRuntimeInjected = true;
    return;
  }

  const { readFile } = await import("node:fs/promises");
  const { fileURLToPath } = await import("node:url");
  const wasmExecUrl = new URL("./wasm_exec.js", import.meta.url);
  const filePath = fileURLToPath(wasmExecUrl);
  const jsText = await readFile(filePath, "utf8");
  new Function(jsText)();
  goRuntimeInjected = true;
}

export async function resolvePackagedWasmURL(
  override?: string | URL
): Promise<string | URL> {
  if (override) return override;

  if (isBrowser) {
    if (!wasmUrlBundled) await tryLoadBundlerAssets();
    if (wasmUrlBundled) return wasmUrlBundled;
    return new URL("./loom.wasm", import.meta.url);
  }

  return new URL("./loom.wasm", import.meta.url);
}

export async function instantiateGoWasm(go: any, wasmUrl: string | URL) {
  const asString = typeof wasmUrl === "string" ? wasmUrl : wasmUrl.toString();

  if (isBrowser) {
    if (typeof WebAssembly.instantiateStreaming === "function") {
      const res = await fetch(asString);
      try {
        const { instance } = await WebAssembly.instantiateStreaming(
          res,
          go.importObject
        );
        return instance;
      } catch {
        const buf = await res.arrayBuffer();
        const result = (await WebAssembly.instantiate(
          buf,
          go.importObject
        )) as any;
        const instance: WebAssembly.Instance = result.instance ?? result;
        return instance;
      }
    } else {
      const res = await fetch(asString);
      const buf = await res.arrayBuffer();
      const result = (await WebAssembly.instantiate(
        buf,
        go.importObject
      )) as any;
      const instance: WebAssembly.Instance = result.instance ?? result;
      return instance;
    }
  }

  // Node/Bun
  const { readFile } = await import("node:fs/promises");
  const { fileURLToPath } = await import("node:url");
  const url =
    typeof wasmUrl === "string" ? new URL(wasmUrl, import.meta.url) : wasmUrl;
  const filePath = fileURLToPath(url);
  const buf = await readFile(filePath);
  const result = (await WebAssembly.instantiate(buf, go.importObject)) as any;
  const instance: WebAssembly.Instance = result.instance ?? result;
  return instance;
}
