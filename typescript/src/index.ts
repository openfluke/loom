/**
 * @openfluke/welvet - Isomorphic WASM Wrapper
 *
 * Direct wrapper around Loom WASM that mirrors main.go exports exactly.
 * Provides the same API in both Node.js and browser environments.
 */

import { Network, StepState } from "./types.js";
import { loadLoomWASM } from "./loader.js";
import { loadLoomWASMBrowser } from "./loader.browser.js";

export * from "./types.js";
export { loadLoomWASM, loadLoomWASMBrowser };

/**
 * Initialize WASM for Node.js environment
 */
export async function init(): Promise<void> {
  await loadLoomWASM();
}

/**
 * Initialize WASM for Browser environment
 */
export async function initBrowser(): Promise<void> {
  await loadLoomWASMBrowser();
}

/**
 * Create a network from JSON config
 * Wrapper around the global createLoomNetwork function exposed by WASM
 */
export function createNetwork(config: object | string): Network {
  const jsonConfig = typeof config === "string"
    ? config
    : JSON.stringify(config);

  return createLoomNetwork(jsonConfig) as unknown as Network;
}



/**
 * Default export with all functions
 */
export default {
  init,
  initBrowser,
  createNetwork
};

