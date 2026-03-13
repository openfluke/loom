/**
 * @openfluke/welvet - Browser Entry Point
 *
 * Browser-only build without Node.js dependencies
 * Isomorphic wrapper that mirrors main.go WASM exports
 */

import { Network } from "./types.js";
import { loadLoomWASMBrowser } from "./loader.browser.js";

export * from "./types.js";
export { loadLoomWASMBrowser };

/**
 * Initialize WASM for Browser environment
 */
export async function initBrowser(): Promise<void> {
  await loadLoomWASMBrowser();
}

/**
 * Initialize - alias for initBrowser in browser context
 */
export async function init(): Promise<void> {
  return initBrowser();
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
