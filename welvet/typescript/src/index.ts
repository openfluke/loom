/**
 * @openfluke/welvet — M-POLY-VTD AI Engine
 *
 * Isomorphic TypeScript wrapper for the Loom v0.73.0 WASM module.
 * Supports Node.js and browser environments.
 *
 * @example
 * ```ts
 * import { init, createNetwork, DType } from "@openfluke/welvet";
 *
 * await init();
 *
 * const net = createNetwork({
 *   layers: [
 *     { type: "dense", input_height: 4, output_height: 8 },
 *     { type: "dense", input_height: 8, output_height: 2 },
 *   ]
 * });
 *
 * const output = net.sequentialForward(new Float32Array([1, 0, 0, 1]));
 * console.log(output); // Float32Array [...]
 * ```
 */

import type {
  Network,
  NEATPopulation,
  SystolicState,
  TargetPropState,
  TrainingBatch,
  TrainingResult,
  DNACompareResult,
} from "./types.js";

import { loadLoomWASMBrowser } from "./loader.browser.js";

// Re-export all types and constants
export * from "./types.js";
export { loadLoomWASMBrowser };

// ──────────────────────────────────────────────────────────────────────────────
// Initialization
// ──────────────────────────────────────────────────────────────────────────────

/**
 * Initialize the welvet WASM module.
 * Auto-detects Node.js vs browser environment.
 *
 * @param wasmUrl Optional custom URL for browser WASM loading (default: /dist/main.wasm)
 */
export async function init(wasmUrl?: string): Promise<void> {
  if (typeof window !== "undefined" && typeof document !== "undefined") {
    return loadLoomWASMBrowser(wasmUrl);
  }
  const mod = await import("./loader.js");
  return mod.loadLoomWASM();
}

/**
 * Initialize explicitly for browser environments.
 * @param wasmUrl Optional custom URL for WASM binary
 */
export async function initBrowser(wasmUrl?: string): Promise<void> {
  return loadLoomWASMBrowser(wasmUrl);
}

// ──────────────────────────────────────────────────────────────────────────────
// Network Lifecycle
// ──────────────────────────────────────────────────────────────────────────────

/**
 * Build a VolumetricNetwork from a JSON configuration object or string.
 *
 * @example
 * ```ts
 * const net = createNetwork({
 *   layers: [
 *     { type: "dense", input_height: 784, output_height: 256 },
 *     { type: "dense", input_height: 256, output_height: 10, activation: "silu" },
 *   ]
 * });
 * ```
 */
export function createNetwork(config: object | string): Network {
  const jsonConfig = typeof config === "string" ? config : JSON.stringify(config);
  return createLoomNetwork(jsonConfig) as unknown as Network;
}

/**
 * Load a pre-trained network from a SafeTensors file path.
 * (Node.js only — requires file system access)
 *
 * @param path Absolute or relative path to a .safetensors file
 */
export function loadNetwork(path: string): Network {
  return loadLoomNetwork(path) as unknown as Network;
}

// ──────────────────────────────────────────────────────────────────────────────
// WebGPU
// ──────────────────────────────────────────────────────────────────────────────

/**
 * Initialize WebGPU (browser only).
 * Sets window.webgpuAdapter, window.webgpuDevice, window.webgpuQueue.
 */
export async function setupWebGPU(): Promise<string> {
  // @ts-ignore — setupWebGPU is injected by the WASM module
  return setupWebGPU() as Promise<string>;
}

// ──────────────────────────────────────────────────────────────────────────────
// DNA / Introspection
// ──────────────────────────────────────────────────────────────────────────────

/**
 * Compare the architectural DNA of two networks.
 *
 * @param dnaA JSON string from network.extractDNA()
 * @param dnaB JSON string from network.extractDNA()
 * @returns DNACompareResult
 */
export function compareDNA(dnaA: string, dnaB: string): DNACompareResult {
  return JSON.parse(compareLoomDNA(dnaA, dnaB)) as DNACompareResult;
}

/**
 * Get the default TargetPropConfig.
 */
export function defaultTargetPropConfig(): object {
  return JSON.parse(getDefaultTargetPropConfig());
}

// ──────────────────────────────────────────────────────────────────────────────
// Training Helpers
// ──────────────────────────────────────────────────────────────────────────────

/**
 * Convenience wrapper for network.train() that handles JSON serialization.
 *
 * @param network The network to train
 * @param batches Array of {input, target} training pairs
 * @param epochs Number of training epochs
 * @param lr Learning rate
 */
export function trainNetwork(
  network: Network,
  batches: TrainingBatch[],
  epochs: number,
  lr: number
): TrainingResult {
  const serialized = batches.map((b) => {
    const inp = b.input instanceof Float32Array ? b.input : new Float32Array(b.input as number[]);
    const tgt = b.target instanceof Float32Array ? b.target : new Float32Array(b.target as number[]);
    const inShape  = b.inputShape  ?? [1, inp.length];
    const tgtShape = b.targetShape ?? [1, tgt.length];
    return {
      input:  { shape: inShape,  data: Array.from(inp) },
      target: { shape: tgtShape, data: Array.from(tgt) },
    };
  });
  return JSON.parse(network.train(JSON.stringify(serialized), epochs, lr)) as TrainingResult;
}

// ──────────────────────────────────────────────────────────────────────────────
// Evolution / DNA
// ──────────────────────────────────────────────────────────────────────────────

/** Get the default SpliceConfig as a plain object. */
export function getSpliceConfig(): object {
  return JSON.parse((globalThis as unknown as Record<string, () => string>)["defaultSpliceConfig"]());
}

/** Get the default NEATConfig as a plain object. */
export function getNEATConfig(dModel: number): object {
  return JSON.parse((globalThis as unknown as Record<string, (n: number) => string>)["defaultNEATConfig"](dModel));
}

/**
 * Create a NEAT population from a seed network.
 * @param network Seed network (its _id is used)
 * @param size Population size
 * @param cfg NEATConfig object or JSON string (defaults to getNEATConfig(64))
 */
export function createNEATPopulation(
  network: Network,
  size: number,
  cfg?: object | string
): NEATPopulation {
  const cfgJSON = cfg
    ? (typeof cfg === "string" ? cfg : JSON.stringify(cfg))
    : (globalThis as unknown as Record<string, (n: number) => string>)["defaultNEATConfig"](64);
  return (globalThis as unknown as Record<string, (...a: unknown[]) => NEATPopulation>)
    ["createLoomNEATPopulation"](network._id, size, cfgJSON);
}

// ──────────────────────────────────────────────────────────────────────────────
// Default export
// ──────────────────────────────────────────────────────────────────────────────

export default {
  init,
  initBrowser,
  createNetwork,
  loadNetwork,
  setupWebGPU,
  compareDNA,
  defaultTargetPropConfig,
  trainNetwork,
  getSpliceConfig,
  getNEATConfig,
  createNEATPopulation,
  // Re-export constants for convenience
  DType: {
    FLOAT64: 0, FLOAT32: 1, FLOAT16: 2, BFLOAT16: 3,
    FP8_E4M3: 4, FP8_E5M2: 5,
    INT64: 6, INT32: 7, INT16: 8, INT8: 9,
    UINT64: 10, UINT32: 11, UINT16: 12, UINT8: 13,
    INT4: 14, UINT4: 15, FP4: 16,
    INT2: 17, UINT2: 18, TERNARY: 19, BINARY: 20,
  },
};

// Suppress unused import warnings for re-exported types
export type { Network, NEATPopulation, SystolicState, TargetPropState, TrainingBatch, TrainingResult, DNACompareResult };
