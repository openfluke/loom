/**
 * @openfluke/welvet - Isomorphic WASM Wrapper
 *
 * Direct wrapper around Loom WASM that mirrors main.go exports exactly.
 * Provides the same API in both Node.js and browser environments.
 */

import {
  Network,
  StepState,
  GraftResult,
  KMeansResult,
  CorrelationResult,
  EnsembleMatch,
  AdaptationTracker
} from "./types.js";
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
 * Create a network handle for grafting
 */
export function createKHandle(config: object | string): number {
  const jsonConfig = typeof config === "string" ? config : JSON.stringify(config);
  return createNetworkForGraft(jsonConfig);
}

/**
 * Graft multiple networks together
 */
export function graft(ids: number[], combineMode: string): GraftResult {
  const idsJSON = JSON.stringify(ids);
  const resJSON = graftNetworks(idsJSON, combineMode);
  return JSON.parse(resJSON) as GraftResult;
}

/**
 * Perform K-Means Clustering
 */
export function kmeans(data: number[][], k: number, iter: number): KMeansResult {
  const resJSON = kmeansCluster(JSON.stringify(data), k, iter);
  return JSON.parse(resJSON) as KMeansResult;
}

/**
 * Compute Correlation Matrix
 */
export function correlation(matrixA: number[][], matrixB?: number[][]): CorrelationResult {
  const jsonA = JSON.stringify(matrixA);
  const jsonB = matrixB ? JSON.stringify(matrixB) : "null"; // Use "null" string for nil
  const resJSON = computeCorrelation(jsonA, jsonB);
  const raw = JSON.parse(resJSON);

  // Transform to match interface
  return {
    pearson: raw.correlation?.matrix || raw.Correlation?.Matrix || raw.matrix || [],
    spearman: raw.spearman?.matrix || raw.Spearman?.Matrix || []
  };
}

/**
 * Find Complementary Ensemble Matches
 */
export function ensemble(models: object[], minCoverage: number): EnsembleMatch[] {
  const resJSON = findComplementaryMatches(JSON.stringify(models), minCoverage);
  return JSON.parse(resJSON) as EnsembleMatch[];
}



/**
 * Create Adaptation Tracker
 */
export function tracker(windowMs: number, totalMs: number): AdaptationTracker {
  return createAdaptationTracker(windowMs, totalMs);
}

/**
 * Default export with all functions
 */
export default {
  init,
  initBrowser,
  createNetwork,
  createKHandle,
  graft,
  kmeans,
  correlation,
  ensemble,
  tracker
};

