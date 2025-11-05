import type { InitOptions, LoomAPI } from "./types";
import {
  ensureGoRuntime,
  resolvePackagedWasmURL,
  instantiateGoWasm,
} from "./loader";

// tiny helper that waits until WASM has placed symbols on globalThis
async function waitForExports(keys: string[], timeoutMs = 5000) {
  const t0 = performance.now();
  for (;;) {
    const ok = keys.every((k) => (globalThis as any)[k]);
    if (ok) return;
    if (performance.now() - t0 > timeoutMs) {
      throw new Error(
        `loom: timed out waiting for exports: ${keys.join(", ")}`
      );
    }
    await new Promise((r) => setTimeout(r, 10));
  }
}

/**
 * Initialize the LOOM WASM module and return the API
 *
 * @example
 * ```typescript
 * import { initLoom, ActivationType } from '@openfluke/loom';
 *
 * const loom = await initLoom();
 *
 * // Create a network: 784 → 392 → 10
 * const network = loom.NewNetwork(784, 1, 1, 2);
 *
 * // Configure layers
 * const layer0 = loom.InitDenseLayer(784, 392, ActivationType.ReLU);
 * const layer1 = loom.InitDenseLayer(392, 10, ActivationType.Sigmoid);
 *
 * network.SetLayer(JSON.stringify([0, 0, 0, JSON.parse(layer0)]));
 * network.SetLayer(JSON.stringify([0, 0, 1, JSON.parse(layer1)]));
 *
 * // Forward pass
 * const input = new Array(784).fill(0).map(() => Math.random());
 * const resultJSON = network.ForwardCPU(JSON.stringify([input]));
 * const [output, duration] = JSON.parse(resultJSON);
 *
 * console.log('Output:', output);
 * ```
 */
export async function initLoom(opts: InitOptions = {}): Promise<LoomAPI> {
  await ensureGoRuntime(opts.injectGoRuntime !== false);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const go = new (globalThis as any).Go();

  const wasmUrl = await resolvePackagedWasmURL(opts.wasmUrl);
  const instance = await instantiateGoWasm(go, wasmUrl);

  // IMPORTANT: don't await this — it resolves when the Go program exits.
  // Let it run; then poll for the exported symbols.
  // no await:
  // eslint-disable-next-line @typescript-eslint/no-floating-promises
  Promise.resolve().then(() => go.run(instance));

  // wait until your Go code has installed the globals
  await waitForExports(["NewNetwork", "LoadModelFromString", "CallLayerInit"]);

  const g = globalThis as any;

  // Helper function to call layer init functions from the registry
  const callLayerInit = (funcName: string, ...params: any[]): string => {
    return g.CallLayerInit(funcName, JSON.stringify(params));
  };

  const api: LoomAPI = {
    NewNetwork: g.NewNetwork,
    LoadModelFromString: g.LoadModelFromString,

    // Layer initialization functions using CallLayerInit (registry-based)
    InitDenseLayer: (
      inputSize: number,
      outputSize: number,
      activation: number
    ) => {
      return callLayerInit("InitDenseLayer", inputSize, outputSize, activation);
    },
    InitConv2DLayer: (
      inChannels: number,
      outChannels: number,
      kernelSize: number,
      stride: number,
      padding: number,
      inputH: number,
      inputW: number,
      activation: number
    ) => {
      return callLayerInit(
        "InitConv2DLayer",
        inChannels,
        outChannels,
        kernelSize,
        stride,
        padding,
        inputH,
        inputW,
        activation
      );
    },
    InitMultiHeadAttentionLayer: (
      dModel: number,
      numHeads: number,
      seqLength: number,
      activation: number
    ) => {
      return callLayerInit(
        "InitMultiHeadAttentionLayer",
        dModel,
        numHeads,
        seqLength,
        activation
      );
    },
    InitRNNLayer: (
      inputSize: number,
      hiddenSize: number,
      seqLength: number,
      outputSize: number
    ) => {
      return callLayerInit(
        "InitRNNLayer",
        inputSize,
        hiddenSize,
        seqLength,
        outputSize
      );
    },
    InitLSTMLayer: (
      inputSize: number,
      hiddenSize: number,
      seqLength: number,
      outputSize: number
    ) => {
      return callLayerInit(
        "InitLSTMLayer",
        inputSize,
        hiddenSize,
        seqLength,
        outputSize
      );
    },
  };

  if (!api.NewNetwork) {
    throw new Error("loom: NewNetwork not found after WASM init");
  }

  return api;
}

export type { LoomAPI, LoomNetwork, InitOptions } from "./types";
export { ActivationType } from "./types";
