/**
 * benchmark.ts
 * TypeScript port of benchmark_training.html
 */

import { loadLoomWASM } from "../src/loader.js";

const TRAINING_CASES = [
  {
    name: 'Dense (Linear)', iters: 5, inDim: 512, outDim: 512,
    cfg: JSON.stringify({ depth: 1, rows: 1, cols: 1, layers_per_cell: 1, layers: [
      { z: 0, y: 0, x: 0, l: 0, type: "Dense", input_height: 512, output_height: 512, activation: "Linear", dtype: "F32" }
    ]})
  },
  {
    name: 'RMSNorm', iters: 5, inDim: 512, outDim: 512,
    cfg: JSON.stringify({ depth: 1, rows: 1, cols: 1, layers_per_cell: 1, layers: [
      { z: 0, y: 0, x: 0, l: 0, type: "RMSNorm", input_height: 512, output_height: 512, dtype: "F32" }
    ]})
  },
  {
    name: 'SwiGLU (MLP)', iters: 5, inDim: 512, outDim: 1024,
    cfg: JSON.stringify({ depth: 1, rows: 1, cols: 1, layers_per_cell: 1, layers: [
      { z: 0, y: 0, x: 0, l: 0, type: "SwiGLU", input_height: 512, output_height: 1024, dtype: "F32" }
    ]})
  },
  {
    name: 'Embedding', iters: 5, inDim: 16, outDim: 2048, isEmbedding: true,
    cfg: JSON.stringify({ depth: 1, rows: 1, cols: 1, layers_per_cell: 1, layers: [
      { z: 0, y: 0, x: 0, l: 0, type: "Embedding", vocab_size: 1024, embedding_dim: 128, dtype: "F32" }
    ]})
  },
  {
    name: 'Residual Add', iters: 5, inDim: 512, outDim: 512,
    cfg: JSON.stringify({ depth: 1, rows: 1, cols: 1, layers_per_cell: 1, layers: [
      { z: 0, y: 0, x: 0, l: 0, type: "Residual", input_height: 512, output_height: 512, dtype: "F32" }
    ]})
  },
  {
    name: 'MHA (Fused)', iters: 5, inDim: 128, outDim: 128,
    cfg: JSON.stringify({ depth: 1, rows: 1, cols: 1, layers_per_cell: 1, layers: [
      { z: 0, y: 0, x: 0, l: 0, type: "MHA", input_height: 128, output_height: 128, num_heads: 4, d_model: 128, dtype: "F32" }
    ]})
  }
];

function makeTrainBatches(inDim: number, outDim: number, nBatches: number, batchSize: number, isEmbedding?: boolean) {
  const batches: any[] = [];
  for (let b = 0; b < nBatches; b++) {
    const inp = new Float32Array(batchSize * inDim);
    const tgt = new Float32Array(batchSize * outDim);
    if (isEmbedding) {
      for (let i = 0; i < inp.length; i++) inp[i] = i % 1024;
    } else {
      for (let i = 0; i < inp.length; i++) inp[i] = (Math.random() * 2 - 1) * 0.5;
    }
    for (let i = 0; i < tgt.length; i++) tgt[i] = Math.random() * 0.1;
    batches.push({
      input: { shape: [batchSize, inDim], data: Array.from(inp) },
      target: { shape: [batchSize, outDim], data: Array.from(tgt) }
    });
  }
  return batches;
}

async function runCase(tc: any) {
  // @ts-ignore
  const net = globalThis.createLoomNetwork(tc.cfg);
  const batchSize = 4;
  const nBatches = 4;
  const epochs = 3;

  const batches = makeTrainBatches(tc.inDim, tc.outDim, nBatches, batchSize, tc.isEmbedding);
  const batchesJSON = JSON.stringify(batches);

  const input = new Float32Array(tc.inDim);
  input.fill(0.5);
  if (tc.isEmbedding) for (let i = 0; i < input.length; i++) input[i] = i % 1024;

  // warm-up
  net.sequentialForward(input);

  const t0 = performance.now();
  let lastOut: any;
  for (let i = 0; i < tc.iters; i++) {
    lastOut = net.sequentialForward(input);
  }
  const fwdMs = (performance.now() - t0) / tc.iters;

  let trainMs = -1;
  let initialLoss: number | null = null, finalLoss: number | null = null;
  try {
    const t1 = performance.now();
    const trainResult = await net.train(batchesJSON, epochs, 0.001);
    trainMs = performance.now() - t1;
    if (typeof trainResult === 'string') {
      try {
        const r = JSON.parse(trainResult);
        if (r.loss_history && r.loss_history.length > 0) {
          initialLoss = r.loss_history[0];
          finalLoss = r.loss_history[r.loss_history.length - 1];
        }
      } catch (e) {}
    }
  } catch (e) {
    trainMs = -1;
  }

  const sample = lastOut ? [lastOut[0] || 0, lastOut[1] || 0, lastOut[2] || 0] : null;
  const sanity = sample && sample.some((v: number) => Math.abs(v) > 1e-9);
  net.free();
  return { fwdMs, trainMs, sample, sanity, initialLoss, finalLoss };
}

export async function runBenchmark() {
  console.log("=== M-POLY-VTD Training Showdown Benchmark ===");
  
  // Decide which loader to use
  if (typeof process !== "undefined" && process.versions && process.versions.node) {
    const { loadLoomWASM } = await import("../src/loader.js");
    await loadLoomWASM();
  } else {
    // @ts-ignore
    const { loadLoomWASMBrowser } = await import("../src/loader.browser.js");
    await loadLoomWASMBrowser();
  }

  console.log("Layer".padEnd(15) + " | " + "Fwd ms/it".padEnd(11) + " | " + "Train ms".padEnd(10) +
    " | " + "Init Loss".padEnd(11) + " | " + "Final Loss".padEnd(11) + " | Sanity");
  console.log("-".repeat(85));

  for (const tc of TRAINING_CASES) {
    const res = await runCase(tc);

    const fwdStr = res.fwdMs >= 0 ? res.fwdMs.toFixed(3).padEnd(10) : 'N/A'.padEnd(10);
    const trainStr = res.trainMs >= 0 ? res.trainMs.toFixed(1).padEnd(9) : 'N/A'.padEnd(9);
    const iLoss = res.initialLoss != null ? res.initialLoss.toFixed(4).padEnd(10) : 'N/A'.padEnd(10);
    const fLoss = res.finalLoss != null ? res.finalLoss.toFixed(4).padEnd(10) : 'N/A'.padEnd(10);
    const sanStr = res.sanity ? 'REAL' : 'ZERO';

    console.log(`${tc.name.padEnd(15)} | ${fwdStr} | ${trainStr} | ${iLoss} | ${fLoss} | ${sanStr}`);
  }
}

// Auto-run if executed directly via Node.js/tsx
if (typeof process !== "undefined" && import.meta.url.includes(process.argv[1].replace(/\\/g, '/'))) {
  runBenchmark();
}
