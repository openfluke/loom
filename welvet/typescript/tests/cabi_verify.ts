/**
 * cabi_verify.ts
 * TypeScript port of cabi_verify.html
 */

import { loadLoomWASM } from "../src/loader.js";

const EXPECTED_SYMBOLS = [
  'createLoomNetwork', 'loadLoomNetwork',
  'compareLoomDNA',
  'getDefaultTargetPropConfig', 'defaultSpliceConfig', 'defaultNEATConfig',
  'createLoomNEATPopulation',
  'setupWebGPU',
];

const EXPECTED_NET_METHODS = [
  'sequentialForward', 'extractDNA', 'extractBlueprint', 'getLayerCount',
  'getLayerSpec', 'morphLayer', 'spliceDNA', 'neatMutate',
  'createSystolicState', 'createTargetPropState', 'initGPU', 'syncToGPU',
  'syncToCPU', 'train', 'free', '_id',
];

const EXPECTED_POP_METHODS = [
  '_id', 'size', 'getNetwork', 'evolveWithFitnesses',
  'best', 'bestFitness', 'summary', 'free',
];

const DENSE_3L = JSON.stringify({
  depth: 3, rows: 1, cols: 1, layers_per_cell: 1,
  layers: [
    { z: 0, y: 0, x: 0, l: 0, type: "Dense", input_height: 16, output_height: 16, activation: "ReLU", dtype: "F32" },
    { z: 1, y: 0, x: 0, l: 0, type: "Dense", input_height: 16, output_height: 16, activation: "ReLU", dtype: "F32" },
    { z: 2, y: 0, x: 0, l: 0, type: "Dense", input_height: 16, output_height: 4, activation: "Linear", dtype: "F32" },
  ]
});

const SWIGLU_NET = JSON.stringify({
  depth: 2, rows: 1, cols: 1, layers_per_cell: 1,
  layers: [
    { z: 0, y: 0, x: 0, l: 0, type: "SwiGLU", input_height: 16, output_height: 32, dtype: "F32" },
    { z: 1, y: 0, x: 0, l: 0, type: "Dense", input_height: 32, output_height: 4, activation: "Linear", dtype: "F32" },
  ]
});

export async function runVerify() {
  console.log("=== Loom WASM C-ABI Diagnostic Report ===");
  
  // Decide which loader to use
  if (typeof process !== "undefined" && process.versions && process.versions.node) {
    const { loadLoomWASM } = await import("../src/loader.js");
    await loadLoomWASM();
  } else {
    // @ts-ignore
    const { loadLoomWASMBrowser } = await import("../src/loader.browser.js");
    await loadLoomWASMBrowser();
  }

  let totalPass = 0;
  let totalFail = 0;

  // 1. Global symbol check
  console.log("\n[1] Checking global WASM exports...");
  for (const sym of EXPECTED_SYMBOLS) {
    // @ts-ignore
    if (typeof globalThis[sym] === 'function') {
      console.log(`  [PASS] ${sym}`);
      totalPass++;
    } else {
      console.error(`  [FAIL] ${sym} (missing)`);
      totalFail++;
    }
  }

  // 2. Network method check
  console.log("\n[2] Checking network wrapper methods...");
  let net: any = null;
  try {
    // @ts-ignore
    net = globalThis.createLoomNetwork(DENSE_3L);
    if (net) {
      for (const m of EXPECTED_NET_METHODS) {
        if (net[m] !== undefined) {
          console.log(`  [PASS] ${m}`);
          totalPass++;
        } else {
          console.error(`  [FAIL] ${m} (missing)`);
          totalFail++;
        }
      }
    }
  } catch (e) {
    console.error("  [FAIL] createLoomNetwork failed:", e);
    totalFail++;
  }

  // 3. Population method check
  if (net) {
    console.log("\n[3] Checking NEAT population wrapper methods...");
    try {
      // @ts-ignore
      const cfg = globalThis.defaultNEATConfig(16);
      // @ts-ignore
      const pop = globalThis.createLoomNEATPopulation(net._id, 4, cfg);
      if (pop) {
        for (const m of EXPECTED_POP_METHODS) {
          if ((pop as any)[m] !== undefined) {
            console.log(`  [PASS] ${m}`);
            totalPass++;
          } else {
            console.error(`  [FAIL] ${m} (missing)`);
            totalFail++;
          }
        }
        pop.free();
      }
    } catch (e) {
      console.error("  [FAIL] Population creation failed:", e);
      totalFail++;
    }
  }

  // 4. Functional smoke tests
  console.log("\n[4] Running functional smoke tests...");

  const smokeTest = (name: string, fn: () => any) => {
    try {
      const result = fn();
      console.log(`  [PASS] ${name}${result ? " → " + result : ""}`);
      totalPass++;
    } catch (e: any) {
      console.error(`  [FAIL] ${name} → ${e.message}`);
      totalFail++;
    }
  };

  smokeTest("sequentialForward", () => {
    // @ts-ignore
    const n = globalThis.createLoomNetwork(DENSE_3L);
    const input = new Float32Array(16).fill(0.5);
    const out = n.sequentialForward(input);
    n.free();
    if (!out || out.length === 0) throw new Error("empty output");
    return "out[0]=" + out[0].toFixed(4);
  });

  smokeTest("extractDNA", () => {
    // @ts-ignore
    const n = globalThis.createLoomNetwork(DENSE_3L);
    const dna = n.extractDNA();
    n.free();
    const parsed = JSON.parse(dna);
    return "sigs=" + parsed.length;
  });

  smokeTest("compareLoomDNA", () => {
    // @ts-ignore
    const n1 = globalThis.createLoomNetwork(DENSE_3L);
    // @ts-ignore
    const n2 = globalThis.createLoomNetwork(DENSE_3L);
    const dna1 = n1.extractDNA();
    const dna2 = n2.extractDNA();
    // @ts-ignore
    const result = JSON.parse(globalThis.compareLoomDNA(dna1, dna2));
    n1.free(); n2.free();
    return "overlap=" + (result.overall_overlap || result.OverallOverlap || "?");
  });

  smokeTest("createLoomNetwork (SwiGLU)", () => {
    // @ts-ignore
    const n = globalThis.createLoomNetwork(SWIGLU_NET);
    const c = n.getLayerCount();
    n.free();
    return "layers=" + c;
  });

  console.log("\n[5] Final Summary");
  console.log("================================");
  if (totalFail === 0) {
    console.log(`SUCCESS: All ${totalPass} checks passed.`);
  } else {
    console.warn(`PARTIAL: ${totalPass} passed, ${totalFail} FAILED.`);
    process.exit(1);
  }
  console.log("================================");

  if (net) net.free();
}

// Auto-run if executed directly via Node.js/tsx
if (typeof process !== "undefined" && import.meta.url.includes(process.argv[1].replace(/\\/g, '/'))) {
  runVerify();
}
