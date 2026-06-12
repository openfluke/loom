/**
 * Seven-layer CPU suite — Node.js + TypeScript loader → WASM bindings → Loom.
 * Same runner as the browser page (assets/seven_layer/runner.js).
 *
 * Prereq: welvet/wasm/build.sh (produces assets/main.wasm) then npm run build
 *
 * Full suite (no --layer): each layer runs in a fresh Node process so WASM heap
 * resets between CNN/MHA/RNN stacks (avoids Node WASM OOM on long runs).
 */

import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { loadLoomWASM } from "../src/loader.js";
import { LAYER_SUITES } from "../assets/seven_layer/bench.js";
import { runAllSuites } from "../assets/seven_layer/runner.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const layer = process.argv.find((a) => a.startsWith("--layer="))?.split("=")[1]
  ?? (process.argv.includes("--layer") ? process.argv[process.argv.indexOf("--layer") + 1] : null);

function runLayerSubprocess(layerName: string): Promise<number> {
  const scriptPath = path.join(__dirname, "seven_layer.ts");
  const pkgRoot = path.join(__dirname, "..");
  return new Promise((resolve) => {
    const child = spawn(
      process.execPath,
      ["--import", "tsx", scriptPath, "--layer", layerName],
      { cwd: pkgRoot, stdio: "inherit", env: process.env },
    );
    child.on("close", (code) => resolve(code ?? 1));
    child.on("error", () => resolve(1));
  });
}

async function runInProcess(filter: string | null): Promise<boolean> {
  console.log("Loading WASM (dist/main.wasm)…");
  await loadLoomWASM();

  if (typeof (globalThis as Record<string, unknown>).createLoomNetwork !== "function") {
    throw new Error("createLoomNetwork missing — rebuild wasm (welvet/wasm/build.sh)");
  }

  console.log("=== welvet seven-layer — TypeScript (Node) → WASM → Loom ===\n");
  return runAllSuites((m) => console.log(m), filter);
}

async function main() {
  if (layer) {
    const ok = await runInProcess(layer);
    console.log(ok ? "\n✅ ALL PASSED" : "\n❌ FAILURES");
    process.exit(ok ? 0 : 1);
    return;
  }

  console.log("=== welvet seven-layer — full suite (one WASM process per layer) ===\n");
  let ok = true;
  for (const suite of LAYER_SUITES) {
    console.log(`\n▶ Layer: ${suite.name}`);
    const code = await runLayerSubprocess(suite.name);
    if (code !== 0) ok = false;
  }
  console.log(ok ? "\n✅ ALL PASSED" : "\n❌ FAILURES");
  process.exit(ok ? 0 : 1);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
