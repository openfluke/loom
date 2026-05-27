/**
 * Seven-layer CPU suite — Node.js + TypeScript loader → WASM bindings → Loom.
 * Same runner as the browser page (assets/seven_layer/runner.js).
 *
 * Prereq: welvet/wasm/build.sh (produces assets/main.wasm) then npm run build
 */

import { loadLoomWASM } from "../src/loader.js";
import { LAYER_SUITES } from "../assets/seven_layer/bench.js";
import { runAllSuites } from "../assets/seven_layer/runner.js";

const layer = process.argv.find((a) => a.startsWith("--layer="))?.split("=")[1]
  ?? (process.argv.includes("--layer") ? process.argv[process.argv.indexOf("--layer") + 1] : null);

async function main() {
  console.log("Loading WASM (dist/main.wasm)…");
  await loadLoomWASM();

  if (typeof (globalThis as Record<string, unknown>).createLoomNetwork !== "function") {
    throw new Error("createLoomNetwork missing — rebuild wasm (welvet/wasm/build.sh)");
  }

  console.log("=== welvet seven-layer — TypeScript (Node) → WASM → Loom ===\n");
  const ok = await runAllSuites((m) => console.log(m), layer || null);
  console.log(ok ? "\n✅ ALL PASSED" : "\n❌ FAILURES");
  process.exit(ok ? 0 : 1);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
