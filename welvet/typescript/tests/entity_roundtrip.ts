/**
 * .entity save/load — all layers × 21 dtypes (no training).
 * npm run test:entity-roundtrip
 */

import { loadLoomWASM } from "../src/loader.js";
import { runEntityRoundtripAll } from "../assets/seven_layer/entity_roundtrip.js";

const layer = process.argv.find((a) => a.startsWith("--layer="))?.split("=")[1]
  ?? (process.argv.includes("--layer") ? process.argv[process.argv.indexOf("--layer") + 1] : null);

async function main() {
  await loadLoomWASM();
  const ok = runEntityRoundtripAll((m) => console.log(m), layer || null);
  process.exit(ok ? 0 : 1);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
