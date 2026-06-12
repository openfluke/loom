/**
 * Verifies README / npm consumer flow: init → createNetwork → forwardPolymorphic →
 * morph → train → serialize/reload. Same logic as welvet/tmp-npm-demo/demo.ts.
 */

import {
  init,
  createNetwork,
  trainNetwork,
  DType,
  LOOM_ENGINE_VERSION,
  type Network,
} from "../src/index.js";

const TrainingMode = { CPUSC: 1, CPUMC: 2 } as const;

function forward(net: Network, data: Float32Array, shape: number[]): Float32Array {
  return net.forwardPolymorphic(data, JSON.stringify(shape));
}

function maxAbs(a: Float32Array, b: Float32Array): number {
  let m = 0;
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > m) m = d;
  }
  return m;
}

function assert(cond: boolean, msg: string): void {
  if (!cond) throw new Error(msg);
}

const WASM_REBUILD_HINT =
  "Rebuild WASM: bash welvet/wasm/build.sh && cd welvet/typescript && npm run build";

function wasmEngineVersion(): string | undefined {
  const fn = (globalThis as { loomEngineVersion?: () => string }).loomEngineVersion;
  return typeof fn === "function" ? fn() : undefined;
}

export async function runConsumerDemo(): Promise<void> {
  assert(LOOM_ENGINE_VERSION === "0.80.0", `expected Loom 0.80.0, got ${LOOM_ENGINE_VERSION}`);

  await init();

  const wasmVer = wasmEngineVersion();
  assert(
    wasmVer === LOOM_ENGINE_VERSION,
    wasmVer
      ? `stale main.wasm: WASM reports ${wasmVer}, package is ${LOOM_ENGINE_VERSION}. ${WASM_REBUILD_HINT}`
      : `stale main.wasm (no loomEngineVersion export). ${WASM_REBUILD_HINT}`,
  );

  const inShape = [1, 16];
  const outShape = [1, 8];

  const net = createNetwork({
    id: "consumer-demo-dense",
    depth: 1,
    rows: 1,
    cols: 1,
    layers_per_cell: 1,
    layers: [
      {
        z: 0, y: 0, x: 0, l: 0,
        type: "DENSE",
        dtype: "FLOAT32",
        input_height: 16,
        output_height: 8,
        activation: "RELU",
      },
    ],
  });

  const info = JSON.parse(net.getInfo());
  assert(info.total_layers === 1, `expected 1 layer, got ${info.total_layers}`);

  const input = new Float32Array(16);
  for (let i = 0; i < 16; i++) input[i] = 0.2 * Math.sin(i * 0.3);

  net.setTrainingMode(TrainingMode.CPUMC);
  const out0 = forward(net, input, inShape);
  assert(out0.length === 8, `forward out len ${out0.length}`);
  assert(Number.isFinite(out0[0]), "forward output non-finite");

  const morphStatus = JSON.parse(net.morphLayer(0, DType.INT8));
  assert(!morphStatus.error, morphStatus.error ?? "morph failed");

  const out1 = forward(net, input, inShape);
  assert(out1.length === 8, "forward after morph");

  const target = new Float32Array(8);
  for (let i = 0; i < 8; i++) target[i] = 0.5 + 0.3 * Math.sin(i * 0.17);

  const trainResult = trainNetwork(
    net,
    [{ input, target, inputShape: inShape, targetShape: outShape }],
    5,
    0.05,
  );
  assert((trainResult.loss_history?.length ?? 0) >= 1, "missing loss_history");
  assert(Number.isFinite(trainResult.final_loss), "non-finite final_loss");

  const wire = net.serialize();
  assert(typeof wire === "string" && wire.length > 10, "serialize empty");

  const reloaded = (globalThis as unknown as { deserializeLoomNetwork: (w: string) => Network })
    .deserializeLoomNetwork(wire);
  const out2 = forward(reloaded, input, inShape);
  assert(maxAbs(out1, out2) < 0.25, `reload drift too large: ${maxAbs(out1, out2)}`);
  reloaded.free();
  net.free();
}

async function main() {
  console.log("consumer_demo — README / npm smoke (Loom", LOOM_ENGINE_VERSION, ")\n");
  await runConsumerDemo();
  console.log("✅ consumer_demo passed\n");
}

const isMain = process.argv[1]?.endsWith("consumer_demo.ts");
if (isMain) {
  main().catch((e) => {
    console.error("❌", e);
    process.exit(1);
  });
}
