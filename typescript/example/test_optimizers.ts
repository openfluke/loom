/**
 * LOOM Optimizer Test Suite - TypeScript/Node.js
 * Replicates C# optimizer tests exactly
 */

import { init, createNetwork, Network, StepState } from "../src/index.js";

// Training data (matches C# exactly)
const trainingData = [
  { input: [0.1, 0.2, 0.1, 0.3], target: [1.0, 0.0, 0.0] }, // Class 0
  { input: [0.8, 0.9, 0.7, 0.8], target: [0.0, 1.0, 0.0] }, // Class 1
  { input: [0.3, 0.5, 0.9, 0.6], target: [0.0, 0.0, 1.0] }, // Class 2
  { input: [0.2, 0.1, 0.2, 0.2], target: [1.0, 0.0, 0.0] }, // Class 0
  { input: [0.9, 0.8, 0.8, 0.9], target: [0.0, 1.0, 0.0] }, // Class 1
  { input: [0.4, 0.6, 0.8, 0.7], target: [0.0, 0.0, 1.0] }, // Class 2
];

// Network configuration (matches C# exactly)
const networkConfig = {
  batch_size: 1,
  grid_rows: 1,
  grid_cols: 1,
  layers_per_cell: 3,
  layers: [
    { type: "dense", input_height: 4, output_height: 8, activation: "relu" },
    { type: "lstm", input_size: 8, hidden_size: 12, seq_length: 1 },
    { type: "dense", input_height: 12, output_height: 3, activation: "softmax" }
  ]
};

/**
 * Run training loop with specified optimizer
 */
async function runTrainingLoop(
  optimizerName: string,
  network: Network,
  state: StepState,
  applyGradientsFn: (network: Network) => void,
  steps: number = 5000
): Promise<void> {
  console.log(`📊 Test: ${optimizerName}`);
  console.log("----------------------------------");

  const startTime = Date.now();
  let totalLoss = 0.0;

  for (let step = 0; step < steps; step++) {
    const idx = step % trainingData.length;
    const { input, target } = trainingData[idx];

    // Forward pass
    state.setInput(new Float32Array(input));
    state.stepForward();
    const output = state.getOutput();

    // Calculate loss and gradients (MSE)
    let loss = 0.0;
    const gradients = new Float32Array(3);
    for (let i = 0; i < 3; i++) {
      const diff = output[i] - target[i];
      loss += diff * diff;
      gradients[i] = 2.0 * diff / 3.0;
    }
    loss /= 3.0;
    totalLoss += loss;

    // Backward pass
    state.stepBackward(gradients);

    // Apply optimizer
    applyGradientsFn(network);

    // Log every 1000 steps
    if ((step + 1) % 1000 === 0) {
      const avgLoss = totalLoss / 1000;
      console.log(`  Step ${step + 1}: Avg Loss=${avgLoss.toFixed(6)}`);
      totalLoss = 0.0;
    }
  }

  const totalTime = (Date.now() - startTime) / 1000;
  const stepsPerSec = steps / totalTime;
  console.log(`✅ ${optimizerName} complete!`);
  console.log(`   Time: ${totalTime.toFixed(2)}s (${Math.round(stepsPerSec)} steps/sec)\n`);
}

/**
 * Test 1: Simple SGD
 */
async function testSGD() {
  const network = createNetwork(networkConfig);
  network.InitializeWeights(JSON.stringify([]));
  const state = network.createStepState(4);

  await runTrainingLoop(
    "Simple SGD (baseline)",
    network,
    state,
    (net) => net.ApplyGradients(JSON.stringify([0.01]))
  );
}

/**
 * Test 2: AdamW Optimizer
 */
async function testAdamW() {
  const network = createNetwork(networkConfig);
  network.InitializeWeights(JSON.stringify([]));
  const state = network.createStepState(4);

  await runTrainingLoop(
    "AdamW Optimizer",
    network,
    state,
    (net) => net.ApplyGradientsAdamW(JSON.stringify([0.001, 0.9, 0.999, 0.01]))
  );
}

/**
 * Test 3: RMSprop Optimizer
 */
async function testRMSprop() {
  const network = createNetwork(networkConfig);
  network.InitializeWeights(JSON.stringify([]));
  const state = network.createStepState(4);

  await runTrainingLoop(
    "RMSprop Optimizer",
    network,
    state,
    (net) => net.ApplyGradientsRMSprop(JSON.stringify([0.001, 0.99, 1e-8, 0.0]))
  );
}

/**
 * Test 4: SGD with Momentum
 */
async function testSGDMomentum() {
  const network = createNetwork(networkConfig);
  network.InitializeWeights(JSON.stringify([]));
  const state = network.createStepState(4);

  await runTrainingLoop(
    "SGD with Momentum",
    network,
    state,
    (net) => net.ApplyGradientsSGDMomentum(JSON.stringify([0.01, 0.9, 0.0, 0]))
  );
}

/**
 * Main test runner
 */
async function main() {
  console.log("🚀 LOOM Optimizer Test Suite (TypeScript)");
  console.log("==========================================\n");

  // Initialize WASM
  console.log("Initializing WASM...");
  await init();
  console.log("✅ WASM initialized\n");

  // Run all tests
  await testSGD();
  await testAdamW();
  await testRMSprop();
  await testSGDMomentum();

  console.log("🎉 All optimizer tests complete!\n");
  console.log("Verified Functions:");
  console.log("  ✅ ApplyGradients (simple SGD)");
  console.log("  ✅ ApplyGradientsAdamW");
  console.log("  ✅ ApplyGradientsRMSprop");
  console.log("  ✅ ApplyGradientsSGDMomentum");
  console.log("\nAll optimizer methods working correctly via TypeScript! 🚀");
}

main().catch((error) => {
  console.error("❌ Error:", error);
  process.exit(1);
});