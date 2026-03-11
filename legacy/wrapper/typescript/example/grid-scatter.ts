/**
 * Grid Scatter Multi-Agent Training Demo
 * TypeScript version of Example 4 from json_grid_scatter_agents.go
 */

import { init, createNetwork } from "../dist/index.js";
import type {
  NetworkConfig,
  TrainingBatch,
  TrainingConfig,
} from "../dist/types.js";

async function runGridScatterTraining() {
  console.log("🤖 Running Grid Scatter Multi-Agent Training...");
  console.log(
    "Task: 3 agents learn to collaborate for binary classification\n"
  );

  // Initialize LOOM WASM
  await init();

  // Training Multi-Agent Collaboration Network
  const agentConfig: NetworkConfig = {
    batch_size: 1,
    grid_rows: 1,
    grid_cols: 3,
    layers_per_cell: 1,
    layers: [
      {
        type: "dense",
        input_size: 8,
        output_size: 16,
        activation: "relu",
      },
      {
        type: "parallel",
        combine_mode: "grid_scatter",
        grid_output_rows: 3,
        grid_output_cols: 1,
        grid_output_layers: 1,
        grid_positions: [
          { branch_index: 0, target_row: 0, target_col: 0, target_layer: 0 },
          { branch_index: 1, target_row: 1, target_col: 0, target_layer: 0 },
          { branch_index: 2, target_row: 2, target_col: 0, target_layer: 0 },
        ],
        branches: [
          {
            type: "parallel",
            combine_mode: "add",
            branches: [
              {
                type: "dense",
                input_size: 16,
                output_size: 8,
                activation: "relu",
              },
              {
                type: "dense",
                input_size: 16,
                output_size: 8,
                activation: "gelu",
              },
            ],
          },
          {
            type: "lstm",
            input_size: 16,
            hidden_size: 8,
            seq_length: 1,
          },
          {
            type: "rnn",
            input_size: 16,
            hidden_size: 8,
            seq_length: 1,
          },
        ],
      },
      {
        type: "dense",
        input_size: 24,
        output_size: 2,
        activation: "sigmoid",
      },
    ],
  };

  console.log("Architecture:");
  console.log("  Shared Layer → Grid Scatter (3 agents) → Decision");
  console.log("  Agent 0: Feature Extractor (ensemble of 2 dense)");
  console.log("  Agent 1: Transformer (LSTM)");
  console.log("  Agent 2: Integrator (RNN)");
  console.log("Task: Binary classification (sum comparison)\n");

  console.log("Building network from JSON...");
  const agentNetwork = createNetwork(agentConfig);
  console.log("✅ Agent network created!\n");

  // Create training data
  const batches: TrainingBatch[] = [
    { Input: [0.2, 0.2, 0.2, 0.2, 0.8, 0.8, 0.8, 0.8], Target: [1.0, 0.0] },
    { Input: [0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1], Target: [0.0, 1.0] },
    { Input: [0.7, 0.7, 0.7, 0.7, 0.3, 0.3, 0.3, 0.3], Target: [0.0, 1.0] },
    { Input: [0.3, 0.3, 0.3, 0.3, 0.7, 0.7, 0.7, 0.7], Target: [1.0, 0.0] },
  ];

  // Training configuration
  const config: TrainingConfig = {
    Epochs: 800,
    LearningRate: 0.15,
    LossType: "mse",
    Verbose: false,
  };

  console.log(
    `Training for ${
      config.Epochs
    } epochs with learning rate ${config.LearningRate.toFixed(3)}`
  );

  // Train using the proper Train method
  const startTime = performance.now();
  const trainResult = agentNetwork.Train(JSON.stringify([batches, config]));
  const endTime = performance.now();

  const result = JSON.parse(trainResult);
  const trainingData = result[0];

  console.log("\n✅ Training complete!");
  console.log(
    `Training time: ${((endTime - startTime) / 1000).toFixed(2)} seconds`
  );
  console.log(`Initial Loss: ${trainingData.LossHistory[0].toFixed(6)}`);
  console.log(`Final Loss: ${trainingData.FinalLoss.toFixed(6)}`);

  const improvement = (
    ((trainingData.LossHistory[0] - trainingData.FinalLoss) /
      trainingData.LossHistory[0]) *
    100
  ).toFixed(2);
  console.log(`Improvement: ${improvement}%`);
  console.log(`Total Epochs: ${trainingData.LossHistory.length}`);

  // Test final predictions
  console.log("\nFinal predictions:");
  const originalPredictions: number[][] = [];
  for (let i = 0; i < 4; i++) {
    const testResult = agentNetwork.ForwardCPU(
      JSON.stringify([batches[i].Input])
    );
    const testParsed = JSON.parse(testResult);
    const pred = testParsed[0];
    originalPredictions.push(pred);

    const predClass = pred[0] > pred[1] ? 0 : 1;
    const expectedClass = batches[i].Target[0] > batches[i].Target[1] ? 0 : 1;
    const correct = predClass === expectedClass ? "✓" : "✗";

    console.log(
      `Sample ${i}: [${pred[0].toFixed(3)}, ${pred[1].toFixed(3)}] → ` +
        `Class ${predClass} (expected ${expectedClass}) ${correct}`
    );
  }

  console.log("\n✅ Multi-agent training complete!");

  // Save and reload model to verify serialization
  console.log("\n💾 Testing model save/load...");
  const savedModelResult = agentNetwork.SaveModelToString(
    JSON.stringify(["grid_scatter_test"])
  );

  // WASM returns results as [value, error] array
  const parsedResult = JSON.parse(savedModelResult);
  const savedModel = parsedResult[0]; // Get the actual JSON string from the result array

  console.log(`✓ Model saved (${savedModel.length} bytes)`);

  // Load the model back using loadLoomNetwork (global WASM function)
  console.log("Loading model from saved state...");
  // @ts-ignore - loadLoomNetwork is a global function from WASM
  const reloadedNetwork = globalThis.loadLoomNetwork(
    savedModel,
    "grid_scatter_test"
  );
  console.log("✓ Model loaded"); // Test predictions with reloaded model
  console.log("\nVerifying predictions match:");
  let allMatch = true;
  for (let i = 0; i < 4; i++) {
    const testResult = reloadedNetwork.ForwardCPU(
      JSON.stringify([batches[i].Input])
    );
    const testParsed = JSON.parse(testResult);
    const pred = testParsed[0];

    // Compare with original predictions
    const diff0 = Math.abs(pred[0] - originalPredictions[i][0]);
    const diff1 = Math.abs(pred[1] - originalPredictions[i][1]);
    const maxDiff = Math.max(diff0, diff1);

    const match = maxDiff < 1e-6;
    allMatch = allMatch && match;

    console.log(
      `Sample ${i}: [${pred[0].toFixed(3)}, ${pred[1].toFixed(
        3
      )}] (diff: ${maxDiff.toExponential(2)}) ${match ? "✓" : "✗"}`
    );
  }

  if (allMatch) {
    console.log("\n✅ Save/Load verification passed! All predictions match.");
  } else {
    console.log("\n❌ Save/Load verification failed! Predictions don't match.");
  }
}

runGridScatterTraining().catch(console.error);
