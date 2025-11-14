// Grid Scatter Multi-Agent Training Demo
// Example 4 from json_grid_scatter_agents.go

function runGridScatterTest() {
  clearLog();
  log("info", "🤖 Running Grid Scatter Multi-Agent Training...");
  log("info", "Task: 3 agents learn to collaborate for binary classification");

  // Training Multi-Agent Collaboration
  const agentConfig = {
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

  try {
    log("info", "\nArchitecture:");
    log("info", "  Shared Layer → Grid Scatter (3 agents) → Decision");
    log("info", "  Agent 0: Feature Extractor (ensemble of 2 dense)");
    log("info", "  Agent 1: Transformer (LSTM)");
    log("info", "  Agent 2: Integrator (RNN)");
    log("info", "Task: Binary classification (sum comparison)");

    log("info", "\nBuilding network from JSON...");
    const agentNetwork = createLoomNetwork(JSON.stringify(agentConfig));

    if (typeof agentNetwork === "string") {
      log("error", "❌ Error creating network: " + agentNetwork);
      return;
    }

    log("success", "✅ Agent network created!");

    // Create training data: 4 samples, 8 features each, binary classification
    // Pattern: if sum of first 4 elements > sum of last 4 elements, class 0, else class 1
    const batches = [
      {
        Input: [0.2, 0.2, 0.2, 0.2, 0.8, 0.8, 0.8, 0.8],
        Target: [1.0, 0.0],
      },
      {
        Input: [0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1],
        Target: [0.0, 1.0],
      },
      {
        Input: [0.7, 0.7, 0.7, 0.7, 0.3, 0.3, 0.3, 0.3],
        Target: [0.0, 1.0],
      },
      {
        Input: [0.3, 0.3, 0.3, 0.3, 0.7, 0.7, 0.7, 0.7],
        Target: [1.0, 0.0],
      },
    ];

    // Training configuration
    const config = {
      Epochs: 800,
      LearningRate: 0.15,
      UseGPU: false,
      PrintEveryBatch: 0,
      GradientClip: 1.0,
      LossType: "mse",
      Verbose: false,
    };

    log(
      "info",
      `\nTraining for ${
        config.Epochs
      } epochs with learning rate ${config.LearningRate.toFixed(3)}`
    );
    log("info", "Architecture:");
    log("info", "  Shared Layer → Grid Scatter (3 agents) → Decision");
    log("info", "  Agent 0: Feature Extractor (ensemble of 2 dense)");
    log("info", "  Agent 1: Transformer (LSTM)");
    log("info", "  Agent 2: Integrator (RNN)");
    log("info", "Task: Binary classification (sum comparison)\n");

    // Train using the proper Train method
    const startTime = performance.now();

    let trainResult;
    try {
      trainResult = agentNetwork.Train(JSON.stringify([batches, config]));
    } catch (e) {
      log("error", "❌ Training crashed: " + e.message);
      console.error(e);
      return;
    }

    const endTime = performance.now();

    // Check for empty or error result
    if (!trainResult || trainResult === "") {
      log("error", "❌ Training returned empty result (possible panic)");
      return;
    }

    if (typeof trainResult === "string" && trainResult.startsWith("Error:")) {
      log("error", "❌ Training failed: " + trainResult);
      return;
    }

    let result;
    try {
      result = JSON.parse(trainResult);
    } catch (e) {
      log("error", "❌ Failed to parse training result: " + e.message);
      log("error", "Raw result: " + trainResult.substring(0, 200));
      return;
    }

    if (!result || !result[0] || result[0].FinalLoss === undefined) {
      log("error", "❌ Invalid training result: " + trainResult);
      return;
    }

    const trainingData = result[0];

    log("success", `\n✅ Training complete!`);
    log(
      "info",
      `Training time: ${((endTime - startTime) / 1000).toFixed(2)} seconds`
    );
    log("info", `Initial Loss: ${trainingData.LossHistory[0].toFixed(6)}`);
    log("info", `Final Loss: ${trainingData.FinalLoss.toFixed(6)}`);

    const improvement = (
      ((trainingData.LossHistory[0] - trainingData.FinalLoss) /
        trainingData.LossHistory[0]) *
      100
    ).toFixed(2);
    log("info", `Improvement: ${improvement}%`);
    log("info", `Total Epochs: ${trainingData.LossHistory.length}`);

    // Test final predictions on each sample
    log("info", "\nFinal predictions:");
    for (let i = 0; i < 4; i++) {
      const testResult = agentNetwork.ForwardCPU(
        JSON.stringify([batches[i].Input])
      );
      const testParsed = JSON.parse(testResult);
      const pred = testParsed[0];

      const predClass = pred[0] > pred[1] ? 0 : 1;
      const expectedClass = batches[i].Target[0] > batches[i].Target[1] ? 0 : 1;
      const correct = predClass === expectedClass ? "✓" : "✗";

      log(
        "success",
        `Sample ${i}: [${pred[0].toFixed(3)}, ${pred[1].toFixed(3)}] → ` +
          `Class ${predClass} (expected ${expectedClass}) ${correct}`
      );
    }

    log("success", "\n✅ Multi-agent training complete!");
  } catch (e) {
    log("error", "❌ Exception: " + e.message);
    console.error(e);
  }
}
