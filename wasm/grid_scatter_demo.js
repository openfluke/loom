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

    // Use EvaluateNetwork for detailed accuracy metrics
    log("info", "\n📊 Evaluating with EvaluateNetwork...");

    // Prepare inputs and expected outputs for evaluation
    // Note: inputs must be [][]float32 and expectedOutputs must be []float64
    const inputs = batches.map((b) => b.Input);
    const expectedOutputs = batches.map((b) =>
      b.Target[0] > b.Target[1] ? 0.0 : 1.0
    ); // Convert to class labels

    try {
      // Pass as array of two elements: [inputs_2d_array, expected_outputs_1d_array]
      const evalResult = agentNetwork.EvaluateNetwork(
        JSON.stringify([inputs, expectedOutputs])
      );

      // Check for error string
      if (
        typeof evalResult === "string" &&
        (evalResult.includes("Error:") || evalResult.includes("parameter"))
      ) {
        log("error", "EvaluateNetwork error: " + evalResult);
        return;
      }

      // Parse result - should be [metrics_object, error_or_null]
      const parsed = JSON.parse(evalResult);

      // Check if we got an error as second element
      if (parsed[1] !== null && parsed[1] !== undefined) {
        log("error", "Evaluation error: " + JSON.stringify(parsed[1]));
        return;
      }

      const metrics = parsed[0];

      if (!metrics || !metrics.total_samples) {
        log(
          "error",
          "Invalid metrics returned: " +
            JSON.stringify(parsed).substring(0, 200)
        );
        return;
      }

      log("success", "\n=== Evaluation Metrics ===");
      log("info", `Total Samples: ${metrics.total_samples}`);
      log("info", `Quality Score: ${metrics.score.toFixed(2)}/100`);
      log("info", `Average Deviation: ${metrics.avg_deviation.toFixed(2)}%`);
      log("info", `Failures (>100% deviation): ${metrics.failures}`);

      log("info", "\nDeviation Distribution:");
      const bucketOrder = [
        "0-10%",
        "10-20%",
        "20-30%",
        "30-40%",
        "40-50%",
        "50-100%",
        "100%+",
      ];
      for (const bucketName of bucketOrder) {
        const bucket = metrics.buckets[bucketName];
        if (bucket && bucket.count > 0) {
          const percentage = (
            (bucket.count / metrics.total_samples) *
            100
          ).toFixed(1);
          const bar = "█".repeat(
            Math.floor((bucket.count / metrics.total_samples) * 20)
          );
          log(
            "info",
            `  ${bucketName.padEnd(8)}: ${
              bucket.count
            } samples (${percentage}%) ${bar}`
          );
        }
      }

      // Show worst predictions if any
      if (metrics.results && metrics.results.length > 0) {
        const sorted = metrics.results.sort(
          (a, b) => b.deviation - a.deviation
        );
        const worst = sorted.slice(0, 2);
        if (worst.length > 0 && worst[0].deviation > 10) {
          log("info", "\nWorst predictions:");
          for (const r of worst) {
            log(
              "info",
              `  Sample ${r.sample_index}: Expected ${r.expected.toFixed(
                2
              )}, Got ${r.actual.toFixed(2)}, Deviation ${r.deviation.toFixed(
                2
              )}%`
            );
          }
        }
      }
    } catch (e) {
      log("error", "Failed to evaluate network: " + e.message);
      console.error(e);
    }

    log("success", "\n✅ Multi-agent training complete!");
  } catch (e) {
    log("error", "❌ Exception: " + e.message);
    console.error(e);
  }
}
