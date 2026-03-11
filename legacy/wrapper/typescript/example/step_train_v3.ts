/**
 * LOOM Stepping Example - TypeScript
 * Replicates step_train_v3.go logic
 */

import { init, createNetwork, StepState } from "../src/index.js";

interface TrainingSample {
  input: number[];
  target: number[];
  label: string;
}

async function main() {
  console.log("=== LOOM Stepping Neural Network (TypeScript) ===");
  console.log("3-Layer Network: Dense -> LSTM -> Dense\n");

  // Initialize WASM
  await init();

  // Network configuration
  const config = {
    batch_size: 1,
    grid_rows: 1,
    grid_cols: 3,
    layers_per_cell: 1,
    layers: [
      {
        type: "dense",
        input_height: 4,
        output_height: 8,
        activation: "relu"
      },
      {
        type: "lstm",
        input_size: 8,
        hidden_size: 12,
        seq_length: 1,
        activation: "tanh"
      },
      {
        type: "dense",
        input_height: 12,
        output_height: 3,
        activation: "softmax"
      }
    ]
  };

  // Create network
  const network = createNetwork(config);
  
  // Initialize weights
  network.InitializeWeights(JSON.stringify([]));

  // Create stepping state
  const state: StepState = network.createStepState(4);

  // Training data
  const trainingData: TrainingSample[] = [
    { input: [0.1, 0.2, 0.1, 0.3], target: [1, 0, 0], label: "Low" },
    { input: [0.2, 0.1, 0.3, 0.2], target: [1, 0, 0], label: "Low" },
    { input: [0.8, 0.9, 0.8, 0.7], target: [0, 1, 0], label: "High" },
    { input: [0.9, 0.8, 0.7, 0.9], target: [0, 1, 0], label: "High" },
    { input: [0.1, 0.9, 0.1, 0.9], target: [0, 0, 1], label: "Mix" },
    { input: [0.9, 0.1, 0.9, 0.1], target: [0, 0, 1], label: "Mix" }
  ];

  // Training parameters
  const totalSteps = 100000;
  const targetDelay = 3;
  const targetQueue: number[][] = [];
  let learningRate = 0.015;
  const minLearningRate = 0.001;
  const decayRate = 0.99995;
  const gradientClipValue = 1.0;

  let currentSampleIdx = 0;
  const startTime = Date.now();

  console.log(`Training for ${totalSteps} steps`);
  console.log(`Target Delay: ${targetDelay} steps`);
  console.log(`LR Decay: ${decayRate} per step (min ${minLearningRate})`);
  console.log(`Gradient Clipping: ${gradientClipValue}\n`);

  console.log("Step   Input      Output (ArgMax)              Loss      LR");
  console.log("──────────────────────────────────────────────────────────────");

  for (let stepCount = 0; stepCount < totalSteps; stepCount++) {
    // Rotate sample every 20 steps
    if (stepCount % 20 === 0) {
      currentSampleIdx = Math.floor(Math.random() * trainingData.length);
    }
    const sample = trainingData[currentSampleIdx];

    // Set input
    state.setInput(new Float32Array(sample.input));

    // Step forward
    state.stepForward();

    // Manage target queue
    targetQueue.push(sample.target);

    if (targetQueue.length >= targetDelay) {
      const delayedTarget = targetQueue.shift()!;
      const output = state.getOutput();

      // Calculate loss & gradient
      let loss = 0.0;
      const gradOutput = new Float32Array(3);

      for (let i = 0; i < 3; i++) {
        let p = output[i];
        if (p < 1e-7) p = 1e-7;
        if (p > 1.0 - 1e-7) p = 1.0 - 1e-7;

        if (delayedTarget[i] > 0.5) {
          loss -= Math.log(p);
        }

        gradOutput[i] = output[i] - delayedTarget[i];
      }

      // Gradient clipping
      let gradNorm = 0.0;
      for (let i = 0; i < 3; i++) {
        gradNorm += gradOutput[i] * gradOutput[i];
      }
      gradNorm = Math.sqrt(gradNorm);

      if (gradNorm > gradientClipValue) {
        const scale = gradientClipValue / gradNorm;
        for (let i = 0; i < 3; i++) {
          gradOutput[i] *= scale;
        }
      }

      // Backward pass
      state.stepBackward(gradOutput);

      // Update weights
      network.ApplyGradients(JSON.stringify([learningRate]));

      // Decay learning rate
      learningRate *= decayRate;
      if (learningRate < minLearningRate) {
        learningRate = minLearningRate;
      }

      // Logging
      if (stepCount % 500 === 0) {
        let maxIdx = 0;
        let maxVal = output[0];
        for (let i = 1; i < 3; i++) {
          if (output[i] > maxVal) {
            maxVal = output[i];
            maxIdx = i;
          }
        }

        let tMaxIdx = 0;
        for (let i = 1; i < 3; i++) {
          if (delayedTarget[i] > 0.5) {
            tMaxIdx = i;
          }
        }

        const mark = maxIdx === tMaxIdx ? "✓" : "✗";
        const stepStr = stepCount.toString().padEnd(6);
        const inputStr = sample.label.padEnd(10);
        const outStr = `Class ${maxIdx} (${maxVal.toFixed(2)}) [${mark}] Exp: ${tMaxIdx}`.padEnd(28);
        const lossStr = `Loss: ${loss.toFixed(4)}`;
        const lrStr = `LR: ${learningRate.toFixed(5)}`;

        console.log(`${stepStr} ${inputStr} ${outStr} ${lossStr}  ${lrStr}`);
      }
    }
  }

  const totalTime = Date.now() - startTime;
  console.log("\n=== Training Complete ===");
  console.log(`Total Time: ${(totalTime / 1000).toFixed(2)}s`);
  console.log(`Speed: ${(totalSteps / (totalTime / 1000)).toFixed(2)} steps/sec\n`);

  // Final evaluation
  console.log("Evaluating on all samples (with settling time)...");
  let correct = 0;
  const settlingSteps = 10;

  for (const sample of trainingData) {
    state.setInput(new Float32Array(sample.input));
    // Settle
    for (let i = 0; i < settlingSteps; i++) {
      state.stepForward();
    }
    const output = state.getOutput();

    let maxIdx = 0;
    let maxVal = output[0];
    for (let i = 1; i < 3; i++) {
      if (output[i] > maxVal) {
        maxVal = output[i];
        maxIdx = i;
      }
    }

    let tMaxIdx = 0;
    for (let i = 1; i < 3; i++) {
      if (sample.target[i] > 0.5) {
        tMaxIdx = i;
      }
    }

    const mark = maxIdx === tMaxIdx ? "✓" : "✗";
    if (maxIdx === tMaxIdx) correct++;

    console.log(`${mark} ${sample.label}: Pred ${maxIdx} (${maxVal.toFixed(2)}) Exp ${tMaxIdx}`);
  }

  console.log(`Final Accuracy: ${correct}/${trainingData.length}`);
}

main().catch(console.error);
