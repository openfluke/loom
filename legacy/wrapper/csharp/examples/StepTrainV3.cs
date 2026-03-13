using System;
using System.Collections.Generic;
using System.Linq;
using Welvet;

namespace StepTrainV3
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("=== LOOM Stepping Neural Network v3: LSTM Middle Layer ===");
            Console.WriteLine("3-Layer Network: Dense -> LSTM -> Dense");
            Console.WriteLine();

            // 1. Define Network Architecture
            var config = new
            {
                batch_size = 1,
                grid_rows = 1,
                grid_cols = 3,
                layers_per_cell = 1,
                layers = new object[]
                {
                    new { type = "dense", input_height = 4, output_height = 8, activation = "relu" },
                    new { type = "lstm", input_size = 8, hidden_size = 12, seq_length = 1, activation = "tanh" },
                    new { type = "dense", input_height = 12, output_height = 3, activation = "softmax" }
                }
            };

            string jsonConfig = System.Text.Json.JsonSerializer.Serialize(config);
            Network.CreateFromJson(jsonConfig);

            // Initialize stepping state
            int inputSize = 4;
            using var state = new StepState(inputSize);

            // 2. Create Training Data (3 Classes)
            var trainingData = new[]
            {
                new { Input = new float[] { 0.1f, 0.2f, 0.1f, 0.3f }, Target = new float[] { 1.0f, 0.0f, 0.0f }, Label = "Low" },
                new { Input = new float[] { 0.2f, 0.1f, 0.3f, 0.2f }, Target = new float[] { 1.0f, 0.0f, 0.0f }, Label = "Low" },
                new { Input = new float[] { 0.8f, 0.9f, 0.8f, 0.7f }, Target = new float[] { 0.0f, 1.0f, 0.0f }, Label = "High" },
                new { Input = new float[] { 0.9f, 0.8f, 0.7f, 0.9f }, Target = new float[] { 0.0f, 1.0f, 0.0f }, Label = "High" },
                new { Input = new float[] { 0.1f, 0.9f, 0.1f, 0.9f }, Target = new float[] { 0.0f, 0.0f, 1.0f }, Label = "Mix" },
                new { Input = new float[] { 0.9f, 0.1f, 0.9f, 0.1f }, Target = new float[] { 0.0f, 0.0f, 1.0f }, Label = "Mix" }
            };

            // 3. Setup Continuous Training Loop
            int totalSteps = 100000;
            int targetDelay = 3;
            var targetQueue = new Queue<float[]>();

            float learningRate = 0.015f;
            float minLearningRate = 0.001f;
            float decayRate = 0.99995f;
            float gradientClipValue = 1.0f;

            Console.WriteLine($"Training for {totalSteps} steps (Max Speed)");
            Console.WriteLine($"Target Delay: {targetDelay} steps (accounts for LSTM internal state)");
            Console.WriteLine($"LR Decay: {decayRate:F4} per step (min {minLearningRate:F4})");
            Console.WriteLine($"Gradient Clipping: {gradientClipValue:F2}");
            Console.WriteLine();

            var startTime = DateTime.Now;
            int stepCount = 0;
            int currentSampleIdx = 0;
            var random = new Random();

            Console.WriteLine($"{"Step",-6} {"Input",-10} {"Output (ArgMax)",-25} {"Loss",-10}");
            Console.WriteLine("──────────────────────────────────────────────────────────");

            while (stepCount < totalSteps)
            {
                if (stepCount % 20 == 0)
                {
                    currentSampleIdx = random.Next(trainingData.Length);
                }

                var sample = trainingData[currentSampleIdx];

                // B. Set Input
                state.SetInput(sample.Input);

                // C. Step Forward
                state.StepForward();

                // D. Manage Target Queue
                targetQueue.Enqueue(sample.Target);

                if (targetQueue.Count >= targetDelay)
                {
                    var delayedTarget = targetQueue.Dequeue();
                    var output = state.GetOutput();

                    // F. Calculate Loss & Gradient
                    float loss = 0.0f;
                    var gradOutput = new float[output.Length];

                    for (int i = 0; i < output.Length; i++)
                    {
                        float p = output[i];
                        if (p < 1e-7f) p = 1e-7f;
                        if (p > 1.0f - 1e-7f) p = 1.0f - 1e-7f;

                        if (delayedTarget[i] > 0.5f)
                        {
                            loss -= (float)Math.Log(p);
                        }

                        gradOutput[i] = output[i] - delayedTarget[i];
                    }

                    // Apply gradient clipping
                    float gradNorm = 0.0f;
                    foreach (var g in gradOutput) gradNorm += g * g;
                    gradNorm = (float)Math.Sqrt(gradNorm);

                    if (gradNorm > gradientClipValue)
                    {
                        float scale = gradientClipValue / gradNorm;
                        for (int i = 0; i < gradOutput.Length; i++)
                        {
                            gradOutput[i] *= scale;
                        }
                    }

                    // G. Backward Pass
                    state.StepBackward(gradOutput);

                    // H. Update Weights
                    Network.ApplyGradients(learningRate);

                    // Decay Learning Rate
                    learningRate *= decayRate;
                    if (learningRate < minLearningRate) learningRate = minLearningRate;

                    // I. Logging
                    if (stepCount % 500 == 0)
                    {
                        int maxIdx = 0;
                        float maxVal = output[0];
                        for (int i = 1; i < output.Length; i++)
                        {
                            if (output[i] > maxVal)
                            {
                                maxVal = output[i];
                                maxIdx = i;
                            }
                        }

                        int tMaxIdx = 0;
                        for (int i = 1; i < delayedTarget.Length; i++)
                        {
                            if (delayedTarget[i] > 0.5f) tMaxIdx = i;
                        }

                        string mark = maxIdx == tMaxIdx ? "✓" : "✗";

                        Console.WriteLine($"{stepCount,-6} {sample.Label,-10} Class {maxIdx} ({maxVal:F2}) [{mark}] Exp: {tMaxIdx}  Loss: {loss:F4}  LR: {learningRate:F5}");
                    }
                }

                stepCount++;
            }

            var totalTime = DateTime.Now - startTime;
            Console.WriteLine();
            Console.WriteLine("=== Training Complete ===");
            Console.WriteLine($"Total Time: {totalTime.TotalSeconds:F3}s");
            Console.WriteLine($"Speed: {totalSteps / totalTime.TotalSeconds:F2} steps/sec");
            Console.WriteLine();

            // Final Evaluation
            Console.WriteLine("Evaluating on all samples (with settling time)...");
            int correct = 0;
            int settlingSteps = 10;

            foreach (var sample in trainingData)
            {
                state.SetInput(sample.Input);
                for (int i = 0; i < settlingSteps; i++)
                {
                    state.StepForward();
                }

                var output = state.GetOutput();
                int maxIdx = 0;
                float maxVal = output[0];
                for (int i = 1; i < output.Length; i++)
                {
                    if (output[i] > maxVal)
                    {
                        maxVal = output[i];
                        maxIdx = i;
                    }
                }

                int tMaxIdx = 0;
                for (int i = 1; i < sample.Target.Length; i++)
                {
                    if (sample.Target[i] > 0.5f) tMaxIdx = i;
                }

                string mark = "✗";
                if (maxIdx == tMaxIdx)
                {
                    correct++;
                    mark = "✓";
                }

                Console.WriteLine($"{mark} {sample.Label}: Pred {maxIdx} ({maxVal:F2}) Exp {tMaxIdx}");
            }

            Console.WriteLine($"Final Accuracy: {correct}/{trainingData.Length}");
        }
    }
}
