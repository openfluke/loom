using System;
using System.Text.Json;
using System.Runtime.InteropServices;

namespace Welvet.Examples;

/// <summary>
/// Grid Scatter Multi-Agent Training Demo
/// Demonstrates the new simple CABI API with save/load verification
/// </summary>
public class GridScatterDemo
{
    private const string LibName = "libloom";

    // Import the new simple API functions
    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    private static extern IntPtr CreateLoomNetwork([MarshalAs(UnmanagedType.LPStr)] string jsonConfig);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr LoomForward([MarshalAs(UnmanagedType.LPArray)] float[] inputs, int length);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    private static extern IntPtr LoomTrain(
        [MarshalAs(UnmanagedType.LPStr)] string batchesJSON,
        [MarshalAs(UnmanagedType.LPStr)] string configJSON);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    private static extern IntPtr LoomSaveModel([MarshalAs(UnmanagedType.LPStr)] string modelID);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    private static extern IntPtr LoomLoadModel(
        [MarshalAs(UnmanagedType.LPStr)] string jsonString,
        [MarshalAs(UnmanagedType.LPStr)] string modelID);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    private static extern IntPtr LoomEvaluateNetwork(
        [MarshalAs(UnmanagedType.LPStr)] string inputsJSON,
        [MarshalAs(UnmanagedType.LPStr)] string expectedOutputsJSON);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    private static extern void FreeLoomString(IntPtr str);

    private static string PtrToStringAndFree(IntPtr ptr)
    {
        if (ptr == IntPtr.Zero)
            return string.Empty;
        try
        {
            return Marshal.PtrToStringAnsi(ptr) ?? string.Empty;
        }
        finally
        {
            FreeLoomString(ptr);
        }
    }

    public static void Run()
    {
        Console.WriteLine("🤖 LOOM C# - Grid Scatter Multi-Agent Training");
        Console.WriteLine("Task: 3 agents learn to collaborate for binary classification\n");

        // Network configuration
        string config = @"{
            ""batch_size"": 1,
            ""grid_rows"": 1,
            ""grid_cols"": 3,
            ""layers_per_cell"": 1,
            ""layers"": [
                {
                    ""type"": ""dense"",
                    ""input_size"": 8,
                    ""output_size"": 16,
                    ""activation"": ""relu""
                },
                {
                    ""type"": ""parallel"",
                    ""combine_mode"": ""grid_scatter"",
                    ""grid_output_rows"": 3,
                    ""grid_output_cols"": 1,
                    ""grid_output_layers"": 1,
                    ""grid_positions"": [
                        {""branch_index"": 0, ""target_row"": 0, ""target_col"": 0, ""target_layer"": 0},
                        {""branch_index"": 1, ""target_row"": 1, ""target_col"": 0, ""target_layer"": 0},
                        {""branch_index"": 2, ""target_row"": 2, ""target_col"": 0, ""target_layer"": 0}
                    ],
                    ""branches"": [
                        {
                            ""type"": ""parallel"",
                            ""combine_mode"": ""add"",
                            ""branches"": [
                                {""type"": ""dense"", ""input_size"": 16, ""output_size"": 8, ""activation"": ""relu""},
                                {""type"": ""dense"", ""input_size"": 16, ""output_size"": 8, ""activation"": ""gelu""}
                            ]
                        },
                        {""type"": ""lstm"", ""input_size"": 16, ""hidden_size"": 8, ""seq_length"": 1},
                        {""type"": ""rnn"", ""input_size"": 16, ""hidden_size"": 8, ""seq_length"": 1}
                    ]
                },
                {
                    ""type"": ""dense"",
                    ""input_size"": 24,
                    ""output_size"": 2,
                    ""activation"": ""sigmoid""
                }
            ]
        }";

        Console.WriteLine("Architecture:");
        Console.WriteLine("  Shared Layer → Grid Scatter (3 agents) → Decision");
        Console.WriteLine("  Agent 0: Feature Extractor (ensemble of 2 dense)");
        Console.WriteLine("  Agent 1: Transformer (LSTM)");
        Console.WriteLine("  Agent 2: Integrator (RNN)");
        Console.WriteLine("Task: Binary classification (sum comparison)\n");

        Console.WriteLine("Building network from JSON...");
        var resultPtr = CreateLoomNetwork(config);
        var result = PtrToStringAndFree(resultPtr);
        Console.WriteLine("✅ Agent network created!\n");

        // Training data
        var batches = new[]
        {
            new { Input = new[] { 0.2f, 0.2f, 0.2f, 0.2f, 0.8f, 0.8f, 0.8f, 0.8f }, Target = new[] { 1.0f, 0.0f } },
            new { Input = new[] { 0.9f, 0.9f, 0.9f, 0.9f, 0.1f, 0.1f, 0.1f, 0.1f }, Target = new[] { 0.0f, 1.0f } },
            new { Input = new[] { 0.7f, 0.7f, 0.7f, 0.7f, 0.3f, 0.3f, 0.3f, 0.3f }, Target = new[] { 0.0f, 1.0f } },
            new { Input = new[] { 0.3f, 0.3f, 0.3f, 0.3f, 0.7f, 0.7f, 0.7f, 0.7f }, Target = new[] { 1.0f, 0.0f } }
        };

        string batchesJson = JsonSerializer.Serialize(batches);
        string trainingConfig = @"{
            ""Epochs"": 800,
            ""LearningRate"": 0.15,
            ""UseGPU"": false,
            ""PrintEveryBatch"": 0,
            ""GradientClip"": 1.0,
            ""LossType"": ""mse"",
            ""Verbose"": false
        }";

        Console.WriteLine("Training for 800 epochs with learning rate 0.150\n");
        var trainStart = DateTime.Now;
        var trainResultPtr = LoomTrain(batchesJson, trainingConfig);
        var trainResult = PtrToStringAndFree(trainResultPtr);
        var trainDuration = (DateTime.Now - trainStart).TotalSeconds;

        // Parse training result
        using (var doc = JsonDocument.Parse(trainResult))
        {
            var root = doc.RootElement;
            if (root.ValueKind == JsonValueKind.Array && root.GetArrayLength() > 0)
            {
                var data = root[0];
                var lossHistory = data.GetProperty("LossHistory");
                float initialLoss = lossHistory[0].GetSingle();
                float finalLoss = data.GetProperty("FinalLoss").GetSingle();
                float improvement = ((initialLoss - finalLoss) / initialLoss) * 100;

                Console.WriteLine("✅ Training complete!");
                Console.WriteLine($"Training time: {trainDuration:F2} seconds");
                Console.WriteLine($"Initial Loss: {initialLoss:F6}");
                Console.WriteLine($"Final Loss: {finalLoss:F6}");
                Console.WriteLine($"Improvement: {improvement:F2}%");
                Console.WriteLine($"Total Epochs: {lossHistory.GetArrayLength()}\n");
            }
        }

        // Test predictions
        Console.WriteLine("📊 After Training:");
        var originalPredictions = new float[4][];

        for (int i = 0; i < 4; i++)
        {
            var predPtr = LoomForward(batches[i].Input, batches[i].Input.Length);
            var predJson = PtrToStringAndFree(predPtr);

            using var doc = JsonDocument.Parse(predJson);
            var pred = doc.RootElement;
            float pred0 = pred[0].GetSingle();
            float pred1 = pred[1].GetSingle();
            originalPredictions[i] = new[] { pred0, pred1 };

            int predClass = pred1 > pred0 ? 1 : 0;
            int expectedClass = batches[i].Target[1] > batches[i].Target[0] ? 1 : 0;
            string check = predClass == expectedClass ? "✓" : "✗";

            Console.WriteLine($"Sample {i}: [{pred0:F3}, {pred1:F3}] → Class {predClass} (expected {expectedClass}) {check}");
        }

        // Evaluation
        Console.WriteLine("\n📊 Evaluating with EvaluateNetwork...");
        var inputsJson = JsonSerializer.Serialize(batches.Select(b => b.Input).ToArray());
        var expectedJson = "[0.0, 1.0, 1.0, 0.0]";

        var evalPtr = LoomEvaluateNetwork(inputsJson, expectedJson);
        var evalResult = PtrToStringAndFree(evalPtr);

        if (evalResult.Contains("error") || evalResult.Contains("Error"))
        {
            Console.WriteLine($"⚠️ Evaluation error: {evalResult}");
        }
        else
        {
            using (var doc = JsonDocument.Parse(evalResult))
            {
                var root = doc.RootElement;

                // Check if it's an array (old format) or direct object (new format)
                JsonElement metrics;
                if (root.ValueKind == JsonValueKind.Array && root.GetArrayLength() > 0)
                {
                    metrics = root[0];
                }
                else
                {
                    metrics = root;
                }

                Console.WriteLine("\n=== Evaluation Metrics ===");
                Console.WriteLine($"Total Samples: {metrics.GetProperty("total_samples").GetInt32()}");
                Console.WriteLine($"Quality Score: {metrics.GetProperty("score").GetSingle():F2}/100");
                Console.WriteLine($"Average Deviation: {metrics.GetProperty("avg_deviation").GetSingle():F2}%");
                Console.WriteLine($"Failures (>100% deviation): {metrics.GetProperty("failures").GetInt32()}");

                Console.WriteLine("\nDeviation Distribution:");
                var buckets = metrics.GetProperty("buckets");
                string[] bucketNames = { "0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-100%", "100%+" };
                int totalSamples = metrics.GetProperty("total_samples").GetInt32();

                foreach (var bucketName in bucketNames)
                {
                    if (buckets.TryGetProperty(bucketName, out var bucket))
                    {
                        int count = bucket.GetProperty("count").GetInt32();
                        if (count > 0)
                        {
                            float percentage = (count * 100.0f) / totalSamples;
                            int barLength = (int)((count * 20.0f) / totalSamples);
                            string bar = new string('█', barLength);
                            Console.WriteLine($"  {bucketName,-8}: {count} samples ({percentage:F1}%) {bar}");
                        }
                    }
                }
            }
        }

        Console.WriteLine("\n✅ Multi-agent training complete!");

        // Save and reload model to verify serialization
        Console.WriteLine("\n💾 Testing model save/load...");

        var savedPtr = LoomSaveModel("grid_scatter_test");
        var savedModel = PtrToStringAndFree(savedPtr);
        Console.WriteLine($"✓ Model saved ({savedModel.Length} bytes)");

        Console.WriteLine("Loading model from saved state...");
        var loadPtr = LoomLoadModel(savedModel, "grid_scatter_test");
        var loadResult = PtrToStringAndFree(loadPtr);

        if (loadResult.Contains("error"))
        {
            Console.WriteLine($"❌ Failed to load model: {loadResult}");
            return;
        }

        Console.WriteLine("✓ Model loaded\n");

        // Verify predictions match
        Console.WriteLine("Verifying predictions match:");
        bool allMatch = true;

        for (int i = 0; i < 4; i++)
        {
            var predPtr = LoomForward(batches[i].Input, batches[i].Input.Length);
            var predJson = PtrToStringAndFree(predPtr);

            using var doc = JsonDocument.Parse(predJson);
            var pred = doc.RootElement;
            float pred0 = pred[0].GetSingle();
            float pred1 = pred[1].GetSingle();

            float diff0 = Math.Abs(pred0 - originalPredictions[i][0]);
            float diff1 = Math.Abs(pred1 - originalPredictions[i][1]);
            float maxDiff = Math.Max(diff0, diff1);

            bool match = maxDiff < 1e-6f;
            allMatch &= match;

            string check = match ? "✓" : "✗";
            Console.WriteLine($"Sample {i}: [{pred0:F3}, {pred1:F3}] (diff: {maxDiff:E2}) {check}");
        }

        if (allMatch)
        {
            Console.WriteLine("\n✅ Save/Load verification passed! All predictions match.");
        }
        else
        {
            Console.WriteLine("\n❌ Save/Load verification failed! Predictions don't match.");
        }
    }
}
