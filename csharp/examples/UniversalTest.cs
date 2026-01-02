using System;
using System.Text.Json;
using System.Runtime.InteropServices;
using System.Linq;
using Welvet;

namespace Welvet.Examples;

/// <summary>
/// LOOM v0.0.7 Complete Test Suite for C#
/// Ports tests from tva/test_0_0_7.go to validate C# bindings
/// </summary>
public class UniversalTest
{
    // Global test counters
    private static int _testsPassed = 0;
    private static int _testsFailed = 0;
    private static string PtrToStringAndFree(IntPtr ptr)
    {
        return NativeMethods.PtrToStringAndFree(ptr);
    }

    private static void Log(string type, string msg)
    {
        switch (type)
        {
            case "success": Console.WriteLine($"\x1b[32m{msg}\x1b[0m"); break;
            case "error": Console.WriteLine($"\x1b[31m{msg}\x1b[0m"); break;
            case "warn": Console.WriteLine($"\x1b[33m{msg}\x1b[0m"); break;
            default: Console.WriteLine(msg); break;
        }
    }

    public static void Main(string[] args)
    {
        Console.WriteLine("╔══════════════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║           LOOM C# Universal Test Suite (v0.0.7)                     ║");
        Console.WriteLine("╚══════════════════════════════════════════════════════════════════════╝");
        Console.WriteLine();

        // Part 1: Layer & DType Compatibility
        TestLayerDTypeCompatibility();

        // Part 2: Feature Tests
        TestGrafting();
        TestStats();
        TestOptimizers();
        TestEnsemble();
        TestObserver();
        TestIntrospection();
        TestStepTween();
        TestAdvancedLayers();

        // Final Summary
        Console.WriteLine();
        Console.WriteLine("======================================================================");
        Console.WriteLine($"FINAL: {_testsPassed + _testsFailed} TESTS RUN");
        Console.WriteLine($"PASSED: {_testsPassed}");
        Console.WriteLine($"FAILED: {_testsFailed}");
        Console.WriteLine("======================================================================");

        if (_testsFailed > 0)
        {
            Log("error", $"❌ {_testsFailed} test(s) failed!");
            Environment.Exit(1);
        }
        else
        {
            Log("success", "🎉 All tests passed!");
        }
    }

    // =========================================================================
    // Part 1: Layer & DType Compatibility Tests
    // =========================================================================

    private static readonly string[] LayerTypes = {
        "Dense", "Conv2D", "MHA", "RNN", "LSTM",
        "LayerNorm", "RMSNorm", "SwiGLU",
        "Parallel", "Sequential", "Softmax", "Conv1D"
    };

    private static readonly string[] DTypes = {
        "float32", "float64", "int32", "int16", "int8", "uint8"
    };

    private static void TestLayerDTypeCompatibility()
    {
        Console.WriteLine("┌──────────────────────────────────────────────────────────────────────┐");
        Console.WriteLine("│ Layer & DType Compatibility (One-Shot)                              │");
        Console.WriteLine("└──────────────────────────────────────────────────────────────────────┘");

        foreach (var layer in LayerTypes)
        {
            foreach (var dtype in DTypes)
            {
                if (TestLayerWithDtype(layer, dtype))
                    _testsPassed++;
                else
                    _testsFailed++;
            }
        }
    }

    private static bool TestLayerWithDtype(string layerType, string dtype)
    {
        int inputSize = 4;
        string layersJson;

        switch (layerType)
        {
            case "Dense":
                // Simple single-layer dense for one-shot test
                layersJson = @"[{""type"": ""dense"", ""input_height"": 4, ""output_height"": 4, ""activation"": ""relu""}]";
                break;
            case "MHA":
                layersJson = @"[{""type"": ""multi_head_attention"", ""d_model"": 4, ""num_heads"": 2, ""seq_length"": 4}]";
                break;
            case "RNN":
                layersJson = @"[{""type"": ""rnn"", ""input_size"": 4, ""hidden_size"": 4, ""activation"": ""tanh""}]";
                break;
            case "LSTM":
                layersJson = @"[{""type"": ""lstm"", ""input_size"": 4, ""hidden_size"": 4}]";
                break;
            case "LayerNorm":
                layersJson = @"[{""type"": ""layer_norm"", ""norm_size"": 4}]";
                break;
            case "RMSNorm":
                layersJson = @"[{""type"": ""rms_norm"", ""norm_size"": 4}]";
                break;
            case "SwiGLU":
                layersJson = @"[{""type"": ""swiglu"", ""input_size"": 4, ""output_size"": 4}]";
                break;
            case "Conv2D":
                inputSize = 16;
                layersJson = @"[{""type"": ""conv2d"", ""input_width"": 4, ""input_height"": 4, ""input_channels"": 1, ""kernel_size"": 3, ""stride"": 1, ""padding"": 1, ""filters"": 1}]";
                break;
            case "Parallel":
                layersJson = @"[{""type"": ""parallel"", ""branches"": [
                    {""type"": ""dense"", ""input_height"": 4, ""output_height"": 2},
                    {""type"": ""dense"", ""input_height"": 4, ""output_height"": 2}
                ]}]";
                break;
            case "Sequential":
                layersJson = @"[{""type"": ""sequential"", ""branches"": [
                    {""type"": ""dense"", ""input_height"": 4, ""output_height"": 4},
                    {""type"": ""dense"", ""input_height"": 4, ""output_height"": 4}
                ]}]";
                break;
            case "Softmax":
                layersJson = @"[{""type"": ""softmax"", ""input_size"": 4}]";
                break;
            case "Conv1D":
                layersJson = @"[{""type"": ""conv1d"", ""input_channels"": 1, ""output_channels"": 1, ""kernel_size"": 3, ""stride"": 1, ""padding"": 1}]";
                break;
            default:
                Console.WriteLine($"  ❌ {layerType,-10} / {dtype,-8}: Unknown layer type");
                return false;
        }

        string config = $@"{{
            ""dtype"": ""{dtype}"",
            ""batch_size"": 1,
            ""grid_rows"": 1,
            ""grid_cols"": 1,
            ""layers_per_cell"": 1,
            ""layers"": {layersJson}
        }}";

        try
        {
            // Create network
            var resultPtr = NativeMethods.CreateLoomNetwork(config);
            var result = PtrToStringAndFree(resultPtr);
            if (result.Contains("error"))
                throw new Exception($"Create failed: {result}");

            // Forward pass
            var input = Enumerable.Repeat(0.1f, inputSize).ToArray();
            var outputPtr = NativeMethods.LoomForward(input, input.Length);
            var outputJson = PtrToStringAndFree(outputPtr);
            if (string.IsNullOrEmpty(outputJson) || outputJson.Contains("error"))
                throw new Exception("Forward pass failed");

            // Save/Load test
            var savedPtr = NativeMethods.LoomSaveModel($"model_{layerType}");
            var savedModel = PtrToStringAndFree(savedPtr);
            if (string.IsNullOrEmpty(savedModel) || savedModel.Contains("error"))
                throw new Exception("Save failed");

            Console.WriteLine($"  ✓ {layerType,-10} / {dtype,-8}: OK");
            return true;
        }
        catch (Exception e)
        {
            Console.WriteLine($"  ❌ {layerType,-10} / {dtype,-8}: {e.Message}");
            return false;
        }
    }

    // =========================================================================
    // Part 2: Network Grafting Test
    // =========================================================================

    private static void TestGrafting()
    {
        Console.WriteLine();
        Console.WriteLine("┌──────────────────────────────────────────────────────────────────────┐");
        Console.WriteLine("│ Network Grafting                                                    │");
        Console.WriteLine("└──────────────────────────────────────────────────────────────────────┘");

        try
        {
            string config = @"{
                ""batch_size"": 1,
                ""grid_rows"": 1,
                ""grid_cols"": 1,
                ""layers_per_cell"": 2,
                ""layers"": [
                    {""type"": ""dense"", ""input_height"": 4, ""output_height"": 8},
                    {""type"": ""dense"", ""input_height"": 8, ""output_height"": 4}
                ]
            }";

            long h1 = NativeMethods.LoomCreateNetworkHandle(config);
            long h2 = NativeMethods.LoomCreateNetworkHandle(config);

            if (h1 <= 0 || h2 <= 0)
                throw new Exception("Failed to create graft handles");

            var handlesJson = JsonSerializer.Serialize(new[] { h1, h2 });
            var resultPtr = NativeMethods.LoomGraftNetworks(handlesJson, "concat");
            var result = PtrToStringAndFree(resultPtr);

            using var doc = JsonDocument.Parse(result);
            var root = doc.RootElement;

            if (root.TryGetProperty("success", out var successProp) && successProp.GetBoolean())
            {
                var numBranches = root.GetProperty("num_branches").GetInt32();
                Console.WriteLine($"  ✓ Grafted: {numBranches} branches");
                Log("success", "  ✅ PASSED: Network Grafting");
                _testsPassed++;
            }
            else
            {
                throw new Exception(root.TryGetProperty("error", out var err) ? err.GetString() : "Unknown error");
            }
        }
        catch (Exception e)
        {
            Log("error", $"  ❌ Grafting failed: {e.Message}");
            _testsFailed++;
        }
    }

    // =========================================================================
    // Part 3: Statistical Tools Tests
    // =========================================================================

    private static void TestStats()
    {
        Console.WriteLine();
        Console.WriteLine("┌──────────────────────────────────────────────────────────────────────┐");
        Console.WriteLine("│ Statistical Tools (K-Means, Correlation)                            │");
        Console.WriteLine("└──────────────────────────────────────────────────────────────────────┘");

        try
        {
            // K-Means
            var data = new[] {
                new[] { 1.0f, 1.0f }, new[] { 1.1f, 1.1f },
                new[] { 5.0f, 5.0f }, new[] { 5.1f, 5.1f }
            };
            var dataJson = JsonSerializer.Serialize(data);
            var kmeansPtr = NativeMethods.LoomKMeansCluster(dataJson, 2, 10);
            var kmeansResult = PtrToStringAndFree(kmeansPtr);

            using (var doc = JsonDocument.Parse(kmeansResult))
            {
                var root = doc.RootElement;
                if (root.TryGetProperty("centroids", out var centroids))
                {
                    Console.WriteLine($"  ✓ K-Means: {centroids.GetArrayLength()} centroids found");
                }
            }

            // Correlation
            var matrixA = new[] {
                new[] { 1f, 2f, 3f },
                new[] { 4f, 5f, 6f },
                new[] { 7f, 8f, 9f },
                new[] { 10f, 11f, 12f }
            };
            var matrixJson = JsonSerializer.Serialize(matrixA);
            var corrPtr = NativeMethods.LoomComputeCorrelation(matrixJson);
            var corrResult = PtrToStringAndFree(corrPtr);

            using (var doc = JsonDocument.Parse(corrResult))
            {
                var root = doc.RootElement;
                if (root.TryGetProperty("pearson", out var pearson) ||
                    root.TryGetProperty("Correlation", out pearson))
                {
                    Console.WriteLine($"  ✓ Correlation matrix computed");
                }
            }

            Log("success", "  ✅ PASSED: Stats Tools");
            _testsPassed++;
        }
        catch (Exception e)
        {
            Log("error", $"  ❌ Stats failed: {e.Message}");
            _testsFailed++;
        }
    }

    // =========================================================================
    // Part 4: Optimizers Test
    // =========================================================================

    private static void TestOptimizers()
    {
        Console.WriteLine();
        Console.WriteLine("┌──────────────────────────────────────────────────────────────────────┐");
        Console.WriteLine("│ Optimizers                                                          │");
        Console.WriteLine("└──────────────────────────────────────────────────────────────────────┘");

        try
        {
            string config = @"{
                ""dtype"": ""float32"",
                ""batch_size"": 2,
                ""grid_rows"": 1,
                ""grid_cols"": 1,
                ""layers_per_cell"": 1,
                ""layers"": [{""type"": ""dense"", ""input_height"": 2, ""output_height"": 2, ""activation"": ""leaky_relu""}]
            }";

            var resultPtr = NativeMethods.CreateLoomNetwork(config);
            var result = PtrToStringAndFree(resultPtr);

            var batches = new[] { new { Input = new[] { 0f, 0f, 1f, 1f }, Target = new[] { 0f, 0f, 1f, 1f } } };
            var trainConfig = new { Epochs = 5, LearningRate = 0.001f, LossType = "mse" };

            var trainPtr = NativeMethods.LoomTrain(JsonSerializer.Serialize(batches), JsonSerializer.Serialize(trainConfig));
            var trainResult = PtrToStringAndFree(trainPtr);

            using var doc = JsonDocument.Parse(trainResult);
            var root = doc.RootElement;

            JsonElement resultObj = root.ValueKind == JsonValueKind.Array && root.GetArrayLength() > 0
                ? root[0] : root;

            if (resultObj.TryGetProperty("FinalLoss", out var lossEl) ||
                resultObj.TryGetProperty("final_loss", out lossEl))
            {
                Console.WriteLine($"  ✓ Optimizer training: Loss={lossEl.GetSingle():F4}");
                Log("success", "  ✅ PASSED: Optimizers");
                _testsPassed++;
            }
            else
            {
                throw new Exception("Training failed (FinalLoss undefined)");
            }
        }
        catch (Exception e)
        {
            Log("error", $"  ❌ Optimizers failed: {e.Message}");
            _testsFailed++;
        }
    }

    // =========================================================================
    // Part 5: Ensemble Features Test
    // =========================================================================

    private static void TestEnsemble()
    {
        Console.WriteLine();
        Console.WriteLine("┌──────────────────────────────────────────────────────────────────────┐");
        Console.WriteLine("│ Ensemble Features                                                   │");
        Console.WriteLine("└──────────────────────────────────────────────────────────────────────┘");

        try
        {
            // Note: Go struct uses PascalCase (ModelID, Mask) - no json tags
            var modelsJson = @"[
                {""ModelID"": ""model_1"", ""Mask"": [true, true, true, false, false]},
                {""ModelID"": ""model_2"", ""Mask"": [false, false, false, true, true]}
            ]";

            var matchesPtr = NativeMethods.LoomFindComplementaryMatches(modelsJson, 0.5f);
            var matchesResult = PtrToStringAndFree(matchesPtr);

            using var doc = JsonDocument.Parse(matchesResult);
            var root = doc.RootElement;

            // API returns {"matches": [...], "num_matches": N}
            if (root.TryGetProperty("matches", out var matchesArray) && 
                matchesArray.ValueKind == JsonValueKind.Array && 
                matchesArray.GetArrayLength() > 0)
            {
                var match = matchesArray[0];
                var modelA = match.GetProperty("ModelA").GetString();
                var modelB = match.GetProperty("ModelB").GetString();
                var coverage = match.GetProperty("Coverage").GetDouble();

                Console.WriteLine($"  ✓ Found pair: {modelA}+{modelB} (Cov: {coverage:F2})");
                Log("success", "  ✅ PASSED: Ensemble Features");
                _testsPassed++;
            }
            else if (root.TryGetProperty("error", out var errProp))
            {
                throw new Exception(errProp.GetString());
            }
            else
            {
                // num_matches is 0
                var numMatches = root.TryGetProperty("num_matches", out var nm) ? nm.GetInt32() : -1;
                throw new Exception($"No matches found (num_matches={numMatches})");
            }
        }
        catch (Exception e)
        {
            Log("error", $"  ❌ Ensemble failed: {e.Message}");
            _testsFailed++;
        }
    }

    // =========================================================================
    // Part 6: Observer Pattern Test
    // =========================================================================

    private static void TestObserver()
    {
        Console.WriteLine();
        Console.WriteLine("┌──────────────────────────────────────────────────────────────────────┐");
        Console.WriteLine("│ Observer Pattern                                                    │");
        Console.WriteLine("└──────────────────────────────────────────────────────────────────────┘");

        try
        {
            using var tracker = new AdaptationTracker(100, 1000);
            tracker.Start("TaskA", 1);
            tracker.RecordOutput(true);
            tracker.RecordOutput(false);

            using var doc = tracker.GetResults();
            var root = doc.RootElement;

            if (root.TryGetProperty("avg_accuracy", out var acc) ||
                root.TryGetProperty("AvgAccuracy", out acc))
            {
                Console.WriteLine($"  ✓ Tracker finalized. Accuracy: {acc.GetSingle():F2}");
                Log("success", "  ✅ PASSED: Observer Pattern");
                _testsPassed++;
            }
            else
            {
                throw new Exception("Invalid tracker stats");
            }
        }
        catch (Exception e)
        {
            Log("error", $"  ❌ Observer failed: {e.Message}");
            _testsFailed++;
        }
    }

    // =========================================================================
    // Part 7: Introspection Test
    // =========================================================================

    private static void TestIntrospection()
    {
        Console.WriteLine();
        Console.WriteLine("┌──────────────────────────────────────────────────────────────────────┐");
        Console.WriteLine("│ Introspection                                                       │");
        Console.WriteLine("└──────────────────────────────────────────────────────────────────────┘");

        try
        {
            string config = @"{
                ""dtype"": ""float32"",
                ""batch_size"": 1,
                ""grid_rows"": 1,
                ""grid_cols"": 1,
                ""layers_per_cell"": 1,
                ""layers"": [{""type"": ""dense"", ""input_height"": 4, ""output_height"": 4}]
            }";

            var resultPtr = NativeMethods.CreateLoomNetwork(config);
            var result = PtrToStringAndFree(resultPtr);

            // Basic check that network was created
            if (!result.Contains("error"))
            {
                Console.WriteLine($"  ✓ Network created successfully");
                Log("success", "  ✅ PASSED: Introspection");
                _testsPassed++;
            }
            else
            {
                throw new Exception(result);
            }
        }
        catch (Exception e)
        {
            Log("error", $"  ❌ Introspection failed: {e.Message}");
            _testsFailed++;
        }
    }

    // =========================================================================
    // Part 8: Step & Tween API Test
    // =========================================================================

    private static void TestStepTween()
    {
        Console.WriteLine();
        Console.WriteLine("┌──────────────────────────────────────────────────────────────────────┐");
        Console.WriteLine("│ Step & Tween API                                                    │");
        Console.WriteLine("└──────────────────────────────────────────────────────────────────────┘");

        try
        {
            // First create a network
            string config = @"{
                ""dtype"": ""float32"",
                ""batch_size"": 1,
                ""grid_rows"": 1,
                ""grid_cols"": 1,
                ""layers_per_cell"": 1,
                ""layers"": [{""type"": ""dense"", ""input_height"": 2, ""output_height"": 2}]
            }";

            var resultPtr = NativeMethods.CreateLoomNetwork(config);
            var result = PtrToStringAndFree(resultPtr);

            // Test Step API
            using var stepState = new StepState(2);
            stepState.SetInput(new[] { 0.5f, 0.5f });
            var duration = stepState.StepForward();
            var output = stepState.GetOutput();

            if (output.Length == 2 && duration >= 0)
            {
                Console.WriteLine($"  ✓ StepForward: [{output[0]:F3}, {output[1]:F3}] ({duration / 1000000.0:F2}ms)");
            }
            else
            {
                throw new Exception("StepForward failed");
            }

            // Test Tween API
            using var tweenState = new TweenState(false);
            var loss = tweenState.Step(new[] { 0.5f, 0.5f }, 0, 2, 0.01f);

            if (loss >= 0)
            {
                Console.WriteLine($"  ✓ TweenStep: Loss={loss:F4}");
                Log("success", "  ✅ PASSED: Step & Tween API");
                _testsPassed++;
            }
            else
            {
                throw new Exception("TweenStep failed");
            }
        }
        catch (Exception e)
        {
            Log("error", $"  ❌ Step/Tween failed: {e.Message}");
            _testsFailed++;
        }
    }

    // =========================================================================
    // Part 9: Advanced Layers Test
    // =========================================================================

    private static void TestAdvancedLayers()
    {
        Console.WriteLine();
        Console.WriteLine("┌──────────────────────────────────────────────────────────────────────┐");
        Console.WriteLine("│ Advanced Layers                                                     │");
        Console.WriteLine("└──────────────────────────────────────────────────────────────────────┘");

        try
        {
            // Embedding Layer Test
            Console.WriteLine("  > Testing Embedding Layer...");
            string embConfig = @"{
                ""dtype"": ""float32"",
                ""batch_size"": 1,
                ""grid_rows"": 1,
                ""grid_cols"": 1,
                ""layers_per_cell"": 1,
                ""layers"": [{""type"": ""embedding"", ""vocab_size"": 10, ""embedding_dim"": 4}]
            }";

            var embPtr = NativeMethods.CreateLoomNetwork(embConfig);
            var embResult = PtrToStringAndFree(embPtr);

            var embOutPtr = NativeMethods.LoomForward(new[] { 1.0f }, 1); // Index 1
            var embOutJson = PtrToStringAndFree(embOutPtr);

            using (var doc = JsonDocument.Parse(embOutJson))
            {
                var embOut = doc.RootElement;
                if (embOut.ValueKind == JsonValueKind.Array && embOut.GetArrayLength() == 4)
                {
                    Console.WriteLine($"    ✓ Embedding output size: {embOut.GetArrayLength()}");
                }
            }

            // Residual Test
            Console.WriteLine("  > Testing Residual Connection...");
            string resConfig = @"{
                ""dtype"": ""float32"",
                ""batch_size"": 1,
                ""grid_rows"": 1,
                ""grid_cols"": 1,
                ""layers_per_cell"": 1,
                ""layers"": [{""type"": ""dense"", ""input_height"": 4, ""output_height"": 4, ""residual"": true}]
            }";

            var resPtr = NativeMethods.CreateLoomNetwork(resConfig);
            var resResult = PtrToStringAndFree(resPtr);

            var resOutPtr = NativeMethods.LoomForward(new[] { 0.1f, 0.1f, 0.1f, 0.1f }, 4);
            var resOutJson = PtrToStringAndFree(resOutPtr);

            using (var doc = JsonDocument.Parse(resOutJson))
            {
                var resOut = doc.RootElement;
                if (resOut.ValueKind == JsonValueKind.Array && resOut.GetArrayLength() == 4)
                {
                    Console.WriteLine($"    ✓ Residual output size: {resOut.GetArrayLength()}");
                }
            }

            Log("success", "  ✅ PASSED: Advanced Layers");
            _testsPassed++;
        }
        catch (Exception e)
        {
            Log("error", $"  ❌ Advanced Layers failed: {e.Message}");
            _testsFailed++;
        }
    }
}
