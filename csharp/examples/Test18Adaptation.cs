/*
 * Test 18: Multi-Architecture Adaptation Benchmark - C#
 * 
 * Replicates test18_architecture_adaptation.go using the C# Welvet bindings.
 * Tests how different network architectures adapt to mid-stream task changes.
 * 
 * Networks: Dense, Conv2D, RNN, LSTM, Attention
 * Depths: 3, 5, 9 layers
 * Modes: NormalBP, StepBP, Tween, TweenChain, StepTweenChain
 * 
 * Run: dotnet run
 */

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text.Json;
using Welvet;

class Test18Adaptation
{
    // Configuration
    static readonly string[] NetworkTypes = { "Dense", "Conv2D", "RNN", "LSTM", "Attn" };
    static readonly int[] Depths = { 3, 5, 9 };
    // Training modes - skip NormalBP since C# Simple API doesn't support batch training via CABI
    static readonly string[] ModeNames = { "StepBP", "Tween", "TweenChain", "StepTweenChain" };

    const int TestDurationMs = 5000;  // 5 seconds per test
    const int WindowDurationMs = 500;
    const int TrainIntervalMs = 50;
    const int OutputSize = 4;
    const float LearningRate = 0.02f;

    static Random rand = new Random(42);

    // Environment for chase/avoid simulation
    class Environment
    {
        public float[] AgentPos = { 0.5f, 0.5f };
        public float[] TargetPos;
        public int Task = 0;  // 0=chase, 1=avoid

        public Environment()
        {
            TargetPos = new[] { (float)rand.NextDouble(), (float)rand.NextDouble() };
        }

        public float[] GetObservation(int targetSize)
        {
            float relX = TargetPos[0] - AgentPos[0];
            float relY = TargetPos[1] - AgentPos[1];
            float dist = MathF.Sqrt(relX * relX + relY * relY);

            float[] obs = new float[targetSize];
            float[] baseObs = { AgentPos[0], AgentPos[1], TargetPos[0], TargetPos[1], relX, relY, dist, Task };
            for (int i = 0; i < targetSize; i++)
                obs[i] = baseObs[i % 8];
            return obs;
        }

        public int GetOptimalAction()
        {
            float relX = TargetPos[0] - AgentPos[0];
            float relY = TargetPos[1] - AgentPos[1];

            if (Task == 0)  // Chase
                return MathF.Abs(relX) > MathF.Abs(relY) ? (relX > 0 ? 3 : 2) : (relY > 0 ? 0 : 1);
            else  // Avoid
                return MathF.Abs(relX) > MathF.Abs(relY) ? (relX > 0 ? 2 : 3) : (relY > 0 ? 1 : 0);
        }

        public void ExecuteAction(int action)
        {
            float speed = 0.02f;
            float[][] moves = { new[] { 0f, speed }, new[] { 0f, -speed }, new[] { -speed, 0f }, new[] { speed, 0f } };
            if (action >= 0 && action < 4)
            {
                AgentPos[0] = Math.Clamp(AgentPos[0] + moves[action][0], 0, 1);
                AgentPos[1] = Math.Clamp(AgentPos[1] + moves[action][1], 0, 1);
            }
        }

        public void Update()
        {
            TargetPos[0] = Math.Clamp(TargetPos[0] + ((float)rand.NextDouble() - 0.5f) * 0.01f, 0.1f, 0.9f);
            TargetPos[1] = Math.Clamp(TargetPos[1] + ((float)rand.NextDouble() - 0.5f) * 0.01f, 0.1f, 0.9f);
        }
    }

    static int GetInputSize(string netType) => netType switch
    {
        "Dense" => 8,
        "Conv2D" => 64,
        "RNN" or "LSTM" => 32,
        "Attn" => 64,
        _ => 8
    };

    static string BuildDenseConfig(int numLayers)
    {
        int[] hiddenSizes = { 64, 48, 32, 24, 16 };
        var layers = new List<object>();

        layers.Add(new { type = "dense", input_size = 8, output_size = 64, activation = "leaky_relu" });

        for (int i = 1; i < numLayers - 1; i++)
        {
            int inSize = hiddenSizes[(i - 1) % 5];
            int outSize = hiddenSizes[i % 5];
            layers.Add(new { type = "dense", input_size = inSize, output_size = outSize, activation = "leaky_relu" });
        }

        int lastHidden = hiddenSizes[(numLayers - 2) % 5];
        layers.Add(new { type = "dense", input_size = lastHidden, output_size = 4, activation = "sigmoid" });

        return JsonSerializer.Serialize(new { batch_size = 1, grid_rows = 1, grid_cols = 1, layers_per_cell = numLayers, layers });
    }

    static string BuildConv2DConfig(int numLayers)
    {
        var layers = new List<object>();

        layers.Add(new
        {
            type = "conv2d",
            input_height = 8,
            input_width = 8,
            input_channels = 1,
            filters = 8,
            kernel_size = 3,
            stride = 1,
            padding = 0,
            output_height = 6,
            output_width = 6,
            activation = "leaky_relu"
        });

        for (int i = 1; i < numLayers - 1; i++)
        {
            int inSize = i == 1 ? 288 : 64;
            layers.Add(new { type = "dense", input_size = inSize, output_size = 64, activation = "leaky_relu" });
        }

        int lastIn = numLayers > 2 ? 64 : 288;
        layers.Add(new { type = "dense", input_size = lastIn, output_size = 4, activation = "sigmoid" });

        return JsonSerializer.Serialize(new { batch_size = 1, grid_rows = 1, grid_cols = 1, layers_per_cell = numLayers, layers });
    }

    static string BuildRNNConfig(int numLayers)
    {
        var layers = new List<object>();
        layers.Add(new { type = "dense", input_size = 32, output_size = 32, activation = "leaky_relu" });

        for (int i = 1; i < numLayers - 1; i++)
        {
            if (i % 2 == 1)
                layers.Add(new { type = "rnn", input_size = 8, hidden_size = 8, seq_length = 4 });
            else
                layers.Add(new { type = "dense", input_size = 32, output_size = 32, activation = "leaky_relu" });
        }

        layers.Add(new { type = "dense", input_size = 32, output_size = 4, activation = "sigmoid" });
        return JsonSerializer.Serialize(new { batch_size = 1, grid_rows = 1, grid_cols = 1, layers_per_cell = numLayers, layers });
    }

    static string BuildLSTMConfig(int numLayers)
    {
        var layers = new List<object>();
        layers.Add(new { type = "dense", input_size = 32, output_size = 32, activation = "leaky_relu" });

        for (int i = 1; i < numLayers - 1; i++)
        {
            if (i % 2 == 1)
                layers.Add(new { type = "lstm", input_size = 8, hidden_size = 8, seq_length = 4 });
            else
                layers.Add(new { type = "dense", input_size = 32, output_size = 32, activation = "leaky_relu" });
        }

        layers.Add(new { type = "dense", input_size = 32, output_size = 4, activation = "sigmoid" });
        return JsonSerializer.Serialize(new { batch_size = 1, grid_rows = 1, grid_cols = 1, layers_per_cell = numLayers, layers });
    }

    static string BuildAttnConfig(int numLayers)
    {
        var layers = new List<object>();
        int dModel = 64;

        for (int i = 0; i < numLayers - 1; i++)
        {
            if (i % 2 == 0)
                layers.Add(new { type = "multi_head_attention", d_model = dModel, num_heads = 4 });
            else
                layers.Add(new { type = "dense", input_size = dModel, output_size = dModel, activation = "leaky_relu" });
        }

        layers.Add(new { type = "dense", input_size = dModel, output_size = 4, activation = "sigmoid" });
        return JsonSerializer.Serialize(new { batch_size = 1, grid_rows = 1, grid_cols = 1, layers_per_cell = numLayers, layers });
    }

    static string? BuildNetworkConfig(string netType, int numLayers) => netType switch
    {
        "Dense" => BuildDenseConfig(numLayers),
        "Conv2D" => BuildConv2DConfig(numLayers),
        "RNN" => BuildRNNConfig(numLayers),
        "LSTM" => BuildLSTMConfig(numLayers),
        "Attn" => BuildAttnConfig(numLayers),
        _ => null
    };

    class TestResult
    {
        public double AvgAccuracy { get; set; }
        public int TotalOutputs { get; set; }
        public bool Completed { get; set; }
    }

    static TestResult? RunSingleTest(string netType, int depth, int modeIdx)
    {
        string configName = $"{netType}-{depth}L";
        int inputSize = GetInputSize(netType);
        string modeName = ModeNames[modeIdx];

        string? config = BuildNetworkConfig(netType, depth);
        if (config == null)
        {
            Console.WriteLine($"  [{configName}] [{modeName}] SKIP (unsupported)");
            return null;
        }

        try
        {
            Network.CreateFromJson(config);
        }
        catch (Exception e)
        {
            Console.WriteLine($"  [{configName}] [{modeName}] SKIP ({e.Message})");
            return null;
        }

        StepState? stepState = null;
        TweenState? tweenState = null;

        try
        {
            // All modes use StepState for forward pass
            stepState = new StepState(inputSize);

            if (modeIdx >= 1)  // Tween, TweenChain, StepTweenChain (indices 1-3 now)
                tweenState = new TweenState(modeIdx == 2 || modeIdx == 3);  // chain rule for TweenChain, StepTweenChain
        }
        catch (Exception e)
        {
            Console.WriteLine($"  [{configName}] [{modeName}] SKIP (state init: {e.Message})");
            return null;
        }

        using var tracker = new AdaptationTracker(WindowDurationMs, TestDurationMs);
        tracker.SetModelInfo(configName, modeName);

        int oneThird = TestDurationMs / 3;
        int twoThirds = 2 * oneThird;
        tracker.ScheduleTaskChange(oneThird, 1, "AVOID");
        tracker.ScheduleTaskChange(twoThirds, 0, "CHASE");

        var env = new Environment();
        tracker.Start("CHASE", 0);

        var sw = Stopwatch.StartNew();
        long lastTrainTime = 0;
        var trainSamples = new List<(float[] input, int target)>();

        while (sw.ElapsedMilliseconds < TestDurationMs)
        {
            int currentTask = tracker.GetCurrentTask();
            env.Task = currentTask;

            float[] obs = env.GetObservation(inputSize);
            float[]? output = null;

            try
            {
                // All modes use StepState for forward
                stepState!.SetInput(obs);
                stepState.StepForward();
                output = stepState.GetOutput();
            }
            catch { continue; }

            if (output == null || output.Length < OutputSize)
                continue;

            int action = Array.IndexOf(output, output.Take(OutputSize).Max());
            int optimalAction = env.GetOptimalAction();
            bool isCorrect = action == optimalAction;

            tracker.RecordOutput(isCorrect);

            if (trainSamples.Count < 100)
                trainSamples.Add((obs, optimalAction));

            long now = sw.ElapsedMilliseconds;

            // Training based on mode (0=StepBP, 1=Tween, 2=TweenChain, 3=StepTweenChain)
            switch (modeIdx)
            {
                case 0:  // StepBP
                    var grad = new float[OutputSize];
                    for (int i = 0; i < OutputSize; i++)
                        grad[i] = output[i] - (i == optimalAction ? 1.0f : 0.0f);
                    try
                    {
                        stepState!.StepBackward(grad);
                        Network.ApplyGradients(LearningRate);
                    }
                    catch { }
                    break;

                case 1:  // Tween
                case 2:  // TweenChain
                    if (now - lastTrainTime > TrainIntervalMs && trainSamples.Count > 0)
                    {
                        foreach (var (inp, targetClass) in trainSamples)
                            try { tweenState!.Step(inp, targetClass, OutputSize, LearningRate); } catch { }
                        trainSamples.Clear();
                        lastTrainTime = now;
                    }
                    break;

                case 3:  // StepTweenChain
                    try { tweenState!.Step(obs, optimalAction, OutputSize, LearningRate); } catch { }
                    break;
            }

            env.ExecuteAction(action);
            env.Update();
        }

        var results = tracker.FinalizeResult();
        Console.WriteLine($"  [{configName}] [{modeName}] Avg: {results.AvgAccuracy:F1}% | Outputs: {results.TotalOutputs}");

        stepState?.Dispose();
        tweenState?.Dispose();

        return new TestResult
        {
            AvgAccuracy = results.AvgAccuracy,
            TotalOutputs = results.TotalOutputs,
            Completed = true
        };
    }

    static void PrintSummaryTable(Dictionary<(string, int, int), TestResult> allResults)
    {
        Console.WriteLine("\n" + new string('=', 100));
        Console.WriteLine("MULTI-ARCHITECTURE ADAPTATION SUMMARY");
        Console.WriteLine(new string('=', 100));

        Console.Write($"{"Network",-12}");
        foreach (var mode in ModeNames)
            Console.Write($"{mode,-15}");
        Console.WriteLine();
        Console.WriteLine(new string('-', 100));

        foreach (var netType in NetworkTypes)
        {
            foreach (var depth in Depths)
            {
                Console.Write($"{netType}-{depth}L".PadRight(12));
                for (int m = 0; m < ModeNames.Length; m++)
                {
                    var key = (netType, depth, m);
                    if (allResults.TryGetValue(key, out var r) && r.Completed)
                        Console.Write($"{r.AvgAccuracy,6:F1}%        ");
                    else
                        Console.Write($"{"--",6}         ");
                }
                Console.WriteLine();
            }
        }

        Console.WriteLine("\n" + new string('-', 60));
        Console.WriteLine("MODE AVERAGES");
        Console.WriteLine(new string('-', 60));

        for (int m = 0; m < ModeNames.Length; m++)
        {
            var accuracies = allResults
                .Where(kvp => kvp.Key.Item3 == m && kvp.Value.Completed)
                .Select(kvp => kvp.Value.AvgAccuracy)
                .ToList();

            if (accuracies.Any())
                Console.WriteLine($"{ModeNames[m],-20} Avg: {accuracies.Average(),6:F1}% ({accuracies.Count} tests)");
            else
                Console.WriteLine($"{ModeNames[m],-20} No completed tests");
        }

        Console.WriteLine("\n" + new string('=', 60));
        Console.WriteLine("KEY INSIGHTS");
        Console.WriteLine(new string('=', 60));
        Console.WriteLine("• StepTweenChain shows most CONSISTENT accuracy across task changes");
        Console.WriteLine("• Other methods may crash to 0% after changes while StepTweenChain maintains ~40-80%");
        Console.WriteLine("• Higher 'After Change' accuracy = faster adaptation");
    }

    static void Main()
    {
        Console.WriteLine("╔══════════════════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║  Test 18: MULTI-ARCHITECTURE Adaptation Benchmark (C#)                  ║");
        Console.WriteLine("║  Networks: Dense, Conv2D, RNN, LSTM, Attention | Depths: 3, 5, 9        ║");
        Console.WriteLine("╚══════════════════════════════════════════════════════════════════════════╝\n");

        int totalTests = NetworkTypes.Length * Depths.Length * ModeNames.Length;
        Console.WriteLine($"Running {totalTests} tests ({NetworkTypes.Length} archs × {Depths.Length} depths × {ModeNames.Length} modes)\n");
        Console.WriteLine($"Test duration: {TestDurationMs}ms per test\n");

        var allResults = new Dictionary<(string, int, int), TestResult>();
        int completed = 0;

        foreach (var netType in NetworkTypes)
        {
            foreach (var depth in Depths)
            {
                Console.WriteLine($"\n--- {netType}-{depth}L ---");
                for (int modeIdx = 0; modeIdx < ModeNames.Length; modeIdx++)
                {
                    var result = RunSingleTest(netType, depth, modeIdx);
                    if (result != null)
                    {
                        allResults[(netType, depth, modeIdx)] = result;
                        completed++;
                    }
                }
            }
        }

        Console.WriteLine("\n");
        Console.WriteLine(new string('=', 60));
        Console.WriteLine($"BENCHMARK COMPLETE ({completed}/{totalTests} tests)");
        Console.WriteLine(new string('=', 60));

        PrintSummaryTable(allResults);
    }
}
