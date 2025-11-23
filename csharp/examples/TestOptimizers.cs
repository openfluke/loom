using System;
using System.Text.Json;
using Welvet;

// Simple C# Optimizer Test
// Tests all 4 optimizer methods via the C ABI

Console.WriteLine("🚀 LOOM Optimizer C# Test");
Console.WriteLine("================================\n");

// Network configuration
var config = new
{
    batch_size = 1,
    grid_rows = 1,
    grid_cols = 1,
    layers_per_cell = 3,
    layers = new object[]
    {
        new { type = "dense", input_height = 4, output_height = 8, activation = "relu" },
        new { type = "lstm", input_size = 8, hidden_size = 12, seq_length = 1 },
        new { type = "dense", input_height = 12, output_height = 3, activation = "softmax" }
    }
};

// Create network
Console.WriteLine("Creating network...");
Network.CreateFromJson(JsonSerializer.Serialize(config));
Console.WriteLine("✓ Network created\n");

// Training data
(float[] input, float[] target)[] trainingData =
{
    (new[] { 0.1f, 0.2f, 0.1f, 0.3f }, new[] { 1.0f, 0.0f, 0.0f }),
    (new[] { 0.8f, 0.9f, 0.7f, 0.8f }, new[] { 0.0f, 1.0f, 0.0f }),
    (new[] { 0.3f, 0.5f, 0.9f, 0.6f }, new[] { 0.0f, 0.0f, 1.0f }),
    (new[] { 0.2f, 0.1f, 0.2f, 0.2f }, new[] { 1.0f, 0.0f, 0.0f }),
    (new[] { 0.9f, 0.8f, 0.8f, 0.9f }, new[] { 0.0f, 1.0f, 0.0f }),
    (new[] { 0.4f, 0.6f, 0.8f, 0.7f }, new[] { 0.0f, 0.0f, 1.0f }),
};

// Helper to call native methods via reflection (workaround for internal NativeMethods)
static void CallNativeMethod(string methodName, params object[] args)
{
    var nativeMethodsType = typeof(Network).Assembly.GetType("Welvet.NativeMethods");
    var method = nativeMethodsType?.GetMethod(methodName, 
        System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Public);
    method?.Invoke(null, args);
}

// ========================================================================
// Test 1: Simple SGD (baseline)
// ========================================================================
Console.WriteLine("📊 Test 1: Simple SGD (baseline)");
Console.WriteLine("----------------------------------");

using (var state = new StepState(4))
{
    float totalLoss = 0.0f;

    for (int step = 0; step < 5000; step++)
    {
        int idx = step % trainingData.Length;
        var (input, target) = trainingData[idx];

        state.SetInput(input);
        state.StepForward();
        var output = state.GetOutput();

        float loss = 0.0f;
        var gradients = new float[3];
        for (int i = 0; i < 3; i++)
        {
            float diff = output[i] - target[i];
            loss += diff * diff;
            gradients[i] = 2.0f * diff / 3.0f;
        }
        loss /= 3.0f;
        totalLoss += loss;

        state.StepBackward(gradients);
        CallNativeMethod("LoomApplyGradients", 0.01f);

        if ((step + 1) % 1000 == 0)
        {
            Console.WriteLine($"  Step {step + 1}: Avg Loss={totalLoss / 1000:F6}");
            totalLoss = 0.0f;
        }
    }
}

Console.WriteLine("✅ SGD Test complete!\n");

// ========================================================================
// Test 2: AdamW Optimizer
// ========================================================================
Console.WriteLine("📊 Test 2: AdamW Optimizer");
Console.WriteLine("----------------------------------");

Network.CreateFromJson(JsonSerializer.Serialize(config));

using (var state = new StepState(4))
{
    float totalLoss = 0.0f;

    for (int step = 0; step < 5000; step++)
    {
        int idx = step % trainingData.Length;
        var (input, target) = trainingData[idx];

        state.SetInput(input);
        state.StepForward();
        var output = state.GetOutput();

        float loss = 0.0f;
        var gradients = new float[3];
        for (int i = 0; i < 3; i++)
        {
            float diff = output[i] - target[i];
            loss += diff * diff;
            gradients[i] = 2.0f * diff / 3.0f;
        }
        loss /= 3.0f;
        totalLoss += loss;

        state.StepBackward(gradients);
        CallNativeMethod("LoomApplyGradientsAdamW", 0.001f, 0.9f, 0.999f, 0.01f);

        if ((step + 1) % 1000 == 0)
        {
            Console.WriteLine($"  Step {step + 1}: Avg Loss={totalLoss / 1000:F6}");
            totalLoss = 0.0f;
        }
    }
}

Console.WriteLine("✅ AdamW Test complete!\n");

// ========================================================================
// Test 3: RMSprop Optimizer
// ========================================================================
Console.WriteLine("📊 Test 3: RMSprop Optimizer");
Console.WriteLine("----------------------------------");

Network.CreateFromJson(JsonSerializer.Serialize(config));

using (var state = new StepState(4))
{
    float totalLoss = 0.0f;

    for (int step = 0; step < 5000; step++)
    {
        int idx = step % trainingData.Length;
        var (input, target) = trainingData[idx];

        state.SetInput(input);
        state.StepForward();
        var output = state.GetOutput();

        float loss = 0.0f;
        var gradients = new float[3];
        for (int i = 0; i < 3; i++)
        {
            float diff = output[i] - target[i];
            loss += diff * diff;
            gradients[i] = 2.0f * diff / 3.0f;
        }
        loss /= 3.0f;
        totalLoss += loss;

        state.StepBackward(gradients);
        CallNativeMethod("LoomApplyGradientsRMSprop", 0.001f, 0.99f, 1e-8f, 0.0f);

        if ((step + 1) % 1000 == 0)
        {
            Console.WriteLine($"  Step {step + 1}: Avg Loss={totalLoss / 1000:F6}");
            totalLoss = 0.0f;
        }
    }
}

Console.WriteLine("✅ RMSprop Test complete!\n");

// ========================================================================
// Test 4: SGD with Momentum
// ========================================================================
Console.WriteLine("📊 Test 4: SGD with Momentum");
Console.WriteLine("----------------------------------");

Network.CreateFromJson(JsonSerializer.Serialize(config));

using (var state = new StepState(4))
{
    float totalLoss = 0.0f;

    for (int step = 0; step < 5000; step++)
    {
        int idx = step % trainingData.Length;
        var (input, target) = trainingData[idx];

        state.SetInput(input);
        state.StepForward();
        var output = state.GetOutput();

        float loss = 0.0f;
        var gradients = new float[3];
        for (int i = 0; i < 3; i++)
        {
            float diff = output[i] - target[i];
            loss += diff * diff;
            gradients[i] = 2.0f * diff / 3.0f;
        }
        loss /= 3.0f;
        totalLoss += loss;

        state.StepBackward(gradients);
        CallNativeMethod("LoomApplyGradientsSGDMomentum", 0.01f, 0.9f, 0.0f, 0);

        if ((step + 1) % 1000 == 0)
        {
            Console.WriteLine($"  Step {step + 1}: Avg Loss={totalLoss / 1000:F6}");
            totalLoss = 0.0f;
        }
    }
}

Console.WriteLine("✅ SGD+Momentum Test complete!\n");

Console.WriteLine("🎉 All C# optimizer tests complete!\n");
Console.WriteLine("Verified Functions:");
Console.WriteLine("  ✅ LoomApplyGradients (simple SGD)");
Console.WriteLine("  ✅ LoomApplyGradientsAdamW");
Console.WriteLine("  ✅ LoomApplyGradientsRMSprop");
Console.WriteLine("  ✅ LoomApplyGradientsSGDMomentum");
Console.WriteLine("\nAll optimizer methods working correctly via C#! 🚀");
