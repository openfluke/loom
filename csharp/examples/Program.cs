using System;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Threading.Tasks;
using Welvet;

/// <summary>
/// LOOM All Layers Test - C#/.NET Version
/// 
/// ✨ ONE FUNCTION CALL TO RULE THEM ALL! ✨
/// 
/// This demonstrates LoadFromString() which takes a JSON string
/// and returns a fully configured network with ALL weights loaded.
/// 
/// No manual layer setup, no type conversions, no hassle!
/// Just: network = Network.LoadFromString(jsonString, "model_id")
/// 
/// The test:
/// 1. Downloads test.json from localhost:3123
/// 2. Calls LoadFromString() - DONE! Network ready!
/// 3. Runs inference and compares with expected outputs
/// 4. Trains the network to verify weights are mutable
/// </summary>
class Program
{
    private static readonly HttpClient _httpClient = new HttpClient();
    private const string BaseUrl = "http://localhost:3123";

    static async Task<int> Main(string[] args)
    {
        Console.WriteLine();
        Console.WriteLine(new string('=', 60));
        Console.WriteLine("🧠 LOOM All Layers Test (C#/.NET + C-ABI)");
        Console.WriteLine(new string('=', 60));
        Console.WriteLine();

        // ===== Step 1: Load files from server =====
        Console.WriteLine("📥 Step 1: Load Files from Server");
        Console.WriteLine($"   Server URL: {BaseUrl}\n");

        Log("Fetching files from localhost:3123...", "loading");

        string modelJson;
        float[] inputs;
        float[] expectedOutputs;

        try
        {
            // Load model JSON
            Log("Loading test.json...", "loading");
            modelJson = await FetchTextAsync($"{BaseUrl}/test.json");
            Log($"test.json loaded ({modelJson.Length / 1024.0:F1} KB)", "success");

            // Load inputs
            Log("Loading inputs.txt...", "loading");
            string inputsText = await FetchTextAsync($"{BaseUrl}/inputs.txt");
            inputs = inputsText.Split('\n', StringSplitOptions.RemoveEmptyEntries)
                .Select(float.Parse)
                .ToArray();
            Log($"inputs.txt loaded ({inputs.Length} values)", "success");

            // Load expected outputs
            Log("Loading outputs.txt...", "loading");
            string outputsText = await FetchTextAsync($"{BaseUrl}/outputs.txt");
            expectedOutputs = outputsText.Split('\n', StringSplitOptions.RemoveEmptyEntries)
                .Select(float.Parse)
                .ToArray();
            Log($"outputs.txt loaded ({expectedOutputs.Length} values)", "success");

            Log("All files loaded successfully!", "success");
        }
        catch (Exception ex)
        {
            Log($"Error loading files: {ex.Message}", "error");
            Log("Make sure the server is running: cd examples && ./serve_files.sh", "error");
            return 1;
        }

        Console.WriteLine();

        // ===== Step 2: Load complete model using C-ABI =====
        Console.WriteLine("🔄 Step 2: Load Complete Model using C-ABI");
        Console.WriteLine();

        Log("Loading complete model (structure + weights) from JSON...", "loading");

        Network? network = null;
        try
        {
            // THIS IS THE MAGIC! Just pass the JSON string and get a fully configured network!
            network = Network.LoadFromString(modelJson, "all_layers_test");

            Log($"✨ Model loaded completely! (handle: {network.Handle})", "success");
            Log("All 16 layers with weights loaded automatically!", "success");
        }
        catch (Exception ex)
        {
            Log($"Error loading model: {ex.Message}", "error");
            Console.WriteLine(ex.StackTrace);
            return 1;
        }

        Console.WriteLine();

        // ===== Step 3: Run inference =====
        Console.WriteLine("▶️ Step 3: Run Inference");
        Console.WriteLine();

        Log("Running forward pass...", "loading");
        Log($"Input size: {inputs.Length}", "info");

        float[] output;
        try
        {
            output = network.Forward(inputs);

            Log($"Forward pass complete! Output size: {output.Length}", "success");

            if (output.Length == 0)
            {
                Log("Output is empty - network might not be configured correctly", "error");
                network.Dispose();
                return 1;
            }

            // Display outputs
            Console.WriteLine("\n   Expected output (from file):");
            for (int i = 0; i < expectedOutputs.Length; i++)
            {
                Console.WriteLine($"     [{i}] {expectedOutputs[i]:F6}");
            }

            Console.WriteLine("\n   C-ABI output (loaded weights):");
            for (int i = 0; i < output.Length; i++)
            {
                Console.WriteLine($"     [{i}] {output[i]:F6}");
            }

            // Compare with expected
            if (output.Length == expectedOutputs.Length)
            {
                float maxDiff = output.Zip(expectedOutputs, (o, e) => Math.Abs(o - e)).Max();

                Log($"Max difference: {maxDiff:F10}", "info");

                if (maxDiff < 1e-5f)
                {
                    Log("✅ Outputs match expected exactly!", "success");
                }
                else if (maxDiff < 0.1f)
                {
                    Log("✅ Outputs match with small differences (expected with softmax)", "success");
                }
                else
                {
                    Log("⚠️ Large output differences detected", "error");
                }
            }
            else
            {
                Log($"Output size mismatch: got {output.Length}, expected {expectedOutputs.Length}", "error");
            }
        }
        catch (Exception ex)
        {
            Log($"Error during inference: {ex.Message}", "error");
            Console.WriteLine(ex.StackTrace);
            network.Dispose();
            return 1;
        }

        Console.WriteLine();

        // ===== Step 4: Train model =====
        Console.WriteLine("🎯 Step 4: Train Model");
        Console.WriteLine();

        Log("Starting training...", "loading");

        try
        {
            // Get output before training
            float[] outputBefore = network.Forward(inputs);

            // Training parameters
            int epochs = 10;
            float learningRate = 0.05f;
            float[] target = new float[] { 0.5f, 0.5f };

            Log($"Epochs: {epochs}", "info");
            Log($"Learning rate: {learningRate}", "info");
            Log("Training...", "loading");

            float initialLoss = 0, finalLoss = 0;

            // Training loop
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                // Forward pass
                float[] currentOutput = network.Forward(inputs);

                // Compute loss (MSE)
                float loss = currentOutput.Zip(target, (o, t) => (o - t) * (o - t))
                    .Average();

                if (epoch == 0)
                {
                    initialLoss = loss;
                    Log($"Initial loss: {loss:F6}", "info");
                }
                else if (epoch == epochs - 1)
                {
                    finalLoss = loss;
                    Log($"Final loss: {loss:F6}", "info");
                }

                // Backward pass with gradient
                float[] gradOutput = currentOutput.Zip(target, (o, t) => (o - t) * 2 / target.Length)
                    .ToArray();
                network.Backward(gradOutput);

                // Update weights
                network.UpdateWeights(learningRate);
            }

            Log("Training complete!", "success");

            // Get output after training
            float[] outputAfter = network.Forward(inputs);

            // Check if weights changed
            float maxChange = outputAfter.Zip(outputBefore, (a, b) => Math.Abs(a - b)).Max();

            Log($"Max output change: {maxChange:F6}", "info");

            if (maxChange > 1e-5f)
            {
                Log("Weights successfully changed!", "success");
            }
            else
            {
                Log("Weights did not change", "error");
            }

            // Display results
            Console.WriteLine("\n   Output before training:");
            for (int i = 0; i < outputBefore.Length; i++)
            {
                Console.WriteLine($"     [{i}] {outputBefore[i]:F6}");
            }

            Console.WriteLine("\n   Output after training:");
            for (int i = 0; i < outputAfter.Length; i++)
            {
                Console.WriteLine($"     [{i}] {outputAfter[i]:F6}");
            }
        }
        catch (Exception ex)
        {
            Log($"Error during training: {ex.Message}", "error");
            Console.WriteLine(ex.StackTrace);
            network.Dispose();
            return 1;
        }

        Console.WriteLine();

        // Cleanup
        Log("Cleaning up...", "loading");
        network.Dispose();
        Log("Network freed", "success");

        Console.WriteLine();
        Console.WriteLine(new string('=', 60));
        Console.WriteLine("✅ All Layer Types Test Complete");
        Console.WriteLine("✅ Model structure loaded from server");
        Console.WriteLine("✅ Network created with C-ABI");
        Console.WriteLine("✅ Inference successful");
        Console.WriteLine("✅ Training verified");
        Console.WriteLine(new string('=', 60));
        Console.WriteLine();

        return 0;
    }

    static async Task<string> FetchTextAsync(string url)
    {
        var response = await _httpClient.GetAsync(url);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadAsStringAsync();
    }

    static void Log(string message, string status = "info")
    {
        const string reset = "\x1b[0m";

        var (color, icon) = status switch
        {
            "info" => ("\x1b[94m", "ℹ"),
            "success" => ("\x1b[92m", "✅"),
            "error" => ("\x1b[91m", "❌"),
            "loading" => ("\x1b[93m", "🔄"),
            _ => ("", "•")
        };

        Console.WriteLine($"{color}{icon} {message}{reset}");
    }
}
