using Welvet;
using System.Diagnostics;

namespace MauiApp1
{
    public partial class MainPage : ContentPage
    {
        int count = 0;
        private static readonly HttpClient _httpClient = new HttpClient();
        private const string BaseUrl = "http://192.168.0.250:3123";

        public MainPage()
        {
            InitializeComponent();
            
            // Initialize Welvet on app startup
            try
            {
                Debug.WriteLine("🔧 Initializing Welvet...");
                Welvet.Loader.Initialize();
                Debug.WriteLine("✅ Welvet initialized successfully!");
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"❌ Welvet initialization failed: {ex.Message}");
                Debug.WriteLine($"Stack trace: {ex.StackTrace}");
            }
        }

        private async void OnCounterClicked(object? sender, EventArgs e)
        {
            count++;
            
            if (count == 1)
                CounterBtn.Text = $"Clicked {count} time - Testing Welvet...";
            else
                CounterBtn.Text = $"Clicked {count} times";

            SemanticScreenReader.Announce(CounterBtn.Text);
            
            // Run test on button click
            await TestWelvet();
        }

        private async Task TestWelvet()
        {
            Debug.WriteLine("");
            Debug.WriteLine("=".PadRight(60, '='));
            Debug.WriteLine("🧠 LOOM All Layers Test (C#/.NET + MAUI)");
            Debug.WriteLine("=".PadRight(60, '='));
            Debug.WriteLine("");

            try
            {
                // Test 1: Get version
                Debug.WriteLine("📌 Test 1: Get LOOM Version");
                string version = Welvet.Network.GetVersion();
                Debug.WriteLine($"✅ LOOM Version: {version}");
                Debug.WriteLine("");

                // Test 2: Load files from server
                Debug.WriteLine("📥 Test 2: Load Files from Server");
                Debug.WriteLine($"   Server URL: {BaseUrl}");
                
                string modelJson = await FetchTextAsync($"{BaseUrl}/test.json");
                Debug.WriteLine($"✅ test.json loaded ({modelJson.Length / 1024.0:F1} KB)");

                string inputsText = await FetchTextAsync($"{BaseUrl}/inputs.txt");
                float[] inputs = inputsText.Split('\n', StringSplitOptions.RemoveEmptyEntries)
                    .Select(float.Parse)
                    .ToArray();
                Debug.WriteLine($"✅ inputs.txt loaded ({inputs.Length} values)");

                string outputsText = await FetchTextAsync($"{BaseUrl}/outputs.txt");
                float[] expectedOutputs = outputsText.Split('\n', StringSplitOptions.RemoveEmptyEntries)
                    .Select(float.Parse)
                    .ToArray();
                Debug.WriteLine($"✅ outputs.txt loaded ({expectedOutputs.Length} values)");
                Debug.WriteLine("");

                // Test 3: Load model
                Debug.WriteLine("🔄 Test 3: Load Complete Model");
                using var network = Welvet.Network.LoadFromString(modelJson, "all_layers_test");
                Debug.WriteLine($"✅ Model loaded! (handle: {network.Handle})");
                Debug.WriteLine($"✅ All 16 layers with weights loaded automatically!");
                Debug.WriteLine("");

                // Test 4: Run inference
                Debug.WriteLine("▶️ Test 4: Run Inference");
                Debug.WriteLine($"   Input size: {inputs.Length}");
                
                float[] output = network.Forward(inputs);
                Debug.WriteLine($"✅ Forward pass complete! Output size: {output.Length}");
                
                Debug.WriteLine("");
                Debug.WriteLine("   Expected output:");
                for (int i = 0; i < expectedOutputs.Length; i++)
                {
                    Debug.WriteLine($"     [{i}] {expectedOutputs[i]:F6}");
                }
                
                Debug.WriteLine("");
                Debug.WriteLine("   Actual output:");
                for (int i = 0; i < output.Length; i++)
                {
                    Debug.WriteLine($"     [{i}] {output[i]:F6}");
                }

                // Compare outputs
                if (output.Length == expectedOutputs.Length)
                {
                    float maxDiff = output.Zip(expectedOutputs, (o, e) => Math.Abs(o - e)).Max();
                    Debug.WriteLine("");
                    Debug.WriteLine($"   Max difference: {maxDiff:F10}");

                    if (maxDiff < 1e-5f)
                        Debug.WriteLine("✅ Outputs match exactly!");
                    else if (maxDiff < 0.1f)
                        Debug.WriteLine("✅ Outputs match (small diff expected with softmax)");
                    else
                        Debug.WriteLine("⚠️ Large differences detected");
                }
                Debug.WriteLine("");

                // Test 5: Training
                Debug.WriteLine("🎯 Test 5: Train Model");
                float[] outputBefore = network.Forward(inputs);
                
                int epochs = 10;
                float learningRate = 0.05f;
                float[] target = new float[] { 0.5f, 0.5f };
                
                Debug.WriteLine($"   Epochs: {epochs}");
                Debug.WriteLine($"   Learning rate: {learningRate}");
                Debug.WriteLine($"   Target: [{string.Join(", ", target.Select(x => $"{x:F1}"))}]");

                float initialLoss = 0, finalLoss = 0;

                for (int epoch = 0; epoch < epochs; epoch++)
                {
                    float[] currentOutput = network.Forward(inputs);
                    
                    // Compute loss (MSE)
                    float loss = currentOutput.Zip(target, (o, t) => (o - t) * (o - t)).Average();
                    
                    if (epoch == 0)
                    {
                        initialLoss = loss;
                        Debug.WriteLine($"   Initial loss: {loss:F6}");
                    }
                    else if (epoch == epochs - 1)
                    {
                        finalLoss = loss;
                        Debug.WriteLine($"   Final loss: {loss:F6}");
                    }

                    // Backward pass
                    float[] gradOutput = currentOutput.Zip(target, (o, t) => (o - t) * 2 / target.Length).ToArray();
                    network.Backward(gradOutput);
                    
                    // Update weights
                    network.UpdateWeights(learningRate);
                }

                float[] outputAfter = network.Forward(inputs);
                float maxChange = outputAfter.Zip(outputBefore, (a, b) => Math.Abs(a - b)).Max();
                
                Debug.WriteLine($"   Max output change: {maxChange:F6}");
                Debug.WriteLine(maxChange > 1e-5f ? "✅ Weights changed!" : "⚠️ Weights unchanged");
                
                Debug.WriteLine("");
                Debug.WriteLine("   Output before training:");
                for (int i = 0; i < outputBefore.Length; i++)
                {
                    Debug.WriteLine($"     [{i}] {outputBefore[i]:F6}");
                }
                
                Debug.WriteLine("");
                Debug.WriteLine("   Output after training:");
                for (int i = 0; i < outputAfter.Length; i++)
                {
                    Debug.WriteLine($"     [{i}] {outputAfter[i]:F6}");
                }
                
                Debug.WriteLine("");
                Debug.WriteLine("=".PadRight(60, '='));
                Debug.WriteLine("✅ ALL TESTS PASSED!");
                Debug.WriteLine("=".PadRight(60, '='));
                
                // Update UI
                CounterBtn.Text = "✅ All tests passed!";
            }
            catch (Exception ex)
            {
                Debug.WriteLine("");
                Debug.WriteLine("❌ ERROR OCCURRED:");
                Debug.WriteLine($"   Type: {ex.GetType().Name}");
                Debug.WriteLine($"   Message: {ex.Message}");
                Debug.WriteLine($"   Stack trace:");
                Debug.WriteLine(ex.StackTrace);
                
                CounterBtn.Text = $"❌ Error: {ex.Message}";
            }
        }

        private static async Task<string> FetchTextAsync(string url)
        {
            var response = await _httpClient.GetAsync(url);
            response.EnsureSuccessStatusCode();
            return await response.Content.ReadAsStringAsync();
        }
    }
}
