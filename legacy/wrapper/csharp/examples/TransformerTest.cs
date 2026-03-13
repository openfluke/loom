using System;
using System.IO;
using System.Linq;
using Welvet;

namespace Welvet.Examples;

/// <summary>
/// Test transformer inference capabilities
/// </summary>
public class TransformerTest
{
    public static void Main(string[] args)
    {
        if (args.Length < 1)
        {
            Console.WriteLine("Usage: dotnet run <model_directory>");
            Console.WriteLine("Example: dotnet run ../../models/SmolLM2-135M-Instruct");
            return;
        }

        string modelDir = args[0];
        Console.WriteLine("=== LOOM C# Transformer Test ===\n");

        try
        {
            // 1. Load tokenizer
            Console.WriteLine("1. Loading tokenizer...");
            string tokenizerPath = Path.Combine(modelDir, "tokenizer.json");
            var tokResult = Transformer.LoadTokenizer(tokenizerPath);
            
            if (!tokResult.Success)
            {
                Console.WriteLine($"   ✗ Error: {tokResult.Error}");
                return;
            }
            Console.WriteLine($"   ✓ Vocab size: {tokResult.VocabSize}\n");

            // 2. Load model
            Console.WriteLine("2. Loading transformer model...");
            var modelResult = Transformer.LoadModelFromDirectory(modelDir);
            
            if (!modelResult.Success)
            {
                Console.WriteLine($"   ✗ Error: {modelResult.Error}");
                return;
            }
            Console.WriteLine($"   ✓ Layers: {modelResult.NumLayers}");
            Console.WriteLine($"   ✓ Hidden size: {modelResult.HiddenSize}");
            Console.WriteLine($"   ✓ Vocab size: {modelResult.VocabSize}\n");

            // 3. Test encoding
            Console.WriteLine("3. Testing encoding...");
            string testText = "Once upon a time";
            var encodeResult = Transformer.Encode(testText, addSpecialTokens: true);
            
            if (!encodeResult.Success || encodeResult.Ids == null)
            {
                Console.WriteLine($"   ✗ Error: {encodeResult.Error}");
                return;
            }
            Console.WriteLine($"   Input: \"{testText}\"");
            Console.WriteLine($"   Token IDs: [{string.Join(", ", encodeResult.Ids)}]\n");

            // 4. Test decoding
            Console.WriteLine("4. Testing decoding...");
            var decodeResult = Transformer.Decode(encodeResult.Ids, skipSpecialTokens: true);
            
            if (!decodeResult.Success)
            {
                Console.WriteLine($"   ✗ Error: {decodeResult.Error}");
                return;
            }
            Console.WriteLine($"   Decoded: \"{decodeResult.Text}\"\n");

            // 5. Test streaming generation
            Console.WriteLine("5. Testing streaming generation...");
            string prompt = "The capital of France is";
            Console.WriteLine($"   Prompt: \"{prompt}\"");
            Console.Write("   Generated: \"");
            Console.Out.Flush();

            int tokenCount = 0;
            foreach (var token in Transformer.GenerateStream(prompt, maxTokens: 30, temperature: 0.7f))
            {
                Console.Write(token);
                Console.Out.Flush();
                tokenCount++;
            }
            Console.WriteLine($"\"\n   Tokens generated: {tokenCount}\n");

            Console.WriteLine("✓ All tests passed!");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"\n✗ Error: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
        }
    }
}
