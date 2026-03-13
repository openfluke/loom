using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.Json;

namespace Welvet;

/// <summary>
/// Response from tokenizer loading
/// </summary>
public class TokenizerLoadResult
{
    public bool Success { get; set; }
    public int VocabSize { get; set; }
    public string? Error { get; set; }
}

/// <summary>
/// Response from model loading
/// </summary>
public class TransformerLoadResult
{
    public bool Success { get; set; }
    public int NumLayers { get; set; }
    public int HiddenSize { get; set; }
    public int VocabSize { get; set; }
    public string? Error { get; set; }
}

/// <summary>
/// Response from text encoding
/// </summary>
public class EncodeResult
{
    public bool Success { get; set; }
    public int[]? Ids { get; set; }
    public string? Error { get; set; }
}

/// <summary>
/// Response from token decoding
/// </summary>
public class DecodeResult
{
    public bool Success { get; set; }
    public string? Text { get; set; }
    public string? Error { get; set; }
}

/// <summary>
/// Response from text generation
/// </summary>
public class GenerateResult
{
    public bool Success { get; set; }
    public string? GeneratedText { get; set; }
    public string? Error { get; set; }
}

/// <summary>
/// Response from next token generation
/// </summary>
public class NextTokenResult
{
    public bool Success { get; set; }
    public int Token { get; set; }
    public bool IsEos { get; set; }
    public string? Error { get; set; }
}

/// <summary>
/// High-level API for transformer inference using LOOM's C-ABI
/// </summary>
public static class Transformer
{
    /// <summary>
    /// Load tokenizer from JSON file
    /// </summary>
    public static TokenizerLoadResult LoadTokenizer(string tokenizerPath)
    {
        byte[] data = File.ReadAllBytes(tokenizerPath);
        return LoadTokenizerFromBytes(data);
    }

    /// <summary>
    /// Load tokenizer from JSON bytes
    /// </summary>
    public static TokenizerLoadResult LoadTokenizerFromBytes(byte[] data)
    {
        IntPtr dataPtr = Marshal.AllocHGlobal(data.Length);
        try
        {
            Marshal.Copy(data, 0, dataPtr, data.Length);
            IntPtr resultPtr = NativeMethods.LoadTokenizerFromBytes(dataPtr, data.Length);
            string json = NativeMethods.PtrToStringAndFree(resultPtr);
            
            return JsonSerializer.Deserialize<TokenizerLoadResult>(json, new JsonSerializerOptions 
            { 
                PropertyNameCaseInsensitive = true 
            }) ?? new TokenizerLoadResult { Success = false, Error = "Failed to parse response" };
        }
        finally
        {
            Marshal.FreeHGlobal(dataPtr);
        }
    }

    /// <summary>
    /// Load transformer model from config and weights files
    /// </summary>
    public static TransformerLoadResult LoadModel(string configPath, string weightsPath)
    {
        byte[] configData = File.ReadAllBytes(configPath);
        byte[] weightsData = File.ReadAllBytes(weightsPath);
        return LoadModelFromBytes(configData, weightsData);
    }

    /// <summary>
    /// Load transformer model from directory (expects config.json and model.safetensors)
    /// </summary>
    public static TransformerLoadResult LoadModelFromDirectory(string modelDir)
    {
        string configPath = Path.Combine(modelDir, "config.json");
        string weightsPath = Path.Combine(modelDir, "model.safetensors");
        return LoadModel(configPath, weightsPath);
    }

    /// <summary>
    /// Load transformer model from config and weights bytes
    /// </summary>
    public static TransformerLoadResult LoadModelFromBytes(byte[] configData, byte[] weightsData)
    {
        IntPtr configPtr = Marshal.AllocHGlobal(configData.Length);
        IntPtr weightsPtr = Marshal.AllocHGlobal(weightsData.Length);
        try
        {
            Marshal.Copy(configData, 0, configPtr, configData.Length);
            Marshal.Copy(weightsData, 0, weightsPtr, weightsData.Length);
            
            IntPtr resultPtr = NativeMethods.LoadTransformerFromBytes(
                configPtr, configData.Length,
                weightsPtr, weightsData.Length
            );
            
            string json = NativeMethods.PtrToStringAndFree(resultPtr);
            
            return JsonSerializer.Deserialize<TransformerLoadResult>(json, new JsonSerializerOptions 
            { 
                PropertyNameCaseInsensitive = true 
            }) ?? new TransformerLoadResult { Success = false, Error = "Failed to parse response" };
        }
        finally
        {
            Marshal.FreeHGlobal(configPtr);
            Marshal.FreeHGlobal(weightsPtr);
        }
    }

    /// <summary>
    /// Encode text to token IDs
    /// </summary>
    public static EncodeResult Encode(string text, bool addSpecialTokens = true)
    {
        IntPtr resultPtr = NativeMethods.EncodeText(text, addSpecialTokens);
        string json = NativeMethods.PtrToStringAndFree(resultPtr);
        
        return JsonSerializer.Deserialize<EncodeResult>(json, new JsonSerializerOptions 
        { 
            PropertyNameCaseInsensitive = true 
        }) ?? new EncodeResult { Success = false, Error = "Failed to parse response" };
    }

    /// <summary>
    /// Decode token IDs to text
    /// </summary>
    public static DecodeResult Decode(int[] tokenIds, bool skipSpecialTokens = true)
    {
        string idsJson = JsonSerializer.Serialize(tokenIds);
        IntPtr resultPtr = NativeMethods.DecodeTokens(idsJson, skipSpecialTokens);
        string json = NativeMethods.PtrToStringAndFree(resultPtr);
        
        return JsonSerializer.Deserialize<DecodeResult>(json, new JsonSerializerOptions 
        { 
            PropertyNameCaseInsensitive = true 
        }) ?? new DecodeResult { Success = false, Error = "Failed to parse response" };
    }

    /// <summary>
    /// Generate text from prompt (blocking, returns all text at once)
    /// </summary>
    public static GenerateResult Generate(string prompt, int maxTokens = 50, float temperature = 0.7f)
    {
        IntPtr resultPtr = NativeMethods.GenerateText(prompt, maxTokens, temperature);
        string json = NativeMethods.PtrToStringAndFree(resultPtr);
        
        return JsonSerializer.Deserialize<GenerateResult>(json, new JsonSerializerOptions 
        { 
            PropertyNameCaseInsensitive = true 
        }) ?? new GenerateResult { Success = false, Error = "Failed to parse response" };
    }

    /// <summary>
    /// Generate text token-by-token (streaming)
    /// </summary>
    public static IEnumerable<string> GenerateStream(string prompt, int maxTokens = 50, float temperature = 0.7f)
    {
        // Encode the prompt
        var encodeResult = Encode(prompt, addSpecialTokens: true);
        if (!encodeResult.Success || encodeResult.Ids == null)
        {
            yield break;
        }

        var tokens = encodeResult.Ids.ToList();

        // Generate tokens one at a time
        for (int i = 0; i < maxTokens; i++)
        {
            // Generate next token
            string tokensJson = JsonSerializer.Serialize(tokens);
            IntPtr resultPtr = NativeMethods.GenerateNextToken(tokensJson, temperature);
            string json = NativeMethods.PtrToStringAndFree(resultPtr);
            
            var result = JsonSerializer.Deserialize<NextTokenResult>(json, new JsonSerializerOptions 
            { 
                PropertyNameCaseInsensitive = true 
            });

            if (result == null || !result.Success)
            {
                yield break;
            }

            tokens.Add(result.Token);

            // Decode just this token
            var decodeResult = Decode(new[] { result.Token }, skipSpecialTokens: true);
            if (decodeResult.Success && decodeResult.Text != null)
            {
                yield return decodeResult.Text;
            }

            // Check for end of sequence
            if (result.IsEos)
            {
                yield break;
            }
        }
    }
}
