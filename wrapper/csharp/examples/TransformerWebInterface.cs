using System;
using System.IO;
using System.Net;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Welvet;

namespace Welvet.Examples;

/// <summary>
/// HTTP server with streaming transformer inference
/// </summary>
public class TransformerWebInterface
{
    private static string modelDirectory = "";

    public static async Task Main(string[] args)
    {
        if (args.Length < 2)
        {
            Console.WriteLine("Usage: dotnet run <model_directory> <port>");
            Console.WriteLine("Example: dotnet run ../../models/SmolLM2-135M-Instruct 8080");
            return;
        }

        modelDirectory = args[0];
        int port = int.Parse(args[1]);

        Console.WriteLine("Loading model...");

        // Load tokenizer
        string tokenizerPath = Path.Combine(modelDirectory, "tokenizer.json");
        var tokResult = Transformer.LoadTokenizer(tokenizerPath);
        if (!tokResult.Success)
        {
            Console.WriteLine($"Failed to load tokenizer: {tokResult.Error}");
            return;
        }
        Console.WriteLine($"✓ Tokenizer loaded\n  Vocab: {tokResult.VocabSize}");

        // Load model
        var modelResult = Transformer.LoadModelFromDirectory(modelDirectory);
        if (!modelResult.Success)
        {
            Console.WriteLine($"Failed to load model: {modelResult.Error}");
            return;
        }
        Console.WriteLine($"✓ Transformer loaded\n  Vocab: {modelResult.VocabSize}\n  Hidden: {modelResult.HiddenSize}\n  Layers: {modelResult.NumLayers}");

        // Start HTTP server
        Console.WriteLine($"\n🚀 LOOM C# Transformer Web Interface");
        Console.WriteLine($"   Model: {Path.GetFileName(modelDirectory)}");
        Console.WriteLine($"   Server: http://localhost:{port}");
        Console.WriteLine($"   Backend: Welvet (NuGet package)\n");
        Console.WriteLine("Press Ctrl+C to stop\n");

        HttpListener listener = new HttpListener();
        listener.Prefixes.Add($"http://localhost:{port}/");
        listener.Start();

        while (true)
        {
            try
            {
                HttpListenerContext context = await listener.GetContextAsync();
                _ = Task.Run(() => HandleRequest(context));
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
            }
        }
    }

    private static async Task HandleRequest(HttpListenerContext context)
    {
        HttpListenerRequest request = context.Request;
        HttpListenerResponse response = context.Response;

        // Enable CORS
        response.Headers.Add("Access-Control-Allow-Origin", "*");
        response.Headers.Add("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        response.Headers.Add("Access-Control-Allow-Headers", "Content-Type");

        if (request.HttpMethod == "OPTIONS")
        {
            response.StatusCode = 204;
            response.Close();
            return;
        }

        try
        {
            string path = request.Url?.AbsolutePath ?? "/";
            Console.WriteLine($"{request.HttpMethod} {path}");

            if (path == "/health")
            {
                await HandleHealth(response);
            }
            else if (path == "/generate" && request.HttpMethod == "POST")
            {
                await HandleGenerate(request, response);
            }
            else if (path == "/inference.html" || path == "/")
            {
                await HandleStaticFile(response, "inference.html");
            }
            else
            {
                response.StatusCode = 404;
                byte[] buffer = Encoding.UTF8.GetBytes("Not Found");
                await response.OutputStream.WriteAsync(buffer, 0, buffer.Length);
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error handling request: {ex.Message}");
            response.StatusCode = 500;
            byte[] buffer = Encoding.UTF8.GetBytes($"{{\"error\": \"{ex.Message}\"}}");
            await response.OutputStream.WriteAsync(buffer, 0, buffer.Length);
        }
        finally
        {
            response.Close();
        }
    }

    private static async Task HandleHealth(HttpListenerResponse response)
    {
        response.ContentType = "application/json";
        string json = "{\"status\": \"ok\", \"model\": \"loaded\"}";
        byte[] buffer = Encoding.UTF8.GetBytes(json);
        await response.OutputStream.WriteAsync(buffer, 0, buffer.Length);
    }

    private static async Task HandleGenerate(HttpListenerRequest request, HttpListenerResponse response)
    {
        // Parse request body
        using StreamReader reader = new StreamReader(request.InputStream, request.ContentEncoding);
        string body = await reader.ReadToEndAsync();
        
        var requestData = JsonSerializer.Deserialize<GenerateRequest>(body);
        if (requestData == null || string.IsNullOrEmpty(requestData.Prompt))
        {
            response.StatusCode = 400;
            byte[] buffer = Encoding.UTF8.GetBytes("{\"error\": \"Missing prompt\"}");
            await response.OutputStream.WriteAsync(buffer, 0, buffer.Length);
            return;
        }

        // Set up SSE
        response.ContentType = "text/event-stream";
        response.Headers.Add("Cache-Control", "no-cache");
        response.Headers.Add("Connection", "keep-alive");

        Console.WriteLine($"Generating from prompt: \"{requestData.Prompt}\"");
        Console.Write("Tokens: ");

        try
        {
            int tokenCount = 0;
            foreach (var token in Transformer.GenerateStream(
                requestData.Prompt, 
                requestData.MaxTokens, 
                requestData.Temperature))
            {
                // Send SSE event
                string eventData = $"data: {JsonSerializer.Serialize(new { token })}\n\n";
                byte[] buffer = Encoding.UTF8.GetBytes(eventData);
                await response.OutputStream.WriteAsync(buffer, 0, buffer.Length);
                await response.OutputStream.FlushAsync();
                
                Console.Write(token);
                tokenCount++;
            }

            // Send done event
            string doneData = "data: {\"done\": true}\n\n";
            byte[] doneBuffer = Encoding.UTF8.GetBytes(doneData);
            await response.OutputStream.WriteAsync(doneBuffer, 0, doneBuffer.Length);
            await response.OutputStream.FlushAsync();

            Console.WriteLine($"\nGenerated {tokenCount} tokens\n");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"\nError during generation: {ex.Message}\n");
            string errorData = $"data: {{\"error\": \"{ex.Message}\"}}\n\n";
            byte[] errorBuffer = Encoding.UTF8.GetBytes(errorData);
            await response.OutputStream.WriteAsync(errorBuffer, 0, errorBuffer.Length);
        }
    }

    private static async Task HandleStaticFile(HttpListenerResponse response, string filename)
    {
        // Try to find inference.html in current directory or parent directories
        string[] searchPaths = new[]
        {
            filename,
            Path.Combine("../../cabi", filename),
            Path.Combine("../cabi", filename),
            Path.Combine("../../python/examples", filename)
        };

        string? filePath = null;
        foreach (var path in searchPaths)
        {
            if (File.Exists(path))
            {
                filePath = path;
                break;
            }
        }

        if (filePath == null)
        {
            response.StatusCode = 404;
            byte[] buffer = Encoding.UTF8.GetBytes($"File not found: {filename}");
            await response.OutputStream.WriteAsync(buffer, 0, buffer.Length);
            return;
        }

        response.ContentType = "text/html";
        byte[] content = File.ReadAllBytes(filePath);
        await response.OutputStream.WriteAsync(content, 0, content.Length);
    }
}

public class GenerateRequest
{
    public string Prompt { get; set; } = "";
    public int MaxTokens { get; set; } = 50;
    public float Temperature { get; set; } = 0.7f;
}
