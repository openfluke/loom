using System;
using Welvet;

class QuickTest
{
    static void Main()
    {
        Console.WriteLine("Testing Welvet v0.0.9...");
        
        try
        {
            Loader.Initialize();
            Console.WriteLine("✅ Loader.Initialize() succeeded!");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"❌ Failed: {ex.Message}");
        }
    }
}
