using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Diagnostics;

namespace Welvet;

/// <summary>
/// Module initializer that runs before any code in the assembly.
/// This ensures the native library resolver is set up ASAP.
/// </summary>
internal static class ModuleInitializer
{
    [ModuleInitializer]
    internal static void Initialize()
    {
        Debug.WriteLine("=== Welvet Module Initializer ===");
        Debug.WriteLine($"Module initializer running...");
        Debug.WriteLine($"AppContext.BaseDirectory: {AppContext.BaseDirectory}");
        Debug.WriteLine($"OS: {RuntimeInformation.OSDescription}");
        Debug.WriteLine($"Architecture: {RuntimeInformation.ProcessArchitecture}");
        
        try
        {
            // Force static constructor to run
            var _ = typeof(NativeMethods);
            Debug.WriteLine("✅ NativeMethods static constructor triggered");
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"❌ Error in module initializer: {ex.Message}");
            Debug.WriteLine($"Stack trace: {ex.StackTrace}");
        }
        
        Debug.WriteLine("=== End Module Initializer ===");
    }
}
