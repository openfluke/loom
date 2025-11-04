#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <stdbool.h>

// Forward declarations of LOOM C ABI functions
extern char* Loom_NewNetwork(int inputSize, int gridRows, int gridCols, int layersPerCell, bool useGPU);
extern char* Loom_InitDenseLayer(int inputSize, int outputSize, int activation);
extern char* Loom_SetLayer(int64_t handle, int row, int col, int layer, char* configJSON);
extern char* Loom_Call(int64_t handle, char* method, char* argsJSON);
extern char* Loom_ListMethods(int64_t handle);
extern char* Loom_GetInfo(int64_t handle);
extern char* Loom_SaveModel(int64_t handle, char* modelID);
extern char* Loom_LoadModel(char* jsonString, char* modelID);
extern void Loom_Free(int64_t handle);
extern void Loom_FreeCString(char* p);
extern char* Loom_GetVersion();

// Helper to extract integer from JSON
int64_t extractHandle(const char* json) {
    const char* handleKey = "\"handle\":";
    const char* pos = strstr(json, handleKey);
    if (!pos) return -1;
    return atoll(pos + strlen(handleKey));
}

// Helper to measure time in milliseconds
double getTime() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

int main() {
    printf("=== LOOM C ABI Simple Benchmark ===\n");
    
    // Get version
    char* version = Loom_GetVersion();
    printf("Version: %s\n\n", version);
    Loom_FreeCString(version);
    
    // Test configuration
    int inputSize = 784;    // MNIST-like input
    int gridRows = 2;
    int gridCols = 1;
    int layersPerCell = 1;
    int iterations = 100;
    
    printf("Network: %dx%dx%d grid, input_size=%d\n", gridRows, gridCols, layersPerCell, inputSize);
    printf("Iterations: %d\n\n", iterations);
    
    // ===== CPU Test =====
    printf("--- CPU Test ---\n");
    double cpuCreateStart = getTime();
    char* cpuNetResult = Loom_NewNetwork(inputSize, gridRows, gridCols, layersPerCell, false);
    double cpuCreateEnd = getTime();
    
    int64_t cpuHandle = extractHandle(cpuNetResult);
    if (cpuHandle < 0) {
        printf("Error: Failed to create CPU network: %s\n", cpuNetResult);
        Loom_FreeCString(cpuNetResult);
        return 1;
    }
    
    printf("CPU Network created in %.2f ms (handle: %ld)\n", cpuCreateEnd - cpuCreateStart, cpuHandle);
    Loom_FreeCString(cpuNetResult);
    
    // Initialize layers for CPU network
    char* layer0Config = Loom_InitDenseLayer(inputSize, 392, 1); // ReLU
    Loom_SetLayer(cpuHandle, 0, 0, 0, layer0Config);
    Loom_FreeCString(layer0Config);
    
    char* layer1Config = Loom_InitDenseLayer(392, 10, 0); // Linear
    Loom_SetLayer(cpuHandle, 1, 0, 0, layer1Config);
    Loom_FreeCString(layer1Config);
    
    printf("Layers initialized\n");
    
    // Get info
    char* cpuInfo = Loom_GetInfo(cpuHandle);
    printf("CPU Info: %s\n\n", cpuInfo);
    Loom_FreeCString(cpuInfo);
    
    // Prepare input (784 zeros)
    char inputJSON[4096];
    strcpy(inputJSON, "[[");
    for (int i = 0; i < inputSize; i++) {
        char buf[16];
        snprintf(buf, sizeof(buf), "%.1f%s", 0.0, i < inputSize-1 ? "," : "");
        strcat(inputJSON, buf);
    }
    strcat(inputJSON, "]]");
    
    // CPU Forward Pass Benchmark
    printf("Running CPU forward passes...\n");
    double cpuForwardStart = getTime();
    for (int i = 0; i < iterations; i++) {
        char* output = Loom_Call(cpuHandle, "ForwardCPU", inputJSON);
        Loom_FreeCString(output);
    }
    double cpuForwardEnd = getTime();
    double cpuTotal = cpuForwardEnd - cpuForwardStart;
    double cpuAvg = cpuTotal / iterations;
    
    printf("CPU Forward: %d iterations in %.2f ms (avg: %.4f ms/iter)\n", iterations, cpuTotal, cpuAvg);
    
    // Single forward pass to show output
    char* cpuOutput = Loom_Call(cpuHandle, "ForwardCPU", inputJSON);
    printf("Sample CPU Output: %.100s...\n\n", cpuOutput);
    Loom_FreeCString(cpuOutput);
    
    // ===== GPU Test =====
    printf("--- GPU Test ---\n");
    double gpuCreateStart = getTime();
    char* gpuNetResult = Loom_NewNetwork(inputSize, gridRows, gridCols, layersPerCell, true);
    double gpuCreateEnd = getTime();
    
    int64_t gpuHandle = extractHandle(gpuNetResult);
    if (gpuHandle < 0) {
        printf("Error: Failed to create GPU network: %s\n", gpuNetResult);
        Loom_FreeCString(gpuNetResult);
        // Continue with CPU only
        Loom_Free(cpuHandle);
        return 0;
    }
    
    printf("GPU Network created in %.2f ms (handle: %ld)\n", gpuCreateEnd - gpuCreateStart, gpuHandle);
    
    // Check if GPU actually initialized
    if (strstr(gpuNetResult, "\"gpu\":false") || strstr(gpuNetResult, "\"gpu\":0")) {
        printf("GPU initialization failed, skipping GPU benchmark\n");
        printf("GPU Result: %s\n", gpuNetResult);
        Loom_FreeCString(gpuNetResult);
        Loom_Free(cpuHandle);
        Loom_Free(gpuHandle);
        return 0;
    }
    
    Loom_FreeCString(gpuNetResult);
    
    // Initialize layers for GPU network (same as CPU)
    layer0Config = Loom_InitDenseLayer(inputSize, 392, 1);
    Loom_SetLayer(gpuHandle, 0, 0, 0, layer0Config);
    Loom_FreeCString(layer0Config);
    
    layer1Config = Loom_InitDenseLayer(392, 10, 0);
    Loom_SetLayer(gpuHandle, 1, 0, 0, layer1Config);
    Loom_FreeCString(layer1Config);
    
    printf("Layers initialized\n");
    
    // Get info
    char* gpuInfo = Loom_GetInfo(gpuHandle);
    printf("GPU Info: %s\n\n", gpuInfo);
    Loom_FreeCString(gpuInfo);
    
    // GPU Forward Pass Benchmark
    printf("Running GPU forward passes...\n");
    double gpuForwardStart = getTime();
    for (int i = 0; i < iterations; i++) {
        char* output = Loom_Call(gpuHandle, "ForwardGPU", inputJSON);
        Loom_FreeCString(output);
    }
    double gpuForwardEnd = getTime();
    double gpuTotal = gpuForwardEnd - gpuForwardStart;
    double gpuAvg = gpuTotal / iterations;
    
    printf("GPU Forward: %d iterations in %.2f ms (avg: %.4f ms/iter)\n", iterations, gpuTotal, gpuAvg);
    
    // Single forward pass to show output
    char* gpuOutput = Loom_Call(gpuHandle, "ForwardGPU", inputJSON);
    printf("Sample GPU Output: %.100s...\n\n", gpuOutput);
    Loom_FreeCString(gpuOutput);
    
    // ===== Comparison =====
    printf("=== Results ===\n");
    printf("CPU Avg: %.4f ms/iter\n", cpuAvg);
    printf("GPU Avg: %.4f ms/iter\n", gpuAvg);
    double speedup = cpuAvg / gpuAvg;
    printf("Speedup: %.2fx %s\n", speedup > 1.0 ? speedup : 1.0/speedup, speedup > 1.0 ? "(GPU faster)" : "(CPU faster)");
    
    // Test method listing
    printf("\n=== Available Methods ===\n");
    char* methods = Loom_ListMethods(cpuHandle);
    printf("%s\n", methods);
    Loom_FreeCString(methods);
    
    // Test model serialization
    printf("\n=== Model Serialization ===\n");
    char* modelJSON = Loom_SaveModel(cpuHandle, "test_model");
    printf("Model JSON (first 200 chars): %.200s...\n", modelJSON);
    printf("Model size: %zu bytes\n", strlen(modelJSON));
    
    // Test loading
    char* loadResult = Loom_LoadModel(modelJSON, "test_model");
    int64_t loadedHandle = extractHandle(loadResult);
    printf("Loaded model handle: %ld\n", loadedHandle);
    Loom_FreeCString(loadResult);
    Loom_FreeCString(modelJSON);
    
    if (loadedHandle >= 0) {
        Loom_Free(loadedHandle);
    }
    
    // Cleanup
    printf("\n=== Cleanup ===\n");
    Loom_Free(cpuHandle);
    Loom_Free(gpuHandle);
    printf("Handles freed\n");
    
    return 0;
}
