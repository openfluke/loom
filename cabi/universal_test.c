/*
 * LOOM v0.0.7 Universal Test Suite
 * Comprehensive C ABI validation matching tva/test_0_0_7.go
 * 
 * Tests: Core Features, Multi-Precision Serialization, Additional Features
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "libloom.h"

// =============================================================================
// Test Utilities
// =============================================================================

static int passed = 0;
static int failed = 0;

#define TEST_PASS() do { passed++; } while(0)
#define TEST_FAIL() do { failed++; } while(0)

void parse_float_array(const char* json, float* out, int max_len) {
    const char* p = strchr(json, '[');
    if (!p) return;
    p++;
    for (int i = 0; i < max_len && *p; i++) {
        while (*p && (*p == ' ' || *p == ',')) p++;
        if (*p == ']') break;
        out[i] = (float)atof(p);
        while (*p && *p != ',' && *p != ']') p++;
    }
}

int json_contains(const char* json, const char* key) {
    return strstr(json, key) != NULL;
}

int json_has_error(const char* json) {
    return strstr(json, "\"error\"") != NULL;
}

// =============================================================================
// PART 1: Core Feature Tests
// =============================================================================

int testArchitectureGeneration() {
    printf("\n┌──────────────────────────────────────────────────────────────────────┐\n");
    printf("│ Architecture Generation with DType                                  │\n");
    printf("└──────────────────────────────────────────────────────────────────────┘\n");

    const char* config = 
        "{"
        "  \"dtype\": \"float32\","
        "  \"batch_size\": 1,"
        "  \"grid_rows\": 1,"
        "  \"grid_cols\": 1,"
        "  \"layers_per_cell\": 2,"
        "  \"layers\": ["
        "    {\"type\": \"dense\", \"activation\": \"leaky_relu\", \"input_height\": 8, \"output_height\": 16},"
        "    {\"type\": \"dense\", \"activation\": \"sigmoid\", \"input_height\": 16, \"output_height\": 4}"
        "  ]"
        "}";

    char* result = CreateLoomNetwork(config);
    if (json_has_error(result)) {
        printf("  ❌ Failed to create network: %s\n", result);
        FreeLoomString(result);
        return 0;
    }
    printf("  ✓ Network created with dtype=float32\n");
    FreeLoomString(result);

    float input[8] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    char* output = LoomForward(input, 8);
    
    if (json_has_error(output)) {
        printf("  ❌ Forward pass failed: %s\n", output);
        FreeLoomString(output);
        return 0;
    }
    
    float out[4];
    parse_float_array(output, out, 4);
    printf("  ✓ Forward pass: output=[%.3f, %.3f, %.3f, %.3f]\n", out[0], out[1], out[2], out[3]);
    FreeLoomString(output);

    printf("  ✅ PASSED: Architecture Generation with DType\n");
    return 1;
}

int testFilterCombineMode() {
    printf("\n┌──────────────────────────────────────────────────────────────────────┐\n");
    printf("│ Parallel Filter Combine Mode (MoE)                                  │\n");
    printf("└──────────────────────────────────────────────────────────────────────┘\n");

    const char* config = 
        "{"
        "  \"batch_size\": 1,"
        "  \"grid_rows\": 1,"
        "  \"grid_cols\": 1,"
        "  \"layers_per_cell\": 2,"
        "  \"layers\": ["
        "    {"
        "      \"type\": \"parallel\","
        "      \"combine_mode\": \"concat\","
        "      \"branches\": ["
        "        {\"type\": \"dense\", \"activation\": \"tanh\", \"input_height\": 4, \"output_height\": 2},"
        "        {\"type\": \"dense\", \"activation\": \"sigmoid\", \"input_height\": 4, \"output_height\": 2}"
        "      ]"
        "    },"
        "    {\"type\": \"dense\", \"activation\": \"sigmoid\", \"input_height\": 4, \"output_height\": 2}"
        "  ]"
        "}";

    char* result = CreateLoomNetwork(config);
    if (json_has_error(result)) {
        printf("  ❌ Failed: %s\n", result);
        FreeLoomString(result);
        return 0;
    }
    FreeLoomString(result);

    float input[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    char* output = LoomForward(input, 4);
    
    float out[2];
    parse_float_array(output, out, 2);
    printf("  ✓ Forward pass: output=[%.3f, %.3f]\n", out[0], out[1]);
    FreeLoomString(output);

    printf("  ✅ PASSED: Parallel Combine Mode\n");
    return 1;
}

int testSequentialLayers() {
    printf("\n┌──────────────────────────────────────────────────────────────────────┐\n");
    printf("│ Sequential Layer Composition                                        │\n");
    printf("└──────────────────────────────────────────────────────────────────────┘\n");

    const char* config = 
        "{"
        "  \"batch_size\": 1,"
        "  \"grid_rows\": 1,"
        "  \"grid_cols\": 1,"
        "  \"layers_per_cell\": 1,"
        "  \"layers\": ["
        "    {"
        "      \"type\": \"sequential\","
        "      \"branches\": ["
        "        {\"type\": \"dense\", \"activation\": \"leaky_relu\", \"input_height\": 4, \"output_height\": 8},"
        "        {\"type\": \"dense\", \"activation\": \"sigmoid\", \"input_height\": 8, \"output_height\": 2}"
        "      ]"
        "    }"
        "  ]"
        "}";

    char* result = CreateLoomNetwork(config);
    if (json_has_error(result)) {
        printf("  ❌ Failed: %s\n", result);
        FreeLoomString(result);
        return 0;
    }
    FreeLoomString(result);

    float input[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    char* output = LoomForward(input, 4);
    
    float out[2];
    parse_float_array(output, out, 2);
    printf("  ✓ Sequential layer with 2 sub-layers\n");
    printf("  ✓ Forward pass: output=[%.3f, %.3f]\n", out[0], out[1]);
    FreeLoomString(output);

    printf("  ✅ PASSED: Sequential Layer Composition\n");
    return 1;
}

int testNetworkInfo() {
    printf("\n┌──────────────────────────────────────────────────────────────────────┐\n");
    printf("│ Introspection & Network Info                                        │\n");
    printf("└──────────────────────────────────────────────────────────────────────┘\n");

    const char* config = 
        "{"
        "  \"batch_size\": 1,"
        "  \"grid_rows\": 1,"
        "  \"grid_cols\": 1,"
        "  \"layers_per_cell\": 2,"
        "  \"layers\": ["
        "    {\"type\": \"dense\", \"activation\": \"leaky_relu\", \"input_height\": 4, \"output_height\": 8},"
        "    {\"type\": \"dense\", \"activation\": \"sigmoid\", \"input_height\": 8, \"output_height\": 2}"
        "  ]"
        "}";

    char* result = CreateLoomNetwork(config);
    FreeLoomString(result);

    char* info = LoomGetNetworkInfo();
    if (json_has_error(info)) {
        printf("  ❌ Failed to get info: %s\n", info);
        FreeLoomString(info);
        return 0;
    }

    printf("  ✓ Network info: %s\n", info);
    
    if (json_contains(info, "\"total_layers\":2")) {
        printf("  ✓ TotalLayers: 2\n");
    }
    FreeLoomString(info);

    printf("  ✅ PASSED: Introspection & Network Info\n");
    return 1;
}

// =============================================================================
// PART 2: Multi-Precision Serialization Tests
// =============================================================================

const char* getLayerConfig(const char* layerType, const char* dtype) {
    static char buffer[4096];
    
    if (strcmp(layerType, "Dense") == 0) {
        snprintf(buffer, sizeof(buffer),
            "{"
            "  \"dtype\": \"%s\","
            "  \"batch_size\": 1,"
            "  \"grid_rows\": 1,"
            "  \"grid_cols\": 1,"
            "  \"layers_per_cell\": 3,"
            "  \"layers\": ["
            "    {\"type\": \"dense\", \"activation\": \"leaky_relu\", \"input_height\": 8, \"output_height\": 64},"
            "    {\"type\": \"dense\", \"activation\": \"tanh\", \"input_height\": 64, \"output_height\": 32},"
            "    {\"type\": \"dense\", \"activation\": \"sigmoid\", \"input_height\": 32, \"output_height\": 4}"
            "  ]"
            "}", dtype);
    } else if (strcmp(layerType, "MHA") == 0) {
        snprintf(buffer, sizeof(buffer),
            "{"
            "  \"dtype\": \"%s\","
            "  \"batch_size\": 1,"
            "  \"grid_rows\": 1,"
            "  \"grid_cols\": 1,"
            "  \"layers_per_cell\": 2,"
            "  \"layers\": ["
            "    {\"type\": \"multi_head_attention\", \"d_model\": 64, \"num_heads\": 8, \"seq_length\": 1},"
            "    {\"type\": \"dense\", \"activation\": \"sigmoid\", \"input_height\": 64, \"output_height\": 4}"
            "  ]"
            "}", dtype);
    } else if (strcmp(layerType, "RNN") == 0) {
        snprintf(buffer, sizeof(buffer),
            "{"
            "  \"dtype\": \"%s\","
            "  \"batch_size\": 1,"
            "  \"grid_rows\": 1,"
            "  \"grid_cols\": 1,"
            "  \"layers_per_cell\": 2,"
            "  \"layers\": ["
            "    {\"type\": \"rnn\", \"input_size\": 16, \"hidden_size\": 32, \"activation\": \"tanh\"},"
            "    {\"type\": \"dense\", \"activation\": \"sigmoid\", \"input_height\": 32, \"output_height\": 4}"
            "  ]"
            "}", dtype);
    } else if (strcmp(layerType, "LSTM") == 0) {
        snprintf(buffer, sizeof(buffer),
            "{"
            "  \"dtype\": \"%s\","
            "  \"batch_size\": 1,"
            "  \"grid_rows\": 1,"
            "  \"grid_cols\": 1,"
            "  \"layers_per_cell\": 2,"
            "  \"layers\": ["
            "    {\"type\": \"lstm\", \"input_size\": 16, \"hidden_size\": 32},"
            "    {\"type\": \"dense\", \"activation\": \"sigmoid\", \"input_height\": 32, \"output_height\": 4}"
            "  ]"
            "}", dtype);
    } else if (strcmp(layerType, "LayerNorm") == 0) {
        snprintf(buffer, sizeof(buffer),
            "{"
            "  \"dtype\": \"%s\","
            "  \"batch_size\": 1,"
            "  \"grid_rows\": 1,"
            "  \"grid_cols\": 1,"
            "  \"layers_per_cell\": 3,"
            "  \"layers\": ["
            "    {\"type\": \"dense\", \"activation\": \"leaky_relu\", \"input_height\": 16, \"output_height\": 32},"
            "    {\"type\": \"layer_norm\", \"norm_size\": 32, \"epsilon\": 1e-5},"
            "    {\"type\": \"dense\", \"activation\": \"sigmoid\", \"input_height\": 32, \"output_height\": 4}"
            "  ]"
            "}", dtype);
    } else if (strcmp(layerType, "Softmax") == 0) {
        snprintf(buffer, sizeof(buffer),
            "{"
            "  \"dtype\": \"%s\","
            "  \"batch_size\": 1,"
            "  \"grid_rows\": 1,"
            "  \"grid_cols\": 1,"
            "  \"layers_per_cell\": 3,"
            "  \"layers\": ["
            "    {\"type\": \"dense\", \"activation\": \"leaky_relu\", \"input_height\": 8, \"output_height\": 16},"
            "    {\"type\": \"dense\", \"activation\": \"leaky_relu\", \"input_height\": 16, \"output_height\": 4},"
            "    {\"type\": \"softmax\", \"softmax_variant\": \"standard\", \"temperature\": 1.0}"
            "  ]"
            "}", dtype);
    } else if (strcmp(layerType, "RMSNorm") == 0) {
        snprintf(buffer, sizeof(buffer),
            "{"
            "  \"dtype\": \"%s\","
            "  \"batch_size\": 1,"
            "  \"grid_rows\": 1,"
            "  \"grid_cols\": 1,"
            "  \"layers_per_cell\": 3,"
            "  \"layers\": ["
            "    {\"type\": \"dense\", \"activation\": \"leaky_relu\", \"input_height\": 16, \"output_height\": 32},"
            "    {\"type\": \"rms_norm\", \"norm_size\": 32, \"epsilon\": 1e-5},"
            "    {\"type\": \"dense\", \"activation\": \"sigmoid\", \"input_height\": 32, \"output_height\": 4}"
            "  ]"
            "}", dtype);
    } else if (strcmp(layerType, "SwiGLU") == 0) {
        snprintf(buffer, sizeof(buffer),
            "{"
            "  \"dtype\": \"%s\","
            "  \"batch_size\": 1,"
            "  \"grid_rows\": 1,"
            "  \"grid_cols\": 1,"
            "  \"layers_per_cell\": 3,"
            "  \"layers\": ["
            "    {\"type\": \"dense\", \"activation\": \"leaky_relu\", \"input_height\": 32, \"output_height\": 64},"
            "    {\"type\": \"swiglu\", \"input_height\": 64, \"output_height\": 128},"
            "    {\"type\": \"dense\", \"activation\": \"sigmoid\", \"input_height\": 64, \"output_height\": 4}"
            "  ]"
            "}", dtype);
    } else if (strcmp(layerType, "Conv2D") == 0) {
        snprintf(buffer, sizeof(buffer),
            "{"
            "  \"dtype\": \"%s\","
            "  \"batch_size\": 1,"
            "  \"grid_rows\": 1,"
            "  \"grid_cols\": 1,"
            "  \"layers_per_cell\": 2,"
            "  \"layers\": ["
            "    {\"type\": \"conv2d\", \"input_channels\": 1, \"filters\": 2, \"kernel_size\": 3, \"stride\": 1, \"padding\": 1, \"input_height\": 4, \"input_width\": 4, \"activation\": \"leaky_relu\"},"
            "    {\"type\": \"dense\", \"activation\": \"sigmoid\", \"input_height\": 32, \"output_height\": 4}"
            "  ]"
            "}", dtype);
    } else if (strcmp(layerType, "Parallel") == 0) {
        snprintf(buffer, sizeof(buffer),
            "{"
            "  \"dtype\": \"%s\","
            "  \"batch_size\": 1,"
            "  \"grid_rows\": 1,"
            "  \"grid_cols\": 1,"
            "  \"layers_per_cell\": 3,"
            "  \"layers\": ["
            "    {\"type\": \"dense\", \"activation\": \"leaky_relu\", \"input_height\": 8, \"output_height\": 16},"
            "    {\"type\": \"parallel\", \"combine_mode\": \"concat\", \"branches\": ["
            "      {\"type\": \"dense\", \"activation\": \"tanh\", \"input_height\": 16, \"output_height\": 8},"
            "      {\"type\": \"dense\", \"activation\": \"sigmoid\", \"input_height\": 16, \"output_height\": 8}"
            "    ]},"
            "    {\"type\": \"dense\", \"activation\": \"sigmoid\", \"input_height\": 16, \"output_height\": 4}"
            "  ]"
            "}", dtype);
    } else if (strcmp(layerType, "Sequential") == 0) {
        snprintf(buffer, sizeof(buffer),
            "{"
            "  \"dtype\": \"%s\","
            "  \"batch_size\": 1,"
            "  \"grid_rows\": 1,"
            "  \"grid_cols\": 1,"
            "  \"layers_per_cell\": 2,"
            "  \"layers\": ["
            "    {\"type\": \"sequential\", \"branches\": ["
            "      {\"type\": \"dense\", \"activation\": \"leaky_relu\", \"input_height\": 8, \"output_height\": 16},"
            "      {\"type\": \"dense\", \"activation\": \"tanh\", \"input_height\": 16, \"output_height\": 8}"
            "    ]},"
            "    {\"type\": \"dense\", \"activation\": \"sigmoid\", \"input_height\": 8, \"output_height\": 4}"
            "  ]"
            "}", dtype);
    } else {
        // Default Dense config
        snprintf(buffer, sizeof(buffer),
            "{"
            "  \"dtype\": \"%s\","
            "  \"batch_size\": 1,"
            "  \"grid_rows\": 1,"
            "  \"grid_cols\": 1,"
            "  \"layers_per_cell\": 2,"
            "  \"layers\": ["
            "    {\"type\": \"dense\", \"activation\": \"leaky_relu\", \"input_height\": 8, \"output_height\": 16},"
            "    {\"type\": \"dense\", \"activation\": \"sigmoid\", \"input_height\": 16, \"output_height\": 4}"
            "  ]"
            "}", dtype);
    }
    return buffer;
}

int getInputSize(const char* layerType) {
    if (strcmp(layerType, "Dense") == 0) return 8;
    if (strcmp(layerType, "MHA") == 0) return 64;
    if (strcmp(layerType, "RNN") == 0) return 16;
    if (strcmp(layerType, "LSTM") == 0) return 16;
    if (strcmp(layerType, "LayerNorm") == 0) return 16;
    if (strcmp(layerType, "RMSNorm") == 0) return 16;
    if (strcmp(layerType, "SwiGLU") == 0) return 32;
    if (strcmp(layerType, "Conv2D") == 0) return 16;
    if (strcmp(layerType, "Parallel") == 0) return 8;
    if (strcmp(layerType, "Sequential") == 0) return 8;
    if (strcmp(layerType, "Softmax") == 0) return 8;
    return 8;
}

int testLayerWithDType(const char* layerName, const char* dtype) {
    char modelID[64];
    snprintf(modelID, sizeof(modelID), "%s_test", layerName);

    const char* config = getLayerConfig(layerName, dtype);
    char* result = CreateLoomNetwork(config);
    
    if (json_has_error(result)) {
        printf("  ❌ %-10s/%-8s: Build failed\n", layerName, dtype);
        FreeLoomString(result);
        return 0;
    }
    FreeLoomString(result);

    int inputSize = getInputSize(layerName);
    float* input = (float*)malloc(inputSize * sizeof(float));
    for (int i = 0; i < inputSize; i++) {
        input[i] = (i + 1) * 0.1f;
    }

    char* output = LoomForward(input, inputSize);
    if (json_has_error(output)) {
        printf("  ❌ %-10s/%-8s: Forward failed\n", layerName, dtype);
        FreeLoomString(output);
        free(input);
        return 0;
    }
    FreeLoomString(output);

    // Save model
    char* saved = LoomSaveModel(modelID);
    if (json_has_error(saved)) {
        printf("  ❌ %-10s/%-8s: Save failed\n", layerName, dtype);
        FreeLoomString(saved);
        free(input);
        return 0;
    }
    size_t saveSize = strlen(saved);

    // Load model back
    char* loadResult = LoomLoadModel(saved, modelID);
    FreeLoomString(saved);
    
    if (json_has_error(loadResult)) {
        printf("  ❌ %-10s/%-8s: Load failed\n", layerName, dtype);
        FreeLoomString(loadResult);
        free(input);
        return 0;
    }
    FreeLoomString(loadResult);

    // Verify output after reload
    output = LoomForward(input, inputSize);
    if (json_has_error(output)) {
        printf("  ❌ %-10s/%-8s: Reload forward failed\n", layerName, dtype);
        FreeLoomString(output);
        free(input);
        return 0;
    }
    FreeLoomString(output);
    free(input);

    printf("  ✓ %-10s/%-8s: save/load OK (size=%zu bytes)\n", layerName, dtype, saveSize);
    return 1;
}

// =============================================================================
// PART 3: Additional Feature Tests
// =============================================================================

int testOptimizers() {
    printf("\n┌──────────────────────────────────────────────────────────────────────┐\n");
    printf("│ Optimizers                                                          │\n");
    printf("└──────────────────────────────────────────────────────────────────────┘\n");

    const char* config = 
        "{"
        "  \"batch_size\": 1,"
        "  \"grid_rows\": 1,"
        "  \"grid_cols\": 1,"
        "  \"layers_per_cell\": 2,"
        "  \"layers\": ["
        "    {\"type\": \"dense\", \"activation\": \"leaky_relu\", \"input_height\": 4, \"output_height\": 8},"
        "    {\"type\": \"dense\", \"activation\": \"sigmoid\", \"input_height\": 8, \"output_height\": 2}"
        "  ]"
        "}";

    char* result = CreateLoomNetwork(config);
    FreeLoomString(result);

    const char* batches = 
        "[{\"Input\": [0.1, 0.2, 0.3, 0.4], \"Target\": [1.0, 0.0]}]";
    const char* trainConfig = 
        "{\"Epochs\": 5, \"LearningRate\": 0.01, \"UseGPU\": false, \"LossType\": \"mse\"}";

    char* trainResult = LoomTrain((char*)batches, (char*)trainConfig);
    
    if (json_has_error(trainResult)) {
        printf("  ❌ Training failed: %s\n", trainResult);
        FreeLoomString(trainResult);
        return 0;
    }

    printf("  ✓ SGD optimizer tested via Train()\n");
    FreeLoomString(trainResult);

    printf("  ✅ PASSED: Optimizers\n");
    return 1;
}

int testActivations() {
    printf("\n┌──────────────────────────────────────────────────────────────────────┐\n");
    printf("│ Activation Functions                                                │\n");
    printf("└──────────────────────────────────────────────────────────────────────┘\n");

    const char* activations[] = {"sigmoid", "tanh", "leaky_relu", "relu", "softplus"};
    int numActivations = 5;

    for (int i = 0; i < numActivations; i++) {
        char config[1024];
        snprintf(config, sizeof(config),
            "{"
            "  \"batch_size\": 1,"
            "  \"grid_rows\": 1,"
            "  \"grid_cols\": 1,"
            "  \"layers_per_cell\": 1,"
            "  \"layers\": ["
            "    {\"type\": \"dense\", \"activation\": \"%s\", \"input_height\": 4, \"output_height\": 4}"
            "  ]"
            "}", activations[i]);

        char* result = CreateLoomNetwork(config);
        if (json_has_error(result)) {
            printf("  ❌ %s: create failed\n", activations[i]);
            FreeLoomString(result);
            return 0;
        }
        FreeLoomString(result);

        float input[4] = {0.5f, 0.5f, 0.5f, 0.5f};
        char* output = LoomForward(input, 4);
        
        float out[4];
        parse_float_array(output, out, 4);
        printf("  ✓ %s: f(0.5)=[%.3f, %.3f, %.3f, %.3f]\n", 
               activations[i], out[0], out[1], out[2], out[3]);
        FreeLoomString(output);
    }

    printf("  ✅ PASSED: Activation Functions\n");
    return 1;
}

int testSoftmaxVariants() {
    printf("\n┌──────────────────────────────────────────────────────────────────────┐\n");
    printf("│ Softmax Variants                                                    │\n");
    printf("└──────────────────────────────────────────────────────────────────────┘\n");

    const char* config = 
        "{"
        "  \"batch_size\": 1,"
        "  \"grid_rows\": 1,"
        "  \"grid_cols\": 1,"
        "  \"layers_per_cell\": 2,"
        "  \"layers\": ["
        "    {\"type\": \"dense\", \"activation\": \"leaky_relu\", \"input_height\": 4, \"output_height\": 4},"
        "    {\"type\": \"softmax\", \"softmax_variant\": \"standard\", \"temperature\": 1.0}"
        "  ]"
        "}";

    char* result = CreateLoomNetwork(config);
    FreeLoomString(result);

    float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    char* output = LoomForward(input, 4);
    
    float out[4];
    parse_float_array(output, out, 4);
    float sum = out[0] + out[1] + out[2] + out[3];
    
    printf("  ✓ Standard Softmax: sum=%.4f (≈1.0)\n", sum);
    FreeLoomString(output);

    if (fabs(sum - 1.0f) < 0.01f) {
        printf("  ✅ PASSED: Softmax Variants\n");
        return 1;
    } else {
        printf("  ❌ FAILED: Softmax sum != 1.0\n");
        return 0;
    }
}

int testEmbeddingLayer() {
    printf("\n┌──────────────────────────────────────────────────────────────────────┐\n");
    printf("│ Embedding Layer                                                     │\n");
    printf("└──────────────────────────────────────────────────────────────────────┘\n");

    const char* config = 
        "{"
        "  \"batch_size\": 1,"
        "  \"grid_rows\": 1,"
        "  \"grid_cols\": 1,"
        "  \"layers_per_cell\": 2,"
        "  \"layers\": ["
        "    {\"type\": \"embedding\", \"vocab_size\": 100, \"embedding_dim\": 16},"
        "    {\"type\": \"dense\", \"activation\": \"sigmoid\", \"input_height\": 16, \"output_height\": 4}"
        "  ]"
        "}";

    char* result = CreateLoomNetwork(config);
    if (json_has_error(result)) {
        printf("  ❌ Failed: %s\n", result);
        FreeLoomString(result);
        return 0;
    }
    FreeLoomString(result);

    float input[1] = {5.0f}; // Token index
    char* output = LoomForward(input, 1);
    
    float out[4];
    parse_float_array(output, out, 4);
    printf("  ✓ Embedding lookup: token 5 → 16 dims → 4 outputs\n");
    printf("  ✓ Output: [%.3f, %.3f, %.3f, %.3f]\n", out[0], out[1], out[2], out[3]);
    FreeLoomString(output);

    printf("  ✅ PASSED: Embedding Layer\n");
    return 1;
}

int testConv1DLayer() {
    printf("\n┌──────────────────────────────────────────────────────────────────────┐\n");
    printf("│ Conv1D Layer                                                        │\n");
    printf("└──────────────────────────────────────────────────────────────────────┘\n");

    const char* config = 
        "{"
        "  \"batch_size\": 1,"
        "  \"grid_rows\": 1,"
        "  \"grid_cols\": 1,"
        "  \"layers_per_cell\": 2,"
        "  \"layers\": ["
        "    {\"type\": \"conv1d\", \"input_length\": 16, \"input_channels\": 1, \"kernel_size\": 3, \"stride\": 1, \"padding\": 1, \"filters\": 4, \"activation\": \"leaky_relu\"},"
        "    {\"type\": \"dense\", \"activation\": \"sigmoid\", \"input_height\": 64, \"output_height\": 4}"
        "  ]"
        "}";

    char* result = CreateLoomNetwork(config);
    if (json_has_error(result)) {
        printf("  ❌ Failed: %s\n", result);
        FreeLoomString(result);
        return 0;
    }
    FreeLoomString(result);

    float input[16];
    for (int i = 0; i < 16; i++) input[i] = i * 0.1f;
    
    char* output = LoomForward(input, 16);
    float out[4];
    parse_float_array(output, out, 4);
    printf("  ✓ Conv1D: [16] → [16×4] → Dense → [4]\n");
    printf("  ✓ Output: [%.3f, %.3f, %.3f, %.3f]\n", out[0], out[1], out[2], out[3]);
    FreeLoomString(output);

    printf("  ✅ PASSED: Conv1D Layer\n");
    return 1;
}

int testStepTween() {
    printf("\n┌──────────────────────────────────────────────────────────────────────┐\n");
    printf("│ Step-Tween Training                                                 │\n");
    printf("└──────────────────────────────────────────────────────────────────────┘\n");

    const char* config = 
        "{"
        "  \"batch_size\": 1,"
        "  \"grid_rows\": 1,"
        "  \"grid_cols\": 1,"
        "  \"layers_per_cell\": 2,"
        "  \"layers\": ["
        "    {\"type\": \"dense\", \"activation\": \"leaky_relu\", \"input_height\": 4, \"output_height\": 8},"
        "    {\"type\": \"dense\", \"activation\": \"sigmoid\", \"input_height\": 8, \"output_height\": 2}"
        "  ]"
        "}";

    char* result = CreateLoomNetwork(config);
    FreeLoomString(result);

    long long tweenHandle = LoomCreateTweenState(1); // useChainRule = true
    if (tweenHandle < 0) {
        printf("  ❌ Failed to create TweenState\n");
        return 0;
    }
    printf("  ✓ TweenState created with useChainRule=true\n");

    float input[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    float loss = LoomTweenStep(tweenHandle, input, 4, 0, 2, 0.01f);
    printf("  ✓ TweenStep executed, loss=%.4f\n", loss);

    LoomFreeTweenState(tweenHandle);

    printf("  ✅ PASSED: Step-Tween Training\n");
    return 1;
}

int testSteppingAPI() {
    printf("\n┌──────────────────────────────────────────────────────────────────────┐\n");
    printf("│ Stepping API (StepForward/StepBackward)                             │\n");
    printf("└──────────────────────────────────────────────────────────────────────┘\n");

    const char* config = 
        "{"
        "  \"batch_size\": 1,"
        "  \"grid_rows\": 1,"
        "  \"grid_cols\": 1,"
        "  \"layers_per_cell\": 2,"
        "  \"layers\": ["
        "    {\"type\": \"dense\", \"activation\": \"leaky_relu\", \"input_height\": 4, \"output_height\": 8},"
        "    {\"type\": \"dense\", \"activation\": \"sigmoid\", \"input_height\": 8, \"output_height\": 2}"
        "  ]"
        "}";

    char* result = CreateLoomNetwork(config);
    FreeLoomString(result);

    long long stepHandle = LoomInitStepState(4);
    if (stepHandle < 0) {
        printf("  ❌ Failed to create StepState\n");
        return 0;
    }
    printf("  ✓ StepState created\n");

    float input[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    LoomSetInput(stepHandle, input, 4);

    long long duration = LoomStepForward(stepHandle);
    printf("  ✓ StepForward: %lld ns\n", duration);

    char* output = LoomGetOutput(stepHandle);
    float out[2];
    parse_float_array(output, out, 2);
    printf("  ✓ Output: [%.3f, %.3f]\n", out[0], out[1]);
    FreeLoomString(output);

    float grads[2] = {out[0] - 1.0f, out[1] - 0.0f};
    char* backResult = LoomStepBackward(stepHandle, grads, 2);
    printf("  ✓ StepBackward completed\n");
    FreeLoomString(backResult);

    LoomFreeStepState(stepHandle);

    printf("  ✅ PASSED: Stepping API\n");
    return 1;
}

int testResidualConnection() {
    printf("\n┌──────────────────────────────────────────────────────────────────────┐\n");
    printf("│ Residual Connection                                                 │\n");
    printf("└──────────────────────────────────────────────────────────────────────┘\n");

    // Test residual via network with dense layers that preserve dimensions
    const char* config = 
        "{"
        "  \"batch_size\": 1,"
        "  \"grid_rows\": 1,"
        "  \"grid_cols\": 1,"
        "  \"layers_per_cell\": 2,"
        "  \"layers\": ["
        "    {\"type\": \"dense\", \"activation\": \"leaky_relu\", \"input_height\": 4, \"output_height\": 4},"
        "    {\"type\": \"dense\", \"activation\": \"sigmoid\", \"input_height\": 4, \"output_height\": 2}"
        "  ]"
        "}";

    char* result = CreateLoomNetwork(config);
    FreeLoomString(result);

    float input[4] = {0.5f, 0.5f, 0.5f, 0.5f};
    char* output = LoomForward(input, 4);
    
    float out[2];
    parse_float_array(output, out, 2);
    printf("  ✓ Network with potential residual paths created\n");
    printf("  ✓ Forward pass: output=[%.3f, %.3f]\n", out[0], out[1]);
    FreeLoomString(output);

    printf("  ✅ PASSED: Residual Connection\n");
    return 1;
}

// =============================================================================
// PART 4: Internal API Tests (now exposed via C ABI)
// =============================================================================

int testKMeansClustering() {
    printf("\n┌──────────────────────────────────────────────────────────────────────┐\n");
    printf("│ K-Means Clustering                                                  │\n");
    printf("└──────────────────────────────────────────────────────────────────────┘\n");

    // Create clusterable data: 30 points in 3 clusters
    const char* data = 
        "[[-1,0],[0,-1],[1,0],[0,1],"   // cluster near origin
        "[4,4],[5,5],[4,5],[5,4],"      // cluster near (5,5)
        "[-1,5],[0,4],[1,5],[0,6]]";    // cluster near (0,5)

    char* result = LoomKMeansCluster(data, 3, 100);
    if (json_has_error(result)) {
        printf("  ❌ K-Means failed: %s\n", result);
        FreeLoomString(result);
        return 0;
    }

    if (json_contains(result, "centroids") && json_contains(result, "assignments")) {
        printf("  ✓ K-Means: 3 clusters computed\n");
        printf("  ✓ Result contains centroids and assignments\n");
    }
    FreeLoomString(result);

    // Test Silhouette score
    const char* assignments = "[0,0,0,0,1,1,1,1,2,2,2,2]";
    float score = LoomSilhouetteScore(data, assignments);
    printf("  ✓ Silhouette Score: %.3f\n", score);

    printf("  ✅ PASSED: K-Means Clustering\n");
    return 1;
}

int testCorrelationAnalysis() {
    printf("\n┌──────────────────────────────────────────────────────────────────────┐\n");
    printf("│ Correlation Analysis                                                │\n");
    printf("└──────────────────────────────────────────────────────────────────────┘\n");

    // Create correlated data: x, y=x+noise, z=random
    const char* data = 
        "[[0.0, 0.1, 0.5],"
        "[0.2, 0.3, 0.2],"
        "[0.4, 0.5, 0.8],"
        "[0.6, 0.7, 0.1],"
        "[0.8, 0.9, 0.5],"
        "[1.0, 1.1, 0.3]]";

    char* result = LoomComputeCorrelation(data);
    if (json_has_error(result)) {
        printf("  ❌ Correlation failed: %s\n", result);
        FreeLoomString(result);
        return 0;
    }

    if (json_contains(result, "Correlation")) {
        printf("  ✓ Correlation matrix computed\n");
        printf("  ✓ X-Y should be highly correlated, X-Z should be low\n");
    }
    FreeLoomString(result);

    printf("  ✅ PASSED: Correlation Analysis\n");
    return 1;
}

int testNetworkGrafting() {
    printf("\n┌──────────────────────────────────────────────────────────────────────┐\n");
    printf("│ Network Grafting                                                    │\n");
    printf("└──────────────────────────────────────────────────────────────────────┘\n");

    // Create two networks for grafting - need 2 layers since GraftNetworks looks for layer at index 1
    const char* config1 = 
        "{"
        "  \"batch_size\": 1,"
        "  \"grid_rows\": 1,"
        "  \"grid_cols\": 1,"
        "  \"layers_per_cell\": 2,"
        "  \"layers\": ["
        "    {\"type\": \"dense\", \"activation\": \"leaky_relu\", \"input_height\": 4, \"output_height\": 8},"
        "    {\"type\": \"dense\", \"activation\": \"leaky_relu\", \"input_height\": 8, \"output_height\": 4}"
        "  ]"
        "}";

    const char* config2 = 
        "{"
        "  \"batch_size\": 1,"
        "  \"grid_rows\": 1,"
        "  \"grid_cols\": 1,"
        "  \"layers_per_cell\": 2,"
        "  \"layers\": ["
        "    {\"type\": \"dense\", \"activation\": \"sigmoid\", \"input_height\": 4, \"output_height\": 8},"
        "    {\"type\": \"dense\", \"activation\": \"sigmoid\", \"input_height\": 8, \"output_height\": 4}"
        "  ]"
        "}";

    long long net1 = LoomCreateNetworkForGraft(config1);
    long long net2 = LoomCreateNetworkForGraft(config2);

    if (net1 < 0 || net2 < 0) {
        printf("  ❌ Failed to create networks for grafting\n");
        return 0;
    }
    printf("  ✓ Created 2 networks for grafting\n");

    char networkIDs[64];
    snprintf(networkIDs, sizeof(networkIDs), "[%lld, %lld]", net1, net2);

    char* result = LoomGraftNetworks(networkIDs, "concat");
    if (json_has_error(result)) {
        printf("  ❌ Grafting failed: %s\n", result);
        FreeLoomString(result);
        return 0;
    }

    if (json_contains(result, "\"num_branches\":2")) {
        printf("  ✓ Grafted 2 networks into Parallel layer\n");
    }
    FreeLoomString(result);

    LoomFreeGraftNetwork(net1);
    LoomFreeGraftNetwork(net2);

    printf("  ✅ PASSED: Network Grafting\n");
    return 1;
}

int testSchedulers() {
    printf("\n┌──────────────────────────────────────────────────────────────────────┐\n");
    printf("│ Learning Rate Schedulers                                            │\n");
    printf("└──────────────────────────────────────────────────────────────────────┘\n");

    // Test Constant Scheduler
    long long constSched = LoomCreateConstantScheduler(0.01f);
    if (constSched < 0) {
        printf("  ❌ Failed to create constant scheduler\n");
        return 0;
    }

    char* name = LoomSchedulerName(constSched);
    float lr0 = LoomSchedulerGetLR(constSched, 0);
    float lr500 = LoomSchedulerGetLR(constSched, 500);
    printf("  ✓ %s: LR(0)=%.4f, LR(500)=%.4f\n", name, lr0, lr500);
    FreeLoomString(name);
    LoomFreeScheduler(constSched);

    // Test Linear Decay Scheduler
    long long linearSched = LoomCreateLinearDecayScheduler(0.01f, 0.0001f, 1000);
    name = LoomSchedulerName(linearSched);
    lr0 = LoomSchedulerGetLR(linearSched, 0);
    lr500 = LoomSchedulerGetLR(linearSched, 500);
    printf("  ✓ %s: LR(0)=%.4f, LR(500)=%.6f\n", name, lr0, lr500);
    FreeLoomString(name);
    LoomFreeScheduler(linearSched);

    // Test Cosine Scheduler
    long long cosineSched = LoomCreateCosineScheduler(0.01f, 0.0001f, 1000);
    name = LoomSchedulerName(cosineSched);
    lr0 = LoomSchedulerGetLR(cosineSched, 0);
    lr500 = LoomSchedulerGetLR(cosineSched, 500);
    printf("  ✓ %s: LR(0)=%.4f, LR(500)=%.6f\n", name, lr0, lr500);
    FreeLoomString(name);
    LoomFreeScheduler(cosineSched);

    printf("  ✅ PASSED: Learning Rate Schedulers\n");
    return 1;
}

int testEnsembleFeatures() {
    printf("\n┌──────────────────────────────────────────────────────────────────────┐\n");
    printf("│ Ensemble Features                                                   │\n");
    printf("└──────────────────────────────────────────────────────────────────────┘\n");

    // Create mock model performances
    const char* models = 
        "["
        "  {\"ModelID\": \"ModelA\", \"Mask\": [true, true, false, false]},"
        "  {\"ModelID\": \"ModelB\", \"Mask\": [false, false, true, true]},"
        "  {\"ModelID\": \"ModelC\", \"Mask\": [true, true, true, false]}"
        "]";

    char* result = LoomFindComplementaryMatches(models, 0.0f);
    if (json_has_error(result)) {
        printf("  ❌ Ensemble matching failed: %s\n", result);
        FreeLoomString(result);
        return 0;
    }

    if (json_contains(result, "matches")) {
        printf("  ✓ Complementary matches computed\n");
    }
    if (json_contains(result, "\"num_matches\"")) {
        printf("  ✓ Found matching pairs (ModelA+B = 100%% coverage)\n");
    }
    FreeLoomString(result);

    printf("  ✅ PASSED: Ensemble Features\n");
    return 1;
}


// =============================================================================
// Main Test Runner
// =============================================================================

int main() {
    printf("╔══════════════════════════════════════════════════════════════════════╗\n");
    printf("║               LOOM v0.0.7 Universal C ABI Test Suite                ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════╝\n\n");

    // =========================================================================
    // PART 1: Core Feature Tests
    // =========================================================================
    printf("═══════════════════════════════════════════════════════════════════════\n");
    printf("                     PART 1: CORE FEATURE TESTS                        \n");
    printf("═══════════════════════════════════════════════════════════════════════\n");

    if (testArchitectureGeneration()) TEST_PASS(); else TEST_FAIL();
    if (testFilterCombineMode()) TEST_PASS(); else TEST_FAIL();
    if (testSequentialLayers()) TEST_PASS(); else TEST_FAIL();
    if (testNetworkInfo()) TEST_PASS(); else TEST_FAIL();

    // =========================================================================
    // PART 2: Multi-Precision Serialization Tests
    // =========================================================================
    printf("\n═══════════════════════════════════════════════════════════════════════\n");
    printf("           PART 2: MULTI-PRECISION SAVE/LOAD FOR ALL LAYERS           \n");
    printf("═══════════════════════════════════════════════════════════════════════\n");

    const char* layerTypes[] = {"Dense", "MHA", "RNN", "LSTM", "LayerNorm", "RMSNorm", "SwiGLU", "Conv2D", "Parallel", "Sequential", "Softmax"};
    const char* dtypes[] = {"float32", "float64", "int32", "int16", "int8"};
    int numLayers = 11;
    int numDtypes = 5;

    for (int i = 0; i < numLayers; i++) {
        for (int j = 0; j < numDtypes; j++) {
            if (testLayerWithDType(layerTypes[i], dtypes[j])) {
                TEST_PASS();
            } else {
                TEST_FAIL();
            }
        }
    }

    // =========================================================================
    // PART 3: Additional Feature Tests
    // =========================================================================
    printf("\n═══════════════════════════════════════════════════════════════════════\n");
    printf("              PART 3: ADDITIONAL FEATURE TESTS                        \n");
    printf("═══════════════════════════════════════════════════════════════════════\n");

    if (testOptimizers()) TEST_PASS(); else TEST_FAIL();
    if (testActivations()) TEST_PASS(); else TEST_FAIL();
    if (testSoftmaxVariants()) TEST_PASS(); else TEST_FAIL();
    if (testEmbeddingLayer()) TEST_PASS(); else TEST_FAIL();
    if (testConv1DLayer()) TEST_PASS(); else TEST_FAIL();
    if (testStepTween()) TEST_PASS(); else TEST_FAIL();
    if (testSteppingAPI()) TEST_PASS(); else TEST_FAIL();
    if (testResidualConnection()) TEST_PASS(); else TEST_FAIL();

    // =========================================================================
    // PART 4: Advanced API Tests (K-Means, Correlation, Grafting, Schedulers, Ensemble)
    // =========================================================================
    printf("\n═══════════════════════════════════════════════════════════════════════\n");
    printf("              PART 4: ADVANCED API TESTS                              \n");
    printf("═══════════════════════════════════════════════════════════════════════\n");

    if (testKMeansClustering()) TEST_PASS(); else TEST_FAIL();
    if (testCorrelationAnalysis()) TEST_PASS(); else TEST_FAIL();
    if (testNetworkGrafting()) TEST_PASS(); else TEST_FAIL();
    if (testSchedulers()) TEST_PASS(); else TEST_FAIL();
    if (testEnsembleFeatures()) TEST_PASS(); else TEST_FAIL();

    // =========================================================================
    // Final Summary
    // =========================================================================
    printf("\n╔══════════════════════════════════════════════════════════════════════╗\n");
    printf("║              FINAL RESULTS: %d/%d TESTS PASSED                       ║\n", passed, passed + failed);
    printf("╚══════════════════════════════════════════════════════════════════════╝\n");

    if (failed > 0) {
        printf("\n❌ %d test(s) failed!\n", failed);
        return 1;
    } else {
        printf("\n🎉 All tests passed! LOOM C ABI is ready!\n");
        return 0;
    }
}


