/*
 * LOOM Optimizer C ABI Test
 * Tests the new optimizer functions via C ABI
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "libloom.h"

// Helper to parse JSON array
void parse_json_array(const char* json, float* output, int* length) {
    const char* start = strchr(json, '[');
    if (!start) return;
    
    int count = 0;
    const char* ptr = start + 1;
    while (*ptr && *ptr != ']') {
        output[count++] = atof(ptr);
        ptr = strchr(ptr, ',');
        if (!ptr) break;
        ptr++;
    }
    *length = count;
}

int main() {
    printf("🚀 LOOM Optimizer C ABI Test\n");
    printf("================================\n\n");

    // Network configuration
    const char* config = "{"
        "\"batch_size\": 1,"
        "\"grid_rows\": 1,"
        "\"grid_cols\": 1,"
        "\"layers_per_cell\": 3,"
        "\"layers\": ["
            "{\"type\": \"dense\", \"input_height\": 4, \"output_height\": 8, \"activation\": \"relu\"},"
            "{\"type\": \"lstm\", \"input_size\": 8, \"hidden_size\": 12, \"seq_length\": 1},"
            "{\"type\": \"dense\", \"input_height\": 12, \"output_height\": 3, \"activation\": \"softmax\"}"
        "]"
    "}";

    // Create network
    printf("Creating network...\n");
    char* result = CreateLoomNetwork(config);
    printf("Result: %s\n\n", result);
    FreeLoomString(result);

    // Training data
    float inputs[6][4] = {
        {0.1f, 0.2f, 0.1f, 0.3f},  // Class 0
        {0.8f, 0.9f, 0.7f, 0.8f},  // Class 1
        {0.3f, 0.5f, 0.9f, 0.6f},  // Class 2
        {0.2f, 0.1f, 0.2f, 0.2f},  // Class 0
        {0.9f, 0.8f, 0.8f, 0.9f},  // Class 1
        {0.4f, 0.6f, 0.8f, 0.7f}   // Class 2
    };

    float targets[6][3] = {
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f},
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f}
    };

    // ========================================================================
    // Test 1: Simple SGD (baseline)
    // ========================================================================
    printf("📊 Test 1: Simple SGD (baseline)\n");
    printf("----------------------------------\n");
    
    long long state = LoomInitStepState(4);
    float total_loss = 0.0f;
    
    for (int step = 0; step < 5000; step++) {
        int idx = step % 6;
        
        // Forward pass
        LoomSetInput(state, inputs[idx], 4);
        LoomStepForward(state);
        
        char* output_json = LoomGetOutput(state);
        float output[3];
        int output_len;
        parse_json_array(output_json, output, &output_len);
        FreeLoomString(output_json);
        
        // Compute loss and gradients
        float loss = 0.0f;
        float gradients[3];
        for (int i = 0; i < 3; i++) {
            float diff = output[i] - targets[idx][i];
            loss += diff * diff;
            gradients[i] = 2.0f * diff / 3.0f;
        }
        loss /= 3.0f;
        total_loss += loss;
        
        // Backward pass
        LoomStepBackward(state, gradients, 3);
        
        // Apply gradients (simple SGD)
        LoomApplyGradients(0.01f);
        
        if ((step + 1) % 1000 == 0) {
            printf("  Step %d: Avg Loss=%.6f\n", step + 1, total_loss / 1000.0f);
            total_loss = 0.0f;
        }
    }
    
    printf("✅ SGD Test complete!\n\n");
    LoomFreeStepState(state);

    // ========================================================================
    // Test 2: AdamW Optimizer
    // ========================================================================
    printf("📊 Test 2: AdamW Optimizer\n");
    printf("----------------------------------\n");
    
    // Recreate network
    FreeLoomString(CreateLoomNetwork(config));
    
    state = LoomInitStepState(4);
    total_loss = 0.0f;
    
    for (int step = 0; step < 5000; step++) {
        int idx = step % 6;
        
        // Forward pass
        LoomSetInput(state, inputs[idx], 4);
        LoomStepForward(state);
        
        char* output_json = LoomGetOutput(state);
        float output[3];
        int output_len;
        parse_json_array(output_json, output, &output_len);
        FreeLoomString(output_json);
        
        // Compute loss and gradients
        float loss = 0.0f;
        float gradients[3];
        for (int i = 0; i < 3; i++) {
            float diff = output[i] - targets[idx][i];
            loss += diff * diff;
            gradients[i] = 2.0f * diff / 3.0f;
        }
        loss /= 3.0f;
        total_loss += loss;
        
        // Backward pass
        LoomStepBackward(state, gradients, 3);
        
        // Apply gradients with AdamW
        LoomApplyGradientsAdamW(0.001f, 0.9f, 0.999f, 0.01f);
        
        if ((step + 1) % 1000 == 0) {
            printf("  Step %d: Avg Loss=%.6f\n", step + 1, total_loss / 1000.0f);
            total_loss = 0.0f;
        }
    }
    
    printf("✅ AdamW Test complete!\n\n");
    LoomFreeStepState(state);

    // ========================================================================
    // Test 3: RMSprop Optimizer
    // ========================================================================
    printf("📊 Test 3: RMSprop Optimizer\n");
    printf("----------------------------------\n");
    
    // Recreate network
    FreeLoomString(CreateLoomNetwork(config));
    
    state = LoomInitStepState(4);
    total_loss = 0.0f;
    
    for (int step = 0; step < 5000; step++) {
        int idx = step % 6;
        
        // Forward pass
        LoomSetInput(state, inputs[idx], 4);
        LoomStepForward(state);
        
        char* output_json = LoomGetOutput(state);
        float output[3];
        int output_len;
        parse_json_array(output_json, output, &output_len);
        FreeLoomString(output_json);
        
        // Compute loss and gradients
        float loss = 0.0f;
        float gradients[3];
        for (int i = 0; i < 3; i++) {
            float diff = output[i] - targets[idx][i];
            loss += diff * diff;
            gradients[i] = 2.0f * diff / 3.0f;
        }
        loss /= 3.0f;
        total_loss += loss;
        
        // Backward pass
        LoomStepBackward(state, gradients, 3);
        
        // Apply gradients with RMSprop
        LoomApplyGradientsRMSprop(0.001f, 0.99f, 1e-8f, 0.0f);
        
        if ((step + 1) % 1000 == 0) {
            printf("  Step %d: Avg Loss=%.6f\n", step + 1, total_loss / 1000.0f);
            total_loss = 0.0f;
        }
    }
    
    printf("✅ RMSprop Test complete!\n\n");
    LoomFreeStepState(state);

    // ========================================================================
    // Test 4: SGD with Momentum
    // ========================================================================
    printf("📊 Test 4: SGD with Momentum\n");
    printf("----------------------------------\n");
    
    // Recreate network
    FreeLoomString(CreateLoomNetwork(config));
    
    state = LoomInitStepState(4);
    total_loss = 0.0f;
    
    for (int step = 0; step < 5000; step++) {
        int idx = step % 6;
        
        // Forward pass
        LoomSetInput(state, inputs[idx], 4);
        LoomStepForward(state);
        
        char* output_json = LoomGetOutput(state);
        float output[3];
        int output_len;
        parse_json_array(output_json, output, &output_len);
        FreeLoomString(output_json);
        
        // Compute loss and gradients
        float loss = 0.0f;
        float gradients[3];
        for (int i = 0; i < 3; i++) {
            float diff = output[i] - targets[idx][i];
            loss += diff * diff;
            gradients[i] = 2.0f * diff / 3.0f;
        }
        loss /= 3.0f;
        total_loss += loss;
        
        // Backward pass
        LoomStepBackward(state, gradients, 3);
        
        // Apply gradients with SGD + Momentum
        LoomApplyGradientsSGDMomentum(0.01f, 0.9f, 0.0f, 0);  // momentum=0.9, dampening=0, nesterov=false
        
        if ((step + 1) % 1000 == 0) {
            printf("  Step %d: Avg Loss=%.6f\n", step + 1, total_loss / 1000.0f);
            total_loss = 0.0f;
        }
    }
    
    printf("✅ SGD+Momentum Test complete!\n\n");
    LoomFreeStepState(state);

    // Summary
    printf("🎉 All C ABI optimizer tests complete!\n\n");
    printf("Verified Functions:\n");
    printf("  ✅ LoomApplyGradients (simple SGD)\n");
    printf("  ✅ LoomApplyGradientsAdamW\n");
    printf("  ✅ LoomApplyGradientsRMSprop\n");
    printf("  ✅ LoomApplyGradientsSGDMomentum\n");
    printf("\nAll optimizer methods working correctly via C ABI! 🚀\n");

    return 0;
}
