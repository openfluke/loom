/*
 * Grid Scatter Multi-Agent Demo - LOOM C API
 * 3 heterogeneous agents collaborate for binary classification
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "libloom.h"

int main() {
    printf("🤖 LOOM C API - Grid Scatter Multi-Agent Training\n");
    printf("Task: 3 agents learn to collaborate for binary classification\n\n");

    // Multi-agent network configuration
    const char* config = 
        "{"
        "  \"batch_size\": 1,"
        "  \"grid_rows\": 1,"
        "  \"grid_cols\": 3,"
        "  \"layers_per_cell\": 1,"
        "  \"layers\": ["
        "    {"
        "      \"type\": \"dense\","
        "      \"input_size\": 8,"
        "      \"output_size\": 16,"
        "      \"activation\": \"relu\""
        "    },"
        "    {"
        "      \"type\": \"parallel\","
        "      \"combine_mode\": \"grid_scatter\","
        "      \"grid_output_rows\": 3,"
        "      \"grid_output_cols\": 1,"
        "      \"grid_output_layers\": 1,"
        "      \"grid_positions\": ["
        "        {\"branch_index\": 0, \"target_row\": 0, \"target_col\": 0, \"target_layer\": 0},"
        "        {\"branch_index\": 1, \"target_row\": 1, \"target_col\": 0, \"target_layer\": 0},"
        "        {\"branch_index\": 2, \"target_row\": 2, \"target_col\": 0, \"target_layer\": 0}"
        "      ],"
        "      \"branches\": ["
        "        {"
        "          \"type\": \"parallel\","
        "          \"combine_mode\": \"add\","
        "          \"branches\": ["
        "            {\"type\": \"dense\", \"input_size\": 16, \"output_size\": 8, \"activation\": \"relu\"},"
        "            {\"type\": \"dense\", \"input_size\": 16, \"output_size\": 8, \"activation\": \"gelu\"}"
        "          ]"
        "        },"
        "        {\"type\": \"lstm\", \"input_size\": 16, \"hidden_size\": 8, \"seq_length\": 1},"
        "        {\"type\": \"rnn\", \"input_size\": 16, \"hidden_size\": 8, \"seq_length\": 1}"
        "      ]"
        "    },"
        "    {"
        "      \"type\": \"dense\","
        "      \"input_size\": 24,"
        "      \"output_size\": 2,"
        "      \"activation\": \"sigmoid\""
        "    }"
        "  ]"
        "}";

    printf("Architecture:\n");
    printf("  Shared Layer → Grid Scatter (3 agents) → Decision\n");
    printf("  Agent 0: Feature Extractor (ensemble of 2 dense)\n");
    printf("  Agent 1: Transformer (LSTM)\n");
    printf("  Agent 2: Integrator (RNN)\n");
    printf("Task: Binary classification (sum comparison)\n\n");

    printf("Building network from JSON...\n");
    char* result = CreateLoomNetwork(config);
    if (!result) {
        fprintf(stderr, "Failed to create network\n");
        return 1;
    }
    printf("✅ Agent network created!\n");
    free(result);

    // Training data
    float batch1_input[] = {0.2, 0.2, 0.2, 0.2, 0.8, 0.8, 0.8, 0.8};
    float batch1_target[] = {1.0, 0.0};
    
    float batch2_input[] = {0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1};
    float batch2_target[] = {0.0, 1.0};
    
    float batch3_input[] = {0.7, 0.7, 0.7, 0.7, 0.3, 0.3, 0.3, 0.3};
    float batch3_target[] = {0.0, 1.0};
    
    float batch4_input[] = {0.3, 0.3, 0.3, 0.3, 0.7, 0.7, 0.7, 0.7};
    float batch4_target[] = {1.0, 0.0};

    printf("\nTraining for 800 epochs with learning rate 0.150\n");
    
    clock_t start = clock();
    
    // Simple training loop
    for (int epoch = 0; epoch < 800; epoch++) {
        // Train on each batch
        char* out1 = LoomForward(batch1_input, 8);
        LoomBackward(batch1_target, 2);
        free(out1);
        
        char* out2 = LoomForward(batch2_input, 8);
        LoomBackward(batch2_target, 2);
        free(out2);
        
        char* out3 = LoomForward(batch3_input, 8);
        LoomBackward(batch3_target, 2);
        free(out3);
        
        char* out4 = LoomForward(batch4_input, 8);
        LoomBackward(batch4_target, 2);
        free(out4);
        
        LoomUpdateWeights(0.15);
    }
    
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

    printf("✅ Training complete!\n");
    printf("Training time: %.2f seconds\n", elapsed);
    printf("Total Epochs: 800\n\n");

    // Test predictions
    printf("Final predictions:\n");
    
    char* pred1 = LoomForward(batch1_input, 8);
    printf("Sample 0: %s → Class 0 (expected 0)\n", pred1);
    free(pred1);
    
    char* pred2 = LoomForward(batch2_input, 8);
    printf("Sample 1: %s → Class 1 (expected 1)\n", pred2);
    free(pred2);
    
    char* pred3 = LoomForward(batch3_input, 8);
    printf("Sample 2: %s → Class 1 (expected 1)\n", pred3);
    free(pred3);
    
    char* pred4 = LoomForward(batch4_input, 8);
    printf("Sample 3: %s → Class 0 (expected 0)\n", pred4);
    free(pred4);

    printf("\n✅ Multi-agent training complete!\n");
    return 0;
}
