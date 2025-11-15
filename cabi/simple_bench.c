/*
 * Grid Scatter Multi-Agent Demo - LOOM C API
 * 3 heterogeneous agents collaborate for binary classification
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "libloom.h"

void parse_output(const char* json, float* out1, float* out2) {
    // Simple JSON parser for [float, float]
    const char* start = strchr(json, '[');
    if (start) {
        sscanf(start, "[%f,%f]", out1, out2);
    }
}

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
    FreeLoomString(result);

    // Training data
    float batch1_input[] = {0.2, 0.2, 0.2, 0.2, 0.8, 0.8, 0.8, 0.8};
    float batch1_target[] = {1.0, 0.0};
    
    float batch2_input[] = {0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1};
    float batch2_target[] = {0.0, 1.0};
    
    float batch3_input[] = {0.7, 0.7, 0.7, 0.7, 0.3, 0.3, 0.3, 0.3};
    float batch3_target[] = {0.0, 1.0};
    
    float batch4_input[] = {0.3, 0.3, 0.3, 0.3, 0.7, 0.7, 0.7, 0.7};
    float batch4_target[] = {1.0, 0.0};

    // Test BEFORE training
    printf("\n📊 Before Training:\n");
    float out[2];
    
    char* pred = LoomForward(batch1_input, 8);
    parse_output(pred, &out[0], &out[1]);
    printf("Sample 0: [%.3f, %.3f] → Class %d (expected 0)\n", out[0], out[1], out[1] > out[0] ? 1 : 0);
    FreeLoomString(pred);
    
    pred = LoomForward(batch2_input, 8);
    parse_output(pred, &out[0], &out[1]);
    printf("Sample 1: [%.3f, %.3f] → Class %d (expected 1)\n", out[0], out[1], out[1] > out[0] ? 1 : 0);
    FreeLoomString(pred);
    
    pred = LoomForward(batch3_input, 8);
    parse_output(pred, &out[0], &out[1]);
    printf("Sample 2: [%.3f, %.3f] → Class %d (expected 1)\n", out[0], out[1], out[1] > out[0] ? 1 : 0);
    FreeLoomString(pred);
    
    pred = LoomForward(batch4_input, 8);
    parse_output(pred, &out[0], &out[1]);
    printf("Sample 3: [%.3f, %.3f] → Class %d (expected 0)\n", out[0], out[1], out[1] > out[0] ? 1 : 0);
    FreeLoomString(pred);

    printf("\nTraining for 800 epochs with learning rate 0.150\n");
    
    // Use LoomTrain like the JavaScript version
    const char* batches_json = 
        "["
        "  {\"Input\": [0.2, 0.2, 0.2, 0.2, 0.8, 0.8, 0.8, 0.8], \"Target\": [1.0, 0.0]},"
        "  {\"Input\": [0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1], \"Target\": [0.0, 1.0]},"
        "  {\"Input\": [0.7, 0.7, 0.7, 0.7, 0.3, 0.3, 0.3, 0.3], \"Target\": [0.0, 1.0]},"
        "  {\"Input\": [0.3, 0.3, 0.3, 0.3, 0.7, 0.7, 0.7, 0.7], \"Target\": [1.0, 0.0]}"
        "]";
    
    const char* config_json = 
        "{"
        "  \"Epochs\": 800,"
        "  \"LearningRate\": 0.15,"
        "  \"UseGPU\": false,"
        "  \"PrintEveryBatch\": 0,"
        "  \"GradientClip\": 1.0,"
        "  \"LossType\": \"mse\","
        "  \"Verbose\": false"
        "}";
    
    clock_t start = clock();
    char* train_result = LoomTrain((char*)batches_json, (char*)config_json);
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    
    // Parse training result
    float initial_loss = 0.0, final_loss = 0.0;
    const char* p = train_result;
    if ((p = strstr(p, "\"LossHistory\":["))) {
        sscanf(p, "\"LossHistory\":[%f", &initial_loss);
    }
    p = train_result;
    if ((p = strstr(p, "\"FinalLoss\":"))) {
        sscanf(p, "\"FinalLoss\":%f", &final_loss);
    }
    
    printf("✅ Training complete!\n");
    printf("Training time: %.2f seconds\n", elapsed);
    if (initial_loss > 0) {
        printf("Initial Loss: %.6f\n", initial_loss);
        printf("Final Loss: %.6f\n", final_loss);
        float improvement = ((initial_loss - final_loss) / initial_loss) * 100.0;
        printf("Improvement: %.2f%%\n", improvement);
    }
    printf("Total Epochs: 800\n");
    
    FreeLoomString(train_result);

    // Test AFTER training
    printf("\n📊 After Training:\n");
    float original_preds[4][2]; // Store original predictions
    
    pred = LoomForward(batch1_input, 8);
    parse_output(pred, &out[0], &out[1]);
    original_preds[0][0] = out[0];
    original_preds[0][1] = out[1];
    printf("Sample 0: [%.3f, %.3f] → Class %d (expected 0) %s\n", 
           out[0], out[1], out[1] > out[0] ? 1 : 0,
           (out[1] > out[0] ? 1 : 0) == 0 ? "✓" : "✗");
    FreeLoomString(pred);
    
    pred = LoomForward(batch2_input, 8);
    parse_output(pred, &out[0], &out[1]);
    original_preds[1][0] = out[0];
    original_preds[1][1] = out[1];
    printf("Sample 1: [%.3f, %.3f] → Class %d (expected 1) %s\n", 
           out[0], out[1], out[1] > out[0] ? 1 : 0,
           (out[1] > out[0] ? 1 : 0) == 1 ? "✓" : "✗");
    FreeLoomString(pred);
    
    pred = LoomForward(batch3_input, 8);
    parse_output(pred, &out[0], &out[1]);
    original_preds[2][0] = out[0];
    original_preds[2][1] = out[1];
    printf("Sample 2: [%.3f, %.3f] → Class %d (expected 1) %s\n", 
           out[0], out[1], out[1] > out[0] ? 1 : 0,
           (out[1] > out[0] ? 1 : 0) == 1 ? "✓" : "✗");
    FreeLoomString(pred);
    
    pred = LoomForward(batch4_input, 8);
    parse_output(pred, &out[0], &out[1]);
    original_preds[3][0] = out[0];
    original_preds[3][1] = out[1];
    printf("Sample 3: [%.3f, %.3f] → Class %d (expected 0) %s\n", 
           out[0], out[1], out[1] > out[0] ? 1 : 0,
           (out[1] > out[0] ? 1 : 0) == 0 ? "✓" : "✗");
    FreeLoomString(pred);

    // Use EvaluateNetwork for detailed accuracy metrics
    printf("\n📊 Evaluating with EvaluateNetwork...\n");
    
    // Prepare inputs and expected outputs as JSON
    const char* inputs_json = 
        "["
        "  [0.2, 0.2, 0.2, 0.2, 0.8, 0.8, 0.8, 0.8],"
        "  [0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1],"
        "  [0.7, 0.7, 0.7, 0.7, 0.3, 0.3, 0.3, 0.3],"
        "  [0.3, 0.3, 0.3, 0.3, 0.7, 0.7, 0.7, 0.7]"
        "]";
    
    const char* expected_json = "[0.0, 1.0, 1.0, 0.0]";  // Class labels
    
    char* eval_result = LoomEvaluateNetwork((char*)inputs_json, (char*)expected_json);
    
    if (strstr(eval_result, "error")) {
        printf("Evaluation error: %s\n", eval_result);
    } else {
        // Parse metrics from JSON (simplified parsing)
        int total_samples = 0;
        float quality_score = 0.0;
        float avg_deviation = 0.0;
        int failures = 0;
        
        // Simple JSON parsing
        const char* p = eval_result;
        if ((p = strstr(p, "\"total_samples\":"))) {
            sscanf(p, "\"total_samples\":%d", &total_samples);
        }
        p = eval_result;
        if ((p = strstr(p, "\"score\":"))) {
            sscanf(p, "\"score\":%f", &quality_score);
        }
        p = eval_result;
        if ((p = strstr(p, "\"avg_deviation\":"))) {
            sscanf(p, "\"avg_deviation\":%f", &avg_deviation);
        }
        p = eval_result;
        if ((p = strstr(p, "\"failures\":"))) {
            sscanf(p, "\"failures\":%d", &failures);
        }
        
        printf("\n=== Evaluation Metrics ===\n");
        printf("Total Samples: %d\n", total_samples);
        printf("Quality Score: %.2f/100\n", quality_score);
        printf("Average Deviation: %.2f%%\n", avg_deviation);
        printf("Failures (>100%% deviation): %d\n", failures);
        
        // Parse bucket counts
        printf("\nDeviation Distribution:\n");
        const char* buckets[] = {"0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-100%", "100%+"};
        for (int i = 0; i < 7; i++) {
            char search[100];
            sprintf(search, "\"%s\":{\"range_min\"", buckets[i]);
            p = strstr(eval_result, search);
            if (p) {
                p = strstr(p, "\"count\":");
                if (p) {
                    int count = 0;
                    sscanf(p, "\"count\":%d", &count);
                    if (count > 0) {
                        float percentage = (count * 100.0) / total_samples;
                        int bar_length = (int)((count * 20.0) / total_samples);
                        printf("  %-8s: %d samples (%.1f%%) ", buckets[i], count, percentage);
                        for (int j = 0; j < bar_length; j++) printf("█");
                        printf("\n");
                    }
                }
            }
        }
    }
    
    FreeLoomString(eval_result);

    printf("\n✅ Multi-agent training complete!\n");

    // Save and reload model to verify serialization
    printf("\n💾 Testing model save/load...\n");
    
    char* saved_model = LoomSaveModel("grid_scatter_test");
    if (strstr(saved_model, "error")) {
        printf("Failed to save model: %s\n", saved_model);
        FreeLoomString(saved_model);
        return 1;
    }
    
    printf("✓ Model saved (%zu bytes)\n", strlen(saved_model));
    
    // Load the model back
    printf("Loading model from saved state...\n");
    char* load_result = LoomLoadModel(saved_model, "grid_scatter_test");
    FreeLoomString(saved_model);
    
    if (strstr(load_result, "error")) {
        printf("Failed to load model: %s\n", load_result);
        FreeLoomString(load_result);
        return 1;
    }
    FreeLoomString(load_result);
    
    printf("✓ Model loaded\n");
    
    // Test predictions with reloaded model
    printf("\nVerifying predictions match:\n");
    int all_match = 1;
    float inputs[4][8] = {
        {0.2, 0.2, 0.2, 0.2, 0.8, 0.8, 0.8, 0.8},
        {0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1},
        {0.7, 0.7, 0.7, 0.7, 0.3, 0.3, 0.3, 0.3},
        {0.3, 0.3, 0.3, 0.3, 0.7, 0.7, 0.7, 0.7}
    };
    
    for (int i = 0; i < 4; i++) {
        char* pred = LoomForward(inputs[i], 8);
        float new_out[2];
        parse_output(pred, &new_out[0], &new_out[1]);
        FreeLoomString(pred);
        
        float diff0 = fabsf(new_out[0] - original_preds[i][0]);
        float diff1 = fabsf(new_out[1] - original_preds[i][1]);
        float max_diff = diff0 > diff1 ? diff0 : diff1;
        
        int match = max_diff < 1e-6;
        all_match = all_match && match;
        
        printf("Sample %d: [%.3f, %.3f] (diff: %.2e) %s\n",
               i, new_out[0], new_out[1], max_diff, match ? "✓" : "✗");
    }
    
    if (all_match) {
        printf("\n✅ Save/Load verification passed! All predictions match.\n");
    } else {
        printf("\n❌ Save/Load verification failed! Predictions don't match.\n");
    }
    
    return 0;
}
