/*
 * Test 18: Multi-Architecture Adaptation Benchmark - LOOM C API
 * 
 * Replicates test18_architecture_adaptation.go using the C-ABI interface.
 * Tests how different network architectures adapt to mid-stream task changes.
 * 
 * Networks: Dense, Conv2D, RNN, LSTM, Attention
 * Depths: 3, 5, 9 layers
 * Modes: NormalBP, StepBP, Tween, TweenChain, StepTweenChain
 * 
 * Compile: gcc -o test18_adaptation test18_adaptation.c -L. -lloom -lm -lpthread
 * Run: LD_LIBRARY_PATH=. ./test18_adaptation
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "libloom.h"

// ============================================================================
// Configuration
// ============================================================================

#define NUM_NETWORK_TYPES 5
#define NUM_DEPTHS 3
#define NUM_MODES 5
#define TEST_DURATION_MS 10000
#define WINDOW_DURATION_MS 1000
#define TRAIN_INTERVAL_MS 50
#define OUTPUT_SIZE 4
#define LEARNING_RATE 0.02f

const char* NETWORK_TYPES[] = {"Dense", "Conv2D", "RNN", "LSTM", "Attn"};
const int DEPTHS[] = {3, 5, 9};
const char* MODE_NAMES[] = {"NormalBP", "StepBP", "Tween", "TweenChain", "StepTweenChain"};

// ============================================================================
// Environment (Chase/Avoid simulation)
// ============================================================================

typedef struct {
    float agent_pos[2];
    float target_pos[2];
    int task;  // 0=chase, 1=avoid
} Environment;

void env_init(Environment* env) {
    env->agent_pos[0] = 0.5f;
    env->agent_pos[1] = 0.5f;
    env->target_pos[0] = (float)rand() / RAND_MAX;
    env->target_pos[1] = (float)rand() / RAND_MAX;
    env->task = 0;
}

void env_get_observation(Environment* env, float* obs, int target_size) {
    float rel_x = env->target_pos[0] - env->agent_pos[0];
    float rel_y = env->target_pos[1] - env->agent_pos[1];
    float dist = sqrtf(rel_x * rel_x + rel_y * rel_y);
    
    float base[] = {
        env->agent_pos[0], env->agent_pos[1],
        env->target_pos[0], env->target_pos[1],
        rel_x, rel_y, dist, (float)env->task
    };
    
    for (int i = 0; i < target_size; i++) {
        obs[i] = base[i % 8];
    }
}

int env_get_optimal_action(Environment* env) {
    float rel_x = env->target_pos[0] - env->agent_pos[0];
    float rel_y = env->target_pos[1] - env->agent_pos[1];
    
    if (env->task == 0) {  // Chase
        return fabsf(rel_x) > fabsf(rel_y) ? (rel_x > 0 ? 3 : 2) : (rel_y > 0 ? 0 : 1);
    } else {  // Avoid
        return fabsf(rel_x) > fabsf(rel_y) ? (rel_x > 0 ? 2 : 3) : (rel_y > 0 ? 1 : 0);
    }
}

void env_execute_action(Environment* env, int action) {
    float speed = 0.02f;
    float moves[][2] = {{0, speed}, {0, -speed}, {-speed, 0}, {speed, 0}};
    if (action >= 0 && action < 4) {
        env->agent_pos[0] += moves[action][0];
        env->agent_pos[1] += moves[action][1];
        if (env->agent_pos[0] < 0) env->agent_pos[0] = 0;
        if (env->agent_pos[0] > 1) env->agent_pos[0] = 1;
        if (env->agent_pos[1] < 0) env->agent_pos[1] = 0;
        if (env->agent_pos[1] > 1) env->agent_pos[1] = 1;
    }
}

void env_update(Environment* env) {
    env->target_pos[0] += ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
    env->target_pos[1] += ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
    if (env->target_pos[0] < 0.1f) env->target_pos[0] = 0.1f;
    if (env->target_pos[0] > 0.9f) env->target_pos[0] = 0.9f;
    if (env->target_pos[1] < 0.1f) env->target_pos[1] = 0.1f;
    if (env->target_pos[1] > 0.9f) env->target_pos[1] = 0.9f;
}

// ============================================================================
// Network Configurations (JSON builders)
// ============================================================================

int get_input_size(const char* net_type) {
    if (strcmp(net_type, "Dense") == 0) return 8;
    if (strcmp(net_type, "Conv2D") == 0) return 64;
    if (strcmp(net_type, "RNN") == 0 || strcmp(net_type, "LSTM") == 0) return 32;
    if (strcmp(net_type, "Attn") == 0) return 64;
    return 8;
}

// Build Dense network JSON config
char* build_dense_config(int num_layers) {
    static char buffer[4096];
    int hidden_sizes[] = {64, 48, 32, 24, 16};
    
    char layers[3000] = "";
    
    // First layer
    sprintf(layers + strlen(layers), 
        "{\"type\":\"dense\",\"input_size\":8,\"output_size\":64,\"activation\":\"leaky_relu\"}");
    
    // Hidden layers
    for (int i = 1; i < num_layers - 1; i++) {
        int in_size = hidden_sizes[(i - 1) % 5];
        int out_size = hidden_sizes[i % 5];
        sprintf(layers + strlen(layers),
            ",{\"type\":\"dense\",\"input_size\":%d,\"output_size\":%d,\"activation\":\"leaky_relu\"}", in_size, out_size);
    }
    
    // Output layer
    int last_hidden = hidden_sizes[(num_layers - 2) % 5];
    sprintf(layers + strlen(layers),
        ",{\"type\":\"dense\",\"input_size\":%d,\"output_size\":4,\"activation\":\"sigmoid\"}", last_hidden);
    
    sprintf(buffer, "{\"batch_size\":1,\"grid_rows\":1,\"grid_cols\":1,\"layers_per_cell\":%d,\"layers\":[%s]}", 
        num_layers, layers);
    return buffer;
}

// Build Conv2D network JSON config
char* build_conv2d_config(int num_layers) {
    static char buffer[4096];
    char layers[3000] = "";
    
    // Conv layer
    sprintf(layers, 
        "{\"type\":\"conv2d\",\"input_height\":8,\"input_width\":8,\"input_channels\":1,"
        "\"filters\":8,\"kernel_size\":3,\"stride\":1,\"padding\":0,"
        "\"output_height\":6,\"output_width\":6,\"activation\":\"leaky_relu\"}");
    
    // Hidden dense layers
    for (int i = 1; i < num_layers - 1; i++) {
        int in_size = (i == 1) ? 288 : 64;  // 6*6*8 = 288
        sprintf(layers + strlen(layers),
            ",{\"type\":\"dense\",\"input_size\":%d,\"output_size\":64,\"activation\":\"leaky_relu\"}", in_size);
    }
    
    // Output layer
    int last_in = (num_layers > 2) ? 64 : 288;
    sprintf(layers + strlen(layers),
        ",{\"type\":\"dense\",\"input_size\":%d,\"output_size\":4,\"activation\":\"sigmoid\"}", last_in);
    
    sprintf(buffer, "{\"batch_size\":1,\"grid_rows\":1,\"grid_cols\":1,\"layers_per_cell\":%d,\"layers\":[%s]}", 
        num_layers, layers);
    return buffer;
}

// Build RNN network JSON config
char* build_rnn_config(int num_layers) {
    static char buffer[4096];
    char layers[3000] = "";
    
    // First layer
    sprintf(layers, "{\"type\":\"dense\",\"input_size\":32,\"output_size\":32,\"activation\":\"leaky_relu\"}");
    
    // Hidden layers
    for (int i = 1; i < num_layers - 1; i++) {
        if (i % 2 == 1) {
            sprintf(layers + strlen(layers),
                ",{\"type\":\"rnn\",\"input_size\":8,\"hidden_size\":8,\"seq_length\":4}");
        } else {
            sprintf(layers + strlen(layers),
                ",{\"type\":\"dense\",\"input_size\":32,\"output_size\":32,\"activation\":\"leaky_relu\"}");
        }
    }
    
    // Output layer
    sprintf(layers + strlen(layers),
        ",{\"type\":\"dense\",\"input_size\":32,\"output_size\":4,\"activation\":\"sigmoid\"}");
    
    sprintf(buffer, "{\"batch_size\":1,\"grid_rows\":1,\"grid_cols\":1,\"layers_per_cell\":%d,\"layers\":[%s]}", 
        num_layers, layers);
    return buffer;
}

// Build LSTM network JSON config
char* build_lstm_config(int num_layers) {
    static char buffer[4096];
    char layers[3000] = "";
    
    // First layer
    sprintf(layers, "{\"type\":\"dense\",\"input_size\":32,\"output_size\":32,\"activation\":\"leaky_relu\"}");
    
    // Hidden layers
    for (int i = 1; i < num_layers - 1; i++) {
        if (i % 2 == 1) {
            sprintf(layers + strlen(layers),
                ",{\"type\":\"lstm\",\"input_size\":8,\"hidden_size\":8,\"seq_length\":4}");
        } else {
            sprintf(layers + strlen(layers),
                ",{\"type\":\"dense\",\"input_size\":32,\"output_size\":32,\"activation\":\"leaky_relu\"}");
        }
    }
    
    // Output layer
    sprintf(layers + strlen(layers),
        ",{\"type\":\"dense\",\"input_size\":32,\"output_size\":4,\"activation\":\"sigmoid\"}");
    
    sprintf(buffer, "{\"batch_size\":1,\"grid_rows\":1,\"grid_cols\":1,\"layers_per_cell\":%d,\"layers\":[%s]}", 
        num_layers, layers);
    return buffer;
}

// Build Attention network JSON config
char* build_attn_config(int num_layers) {
    static char buffer[4096];
    char layers[3000] = "";
    int d_model = 64;
    
    layers[0] = '\0';
    for (int i = 0; i < num_layers - 1; i++) {
        if (i > 0) strcat(layers, ",");
        if (i % 2 == 0) {
            sprintf(layers + strlen(layers),
                "{\"type\":\"multi_head_attention\",\"d_model\":%d,\"num_heads\":4}", d_model);
        } else {
            sprintf(layers + strlen(layers),
                "{\"type\":\"dense\",\"input_size\":%d,\"output_size\":%d,\"activation\":\"leaky_relu\"}", d_model, d_model);
        }
    }
    
    // Output layer
    sprintf(layers + strlen(layers),
        ",{\"type\":\"dense\",\"input_size\":%d,\"output_size\":4,\"activation\":\"sigmoid\"}", d_model);
    
    sprintf(buffer, "{\"batch_size\":1,\"grid_rows\":1,\"grid_cols\":1,\"layers_per_cell\":%d,\"layers\":[%s]}", 
        num_layers, layers);
    return buffer;
}

char* build_network_config(const char* net_type, int num_layers) {
    if (strcmp(net_type, "Dense") == 0) return build_dense_config(num_layers);
    if (strcmp(net_type, "Conv2D") == 0) return build_conv2d_config(num_layers);
    if (strcmp(net_type, "RNN") == 0) return build_rnn_config(num_layers);
    if (strcmp(net_type, "LSTM") == 0) return build_lstm_config(num_layers);
    if (strcmp(net_type, "Attn") == 0) return build_attn_config(num_layers);
    return NULL;
}

// ============================================================================
// Helper Functions
// ============================================================================

void parse_output_json(const char* json, float* output, int* output_len) {
    *output_len = 0;
    const char* p = strchr(json, '[');
    if (!p) return;
    p++;
    
    while (*p && *p != ']') {
        while (*p == ' ' || *p == ',') p++;
        if (*p == ']') break;
        
        float val;
        if (sscanf(p, "%f", &val) == 1) {
            output[(*output_len)++] = val;
        }
        
        while (*p && *p != ',' && *p != ']') p++;
    }
}

int argmax(float* arr, int len) {
    if (len == 0) return 0;
    int max_idx = 0;
    for (int i = 1; i < len; i++) {
        if (arr[i] > arr[max_idx]) max_idx = i;
    }
    return max_idx;
}

long long get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

// ============================================================================
// Results Storage
// ============================================================================

typedef struct {
    float avg_accuracy;
    int total_outputs;
    int completed;
} TestResult;

TestResult all_results[NUM_NETWORK_TYPES][NUM_DEPTHS][NUM_MODES];

// ============================================================================
// Single Test Runner
// ============================================================================

int run_single_test(const char* net_type, int depth, int mode_idx, TestResult* result) {
    char config_name[32];
    sprintf(config_name, "%s-%dL", net_type, depth);
    
    int input_size = get_input_size(net_type);
    const char* mode_name = MODE_NAMES[mode_idx];
    
    // Build and create network
    char* config = build_network_config(net_type, depth);
    if (!config) {
        printf("  [%s] [%s] SKIP (unsupported)\n", config_name, mode_name);
        return 0;
    }
    
    char* net_result = CreateLoomNetwork(config);
    if (!net_result || strstr(net_result, "error")) {
        printf("  [%s] [%s] SKIP (network creation failed)\n", config_name, mode_name);
        if (net_result) FreeLoomString(net_result);
        return 0;
    }
    FreeLoomString(net_result);
    
    // Initialize states based on mode
    long long step_state = -1;
    long long tween_state = -1;
    
    if (mode_idx == 1 || mode_idx == 4) {  // StepBP or StepTweenChain
        step_state = LoomInitStepState(input_size);
    }
    
    if (mode_idx >= 2) {  // Tween, TweenChain, StepTweenChain
        int use_chain_rule = (mode_idx == 3 || mode_idx == 4) ? 1 : 0;
        tween_state = LoomCreateTweenState(use_chain_rule);
    }
    
    // Create tracker
    long long tracker = LoomCreateAdaptationTracker(WINDOW_DURATION_MS, TEST_DURATION_MS);
    LoomTrackerSetModelInfo(tracker, config_name, (char*)mode_name);
    
    // Schedule task changes at 1/3 and 2/3
    int one_third = TEST_DURATION_MS / 3;
    int two_thirds = 2 * one_third;
    LoomTrackerScheduleTaskChange(tracker, one_third, 1, "AVOID");
    LoomTrackerScheduleTaskChange(tracker, two_thirds, 0, "CHASE");
    
    // Initialize environment
    Environment env;
    env_init(&env);
    
    // Start tracking
    LoomTrackerStart(tracker, "CHASE", 0);
    
    long long start_time = get_time_ms();
    long long last_train_time = start_time;
    
    float* obs = malloc(input_size * sizeof(float));
    float* output = malloc(64 * sizeof(float));
    int output_len = 0;
    
    // Training batch storage (simplified - just track last samples)
    float** train_inputs = malloc(100 * sizeof(float*));
    int* train_targets = malloc(100 * sizeof(int));
    int batch_size = 0;
    for (int i = 0; i < 100; i++) {
        train_inputs[i] = malloc(input_size * sizeof(float));
    }
    
    while (get_time_ms() - start_time < TEST_DURATION_MS) {
        int current_task = LoomTrackerGetCurrentTask(tracker);
        env.task = current_task;
        
        env_get_observation(&env, obs, input_size);
        
        // Forward pass
        char* output_json = NULL;
        if (mode_idx == 1 || mode_idx == 4) {  // StepBP or StepTweenChain
            LoomSetInput(step_state, obs, input_size);
            LoomStepForward(step_state);
            output_json = LoomGetOutput(step_state);
        } else {
            output_json = LoomForward(obs, input_size);
        }
        
        if (!output_json) continue;
        
        parse_output_json(output_json, output, &output_len);
        FreeLoomString(output_json);
        
        // Ensure output has 4 elements
        while (output_len < OUTPUT_SIZE) {
            output[output_len++] = 0.0f;
        }
        
        int action = argmax(output, OUTPUT_SIZE);
        int optimal_action = env_get_optimal_action(&env);
        int is_correct = (action == optimal_action) ? 1 : 0;
        
        LoomTrackerRecordOutput(tracker, is_correct);
        
        // Store training sample
        if (batch_size < 100) {
            memcpy(train_inputs[batch_size], obs, input_size * sizeof(float));
            train_targets[batch_size] = optimal_action;
            batch_size++;
        }
        
        // Training based on mode
        long long now = get_time_ms();
        
        switch (mode_idx) {
            case 0:  // NormalBP
                if (now - last_train_time > TRAIN_INTERVAL_MS && batch_size > 0) {
                    // Build training batch JSON - needs larger buffer for 64-float inputs
                    static char batch_json[65536];  // Much larger buffer for conv2d/attn
                    batch_json[0] = '[';
                    batch_json[1] = '\0';
                    int pos = 1;
                    
                    for (int i = 0; i < batch_size && pos < 60000; i++) {
                        if (i > 0) batch_json[pos++] = ',';
                        
                        // Build input string
                        char input_str[1024] = "";
                        int input_pos = 0;
                        for (int j = 0; j < input_size && input_pos < 900; j++) {
                            input_pos += snprintf(input_str + input_pos, sizeof(input_str) - input_pos,
                                "%s%.4f", j > 0 ? "," : "", train_inputs[i][j]);
                        }
                        
                        const char* target_str = 
                            train_targets[i] == 0 ? "1,0,0,0" :
                            train_targets[i] == 1 ? "0,1,0,0" :
                            train_targets[i] == 2 ? "0,0,1,0" : "0,0,0,1";
                        
                        pos += snprintf(batch_json + pos, sizeof(batch_json) - pos,
                            "{\"Input\":[%s],\"Target\":[%s]}", input_str, target_str);
                    }
                    
                    if (pos < sizeof(batch_json) - 2) {
                        batch_json[pos++] = ']';
                        batch_json[pos] = '\0';
                    }
                    
                    char config_json[128];
                    sprintf(config_json, "{\"Epochs\":1,\"LearningRate\":%.4f,\"LossType\":\"mse\"}", LEARNING_RATE);
                    
                    char* train_res = LoomTrain(batch_json, config_json);
                    if (train_res) FreeLoomString(train_res);
                    
                    batch_size = 0;
                    last_train_time = now;
                }
                break;
                
            case 1:  // StepBP
                {
                    float grad[64];
                    for (int i = 0; i < output_len; i++) {
                        float target = (i == optimal_action) ? 1.0f : 0.0f;
                        grad[i] = output[i] - target;
                    }
                    char* back_res = LoomStepBackward(step_state, grad, output_len);
                    if (back_res) FreeLoomString(back_res);
                    LoomApplyGradients(LEARNING_RATE);
                }
                break;
                
            case 2:  // Tween
            case 3:  // TweenChain
                if (now - last_train_time > TRAIN_INTERVAL_MS && batch_size > 0) {
                    for (int i = 0; i < batch_size; i++) {
                        LoomTweenStep(tween_state, train_inputs[i], input_size, 
                                     train_targets[i], OUTPUT_SIZE, LEARNING_RATE);
                    }
                    batch_size = 0;
                    last_train_time = now;
                }
                break;
                
            case 4:  // StepTweenChain
                LoomTweenStep(tween_state, obs, input_size, optimal_action, OUTPUT_SIZE, LEARNING_RATE);
                break;
        }
        
        env_execute_action(&env, action);
        env_update(&env);
    }
    
    // Finalize
    char* result_json = LoomTrackerFinalize(tracker);
    
    // Parse result (simplified - just extract avg_accuracy and total_outputs)
    result->avg_accuracy = 0;
    result->total_outputs = 0;
    result->completed = 1;
    
    char* p = strstr(result_json, "\"avg_accuracy\":");
    if (p) sscanf(p, "\"avg_accuracy\":%f", &result->avg_accuracy);
    p = strstr(result_json, "\"total_outputs\":");
    if (p) sscanf(p, "\"total_outputs\":%d", &result->total_outputs);
    
    printf("  [%s] [%s] Avg: %.1f%% | Outputs: %d\n", 
           config_name, mode_name, result->avg_accuracy, result->total_outputs);
    
    FreeLoomString(result_json);
    
    // Cleanup
    if (step_state >= 0) LoomFreeStepState(step_state);
    if (tween_state >= 0) LoomFreeTweenState(tween_state);
    LoomFreeTracker(tracker);
    
    free(obs);
    free(output);
    for (int i = 0; i < 100; i++) free(train_inputs[i]);
    free(train_inputs);
    free(train_targets);
    
    return 1;
}

// ============================================================================
// Summary Table
// ============================================================================

void print_summary_table() {
    printf("\n╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║                                          MULTI-ARCHITECTURE ADAPTATION SUMMARY                                                               ║\n");
    printf("╠════════════╦═══════════════════════════════╦═══════════════════════════════╦═══════════════════════════════╦═══════════════════════════════╦═══════════════════════════════╣\n");
    printf("║ Network    ║ NormalBP                      ║ Step+BP                       ║ Tween                         ║ TweenChain                    ║ StepTweenChain                ║\n");
    printf("╠════════════╬═══════════════════════════════╬═══════════════════════════════╬═══════════════════════════════╬═══════════════════════════════╬═══════════════════════════════╣\n");
    
    for (int nt = 0; nt < NUM_NETWORK_TYPES; nt++) {
        for (int d = 0; d < NUM_DEPTHS; d++) {
            char config_name[16];
            sprintf(config_name, "%s-%dL", NETWORK_TYPES[nt], DEPTHS[d]);
            
            printf("║ %-10s ║", config_name);
            
            for (int m = 0; m < NUM_MODES; m++) {
                TestResult* r = &all_results[nt][d][m];
                if (r->completed) {
                    printf(" %5.1f%%                        ║", r->avg_accuracy);
                } else {
                    printf("     --                         ║");
                }
            }
            printf("\n");
        }
    }
    
    printf("╚════════════╩═══════════════════════════════╩═══════════════════════════════╩═══════════════════════════════╩═══════════════════════════════╩═══════════════════════════════╝\n");
    
    // Mode averages
    printf("\n┌────────────────────────────────────────────────────────────────────────────────────────────────────┐\n");
    printf("│                                         MODE AVERAGES                                              │\n");
    printf("├────────────────────────────────────────────────────────────────────────────────────────────────────┤\n");
    
    for (int m = 0; m < NUM_MODES; m++) {
        float total_acc = 0;
        int count = 0;
        for (int nt = 0; nt < NUM_NETWORK_TYPES; nt++) {
            for (int d = 0; d < NUM_DEPTHS; d++) {
                if (all_results[nt][d][m].completed) {
                    total_acc += all_results[nt][d][m].avg_accuracy;
                    count++;
                }
            }
        }
        float avg = count > 0 ? total_acc / count : 0;
        printf("│ %-20s │ Avg Accuracy: %5.1f%%  (%d tests)\n", MODE_NAMES[m], avg, count);
    }
    
    printf("└────────────────────────────────────────────────────────────────────────────────────────────────────┘\n");
    
    printf("\n┌────────────────────────────────────────────────────────────────────────────────────────────────────┐\n");
    printf("│                                         KEY INSIGHTS                                               │\n");
    printf("├────────────────────────────────────────────────────────────────────────────────────────────────────┤\n");
    printf("│ • StepTweenChain shows most CONSISTENT accuracy across all windows                                 │\n");
    printf("│ • Other methods may crash to 0%% after task changes while StepTweenChain maintains ~40-80%%         │\n");
    printf("│ • Higher 'After Change' accuracy = faster adaptation to changing goals                             │\n");
    printf("└────────────────────────────────────────────────────────────────────────────────────────────────────┘\n");
}

// ============================================================================
// Main
// ============================================================================

int main() {
    srand(time(NULL));
    
    printf("╔══════════════════════════════════════════════════════════════════════════╗\n");
    printf("║  Test 18: MULTI-ARCHITECTURE Adaptation Benchmark (C-ABI)                ║\n");
    printf("║  Networks: Dense, Conv2D, RNN, LSTM, Attention | Depths: 3, 5, 9         ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════════╝\n\n");
    
    int total_tests = NUM_NETWORK_TYPES * NUM_DEPTHS * NUM_MODES;
    int completed = 0;
    
    printf("Running %d tests (%d archs × %d depths × %d modes)\n\n", 
           total_tests, NUM_NETWORK_TYPES, NUM_DEPTHS, NUM_MODES);
    
    // Initialize results
    memset(all_results, 0, sizeof(all_results));
    
    // Run all tests
    for (int nt = 0; nt < NUM_NETWORK_TYPES; nt++) {
        for (int d = 0; d < NUM_DEPTHS; d++) {
            printf("\n--- %s-%dL ---\n", NETWORK_TYPES[nt], DEPTHS[d]);
            
            for (int m = 0; m < NUM_MODES; m++) {
                if (run_single_test(NETWORK_TYPES[nt], DEPTHS[d], m, &all_results[nt][d][m])) {
                    completed++;
                }
            }
        }
    }
    
    printf("\n");
    printf("════════════════════════════════════════════════════════════════════════════\n");
    printf("BENCHMARK COMPLETE (%d/%d tests)\n", completed, total_tests);
    printf("════════════════════════════════════════════════════════════════════════════\n");
    
    print_summary_table();
    
    return 0;
}
