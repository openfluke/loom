#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "libloom.h"

// Mock libloom.h declarations if not yet updated in the file (since we just modified main.go)
// In a real scenario, we'd update the header or let go build generate it.
// For this test, we'll declare the new functions manually if needed, but let's assume libloom.h is updated or we declare them here.

// Declare the new functions (signatures based on our implementation)
extern long long LoomInitStepState(int inputSize);
extern void LoomSetInput(long long handle, float* input, int length);
extern long long LoomStepForward(long long handle);
extern char* LoomGetOutput(long long handle);
extern char* LoomStepBackward(long long handle, float* gradients, int length);
extern void LoomApplyGradients(float learningRate);
extern void LoomFreeStepState(long long handle);

int main() {
    printf("=== Testing LOOM Stepping C-ABI ===\n");

    // 1. Create Network
    const char* networkJSON = "{"
        "\"batch_size\": 1,"
        "\"grid_rows\": 1,"
        "\"grid_cols\": 1,"
        "\"layers_per_cell\": 1,"
        "\"layers\": ["
            "{\"type\": \"dense\", \"input_height\": 4, \"output_height\": 2, \"activation\": \"relu\"}"
        "]"
    "}";

    printf("Creating network...\n");
    char* result = CreateLoomNetwork(networkJSON);
    printf("Network created: %s\n", result);
    FreeLoomString(result);

    // 2. Initialize Step State
    printf("Initializing step state...\n");
    long long handle = LoomInitStepState(4);
    if (handle < 0) {
        printf("Failed to init step state\n");
        return 1;
    }
    printf("Step state handle: %lld\n", handle);

    // 3. Set Input
    printf("Setting input...\n");
    float input[] = {0.1f, 0.2f, 0.3f, 0.4f};
    LoomSetInput(handle, input, 4);

    // 4. Step Forward
    printf("Stepping forward...\n");
    long long duration = LoomStepForward(handle);
    printf("Forward step took %lld ns\n", duration);

    // 5. Get Output
    printf("Getting output...\n");
    char* outputJSON = LoomGetOutput(handle);
    printf("Output: %s\n", outputJSON);
    FreeLoomString(outputJSON);

    // 6. Step Backward
    printf("Stepping backward...\n");
    float grads[] = {1.0f, -1.0f}; // Gradient for 2 outputs
    char* backwardResult = LoomStepBackward(handle, grads, 2);
    printf("Backward result: %s\n", backwardResult);
    FreeLoomString(backwardResult);

    // 7. Apply Gradients
    printf("Applying gradients...\n");
    LoomApplyGradients(0.01f);

    // 8. Free State
    printf("Freeing state...\n");
    LoomFreeStepState(handle);

    printf("=== Test Complete ===\n");
    return 0;
}
