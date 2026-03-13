#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cabi_test.h"
#include "networks.h"

#ifdef _WIN32
#include <windows.h>
typedef HMODULE lib_handle;
#define load_lib(name) LoadLibraryA(name)
#define get_sym GetProcAddress
#define close_lib FreeLibrary
#else
#include <dlfcn.h>
typedef void* lib_handle;
#define load_lib(name) dlopen(name, RTLD_LAZY)
#define get_sym dlsym
#define close_lib dlclose
#endif

const char* symbols[] = {
    "FreeLoomString",
    "LoomApplyGradients",
    "LoomApplyRecursiveGradients",
    "LoomApplyTargetProp",
    "LoomApplyTargetPropGaps",
    "LoomBuildNetworkFromJSON",
    "LoomCalculateOptimalGPUTileSizeFromLimits",
    "LoomCNN1Backward",
    "LoomCNN1BackwardTiled",
    "LoomCNN1Forward",
    "LoomCNN1ForwardTiled",
    "LoomCNN2Backward",
    "LoomCNN2BackwardTiled",
    "LoomCNN2Forward",
    "LoomCNN2ForwardTiled",
    "LoomCNN3Backward",
    "LoomCNN3BackwardTiled",
    "LoomCNN3Forward",
    "LoomCNN3ForwardTiled",
    "LoomCompareDNA",
    "LoomComputeLayerStats",
    "LoomComputeLossGradient",
    "LoomConvTransposed1DBackward",
    "LoomConvTransposed1DForward",
    "LoomConvTransposed2DBackward",
    "LoomConvTransposed2DForward",
    "LoomConvTransposed3DBackward",
    "LoomConvTransposed3DForward",
    "LoomCreateNetwork",
    "LoomCreateSystolicState",
    "LoomCreateTargetPropState",
    "LoomCreateTransformer",
    "LoomDefaultTargetPropConfig",
    "LoomDenseBackward",
    "LoomDenseForward",
    "LoomDenseForwardTiled",
    "LoomDetokenize",
    "LoomDispatchCNN1",
    "LoomDispatchCNN2",
    "LoomDispatchCNN3",
    "LoomDispatchDense",
    "LoomDispatchDenseQ4",
    "LoomDispatchEmbedding",
    "LoomDispatchKVUpdate",
    "LoomDispatchLayer",
    "LoomDispatchLayerBackward",
    "LoomDispatchLSTMStep",
    "LoomDispatchMHA",
    "LoomDispatchResidual",
    "LoomDispatchRMSNorm",
    "LoomDispatchRNNStep",
    "LoomDispatchRoPE",
    "LoomDispatchSwiGLU",
    "LoomDispatchSwiGLUQ4",
    "LoomEmbeddingBackward",
    "LoomEmbeddingBackwardTiled",
    "LoomEmbeddingForward",
    "LoomEmbeddingForwardTiled",
    "LoomExtractDNA",
    "LoomExtractNetworkBlueprint",
    "LoomForwardTokenIDsWGPU",
    "LoomForwardWGPU",
    "LoomFreeNetwork",
    "LoomFreeSystolicState",
    "LoomFreeTokenizer",
    "LoomGetLayerSpec",
    "LoomGetLayerStats",
    "LoomGetLayerTelemetry",
    "LoomGetMethodsJSON",
    "LoomGetNetworkInfo",
    "LoomGetOutput",
    "LoomInitLayerNormCell",
    "LoomInitWGPU",
    "LoomKMeansBackward",
    "LoomKMeansForward",
    "LoomLayerNormBackward",
    "LoomLayerNormForward",
    "LoomLoadSafetensors",
    "LoomLoadSafetensorsFromBytes",
    "LoomLoadSafetensorsWithShapes",
    "LoomLoadTokenizer",
    "LoomLoadUniversal",
    "LoomLoadUniversalDetailed",
    "LoomLoadWithPrefixes",
    "LoomLSTMBackward",
    "LoomLSTMBackwardTiled",
    "LoomLSTMForward",
    "LoomLSTMForwardTiled",
    "LoomMHABackward",
    "LoomMHAForward",
    "LoomMHAForwardTiled",
    "LoomMorphLayer",
    "LoomNewVolumetricNetwork",
    "LoomParallelBackward",
    "LoomResidualBackward",
    "LoomResidualBackwardTiled",
    "LoomResidualForward",
    "LoomResidualForwardTiled",
    "LoomRMSNormBackward",
    "LoomRMSNormForward",
    "LoomRNNBackward",
    "LoomRNNBackwardTiled",
    "LoomRNNForward",
    "LoomRNNForwardTiled",
    "LoomSequentialForward",
    "LoomSetInput",
    "LoomSoftmaxBackward",
    "LoomSoftmaxForward",
    "LoomSwiGLUBackward",
    "LoomSwiGLUBackwardTiled",
    "LoomSwiGLUForward",
    "LoomSwiGLUForwardTiled",
    "LoomSyncToCPU",
    "LoomSyncToGPU",
    "LoomSystolicBackward",
    "LoomSystolicStep",
    "LoomTargetPropBackward",
    "LoomTargetPropBackwardChainRule",
    "LoomTargetPropBackwardTargetProp",
    "LoomTargetPropForward",
    "LoomTokenize"
};

#define NUM_SYMBOLS (sizeof(symbols) / sizeof(symbols[0]))

void verify_structs() {
    printf("+--------------------------------------------------+-------------+\n");
    printf("| Struct Name                                      | Size (Bytes)|\n");
    printf("+--------------------------------------------------+-------------+\n");
    printf("| LoomLayerSpec                                    | %-11zu |\n", sizeof(LoomLayerSpec));
    printf("| LoomLayerStats                                   | %-11zu |\n", sizeof(LoomLayerStats));
    printf("| LoomTensorMeta                                   | %-11zu |\n", sizeof(LoomTensorMeta));
    printf("| LoomTensorInfo                                   | %-11zu |\n", sizeof(LoomTensorInfo));
    printf("| LoomSafetensorsHeader                            | %-11zu |\n", sizeof(LoomSafetensorsHeader));
    printf("| LoomNetworkBlueprint                             | %-11zu |\n", sizeof(LoomNetworkBlueprint));
    printf("| WGPUDenseParams                                  | %-11zu |\n", sizeof(WGPUDenseParams));
    printf("| WGPUMHAParams                                    | %-11zu |\n", sizeof(WGPUMHAParams));
    printf("| WGPURMSNormParams                                | %-11zu |\n", sizeof(WGPURMSNormParams));
    printf("| WGPUKVParams                                     | %-11zu |\n", sizeof(WGPUKVParams));
    printf("| WGPURoPEParams                                   | %-11zu |\n", sizeof(WGPURoPEParams));
    printf("| WGPUEmbeddingParams                              | %-11zu |\n", sizeof(WGPUEmbeddingParams));
    printf("+--------------------------------------------------+-------------+\n\n");
}

int verify_symbols(lib_handle handle) {
    printf("+------------------------------------------------------+---------+\n");
    printf("| Symbol Name                                          | Status  |\n");
    printf("+------------------------------------------------------+---------+\n");

    int missing = 0;
    for (size_t i = 0; i < NUM_SYMBOLS; i++) {
        void* sym = (void*)get_sym(handle, symbols[i]);
        if (sym) {
            printf("| %-52s | [ PASS ] |\n", symbols[i]);
        } else {
            printf("| %-52s | [ FAIL ] |\n", symbols[i]);
            missing++;
        }
    }
    printf("+------------------------------------------------------+---------+\n");
    printf("| TOTAL SYMBOLS: %-3zu                                   | MISS: %-2d |\n", NUM_SYMBOLS, missing);
    printf("+------------------------------------------------------+---------+\n\n");

    if (missing > 0) {
        printf("[!] WARNING: %d symbols are missing from the library.\n\n", missing);
    } else {
        printf("[+] SUCCESS: All 121 core C-ABI symbols are present.\n\n");
    }
    return missing;
}

int main(int argc, char** argv) {
#ifdef _WIN32
    const char* lib_name = "welvet.dll";
#elif defined(__APPLE__)
    const char* lib_name = "welvet.dylib";
#else
    const char* lib_name = "welvet.so";
#endif

    if (argc > 1) lib_name = argv[1];

    printf("\n=== Loom C-ABI Diagnostic Report ===\n\n");
    printf("[*] Loading library: %s\n", lib_name);
    lib_handle handle = load_lib(lib_name);
    if (!handle) {
        printf("[!] Failed to load library\n");
        return 1;
    }

    verify_structs();

    int missing = verify_symbols(handle);

    // Functional Check: LoomBuildNetworkFromJSON
    typedef long long (*fn_BuildNetwork)(const char*);
    typedef void (*fn_FreeNetwork)(long long);
    typedef char* (*fn_GetNetworkInfo)(long long);
    typedef void (*fn_FreeLoomString)(char*);
    typedef LoomLayerSpec (*fn_GetLayerSpec)(long long, int);

    fn_BuildNetwork build = (fn_BuildNetwork)get_sym(handle, "LoomBuildNetworkFromJSON");
    fn_FreeNetwork free_net = (fn_FreeNetwork)get_sym(handle, "LoomFreeNetwork");
    fn_GetNetworkInfo get_info = (fn_GetNetworkInfo)get_sym(handle, "LoomGetNetworkInfo");
    fn_FreeLoomString free_str = (fn_FreeLoomString)get_sym(handle, "FreeLoomString");
    fn_GetLayerSpec get_spec = (fn_GetLayerSpec)get_sym(handle, "LoomGetLayerSpec");

    if (build && free_net && get_info && free_str && get_spec) {
        printf("[*] Running functional smoke test...\n");
        long long net_handle = build(DENSE_NETWORK_JSON);
        if (net_handle >= 0) {
            printf("[+] Network built successfully. Handle: %lld\n", net_handle);
            char* info = get_info(net_handle);
            printf("[+] Network Info: %s\n", info);
            free_str(info);

            LoomLayerSpec spec = get_spec(net_handle, 0);
            printf("[+] Layer 0 Spec: Type=%d, IH=%d, OH=%d\n", spec.Type, spec.InputHeight, spec.OutputHeight);

            free_net(net_handle);
            printf("[+] Network freed.\n");
        } else {
            printf("[!] Failed to build network\n");
        }
    }

    close_lib(handle);
    printf("[*] Verification complete.\n");
    return missing == 0 ? 0 : 1;
}
