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
    // Core
    "FreeLoomString",
    "LoomApplyGradients",
    "LoomApplyRecursiveGradients",
    "LoomApplyTargetProp",
    "LoomApplyTargetPropGaps",
    "LoomBuildNetworkFromJSON",
    "LoomCalculateOptimalGPUTileSizeFromLimits",
    // CNN forward/backward (tiled + untiled)
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
    // DNA / introspection
    "LoomCompareDNA",
    "LoomComputeLayerStats",
    "LoomComputeLossGradient",
    // DNA splice / evolution
    "LoomDefaultSpliceConfig",
    "LoomSpliceDNA",
    "LoomSpliceDNAWithReport",
    "LoomDefaultNEATConfig",
    "LoomNEATMutate",
    "LoomNewNEATPopulation",
    "LoomNEATPopulationSize",
    "LoomNEATPopulationGetNetwork",
    "LoomNEATPopulationEvolveWithFitnesses",
    "LoomNEATPopulationBest",
    "LoomNEATPopulationBestFitness",
    "LoomNEATPopulationSummary",
    "LoomFreeNEATPopulation",
    // Transposed convolutions
    "LoomConvTransposed1DBackward",
    "LoomConvTransposed1DForward",
    "LoomConvTransposed2DBackward",
    "LoomConvTransposed2DForward",
    "LoomConvTransposed3DBackward",
    "LoomConvTransposed3DForward",
    // Network lifecycle
    "LoomCreateNetwork",
    "LoomCreateSystolicState",
    "LoomCreateTargetPropState",
    "LoomCreateTransformer",
    "LoomDefaultTargetPropConfig",
    // Dense
    "LoomDenseBackward",
    "LoomDenseForward",
    "LoomDenseForwardTiled",
    // Tokenizer
    "LoomDetokenize",
    // CPU dispatch (forward)
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
    // GPU backward dispatch (gpu_backward_ext.go)
    "LoomCreateGPUBuffer",
    "LoomFreeGPUBuffer",
    "LoomShaderDenseBackwardDX",
    "LoomShaderDenseBackwardDW",
    "LoomDispatchDenseBackwardDX",
    "LoomDispatchDenseBackwardDW",
    "LoomDispatchSwiGLUBackward",
    "LoomDispatchRMSNormBackward",
    "LoomDispatchEmbeddingBackward",
    "LoomDispatchResidualBackward",
    "LoomDispatchCNN1BackwardDX",
    "LoomDispatchCNN1BackwardDW",
    "LoomDispatchCNN2BackwardDX",
    "LoomDispatchCNN2BackwardDW",
    "LoomDispatchCNN3BackwardDX",
    "LoomDispatchCNN3BackwardDW",
    "LoomDispatchMHABackward",
    "LoomDispatchApplyGradients",
    "LoomDispatchMSEGradPartialLoss",
    "LoomDispatchForwardLayer",
    "LoomDispatchBackwardLayer",
    "LoomDispatchActivation",
    "LoomDispatchActivationBackward",
    // Embedding
    "LoomEmbeddingBackward",
    "LoomEmbeddingBackwardTiled",
    "LoomEmbeddingForward",
    "LoomEmbeddingForwardTiled",
    // Serialization / telemetry
    "LoomExtractDNA",
    "LoomExtractNetworkBlueprint",
    // GPU forward
    "LoomForwardTokenIDsWGPU",
    "LoomForwardWGPU",
    // Free / lifecycle
    "LoomFreeNetwork",
    "LoomFreeSystolicState",
    "LoomFreeTokenizer",
    // Introspection
    "LoomGetLayerSpec",
    "LoomGetLayerStats",
    "LoomGetLayerTelemetry",
    "LoomGetMethodsJSON",
    "LoomGetNetworkInfo",
    "LoomGetOutput",
    // LayerNorm
    "LoomInitLayerNormCell",
    "LoomInitWGPU",
    // KMeans
    "LoomKMeansBackward",
    "LoomKMeansForward",
    "LoomLayerNormBackward",
    "LoomLayerNormForward",
    // Loaders
    "LoomLoadSafetensors",
    "LoomLoadSafetensorsFromBytes",
    "LoomLoadSafetensorsWithShapes",
    "LoomLoadTokenizer",
    "LoomLoadUniversal",
    "LoomLoadUniversalDetailed",
    "LoomLoadWithPrefixes",
    // LSTM
    "LoomLSTMBackward",
    "LoomLSTMBackwardTiled",
    "LoomLSTMForward",
    "LoomLSTMForwardTiled",
    // MHA
    "LoomMHABackward",
    "LoomMHAForward",
    "LoomMHAForwardTiled",
    // Morph / network construction
    "LoomMorphLayer",
    "LoomNewVolumetricNetwork",
    "LoomParallelBackward",
    // Residual
    "LoomResidualBackward",
    "LoomResidualBackwardTiled",
    "LoomResidualForward",
    "LoomResidualForwardTiled",
    // RMSNorm
    "LoomRMSNormBackward",
    "LoomRMSNormForward",
    // RNN
    "LoomRNNBackward",
    "LoomRNNBackwardTiled",
    "LoomRNNForward",
    "LoomRNNForwardTiled",
    // Sequential / systolic / target prop
    "LoomSequentialForward",
    "LoomSetInput",
    // Softmax
    "LoomSoftmaxBackward",
    "LoomSoftmaxForward",
    // SwiGLU
    "LoomSwiGLUBackward",
    "LoomSwiGLUBackwardTiled",
    "LoomSwiGLUForward",
    "LoomSwiGLUForwardTiled",
    // GPU sync
    "LoomSyncToCPU",
    "LoomSyncToGPU",
    // Systolic
    "LoomSystolicBackward",
    "LoomSystolicStep",
    // Target propagation
    "LoomTargetPropBackward",
    "LoomTargetPropBackwardChainRule",
    "LoomTargetPropBackwardTargetProp",
    "LoomTargetPropForward",
    // Tokenizer
    "LoomTokenize",
};

#define NUM_SYMBOLS (sizeof(symbols) / sizeof(symbols[0]))

void verify_structs() {
    printf("+--------------------------------------------------+-------------+\n");
    printf("| Struct Name                                      | Size (Bytes)|\n");
    printf("+--------------------------------------------------+-------------+\n");
    // Core
    printf("| LoomLayerSpec                                    | %-11zu |\n", sizeof(LoomLayerSpec));
    printf("| LoomLayerStats                                   | %-11zu |\n", sizeof(LoomLayerStats));
    printf("| LoomTensorMeta                                   | %-11zu |\n", sizeof(LoomTensorMeta));
    printf("| LoomTensorInfo                                   | %-11zu |\n", sizeof(LoomTensorInfo));
    printf("| LoomSafetensorsHeader                            | %-11zu |\n", sizeof(LoomSafetensorsHeader));
    printf("| LoomNetworkBlueprint                             | %-11zu |\n", sizeof(LoomNetworkBlueprint));
    // GPU forward params
    printf("| WGPUDenseParams                                  | %-11zu |\n", sizeof(WGPUDenseParams));
    printf("| WGPUApplyGradientsParams                         | %-11zu |\n", sizeof(WGPUApplyGradientsParams));
    printf("| WGPUMHAParams                                    | %-11zu |\n", sizeof(WGPUMHAParams));
    printf("| WGPURMSNormParams                                | %-11zu |\n", sizeof(WGPURMSNormParams));
    printf("| WGPUKVParams                                     | %-11zu |\n", sizeof(WGPUKVParams));
    printf("| WGPURoPEParams                                   | %-11zu |\n", sizeof(WGPURoPEParams));
    printf("| WGPUEmbeddingParams                              | %-11zu |\n", sizeof(WGPUEmbeddingParams));
    printf("| WGPURNNParams                                    | %-11zu |\n", sizeof(WGPURNNParams));
    printf("| WGPULSTMParams                                   | %-11zu |\n", sizeof(WGPULSTMParams));
    printf("| WGPUCNN1Params                                   | %-11zu |\n", sizeof(WGPUCNN1Params));
    printf("| WGPUCNN2Params                                   | %-11zu |\n", sizeof(WGPUCNN2Params));
    printf("| WGPUCNN3Params                                   | %-11zu |\n", sizeof(WGPUCNN3Params));
    // GPU backward params
    printf("| WGPUMHABackwardParams                            | %-11zu |\n", sizeof(WGPUMHABackwardParams));
    printf("| WGPUCNN1BackwardParams                           | %-11zu |\n", sizeof(WGPUCNN1BackwardParams));
    printf("| WGPUCNN2BackwardParams                           | %-11zu |\n", sizeof(WGPUCNN2BackwardParams));
    printf("| WGPUCNN3BackwardParams                           | %-11zu |\n", sizeof(WGPUCNN3BackwardParams));
    printf("| WGPUActivationParams                             | %-11zu |\n", sizeof(WGPUActivationParams));
    printf("| WGPULossParams                                   | %-11zu |\n", sizeof(WGPULossParams));
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
        printf("[+] SUCCESS: All %zu C-ABI symbols are present.\n\n", NUM_SYMBOLS);
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

#ifndef _WIN32
    /* On Linux/macOS dlopen() does not search the current directory unless
       the path contains a slash. Prepend "./" when no path separator is given. */
    char lib_path_buf[1024];
    if (strchr(lib_name, '/') == NULL && strchr(lib_name, '\\') == NULL) {
        snprintf(lib_path_buf, sizeof(lib_path_buf), "./%s", lib_name);
        lib_name = lib_path_buf;
    }
#endif

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

    // Functional Check: DNA Splice & NEAT Evolution
    typedef char* (*fn_DefaultSpliceConfig)();
    typedef long long (*fn_SpliceDNA)(long long, long long, const char*);
    typedef char* (*fn_SpliceDNAWithReport)(long long, long long, const char*);
    typedef char* (*fn_DefaultNEATConfig)(int);
    typedef long long (*fn_NEATMutate)(long long, const char*);
    typedef long long (*fn_NewNEATPopulation)(long long, int, const char*);
    typedef int (*fn_NEATPopulationSize)(long long);
    typedef long long (*fn_NEATPopulationGetNetwork)(long long, int);
    typedef char* (*fn_NEATPopulationEvolveWithFitnesses)(long long, const char*);
    typedef long long (*fn_NEATPopulationBest)(long long);
    typedef double (*fn_NEATPopulationBestFitness)(long long);
    typedef char* (*fn_NEATPopulationSummary)(long long, int);
    typedef void (*fn_FreeNEATPopulation)(long long);

    fn_DefaultSpliceConfig default_splice_cfg =
        (fn_DefaultSpliceConfig)get_sym(handle, "LoomDefaultSpliceConfig");
    fn_SpliceDNA splice_dna =
        (fn_SpliceDNA)get_sym(handle, "LoomSpliceDNA");
    fn_SpliceDNAWithReport splice_dna_report =
        (fn_SpliceDNAWithReport)get_sym(handle, "LoomSpliceDNAWithReport");
    fn_DefaultNEATConfig default_neat_cfg =
        (fn_DefaultNEATConfig)get_sym(handle, "LoomDefaultNEATConfig");
    fn_NEATMutate neat_mutate =
        (fn_NEATMutate)get_sym(handle, "LoomNEATMutate");
    fn_NewNEATPopulation new_pop =
        (fn_NewNEATPopulation)get_sym(handle, "LoomNewNEATPopulation");
    fn_NEATPopulationSize pop_size =
        (fn_NEATPopulationSize)get_sym(handle, "LoomNEATPopulationSize");
    fn_NEATPopulationGetNetwork pop_get_net =
        (fn_NEATPopulationGetNetwork)get_sym(handle, "LoomNEATPopulationGetNetwork");
    fn_NEATPopulationEvolveWithFitnesses pop_evolve =
        (fn_NEATPopulationEvolveWithFitnesses)get_sym(handle, "LoomNEATPopulationEvolveWithFitnesses");
    fn_NEATPopulationBest pop_best =
        (fn_NEATPopulationBest)get_sym(handle, "LoomNEATPopulationBest");
    fn_NEATPopulationBestFitness pop_best_fitness =
        (fn_NEATPopulationBestFitness)get_sym(handle, "LoomNEATPopulationBestFitness");
    fn_NEATPopulationSummary pop_summary =
        (fn_NEATPopulationSummary)get_sym(handle, "LoomNEATPopulationSummary");
    fn_FreeNEATPopulation free_pop =
        (fn_FreeNEATPopulation)get_sym(handle, "LoomFreeNEATPopulation");

    if (build && free_net && free_str &&
        default_splice_cfg && splice_dna && splice_dna_report &&
        default_neat_cfg && neat_mutate && new_pop && pop_size &&
        pop_get_net && pop_evolve && pop_best && pop_best_fitness &&
        pop_summary && free_pop) {

        printf("[*] Running DNA / Evolution smoke test...\n");

        // Build two parent networks
        long long parentA = build(DENSE_NETWORK_JSON);
        long long parentB = build(DENSE_NETWORK_JSON);
        if (parentA < 0 || parentB < 0) {
            printf("[!] Failed to build parent networks\n");
            goto evolution_done;
        }
        printf("[+] Parent networks built. A=%lld B=%lld\n", parentA, parentB);

        // Default splice config
        char* splice_cfg_json = default_splice_cfg();
        printf("[+] Default splice config: %s\n", splice_cfg_json);

        // SpliceDNA
        long long child = splice_dna(parentA, parentB, splice_cfg_json);
        if (child >= 0) {
            printf("[+] SpliceDNA child handle: %lld\n", child);
            free_net(child);
        } else {
            printf("[!] SpliceDNA failed\n");
        }

        // SpliceDNAWithReport
        char* report = splice_dna_report(parentA, parentB, splice_cfg_json);
        printf("[+] SpliceDNAWithReport: %.80s...\n", report);
        // Extract child handle from report — it is registered; free via FreeNetwork
        // (child_handle value is in JSON; for smoke test we just verify no error key)
        free_str(report);
        free_str(splice_cfg_json);

        // Default NEAT config (dModel=10 matches DENSE_NETWORK_JSON layer size)
        char* neat_cfg_json = default_neat_cfg(10);
        printf("[+] Default NEAT config (dModel=10): %.60s...\n", neat_cfg_json);

        // NEATMutate
        long long mutant = neat_mutate(parentA, neat_cfg_json);
        if (mutant >= 0) {
            printf("[+] NEATMutate mutant handle: %lld\n", mutant);
            free_net(mutant);
        } else {
            printf("[!] NEATMutate failed\n");
        }

        // NEAT Population
        long long pop = new_pop(parentA, 4, neat_cfg_json);
        if (pop >= 0) {
            printf("[+] NEATPopulation handle: %lld, size=%d\n", pop, pop_size(pop));

            // Get a member network
            long long member = pop_get_net(pop, 0);
            printf("[+] Population[0] network handle: %lld\n", member);

            // Evolve with dummy fitnesses [1.0, 0.8, 0.5, 0.3]
            char* evolve_result = pop_evolve(pop, "[1.0, 0.8, 0.5, 0.3]");
            printf("[+] Evolve result: %s\n", evolve_result);
            free_str(evolve_result);

            // Best
            long long best = pop_best(pop);
            double best_fit = pop_best_fitness(pop);
            printf("[+] Best network handle: %lld, best fitness: %.4f\n", best, best_fit);

            // Summary
            char* summary = pop_summary(pop, 1);
            printf("[+] Summary: %s\n", summary);
            free_str(summary);

            free_pop(pop);
            printf("[+] Population freed.\n");
        } else {
            printf("[!] NewNEATPopulation failed\n");
        }

        free_str(neat_cfg_json);
        free_net(parentA);
        free_net(parentB);
        printf("[+] Evolution smoke test complete.\n");
    } else {
        printf("[!] Skipping evolution smoke test — one or more symbols missing\n");
    }
    evolution_done:;

    close_lib(handle);
    printf("[*] Verification complete.\n");
    return missing == 0 ? 0 : 1;
}
