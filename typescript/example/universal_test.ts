
import welvet from "../src/index.js";
import { Network, LayerConfig } from "../src/types.js";
import { dirname, join } from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Initialize WASM
await welvet.init();
console.log("✅ WASM Initialized");

// Global counters
let testsPassed = 0;
let testsFailed = 0;
let testsSkipped = 0;

function log(type: string, msg: string) {
    if (type === "success") console.log(`\x1b[32m${msg}\x1b[0m`);
    else if (type === "error") console.log(`\x1b[31m${msg}\x1b[0m`);
    else if (type === "warn") console.log(`\x1b[33m${msg}\x1b[0m`);
    else console.log(msg);
}

// Helper: Create network
function createNetwork(config: object | string): Network | null {
    try {
        const net = welvet.createNetwork(config);
        return net;
    } catch (e) {
        return null; // WASM error handling might throw or return null? Wrapper throws?
    }
}

// ----------------------------------------------------------------------------
// 1. Layer Tests (Auto-Generated 72 Tests)
// ----------------------------------------------------------------------------
const layerTypes = [
    "Dense", "Conv2D", "MHA", "RNN", "LSTM",
    "LayerNorm", "RMSNorm", "SwiGLU",
    "Parallel", "Sequential", "Softmax", "Conv1D"
];
// dtypes matching Go test suite, though WASM currently uses float32 primarily. 
// We test if "dtype" param is accepted in config without crashing.
const dtypes = ["float32", "float64", "int32", "int16", "int8", "uint8"];

console.log("\n┌──────────────────────────────────────────────────────────────────────┐");
console.log("│ Layer & DType Compatibility (One-Shot)                              │");
console.log("└──────────────────────────────────────────────────────────────────────┘");

async function testLayerWithDtype(layerType: string, dtype: string): Promise<boolean> {
    let inputSize = 4;
    let layers: LayerConfig[] = [];

    // Canonical configurations from tva/test_0_0_7.go
    if (layerType === "Dense") {
        layers = [
            { type: "dense", input_height: 4, output_height: 8, activation: "relu" },
            { type: "dense", input_height: 8, output_height: 8, activation: "gelu" },
            { type: "dense", input_height: 8, output_height: 4, activation: "linear" }
        ];
    } else if (layerType === "MHA") {
        layers = [{ type: "multi_head_attention", d_model: 4, num_heads: 2, seq_length: 4 }];
    } else if (layerType === "RNN") {
        layers = [{ type: "rnn", input_size: 4, hidden_size: 4, activation: "tanh" }];
    } else if (layerType === "LSTM") {
        layers = [{ type: "lstm", input_size: 4, hidden_size: 4 }];
    } else if (layerType === "LayerNorm") {
        layers = [{ type: "layer_norm", norm_size: 4 }];
    } else if (layerType === "RMSNorm") {
        layers = [{ type: "rms_norm", norm_size: 4 }];
    } else if (layerType === "SwiGLU") {
        layers = [{ type: "swiglu", input_size: 4, output_size: 4 }];
    } else if (layerType === "Conv2D") {
        inputSize = 16;
        layers = [{ type: "conv2d", input_width: 4, input_height: 4, input_channels: 1, kernel_size: 3, stride: 1, padding: 1, filters: 1 }];
    } else if (layerType === "Parallel") {
        layers = [{
            type: "parallel",
            branches: [
                { type: "dense", input_height: 4, output_height: 2 },
                { type: "dense", input_height: 4, output_height: 2 }
            ]
        }];
    } else if (layerType === "Sequential") {
        layers = [{
            type: "sequential",
            branches: [
                { type: "dense", input_height: 4, output_height: 4 },
                { type: "dense", input_height: 4, output_height: 4 }
            ]
        }];
    } else if (layerType === "Softmax") {
        layers = [{ type: "softmax", input_size: 4 }];
    } else if (layerType === "Conv1D") {
        layers = [{ type: "conv1d", input_channels: 1, output_channels: 1, kernel_size: 3, stride: 1, padding: 1 }];
    }

    const config = {
        dtype: dtype,
        batch_size: 1,
        grid_rows: 1,
        grid_cols: 1,
        layers_per_cell: layers.length,
        layers: layers
    };

    try {
        const net = createNetwork(config);
        if (!net) throw new Error("Failed to create network");

        // Simple forward pass
        const input = new Float32Array(inputSize).fill(0.1);
        // Method wrapper expects arguments as a JSON array [arg1, arg2...]
        // ForwardCPU takes 1 argument (input array), so we wrap it: [[0.1, ...]]
        const outputJSON = net.ForwardCPU(JSON.stringify([Array.from(input)]));
        const output = JSON.parse(outputJSON);
        if (!output || output.length === 0) throw new Error("Forward pass failed");

        // Save/Load test
        const saved = net.SaveModelToString(JSON.stringify(["model_" + layerType]));
        if (!saved || saved.includes("error")) throw new Error("Save failed");

        console.log(`  ✓ ${layerType.padEnd(10)} / ${dtype.padEnd(8)}: OK`);
        return true;

    } catch (e: any) {
        console.log(`  ❌ ${layerType.padEnd(10)} / ${dtype.padEnd(8)}: ${e.message}`);
        return false;
    }
}

for (const layer of layerTypes) {
    for (const dtype of dtypes) {
        if (await testLayerWithDtype(layer, dtype)) testsPassed++;
        else testsFailed++;
    }
}

// ----------------------------------------------------------------------------
// 2. Grafting Test
// ----------------------------------------------------------------------------
console.log("\n┌──────────────────────────────────────────────────────────────────────┐");
console.log("│ Network Grafting                                                    │");
console.log("└──────────────────────────────────────────────────────────────────────┘");

function testGrafting() {
    try {
        const config = JSON.stringify({
            batch_size: 1,
            grid_rows: 1,
            grid_cols: 1,
            layers_per_cell: 2,
            layers: [
                { type: "dense", input_height: 4, output_height: 8 },
                { type: "dense", input_height: 8, output_height: 4 }
            ]
        });

        // Create 2 separate handles
        const h1 = welvet.createKHandle(config);
        const h2 = welvet.createKHandle(config);

        if (h1 <= 0 || h2 <= 0) throw new Error("Failed to create graft handles");

        const result = welvet.graft([h1, h2], "concat");

        if (result.success) {
            console.log(`  ✓ Grafted: ${result.num_branches} branches, type=${result.type}`);
            log("success", "  ✅ PASSED: Network Grafting");
            testsPassed++;
        } else {
            throw new Error(result.error || "Unknown error");
        }

    } catch (e: any) {
        log("error", `  ❌ Grafting failed: ${e.message}`);
        testsFailed++;
    }
}
testGrafting();


// ----------------------------------------------------------------------------
// 3. Stats Tests (K-Means, Correlation)
// ----------------------------------------------------------------------------
console.log("\n┌──────────────────────────────────────────────────────────────────────┐");
console.log("│ Statistical Tools (K-Means, Correlation)                            │");
console.log("└──────────────────────────────────────────────────────────────────────┘");

function testStats() {
    // K-Means
    try {
        const data = [
            [1.0, 1.0], [1.1, 1.1],
            [5.0, 5.0], [5.1, 5.1]
        ];
        const res = welvet.kmeans(data, 2, 10);
        if (res.centroids.length !== 2) throw new Error("Incorrect centroid count");
        if (res.silhouette_score === undefined) {
            // Handle potential key casing mismatch
            const score = (res as any).SilhouetteScore;
            if (score !== undefined) console.log(`  ✓ K-Means: Score=${score.toFixed(3)}`);
            else console.log(`  ✓ K-Means: ${res.centroids.length} centroids found`);
        } else {
            console.log(`  ✓ K-Means: Score=${res.silhouette_score.toFixed(3)}`);
        }
    } catch (e: any) {
        log("error", `  ❌ K-Means failed: ${e.message}`);
        testsFailed++;
        return;
    }

    // Correlation
    try {
        const matrixA = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12]
        ];

        // Wrapper now handles null second arg and transforms result
        const res = welvet.correlation(matrixA);
        const matrix = res.pearson;

        if (matrix && matrix.length === 3) {
            console.log(`  ✓ Correlation: ${matrix.length}x${matrix[0].length} matrix`);
        } else {
            throw new Error("Invalid correlation result format");
        }

        log("success", "  ✅ PASSED: Stats Tools");
        testsPassed++;

    } catch (e: any) {
        log("error", `  ❌ Correlation failed: ${e.message}`);
        testsFailed++;
    }
}
testStats();

// ----------------------------------------------------------------------------
// 4. Optimizers Test
// ----------------------------------------------------------------------------
console.log("\n┌──────────────────────────────────────────────────────────────────────┐");
console.log("│ Optimizers                                                          │");
console.log("└──────────────────────────────────────────────────────────────────────┘");

function testOptimizers() {
    try {
        // Use leaky_relu and small grid for stability
        const config = {
            dtype: "float32", batch_size: 2, grid_rows: 1, grid_cols: 1, layers_per_cell: 1,
            layers: [{ type: "dense", input_height: 2, output_height: 2, activation: "leaky_relu" }]
        };
        const net = createNetwork(config);
        if (!net) throw new Error("Failed to create network");

        const trainConfig = { Epochs: 5, LearningRate: 0.001, LossType: "mse" };

        // Fix: Batch size is 2, so each batch must contain 2 samples concatenated
        // Input: [0,0] and [1,1] -> [0,0, 1,1]
        // Target: [0,0] and [1,1] -> [0,0, 1,1]
        const batches = [
            { Input: [0, 0, 1, 1], Target: [0, 0, 1, 1] }
        ];

        // Train returns JSON string directly from WASM method
        const resStr = net.Train(JSON.stringify([batches, trainConfig]));
        if (!resStr) throw new Error("Net.Train returned empty string");

        const res = JSON.parse(resStr);

        // Handle reflected array return [result, error]
        let resultObj = res;
        if (Array.isArray(res) && res.length > 0) {
            resultObj = res[0];
        }

        if (resultObj && (resultObj.FinalLoss !== undefined || resultObj.final_loss !== undefined)) {
            const loss = resultObj.FinalLoss !== undefined ? resultObj.FinalLoss : resultObj.final_loss;
            console.log(`  ✓ Optimizer training: Loss=${loss.toFixed(4)}`);
            log("success", "  ✅ PASSED: Optimizers");
            testsPassed++;
        } else {
            throw new Error("Training failed (FinalLoss undefined)");
        }
    } catch (e: any) {
        log("error", `  ❌ Optimizers failed: ${e.message}`);
        testsFailed++;
    }
}
testOptimizers();

// ----------------------------------------------------------------------------
// 5. Ensemble Test
// ----------------------------------------------------------------------------
console.log("\n┌──────────────────────────────────────────────────────────────────────┐");
console.log("│ Ensemble Features                                                   │");
console.log("└──────────────────────────────────────────────────────────────────────┘");

function testEnsemble() {
    try {
        // Mock models with Boolean Mask
        const models = [
            { ModelID: "model_1", Mask: [true, true, true, false, false] },
            { ModelID: "model_2", Mask: [false, false, false, true, true] }
        ];

        const matches = welvet.ensemble(models, 0.5);

        if (matches && matches.length > 0) {
            const m = matches[0];
            // Check keys (ModelA vs ModelA - case sensitive interface)
            if (m.ModelA && m.ModelB && m.Coverage >= 0.9) {
                console.log(`  ✓ Found pair: ${m.ModelA}+${m.ModelB} (Cov: ${m.Coverage.toFixed(2)})`);
                log("success", "  ✅ PASSED: Ensemble Features");
                testsPassed++;
                return;
            }
        }
        throw new Error("No valid matches found");
    } catch (e: any) {
        log("error", `  ❌ Ensemble failed: ${e.message}`);
        testsFailed++;
    }
}
testEnsemble();


// ----------------------------------------------------------------------------
// 6. Observer Pattern Test
// ----------------------------------------------------------------------------
console.log("\n┌──────────────────────────────────────────────────────────────────────┐");
console.log("│ Observer Pattern                                                    │");
console.log("└──────────────────────────────────────────────────────────────────────┘");

function testObserver() {
    try {
        const tracker = welvet.tracker(100, 1000);
        tracker.start("TaskA", 1);
        tracker.recordOutput(true);
        tracker.recordOutput(false);

        const resStr = tracker.finalize();
        const res = JSON.parse(resStr);

        // Go struct tag: avg_accuracy
        if (res.avg_accuracy !== undefined || res.AvgAccuracy !== undefined) {
            const acc = res.avg_accuracy !== undefined ? res.avg_accuracy : res.AvgAccuracy;
            console.log(`  ✓ Tracker finalized. Accuracy: ${acc.toFixed(2)}`);
            log("success", "  ✅ PASSED: Observer Pattern");
            testsPassed++;
        } else {
            throw new Error("Invalid tracker stats");
        }
    } catch (e: any) {
        log("error", `  ❌ Observer failed: ${e.message}`);
        testsFailed++;
    }
}
testObserver();


// ----------------------------------------------------------------------------
// 7. Introspection Test
// ----------------------------------------------------------------------------
console.log("\n┌──────────────────────────────────────────────────────────────────────┐");
console.log("│ Introspection                                                       │");
console.log("└──────────────────────────────────────────────────────────────────────┘");

function testIntrospection() {
    try {
        const config = {
            dtype: "float32", batch_size: 1, grid_rows: 1, grid_cols: 1, layers_per_cell: 1,
            layers: [{ type: "dense", input_height: 4, output_height: 4 }]
        };
        const net = welvet.createNetwork(config);

        // Match the HTML logic: net.TotalLayers() vs net.GetNetworkInfo()
        const totalLayersRaw = net.TotalLayers(); // If this exists in your Go code
        const totalLayers = JSON.parse(totalLayersRaw)[0];

        const inputSize = net.getInputSize();

        if (totalLayers >= 0 && inputSize > 0) {
            console.log(`  ✓ Introspection: ${totalLayers} layers, Input Size: ${inputSize}`);
            log("success", "  ✅ PASSED: Introspection");
            testsPassed++;
        } else {
            throw new Error("Invalid introspection data");
        }
    } catch (e: any) {
        log("error", `  ❌ Introspection failed: ${e.message}`);
        testsFailed++;
    }
}
testIntrospection();

// ----------------------------------------------------------------------------
// 8. Step & Tween API Test
// ----------------------------------------------------------------------------
console.log("\n┌──────────────────────────────────────────────────────────────────────┐");
console.log("│ Step & Tween API                                                    │");
console.log("└──────────────────────────────────────────────────────────────────────┘");

function testStepTween() {
    try {
        const config = {
            dtype: "float32", batch_size: 1, grid_rows: 1, grid_cols: 1, layers_per_cell: 1,
            layers: [{ type: "dense", input_height: 2, output_height: 2 }]
        };
        const net = createNetwork(config);

        // Test Step API
        const stepState = net.createStepState(2);
        stepState.setInput([0.5, 0.5]);
        const duration = stepState.stepForward();
        const out = stepState.getOutput();

        if (out.length === 2 && duration >= 0) {
            console.log(`  ✓ StepForward: ${out[0].toFixed(3)}, ${out[1].toFixed(3)} (${duration.toFixed(2)}ms)`);
        } else {
            throw new Error("StepForward failed");
        }

        // Test Tween API
        const tweenState = net.createTweenState(false);
        const loss = tweenState.TweenStep([0.5, 0.5], 0, 2, 0.01);

        if (loss >= 0) {
            console.log(`  ✓ TweenStep: Loss=${loss.toFixed(4)}`);
            log("success", "  ✅ PASSED: Step & Tween API");
            testsPassed++;
        } else {
            throw new Error("TweenStep failed");
        }

    } catch (e: any) {
        log("error", `  ❌ Step/Tween failed: ${e.message}`);
        testsFailed++;
    }
}
testStepTween();


// ----------------------------------------------------------------------------
// 9. Advanced Layers (Embedding, Residual)
// ----------------------------------------------------------------------------
console.log("\n┌──────────────────────────────────────────────────────────────────────┐");
console.log("│ Advanced Layers                                                     │");
console.log("└──────────────────────────────────────────────────────────────────────┘");

function testAdvancedLayers() {
    try {
        // Embedding Layer Test
        console.log("  > Testing Embedding Layer...");
        const embConfig = {
            dtype: "float32", batch_size: 1, grid_rows: 1, grid_cols: 1, layers_per_cell: 1,
            layers: [{ type: "embedding", vocab_size: 10, embedding_dim: 4 }]
        };
        const netEmb = createNetwork(embConfig);
        // Embedding input must be int indices, but wrapper handles float->int conversion for input array?
        // Wait, ForwardCPU takes float32 array.
        // Embedding layer casts float input to int index.
        const embOutStr = netEmb.ForwardCPU(JSON.stringify([[1.0]])); // Index 1
        const embOut = JSON.parse(embOutStr)[0]; // [vector]

        if (embOut.length === 4) {
            console.log(`    ✓ Embedding output size: ${embOut.length}`);
        } else {
            throw new Error(`Embedding output size mismatch: ${embOut.length}`);
        }

        // Residual Test
        console.log("  > Testing Residual Connection...");
        const resConfig = {
            dtype: "float32", batch_size: 1, grid_rows: 1, grid_cols: 1, layers_per_cell: 1,
            layers: [{ type: "dense", input_height: 4, output_height: 4, residual: true }]
        };
        const netRes = createNetwork(resConfig);
        const resOutStr = netRes.ForwardCPU(JSON.stringify([[0.1, 0.1, 0.1, 0.1]]));
        const resOut = JSON.parse(resOutStr)[0];

        if (resOut.length === 4) {
            console.log(`    ✓ Residual output size: ${resOut.length}`);
        } else {
            throw new Error("Residual output failed");
        }

        log("success", "  ✅ PASSED: Advanced Layers");
        testsPassed++;

    } catch (e: any) {
        log("error", `  ❌ Advanced Layers failed: ${e.message}`);
        testsFailed++;
    }
}
testAdvancedLayers();


// Summary
console.log("\n======================================================================");
console.log(`FINAL: ${testsPassed + testsFailed} TESTS RUN`);
console.log(`PASSED: ${testsPassed}`);
console.log(`FAILED: ${testsFailed}`);
console.log("======================================================================");

if (testsFailed > 0) process.exit(1);

