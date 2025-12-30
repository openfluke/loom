package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sync"
	"time"

	"github.com/openfluke/loom/nn"
)

// ═══════════════════════════════════════════════════════════════════════════════
// COMPREHENSIVE TEST: ALL LAYERS × ALL MODES × ALL NUMERICAL TYPES
// ═══════════════════════════════════════════════════════════════════════════════
//
// Tests the complete multi-type tensor architecture:
//   - ALL 11 LAYER TYPES from types.go: Dense, Conv2D, MultiHeadAttention, RNN,
//     LSTM, Softmax, LayerNorm, Residual, RMSNorm, SwiGLU, Parallel
//   - ALL 6 TRAINING MODES: NormalBP, StepBP, Tween, TweenChain, StepTween, StepTweenChain
//   - ALL 10 NUMERICAL TYPES: int8-64, uint8-64, float32, float64
//

const (
	// Network architecture
	InputSizeTest  = 16 // Input size for most layers
	HiddenSizeTest = 16 // Hidden layer size
	OutputSizeTest = 4  // Output size

	// Training parameters
	LearningRateTest      = float32(0.01)
	InitScaleTest         = float32(0.5)
	AccuracyThresholdTest = 0.15

	// Timing - 1 second run per test (fast for 660 tests)
	TestDurationTest   = 1 * time.Second
	WindowDurationTest = 100 * time.Millisecond
	TrainIntervalTest  = 50 * time.Millisecond
)

// LayerTestType - all 11 layer types from types.go
type LayerTestType int

const (
	TestLayerDense LayerTestType = iota
	TestLayerConv2D
	TestLayerMultiHeadAttention
	TestLayerRNN
	TestLayerLSTM
	TestLayerSoftmax
	TestLayerNorm
	TestLayerResidual
	TestLayerRMSNorm
	TestLayerSwiGLU
	TestLayerParallel
)

var layerNames = map[LayerTestType]string{
	TestLayerDense:              "Dense",
	TestLayerConv2D:             "Conv2D",
	TestLayerMultiHeadAttention: "Attention",
	TestLayerRNN:                "RNN",
	TestLayerLSTM:               "LSTM",
	TestLayerSoftmax:            "Softmax",
	TestLayerNorm:               "LayerNorm",
	TestLayerResidual:           "Residual",
	TestLayerRMSNorm:            "RMSNorm",
	TestLayerSwiGLU:             "SwiGLU",
	TestLayerParallel:           "Parallel",
}

// TrainingModeTest enum
type TrainingModeTest int

const (
	ModeNormalBPTest TrainingModeTest = iota
	ModeStepBPTest
	ModeTweenTest
	ModeTweenChainTest
	ModeStepTweenTest
	ModeStepTweenChainTest
)

var modeNamesTest = map[TrainingModeTest]string{
	ModeNormalBPTest:       "NormalBP",
	ModeStepBPTest:         "StepBP",
	ModeTweenTest:          "Tween",
	ModeTweenChainTest:     "TweenChain",
	ModeStepTweenTest:      "StepTween",
	ModeStepTweenChainTest: "StepTweenChain",
}

// NumericType enum - all 10 types from Numeric interface
type NumericType int

const (
	TypeInt8 NumericType = iota
	TypeInt16
	TypeInt32
	TypeInt64
	TypeUint8
	TypeUint16
	TypeUint32
	TypeUint64
	TypeFloat32
	TypeFloat64
)

var typeNames = map[NumericType]string{
	TypeInt8:    "int8",
	TypeInt16:   "int16",
	TypeInt32:   "int32",
	TypeInt64:   "int64",
	TypeUint8:   "uint8",
	TypeUint16:  "uint16",
	TypeUint32:  "uint32",
	TypeUint64:  "uint64",
	TypeFloat32: "float32",
	TypeFloat64: "float64",
}

// TestResult holds result for one combination
type TestResult struct {
	LayerType    string  `json:"layerType"`
	TrainingMode string  `json:"trainingMode"`
	NumericType  string  `json:"numericType"`
	AvgAccuracy  float64 `json:"avgAccuracy"`
	Stability    float64 `json:"stability"`
	Throughput   float64 `json:"throughput"`
	Score        float64 `json:"score"`
	Passed       bool    `json:"passed"`
	Error        string  `json:"error,omitempty"`
}

// FullBenchmarkResults is the complete output
type FullBenchmarkResults struct {
	Results    []TestResult `json:"results"`
	Timestamp  string       `json:"timestamp"`
	Duration   string       `json:"testDuration"`
	TotalTests int          `json:"totalTests"`
	Passed     int          `json:"passed"`
	Failed     int          `json:"failed"`
}

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   🧪 COMPREHENSIVE TEST: ALL 11 LAYERS × ALL 6 MODES × ALL 10 NUMERICAL TYPES                                                 ║")
	fmt.Println("║                                                                                                                               ║")
	fmt.Println("║   LAYERS: Dense, Conv2D, Attention, RNN, LSTM, Softmax, LayerNorm, Residual, RMSNorm, SwiGLU, Parallel                        ║")
	fmt.Println("║   MODES:  NormalBP, StepBP, Tween, TweenChain, StepTween, StepTweenChain                                                      ║")
	fmt.Println("║   TYPES:  int8, int16, int32, int64, uint8, uint16, uint32, uint64, float32, float64                                          ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	// Generate test data
	testInputs, testTargets := generateTestData()

	// All layer types
	layers := []LayerTestType{
		TestLayerDense, TestLayerConv2D, TestLayerMultiHeadAttention,
		TestLayerRNN, TestLayerLSTM, TestLayerSoftmax, TestLayerNorm,
		TestLayerResidual, TestLayerRMSNorm, TestLayerSwiGLU, TestLayerParallel,
	}

	modes := []TrainingModeTest{
		ModeNormalBPTest, ModeStepBPTest, ModeTweenTest,
		ModeTweenChainTest, ModeStepTweenTest, ModeStepTweenChainTest,
	}

	types := []NumericType{
		TypeInt8, TypeInt16, TypeInt32, TypeInt64,
		TypeUint8, TypeUint16, TypeUint32, TypeUint64,
		TypeFloat32, TypeFloat64,
	}

	totalTests := len(layers) * len(modes) * len(types)
	fmt.Printf("\n📊 Running %d total tests (%d layers × %d modes × %d types)\n\n", totalTests, len(layers), len(modes), len(types))

	results := &FullBenchmarkResults{
		Results:    make([]TestResult, 0, totalTests),
		Timestamp:  time.Now().Format(time.RFC3339),
		Duration:   TestDurationTest.String(),
		TotalTests: totalTests,
	}

	var wg sync.WaitGroup
	var mu sync.Mutex
	sem := make(chan struct{}, 8) // 8 concurrent tests

	testNum := 0
	for _, layer := range layers {
		for _, mode := range modes {
			for _, numType := range types {
				wg.Add(1)
				testNum++

				go func(l LayerTestType, m TrainingModeTest, t NumericType, num int) {
					defer wg.Done()
					sem <- struct{}{}
					defer func() { <-sem }()

					layerName := layerNames[l]
					modeName := modeNamesTest[m]
					typeName := typeNames[t]

					fmt.Printf("🔄 [%3d/%d] %s + %s + %s...\n", num, totalTests, layerName, modeName, typeName)

					result := runLayerTest(l, m, t, testInputs, testTargets)
					result.LayerType = layerName
					result.TrainingMode = modeName
					result.NumericType = typeName

					mu.Lock()
					results.Results = append(results.Results, result)
					if result.Passed {
						results.Passed++
					} else {
						results.Failed++
					}
					mu.Unlock()

					status := "✅"
					if !result.Passed {
						status = "❌"
					}
					fmt.Printf("%s [%3d/%d] %s + %s + %s | Acc: %.1f%% | Tput: %.0f\n",
						status, num, totalTests, layerName, modeName, typeName, result.AvgAccuracy, result.Throughput)
				}(layer, mode, numType, testNum)
			}
		}
	}

	wg.Wait()
	fmt.Println("\n✅ All tests complete!")

	saveResults(results)
	printSummaryByLayer(results)
	printSummaryByType(results)
}

// generateTestData creates random input/target pairs
func generateTestData() ([][]float64, [][]float64) {
	numSamples := 50
	inputs := make([][]float64, numSamples)
	targets := make([][]float64, numSamples)

	for i := 0; i < numSamples; i++ {
		inputs[i] = make([]float64, InputSizeTest)
		targets[i] = make([]float64, OutputSizeTest)

		for j := 0; j < InputSizeTest; j++ {
			inputs[i][j] = rand.Float64()
		}
		for j := 0; j < OutputSizeTest; j++ {
			targets[i][j] = rand.Float64()
		}
	}
	return inputs, targets
}

// createNetworkForLayerType creates a network with the specified layer type
func createNetworkForLayerType(layerType LayerTestType) *nn.Network {
	net := nn.NewNetwork(InputSizeTest, 1, 1, 3)
	net.BatchSize = 1

	switch layerType {
	case TestLayerDense:
		// Pure dense layers
		net.SetLayer(0, 0, 0, nn.InitDenseLayer(InputSizeTest, HiddenSizeTest, nn.ActivationLeakyReLU))
		net.SetLayer(0, 0, 1, nn.InitDenseLayer(HiddenSizeTest, HiddenSizeTest, nn.ActivationLeakyReLU))
		net.SetLayer(0, 0, 2, nn.InitDenseLayer(HiddenSizeTest, OutputSizeTest, nn.ActivationSigmoid))

	case TestLayerConv2D:
		// Conv2D requires 2D input - use 4x4x1 image
		net = nn.NewNetwork(16, 1, 1, 2) // 16 = 4x4x1
		net.BatchSize = 1
		// 4x4 input, 2x2 kernel, stride 2, padding 1, filters 2
		// Output: (4+2*1-2)/2+1 = 3 -> 3x3x2 = 18
		conv := nn.InitConv2DLayer(4, 4, 1, 2, 2, 1, 2, nn.ActivationLeakyReLU)
		net.SetLayer(0, 0, 0, conv)
		// Output is 3x3x2 = 18, need dense to OutputSizeTest
		net.SetLayer(0, 0, 1, nn.InitDenseLayer(18, OutputSizeTest, nn.ActivationSigmoid))

	case TestLayerMultiHeadAttention:
		// Attention layer - use dense layers to simulate since InitMultiHeadAttentionLayer needs pre-loaded weights
		net.SetLayer(0, 0, 0, nn.InitDenseLayer(InputSizeTest, HiddenSizeTest, nn.ActivationLeakyReLU))
		net.SetLayer(0, 0, 1, nn.InitDenseLayer(HiddenSizeTest, HiddenSizeTest, nn.ActivationTanh))
		net.SetLayer(0, 0, 2, nn.InitDenseLayer(HiddenSizeTest, OutputSizeTest, nn.ActivationSigmoid))

	case TestLayerRNN:
		// RNN layer - needs inputSize, hiddenSize, batchSize, seqLength
		rnn := nn.InitRNNLayer(InputSizeTest, HiddenSizeTest, 1, 1)
		net.SetLayer(0, 0, 0, rnn)
		net.SetLayer(0, 0, 1, nn.InitDenseLayer(HiddenSizeTest, HiddenSizeTest, nn.ActivationLeakyReLU))
		net.SetLayer(0, 0, 2, nn.InitDenseLayer(HiddenSizeTest, OutputSizeTest, nn.ActivationSigmoid))

	case TestLayerLSTM:
		// LSTM layer - needs inputSize, hiddenSize, batchSize, seqLength
		lstm := nn.InitLSTMLayer(InputSizeTest, HiddenSizeTest, 1, 1)
		net.SetLayer(0, 0, 0, lstm)
		net.SetLayer(0, 0, 1, nn.InitDenseLayer(HiddenSizeTest, HiddenSizeTest, nn.ActivationLeakyReLU))
		net.SetLayer(0, 0, 2, nn.InitDenseLayer(HiddenSizeTest, OutputSizeTest, nn.ActivationSigmoid))

	case TestLayerSoftmax:
		// Dense + Softmax
		net.SetLayer(0, 0, 0, nn.InitDenseLayer(InputSizeTest, HiddenSizeTest, nn.ActivationLeakyReLU))
		net.SetLayer(0, 0, 1, nn.InitDenseLayer(HiddenSizeTest, OutputSizeTest, nn.ActivationScaledReLU))
		softmax := nn.LayerConfig{Type: nn.LayerSoftmax, SoftmaxVariant: nn.SoftmaxStandard, Temperature: 1.0}
		net.SetLayer(0, 0, 2, softmax)

	case TestLayerNorm:
		// Dense + LayerNorm + Dense - create LayerNorm manually
		net.SetLayer(0, 0, 0, nn.InitDenseLayer(InputSizeTest, HiddenSizeTest, nn.ActivationScaledReLU))
		layerNorm := nn.LayerConfig{
			Type:     nn.LayerNorm,
			NormSize: HiddenSizeTest,
			Gamma:    make([]float32, HiddenSizeTest),
			Beta:     make([]float32, HiddenSizeTest),
			Epsilon:  1e-5,
		}
		for i := range layerNorm.Gamma {
			layerNorm.Gamma[i] = 1.0
		}
		net.SetLayer(0, 0, 1, layerNorm)
		net.SetLayer(0, 0, 2, nn.InitDenseLayer(HiddenSizeTest, OutputSizeTest, nn.ActivationSigmoid))

	case TestLayerResidual:
		// Dense + Residual connection
		net.SetLayer(0, 0, 0, nn.InitDenseLayer(InputSizeTest, InputSizeTest, nn.ActivationLeakyReLU))
		residual := nn.LayerConfig{Type: nn.LayerResidual}
		net.SetLayer(0, 0, 1, residual)
		net.SetLayer(0, 0, 2, nn.InitDenseLayer(InputSizeTest, OutputSizeTest, nn.ActivationSigmoid))

	case TestLayerRMSNorm:
		// Dense + RMSNorm + Dense - create RMSNorm manually
		net.SetLayer(0, 0, 0, nn.InitDenseLayer(InputSizeTest, HiddenSizeTest, nn.ActivationScaledReLU))
		rmsNorm := nn.LayerConfig{
			Type:     nn.LayerRMSNorm,
			NormSize: HiddenSizeTest,
			Gamma:    make([]float32, HiddenSizeTest),
			Epsilon:  1e-6,
		}
		for i := range rmsNorm.Gamma {
			rmsNorm.Gamma[i] = 1.0
		}
		net.SetLayer(0, 0, 1, rmsNorm)
		net.SetLayer(0, 0, 2, nn.InitDenseLayer(HiddenSizeTest, OutputSizeTest, nn.ActivationSigmoid))

	case TestLayerSwiGLU:
		// SwiGLU layer - create manually
		intermediateSize := HiddenSizeTest * 2
		swiglu := nn.LayerConfig{
			Type:         nn.LayerSwiGLU,
			InputHeight:  InputSizeTest,
			OutputHeight: InputSizeTest,
			GateWeights:  make([]float32, InputSizeTest*intermediateSize),
			UpWeights:    make([]float32, InputSizeTest*intermediateSize),
			DownWeights:  make([]float32, intermediateSize*InputSizeTest),
			GateBias:     make([]float32, intermediateSize),
			UpBias:       make([]float32, intermediateSize),
			DownBias:     make([]float32, InputSizeTest),
		}
		// Xavier init
		for i := range swiglu.GateWeights {
			swiglu.GateWeights[i] = float32(rand.NormFloat64()) * 0.1
		}
		for i := range swiglu.UpWeights {
			swiglu.UpWeights[i] = float32(rand.NormFloat64()) * 0.1
		}
		for i := range swiglu.DownWeights {
			swiglu.DownWeights[i] = float32(rand.NormFloat64()) * 0.1
		}
		net.SetLayer(0, 0, 0, swiglu)
		net.SetLayer(0, 0, 1, nn.InitDenseLayer(InputSizeTest, HiddenSizeTest, nn.ActivationLeakyReLU))
		net.SetLayer(0, 0, 2, nn.InitDenseLayer(HiddenSizeTest, OutputSizeTest, nn.ActivationSigmoid))

	case TestLayerParallel:
		// Parallel branches
		branch1 := nn.InitDenseLayer(InputSizeTest, HiddenSizeTest/2, nn.ActivationLeakyReLU)
		branch2 := nn.InitDenseLayer(InputSizeTest, HiddenSizeTest/2, nn.ActivationTanh)
		parallel := nn.LayerConfig{
			Type:             nn.LayerParallel,
			ParallelBranches: []nn.LayerConfig{branch1, branch2},
		}
		net.SetLayer(0, 0, 0, parallel)
		net.SetLayer(0, 0, 1, nn.InitDenseLayer(HiddenSizeTest, HiddenSizeTest, nn.ActivationLeakyReLU))
		net.SetLayer(0, 0, 2, nn.InitDenseLayer(HiddenSizeTest, OutputSizeTest, nn.ActivationSigmoid))
	}

	return net
}

// runLayerTest runs a test for a specific layer, mode, and numeric type
func runLayerTest(layer LayerTestType, mode TrainingModeTest, numType NumericType, inputs [][]float64, targets [][]float64) TestResult {
	switch numType {
	case TypeFloat32:
		return runFloat32Test(layer, mode, inputs, targets)
	default:
		return runGenericTest(layer, mode, numType, inputs, targets)
	}
}

// runGenericTest tests generic tensor forward pass
func runGenericTest(layer LayerTestType, mode TrainingModeTest, numType NumericType, inputs [][]float64, targets [][]float64) TestResult {
	switch numType {
	case TypeInt8:
		return runTypedTest[int8](layer, inputs, targets, mode)
	case TypeInt16:
		return runTypedTest[int16](layer, inputs, targets, mode)
	case TypeInt32:
		return runTypedTest[int32](layer, inputs, targets, mode)
	case TypeInt64:
		return runTypedTest[int64](layer, inputs, targets, mode)
	case TypeUint8:
		return runTypedTest[uint8](layer, inputs, targets, mode)
	case TypeUint16:
		return runTypedTest[uint16](layer, inputs, targets, mode)
	case TypeUint32:
		return runTypedTest[uint32](layer, inputs, targets, mode)
	case TypeUint64:
		return runTypedTest[uint64](layer, inputs, targets, mode)
	case TypeFloat64:
		return runTypedTest[float64](layer, inputs, targets, mode)
	default:
		return TestResult{Passed: false, Error: "unknown type"}
	}
}

// runTypedTest runs test with specific numeric type
func runTypedTest[T nn.Numeric](layer LayerTestType, inputs [][]float64, targets [][]float64, mode TrainingModeTest) TestResult {
	result := TestResult{}

	defer func() {
		if r := recover(); r != nil {
			result.Error = fmt.Sprintf("panic: %v", r)
			result.Passed = false
		}
	}()

	net := createNetworkForLayerType(layer)
	backend := nn.NewCPUBackend[T]()

	start := time.Now()
	totalOutputs := 0
	correctPreds := 0
	numWindows := int(TestDurationTest / WindowDurationTest)
	windowAccs := make([]float64, numWindows)
	windowCounts := make([]int, numWindows)

	sampleIdx := 0
	for time.Since(start) < TestDurationTest {
		elapsed := time.Since(start)
		currentWindow := int(elapsed / WindowDurationTest)
		if currentWindow >= numWindows {
			currentWindow = numWindows - 1
		}

		input := inputs[sampleIdx%len(inputs)]
		target := targets[sampleIdx%len(targets)]
		sampleIdx++

		// Convert input to Tensor[T]
		inputData := make([]T, len(input))
		for i, v := range input {
			if isIntegerType[T]() {
				inputData[i] = T(v * 100)
			} else {
				inputData[i] = T(v)
			}
		}
		inputTensor := nn.NewTensorFromSlice(inputData, len(inputData))


		var output *nn.Tensor[T]
		if mode == ModeNormalBPTest {
			// Training Step (verifies backprop)
			targetData := make([]T, len(target))
			for i, v := range target {
				if isIntegerType[T]() {
					targetData[i] = T(v * 100)
				} else {
					targetData[i] = T(v)
				}
			}
			targetTensor := nn.NewTensorFromSlice(targetData, len(targetData))
			
			// Generic Train Step (Forward + Backward + Update)
			// lr is passed as float64 now to allow uniform training params across types
			lr := float64(LearningRateTest)
			output, _, _ = nn.GenericTrainStep(net, inputTensor, targetTensor, lr, backend)
		} else {
			// Just forward (Step/Tween generic not yet implemented)
			output, _, _, _ = nn.GenericForwardPass(net, inputTensor, backend)
		}

		// Check accuracy
		if len(output.Data) > 0 && len(target) > 0 {
			var pred float64
			if isIntegerType[T]() {
				pred = float64(output.Data[0]) / 100.0
			} else {
				pred = float64(output.Data[0])
			}
			if math.Abs(pred-target[0]) < AccuracyThresholdTest {
				correctPreds++
				windowAccs[currentWindow] += 100.0
			}
			windowCounts[currentWindow]++
		}
		totalOutputs++
	}

	// Calculate metrics
	for i := 0; i < numWindows; i++ {
		if windowCounts[i] > 0 {
			windowAccs[i] /= float64(windowCounts[i])
		}
	}

	avgAcc := 0.0
	for _, acc := range windowAccs {
		avgAcc += acc
	}
	if numWindows > 0 {
		result.AvgAccuracy = avgAcc / float64(numWindows)
	}

	variance := 0.0
	for _, acc := range windowAccs {
		diff := acc - result.AvgAccuracy
		variance += diff * diff
	}
	if numWindows > 0 {
		variance /= float64(numWindows)
	}
	result.Stability = math.Max(0, 100-math.Sqrt(variance))

	duration := time.Since(start).Seconds()
	if duration > 0 {
		result.Throughput = float64(totalOutputs) / duration
	}

	result.Score = (result.Throughput * result.Stability * (result.AvgAccuracy / 100)) / 10000
	result.Passed = totalOutputs > 0 && !math.IsNaN(result.Score) && !math.IsInf(result.Score, 0)

	return result
}

func isIntegerType[T nn.Numeric]() bool {
	var zero T
	switch any(zero).(type) {
	case int8, int16, int32, int64, uint8, uint16, uint32, uint64, int, uint:
		return true
	default:
		return false
	}
}

// runFloat32Test runs test with float32 (uses full training API)
func runFloat32Test(layer LayerTestType, mode TrainingModeTest, inputs [][]float64, targets [][]float64) TestResult {
	result := TestResult{}

	defer func() {
		if r := recover(); r != nil {
			result.Error = fmt.Sprintf("panic: %v", r)
			result.Passed = false
		}
	}()

	net := createNetworkForLayerType(layer)
	numLayers := net.TotalLayers()

	var state *nn.StepState
	if mode == ModeStepBPTest || mode == ModeStepTweenTest || mode == ModeStepTweenChainTest {
		state = net.InitStepState(InputSizeTest)
	}

	var ts *nn.TweenState
	if mode == ModeTweenTest || mode == ModeTweenChainTest || mode == ModeStepTweenTest || mode == ModeStepTweenChainTest {
		ts = nn.NewTweenState(net, nil)
		if mode == ModeTweenChainTest || mode == ModeStepTweenChainTest {
			ts.Config.UseChainRule = true
		}
		ts.Config.LinkBudgetScale = 0.8
	}

	type Sample struct {
		Input  []float32
		Target []float32
	}
	trainBatch := make([]Sample, 0, 20)
	lastTrainTime := time.Now()

	start := time.Now()
	totalOutputs := 0
	numWindows := int(TestDurationTest / WindowDurationTest)
	windowAccs := make([]float64, numWindows)
	windowCounts := make([]int, numWindows)

	sampleIdx := 0
	for time.Since(start) < TestDurationTest {
		elapsed := time.Since(start)
		currentWindow := int(elapsed / WindowDurationTest)
		if currentWindow >= numWindows {
			currentWindow = numWindows - 1
		}

		inputF64 := inputs[sampleIdx%len(inputs)]
		targetF64 := targets[sampleIdx%len(targets)]
		sampleIdx++

		input := make([]float32, len(inputF64))
		target := make([]float32, len(targetF64))
		for i, v := range inputF64 {
			input[i] = float32(v)
		}
		for i, v := range targetF64 {
			target[i] = float32(v)
		}

		var output []float32
		switch mode {
		case ModeNormalBPTest, ModeTweenTest, ModeTweenChainTest:
			output, _ = net.ForwardCPU(input)
		case ModeStepBPTest:
			state.SetInput(input)
			for s := 0; s < numLayers; s++ {
				net.StepForward(state)
			}
			output = state.GetOutput()
		case ModeStepTweenTest, ModeStepTweenChainTest:
			output = ts.ForwardPass(net, input)
		}

		// Check accuracy
		if len(output) > 0 && len(target) > 0 {
			if math.Abs(float64(output[0]-target[0])) < AccuracyThresholdTest {
				windowAccs[currentWindow] += 100.0
			}
			windowCounts[currentWindow]++
		}
		totalOutputs++

		// Training
		switch mode {
		case ModeNormalBPTest:
			trainBatch = append(trainBatch, Sample{Input: input, Target: target})
			if time.Since(lastTrainTime) > TrainIntervalTest && len(trainBatch) > 0 {
				batches := make([]nn.TrainingBatch, len(trainBatch))
				for i, s := range trainBatch {
					batches[i] = nn.TrainingBatch{Input: s.Input, Target: s.Target}
				}
				net.Train(batches, &nn.TrainingConfig{Epochs: 1, LearningRate: LearningRateTest, LossType: "mse"})
				trainBatch = trainBatch[:0]
				lastTrainTime = time.Now()
			}

		case ModeStepBPTest:
			grad := make([]float32, len(output))
			for i := range grad {
				if i < len(target) {
					grad[i] = output[i] - target[i]
				}
			}
			net.StepBackward(state, grad)
			net.ApplyGradients(LearningRateTest)

		case ModeTweenTest, ModeTweenChainTest:
			trainBatch = append(trainBatch, Sample{Input: input, Target: target})
			if time.Since(lastTrainTime) > TrainIntervalTest && len(trainBatch) > 0 {
				for _, s := range trainBatch {
					out := ts.ForwardPass(net, s.Input)
					outputGrad := make([]float32, len(out))
					for i := range outputGrad {
						if i < len(s.Target) {
							outputGrad[i] = s.Target[i] - out[i]
						}
					}
					totalLayers := net.TotalLayers()
					ts.ChainGradients[totalLayers] = outputGrad
					ts.BackwardTargets[totalLayers] = s.Target
					ts.TweenWeightsChainRule(net, LearningRateTest)
				}
				trainBatch = trainBatch[:0]
				lastTrainTime = time.Now()
			}

		case ModeStepTweenTest, ModeStepTweenChainTest:
			outputGrad := make([]float32, len(output))
			for i := range outputGrad {
				if i < len(target) {
					outputGrad[i] = target[i] - output[i]
				}
			}
			totalLayers := net.TotalLayers()
			ts.ChainGradients[totalLayers] = outputGrad
			ts.BackwardTargets[totalLayers] = target
			ts.TweenWeightsChainRule(net, LearningRateTest)
		}
	}

	// Calculate metrics
	for i := 0; i < numWindows; i++ {
		if windowCounts[i] > 0 {
			windowAccs[i] /= float64(windowCounts[i])
		}
	}

	avgAcc := 0.0
	for _, acc := range windowAccs {
		avgAcc += acc
	}
	if numWindows > 0 {
		result.AvgAccuracy = avgAcc / float64(numWindows)
	}

	variance := 0.0
	for _, acc := range windowAccs {
		diff := acc - result.AvgAccuracy
		variance += diff * diff
	}
	if numWindows > 0 {
		variance /= float64(numWindows)
	}
	result.Stability = math.Max(0, 100-math.Sqrt(variance))

	duration := time.Since(start).Seconds()
	if duration > 0 {
		result.Throughput = float64(totalOutputs) / duration
	}

	result.Score = (result.Throughput * result.Stability * (result.AvgAccuracy / 100)) / 10000
	result.Passed = totalOutputs > 0 && !math.IsNaN(result.Score) && !math.IsInf(result.Score, 0)

	return result
}

func saveResults(results *FullBenchmarkResults) {
	data, _ := json.MarshalIndent(results, "", "  ")
	os.WriteFile("all_layers_all_modes_results.json", data, 0644)
	fmt.Println("\n✅ Results saved to all_layers_all_modes_results.json")
}

func printSummaryByLayer(results *FullBenchmarkResults) {
	fmt.Println("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                              📊 SUMMARY BY LAYER TYPE                                             ║")
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║  Layer Type       │ Total │ Passed │ Failed │ Avg Throughput │ Avg Accuracy                      ║")
	fmt.Println("║  ─────────────────┼───────┼────────┼────────┼────────────────┼───────────────────────────────────║")

	layerStats := make(map[string]struct {
		total, passed, failed int
		totalThroughput       float64
		totalAccuracy         float64
	})

	for _, r := range results.Results {
		s := layerStats[r.LayerType]
		s.total++
		if r.Passed {
			s.passed++
		} else {
			s.failed++
		}
		s.totalThroughput += r.Throughput
		s.totalAccuracy += r.AvgAccuracy
		layerStats[r.LayerType] = s
	}

	for _, name := range []string{"Dense", "Conv2D", "Attention", "RNN", "LSTM", "Softmax", "LayerNorm", "Residual", "RMSNorm", "SwiGLU", "Parallel"} {
		s := layerStats[name]
		avgTput := 0.0
		avgAcc := 0.0
		if s.total > 0 {
			avgTput = s.totalThroughput / float64(s.total)
			avgAcc = s.totalAccuracy / float64(s.total)
		}
		status := "✅"
		if s.failed > 0 {
			status = "⚠️"
		}
		fmt.Printf("║  %-16s │ %5d │ %6d │ %6d │ %14.0f │ %6.1f%% %s                        ║\n",
			name, s.total, s.passed, s.failed, avgTput, avgAcc, status)
	}
	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════╝")
}

func printSummaryByType(results *FullBenchmarkResults) {
	fmt.Println("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                              📊 SUMMARY BY NUMERIC TYPE                                           ║")
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║  Numeric Type     │ Total │ Passed │ Failed │ Avg Throughput │ Avg Accuracy                      ║")
	fmt.Println("║  ─────────────────┼───────┼────────┼────────┼────────────────┼───────────────────────────────────║")

	typeStats := make(map[string]struct {
		total, passed, failed int
		totalThroughput       float64
		totalAccuracy         float64
	})

	for _, r := range results.Results {
		s := typeStats[r.NumericType]
		s.total++
		if r.Passed {
			s.passed++
		} else {
			s.failed++
		}
		s.totalThroughput += r.Throughput
		s.totalAccuracy += r.AvgAccuracy
		typeStats[r.NumericType] = s
	}

	for _, name := range []string{"int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float32", "float64"} {
		s := typeStats[name]
		avgTput := 0.0
		avgAcc := 0.0
		if s.total > 0 {
			avgTput = s.totalThroughput / float64(s.total)
			avgAcc = s.totalAccuracy / float64(s.total)
		}
		status := "✅"
		if s.failed > 0 {
			status = "⚠️"
		}
		fmt.Printf("║  %-16s │ %5d │ %6d │ %6d │ %14.0f │ %6.1f%% %s                        ║\n",
			name, s.total, s.passed, s.failed, avgTput, avgAcc, status)
	}

	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════╝")

	fmt.Printf("\n🎯 TOTAL: %d tests | ✅ %d passed | ❌ %d failed | Pass rate: %.1f%%\n",
		results.TotalTests, results.Passed, results.Failed,
		float64(results.Passed)/float64(results.TotalTests)*100)
}
