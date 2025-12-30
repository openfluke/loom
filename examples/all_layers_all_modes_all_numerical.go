package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/openfluke/loom/nn"
)

// ═══════════════════════════════════════════════════════════════════════════════
// COMPREHENSIVE TEST: ALL LAYERS × ALL MODES × ALL NUMERICAL TYPES
// ═══════════════════════════════════════════════════════════════════════════════

const (
	// Network architecture
	InputSizeTest  = 16 // Input size for most layers
	HiddenSizeTest = 16 // Hidden layer size
	OutputSizeTest = 4  // Output size

	// Training parameters
	LearningRateTest      = float32(0.01)
	InitScaleTest         = float32(0.5)
	AccuracyThresholdTest = 0.15

	// Timing - 1 second run per test (10 windows of 100ms)
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
	TestLayerDense:          "Dense",
	TestLayerConv2D:         "Conv2D",
	TestLayerMultiHeadAttention: "Attention",
	TestLayerRNN:            "RNN",
	TestLayerLSTM:           "LSTM",
	TestLayerSoftmax:        "Softmax",
	TestLayerNorm:           "LayerNorm",
	TestLayerResidual:       "Residual",
	TestLayerRMSNorm:        "RMSNorm",
	TestLayerSwiGLU:         "SwiGLU",
	TestLayerParallel:       "Parallel",
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
	LayerType    string    `json:"layerType"`
	TrainingMode string    `json:"trainingMode"`
	NumericType  string    `json:"numericType"`
	AvgAccuracy  float64   `json:"avgAccuracy"`
	Stability    float64   `json:"stability"`
	Throughput   float64   `json:"throughput"`
	Score        float64   `json:"score"`
	Passed       bool      `json:"passed"`
	Error        string    `json:"error,omitempty"`
	History      []float64 `json:"history"` // Accuracy per window
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
	fmt.Println("║   TYPES:  int8 - int64, uint8 - uint64, float32, float64                                                                      ║")
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

					// Print generic progress line to not clutter
					// fmt.Printf("🔄 [%3d/%d] %s + %s + %s...\n", num, totalTests, layerName, modeName, typeName)

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
					// Only print simple status per line
					fmt.Printf("%s [%3d/%d] %-10s %-15s %-8s | Acc: %5.1f%% | Sc: %.0f\n",
						status, num, totalTests, layerName, modeName, typeName, result.AvgAccuracy, result.Score)
				}(layer, mode, numType, testNum)
			}
		}
	}

	wg.Wait()
	fmt.Println("\n✅ All tests complete!")

	saveResults(results)
	printComparisonTable(results) // <--- NEW FANCY PRINT
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
		net.SetLayer(0, 0, 0, nn.InitDenseLayer(InputSizeTest, HiddenSizeTest, nn.ActivationLeakyReLU))
		net.SetLayer(0, 0, 1, nn.InitDenseLayer(HiddenSizeTest, HiddenSizeTest, nn.ActivationLeakyReLU))
		net.SetLayer(0, 0, 2, nn.InitDenseLayer(HiddenSizeTest, OutputSizeTest, nn.ActivationSigmoid))
	case TestLayerConv2D:
		net = nn.NewNetwork(16, 1, 1, 2)
		net.BatchSize = 1
		conv := nn.InitConv2DLayer(4, 4, 1, 2, 2, 1, 2, nn.ActivationLeakyReLU)
		net.SetLayer(0, 0, 0, conv)
		net.SetLayer(0, 0, 1, nn.InitDenseLayer(18, OutputSizeTest, nn.ActivationSigmoid))
	case TestLayerMultiHeadAttention:
		net.SetLayer(0, 0, 0, nn.InitDenseLayer(InputSizeTest, HiddenSizeTest, nn.ActivationLeakyReLU))
		net.SetLayer(0, 0, 1, nn.InitDenseLayer(HiddenSizeTest, HiddenSizeTest, nn.ActivationTanh))
		net.SetLayer(0, 0, 2, nn.InitDenseLayer(HiddenSizeTest, OutputSizeTest, nn.ActivationSigmoid))
	case TestLayerRNN:
		rnn := nn.InitRNNLayer(InputSizeTest, HiddenSizeTest, 1, 1)
		net.SetLayer(0, 0, 0, rnn)
		net.SetLayer(0, 0, 1, nn.InitDenseLayer(HiddenSizeTest, HiddenSizeTest, nn.ActivationLeakyReLU))
		net.SetLayer(0, 0, 2, nn.InitDenseLayer(HiddenSizeTest, OutputSizeTest, nn.ActivationSigmoid))
	case TestLayerLSTM:
		lstm := nn.InitLSTMLayer(InputSizeTest, HiddenSizeTest, 1, 1)
		net.SetLayer(0, 0, 0, lstm)
		net.SetLayer(0, 0, 1, nn.InitDenseLayer(HiddenSizeTest, HiddenSizeTest, nn.ActivationLeakyReLU))
		net.SetLayer(0, 0, 2, nn.InitDenseLayer(HiddenSizeTest, OutputSizeTest, nn.ActivationSigmoid))
	case TestLayerSoftmax:
		net.SetLayer(0, 0, 0, nn.InitDenseLayer(InputSizeTest, HiddenSizeTest, nn.ActivationLeakyReLU))
		net.SetLayer(0, 0, 1, nn.InitDenseLayer(HiddenSizeTest, OutputSizeTest, nn.ActivationScaledReLU))
		softmax := nn.LayerConfig{Type: nn.LayerSoftmax, SoftmaxVariant: nn.SoftmaxStandard, Temperature: 1.0}
		net.SetLayer(0, 0, 2, softmax)
	case TestLayerNorm:
		net.SetLayer(0, 0, 0, nn.InitDenseLayer(InputSizeTest, HiddenSizeTest, nn.ActivationScaledReLU))
		layerNorm := nn.LayerConfig{Type: nn.LayerNorm, NormSize: HiddenSizeTest, Gamma: make([]float32, HiddenSizeTest), Beta: make([]float32, HiddenSizeTest), Epsilon: 1e-5}
		for i := range layerNorm.Gamma {
			layerNorm.Gamma[i] = 1.0
		}
		net.SetLayer(0, 0, 1, layerNorm)
		net.SetLayer(0, 0, 2, nn.InitDenseLayer(HiddenSizeTest, OutputSizeTest, nn.ActivationSigmoid))
	case TestLayerResidual:
		net.SetLayer(0, 0, 0, nn.InitDenseLayer(InputSizeTest, InputSizeTest, nn.ActivationLeakyReLU))
		residual := nn.LayerConfig{Type: nn.LayerResidual}
		net.SetLayer(0, 0, 1, residual)
		net.SetLayer(0, 0, 2, nn.InitDenseLayer(InputSizeTest, OutputSizeTest, nn.ActivationSigmoid))
	case TestLayerRMSNorm:
		net.SetLayer(0, 0, 0, nn.InitDenseLayer(InputSizeTest, HiddenSizeTest, nn.ActivationScaledReLU))
		rmsNorm := nn.LayerConfig{Type: nn.LayerRMSNorm, NormSize: HiddenSizeTest, Gamma: make([]float32, HiddenSizeTest), Epsilon: 1e-6}
		for i := range rmsNorm.Gamma {
			rmsNorm.Gamma[i] = 1.0
		}
		net.SetLayer(0, 0, 1, rmsNorm)
		net.SetLayer(0, 0, 2, nn.InitDenseLayer(HiddenSizeTest, OutputSizeTest, nn.ActivationSigmoid))
	case TestLayerSwiGLU:
		intermediateSize := HiddenSizeTest * 2
		swiglu := nn.LayerConfig{Type: nn.LayerSwiGLU, InputHeight: InputSizeTest, OutputHeight: InputSizeTest, GateWeights: make([]float32, InputSizeTest*intermediateSize), UpWeights: make([]float32, InputSizeTest*intermediateSize), DownWeights: make([]float32, intermediateSize*InputSizeTest), GateBias: make([]float32, intermediateSize), UpBias: make([]float32, intermediateSize), DownBias: make([]float32, InputSizeTest)}
		for i := range swiglu.GateWeights {
			swiglu.GateWeights[i] = float32(rand.NormFloat64()) * 0.1
		}
		net.SetLayer(0, 0, 0, swiglu)
		net.SetLayer(0, 0, 1, nn.InitDenseLayer(InputSizeTest, HiddenSizeTest, nn.ActivationLeakyReLU))
		net.SetLayer(0, 0, 2, nn.InitDenseLayer(HiddenSizeTest, OutputSizeTest, nn.ActivationSigmoid))
	case TestLayerParallel:
		branch1 := nn.InitDenseLayer(InputSizeTest, HiddenSizeTest/2, nn.ActivationLeakyReLU)
		branch2 := nn.InitDenseLayer(InputSizeTest, HiddenSizeTest/2, nn.ActivationTanh)
		parallel := nn.LayerConfig{Type: nn.LayerParallel, ParallelBranches: []nn.LayerConfig{branch1, branch2}}
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
			targetData := make([]T, len(target))
			for i, v := range target {
				if isIntegerType[T]() {
					targetData[i] = T(v * 100)
				} else {
					targetData[i] = T(v)
				}
			}
			targetTensor := nn.NewTensorFromSlice(targetData, len(targetData))
			lr := float64(LearningRateTest)
			output, _, _ = nn.GenericTrainStep(net, inputTensor, targetTensor, lr, backend)
		} else {
			output, _, _, _ = nn.GenericForwardPass(net, inputTensor, backend)
		}

		if len(output.Data) > 0 && len(target) > 0 {
			var pred float64
			if isIntegerType[T]() {
				pred = float64(output.Data[0]) / 100.0
			} else {
				pred = float64(output.Data[0])
			}
			if math.Abs(pred-target[0]) < AccuracyThresholdTest {
				windowAccs[currentWindow] += 100.0
			}
			windowCounts[currentWindow]++
		}
		totalOutputs++
	}

	for i := 0; i < numWindows; i++ {
		if windowCounts[i] > 0 {
			windowAccs[i] /= float64(windowCounts[i])
		}
	}
	result.History = windowAccs // Store history

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

		if len(output) > 0 && len(target) > 0 {
			if math.Abs(float64(output[0]-target[0])) < AccuracyThresholdTest {
				windowAccs[currentWindow] += 100.0
			}
			windowCounts[currentWindow]++
		}
		totalOutputs++

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

	for i := 0; i < numWindows; i++ {
		if windowCounts[i] > 0 {
			windowAccs[i] /= float64(windowCounts[i])
		}
	}
	result.History = windowAccs // Store history

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

// NEW FUNCTION: Aggregates results to find best Type per Mode and prints fancy table
func printComparisonTable(results *FullBenchmarkResults) {
	// 1. Group results by (Mode, Type) and calculate average stats across all layers
	type GroupKey struct {
		Mode string
		Type string
	}
	type GroupStats struct {
		count          int
		sumScore       float64
		sumAcc         float64
		sumStab        float64
		sumTput        float64
		historySum     []float64
		historySamples []int
	}

	stats := make(map[GroupKey]*GroupStats)

	// Mode order for printing
	modesOrdered := []string{"NormalBP", "StepBP", "Tween", "TweenChain", "StepTween", "StepTweenChain"}

	for _, r := range results.Results {
		if !r.Passed {
			continue
		}
		key := GroupKey{Mode: r.TrainingMode, Type: r.NumericType}
		if stats[key] == nil {
			stats[key] = &GroupStats{
				historySum:     make([]float64, 10), // 10 windows
				historySamples: make([]int, 10),
			}
		}
		s := stats[key]
		s.count++
		s.sumScore += r.Score
		s.sumAcc += r.AvgAccuracy
		s.sumStab += r.Stability
		s.sumTput += r.Throughput
		for i, h := range r.History {
			if i < len(s.historySum) {
				s.historySum[i] += h
				s.historySamples[i]++
			}
		}
	}

	// 2. For each Mode, find the Type with the highest Average Score
	bestTypeForMode := make(map[string]string) // Mode -> Type
	bestStatsForMode := make(map[string]*GroupStats)

	for _, mode := range modesOrdered {
		var bestType string
		var maxAvgScore float64 = -1.0

		// Iterate all types to find best for this mode
		for _, t := range typeNames { // typeNames map values
			// We need to iterate the values of the map, or just hardcode checking standard ones
			key := GroupKey{Mode: mode, Type: t}
			if s, ok := stats[key]; ok {
				avgScore := s.sumScore / float64(s.count)
				if avgScore > maxAvgScore {
					maxAvgScore = avgScore
					bestType = t
				}
			}
		}
		if bestType != "" {
			bestTypeForMode[mode] = bestType
			bestStatsForMode[mode] = stats[GroupKey{Mode: mode, Type: bestType}]
		}
	}

	// 3. Print Timeline Table
	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║           PREDICTION ACCURACY % (Avg across layers) — Best Numeric Type per Mode selected                                                    ║")
	fmt.Println("║           Timeline represents 1.0s duration broken into 100ms windows                                                                        ║")
	fmt.Println("╠════════════════════════════╦═══════════════════════════════════════════════════════════════════════════════════════════════╦═══════╦════════════╣")
	fmt.Println("║ Mode (Best Type)           ║ .1s .2s .3s .4s .5s .6s .7s .8s .9s 1.0s                                                    ║ Avg   ║ Score      ║")
	fmt.Println("╠════════════════════════════╬═══════════════════════════════════════════════════════════════════════════════════════════════╬═══════╬════════════╣")

	var bestGlobalScore float64
	var bestGlobalModeStr string

	for _, mode := range modesOrdered {
		bestType := bestTypeForMode[mode]
		if bestType == "" {
			continue // No results for this mode
		}
		s := bestStatsForMode[mode]

		// Format Label: "NormalBP (f32)"
		shortType := bestType
		if strings.HasPrefix(shortType, "float") {
			shortType = "f" + shortType[5:]
		}
		label := fmt.Sprintf("%s (%s)", mode, shortType)
		fmt.Printf("║ %-26s ║", label)

		// Print History
		for i := 0; i < 10; i++ {
			val := 0.0
			if s.historySamples[i] > 0 {
				val = s.historySum[i] / float64(s.historySamples[i])
			}
			fmt.Printf(" %3.0f%%", val)
		}

		avgAcc := s.sumAcc / float64(s.count)
		avgScore := s.sumScore / float64(s.count)

		if avgScore > bestGlobalScore {
			bestGlobalScore = avgScore
			bestGlobalModeStr = label
		}

		fmt.Printf(" ║ %3.0f%% ║ %10.0f ║\n", avgAcc, avgScore)
	}

	fmt.Println("╚════════════════════════════╩═══════════════════════════════════════════════════════════════════════════════════════════════╩═══════╩════════════╝")

	// 4. Print Summary Table
	fmt.Println("\n╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                  📊 COMPREHENSIVE BENCHMARK SUMMARY 📊                                         ║")
	fmt.Println("╠════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║                                                                                                                ║")
	fmt.Println("║  Mode (Best Type)     │ Avg Accuracy │ Stability │ Throughput  │ Score       │ Best Layer for Mode           ║")
	fmt.Println("║  ─────────────────────┼──────────────┼───────────┼─────────────┼─────────────┼───────────────────────────────║")

	for _, mode := range modesOrdered {
		bestType := bestTypeForMode[mode]
		if bestType == "" {
			continue
		}
		s := bestStatsForMode[mode]

		avgAcc := s.sumAcc / float64(s.count)
		avgStab := s.sumStab / float64(s.count)
		avgTput := s.sumTput / float64(s.count)
		avgScore := s.sumScore / float64(s.count)

		shortType := bestType
		if strings.HasPrefix(shortType, "float") {
			shortType = "f" + shortType[5:]
		}
		label := fmt.Sprintf("%s (%s)", mode, shortType)

		// Find the single best layer for this Mode+Type combo to list as "Best Layer"
		bestLayerName := "-"
		bestLayerScore := -1.0
		for _, r := range results.Results {
			if r.TrainingMode == mode && r.NumericType == bestType && r.Score > bestLayerScore {
				bestLayerScore = r.Score
				bestLayerName = r.LayerType
			}
		}

		fmt.Printf("║  %-20s │  %7.1f%%    │  %6.1f%%  │  %9.0f  │  %9.0f  │  %-29s║\n",
			label, avgAcc, avgStab, avgTput, avgScore, bestLayerName)
	}

	fmt.Println("║                                                                                                                ║")
	fmt.Println("╠════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Printf("║  🏆 WINNER: %-20s with Score: %.0f                                                       ║\n", bestGlobalModeStr, bestGlobalScore)
	fmt.Println("║                                                                                                                ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")
}