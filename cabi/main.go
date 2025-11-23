package main

/*
#include <stdlib.h>
#include <stdint.h>
*/
import "C"

import (
	"encoding/json"
	"fmt"
	"sync"
	"unsafe"

	"github.com/openfluke/loom/nn"
)

// Helper functions for JSON responses
func errJSON(msg string) *C.char {
	return C.CString(fmt.Sprintf(`{"error": "%s"}`, msg))
}

func asJSON(v interface{}) *C.char {
	data, err := json.Marshal(v)
	if err != nil {
		return errJSON(err.Error())
	}
	return C.CString(string(data))
}

// Global network instance (simplified single-network API)
var currentNetwork *nn.Network

//export CreateLoomNetwork
func CreateLoomNetwork(jsonConfig *C.char) *C.char {
	config := C.GoString(jsonConfig)

	network, err := nn.BuildNetworkFromJSON(config)
	if err != nil {
		errMsg := fmt.Sprintf(`{"error": "failed to create network: %v"}`, err)
		return C.CString(errMsg)
	}

	network.InitializeWeights()
	currentNetwork = network

	return C.CString(`{"status": "success", "message": "network created"}`)
}

//export LoomForward
func LoomForward(inputs *C.float, length C.int) *C.char {
	if currentNetwork == nil {
		return C.CString(`{"error": "no network created"}`)
	}

	// Convert C array to Go slice
	inputSlice := (*[1 << 30]float32)(unsafe.Pointer(inputs))[:length:length]
	goInputs := make([]float32, length)
	copy(goInputs, inputSlice)

	// Forward pass
	output, _ := currentNetwork.ForwardCPU(goInputs)

	// Convert to JSON
	result, err := json.Marshal(output)
	if err != nil {
		return C.CString(fmt.Sprintf(`{"error": "%v"}`, err))
	}

	return C.CString(string(result))
}

//export LoomBackward
func LoomBackward(gradients *C.float, length C.int) *C.char {
	if currentNetwork == nil {
		return C.CString(`{"error": "no network created"}`)
	}

	// Convert C array to Go slice
	gradSlice := (*[1 << 30]float32)(unsafe.Pointer(gradients))[:length:length]
	goGrads := make([]float32, length)
	copy(goGrads, gradSlice)

	// Backward pass
	_, _ = currentNetwork.BackwardCPU(goGrads)

	return C.CString(`{"status": "success"}`)
}

//export LoomUpdateWeights
func LoomUpdateWeights(learningRate C.float) {
	if currentNetwork != nil {
		currentNetwork.UpdateWeights(float32(learningRate))
	}
}

//export LoomTrain
func LoomTrain(batchesJSON *C.char, configJSON *C.char) *C.char {
	if currentNetwork == nil {
		return C.CString(`{"error": "no network created"}`)
	}

	// Parse batches
	var batches []nn.TrainingBatch
	if err := json.Unmarshal([]byte(C.GoString(batchesJSON)), &batches); err != nil {
		return C.CString(fmt.Sprintf(`{"error": "invalid batches: %v"}`, err))
	}

	// Parse config
	var config nn.TrainingConfig
	if err := json.Unmarshal([]byte(C.GoString(configJSON)), &config); err != nil {
		return C.CString(fmt.Sprintf(`{"error": "invalid config: %v"}`, err))
	}

	// Train
	result, err := currentNetwork.Train(batches, &config)
	if err != nil {
		return C.CString(fmt.Sprintf(`{"error": "%v"}`, err))
	}

	// Convert result to JSON
	resultJSON, _ := json.Marshal(result)
	return C.CString(string(resultJSON))
}

//export LoomSaveModel
func LoomSaveModel(modelID *C.char) *C.char {
	if currentNetwork == nil {
		return C.CString(`{"error": "no network created"}`)
	}

	id := C.GoString(modelID)
	jsonStr, err := currentNetwork.SaveModelToString(id)
	if err != nil {
		return C.CString(fmt.Sprintf(`{"error": "%v"}`, err))
	}

	return C.CString(jsonStr)
}

//export LoomLoadModel
func LoomLoadModel(jsonString *C.char, modelID *C.char) *C.char {
	jsonStr := C.GoString(jsonString)
	id := C.GoString(modelID)

	network, err := nn.LoadModelFromString(jsonStr, id)
	if err != nil {
		return C.CString(fmt.Sprintf(`{"error": "%v"}`, err))
	}

	// Replace current network with loaded one
	currentNetwork = network

	return C.CString(`{"success": true}`)
}

//export LoomGetNetworkInfo
func LoomGetNetworkInfo() *C.char {
	if currentNetwork == nil {
		return C.CString(`{"error": "no network created"}`)
	}

	info := map[string]interface{}{
		"grid_rows":       currentNetwork.GridRows,
		"grid_cols":       currentNetwork.GridCols,
		"layers_per_cell": currentNetwork.LayersPerCell,
		"batch_size":      currentNetwork.BatchSize,
		"total_layers":    currentNetwork.TotalLayers(),
	}

	infoJSON, _ := json.Marshal(info)
	return C.CString(string(infoJSON))
}

//export LoomEvaluateNetwork
func LoomEvaluateNetwork(inputsJSON *C.char, expectedOutputsJSON *C.char) *C.char {
	if currentNetwork == nil {
		return C.CString(`{"error": "no network created"}`)
	}

	// Parse inputs (2D array of float32)
	var inputs [][]float32
	if err := json.Unmarshal([]byte(C.GoString(inputsJSON)), &inputs); err != nil {
		return C.CString(fmt.Sprintf(`{"error": "invalid inputs JSON: %v"}`, err))
	}

	// Parse expected outputs (1D array of float64)
	var expectedOutputs []float64
	if err := json.Unmarshal([]byte(C.GoString(expectedOutputsJSON)), &expectedOutputs); err != nil {
		return C.CString(fmt.Sprintf(`{"error": "invalid expected outputs JSON: %v"}`, err))
	}

	// Evaluate
	metrics, err := currentNetwork.EvaluateNetwork(inputs, expectedOutputs)
	if err != nil {
		return C.CString(fmt.Sprintf(`{"error": "%v"}`, err))
	}

	// Convert metrics to JSON
	metricsJSON, err := json.Marshal(metrics)
	if err != nil {
		return C.CString(fmt.Sprintf(`{"error": "failed to marshal metrics: %v"}`, err))
	}

	return C.CString(string(metricsJSON))
}

//export FreeLoomString
func FreeLoomString(str *C.char) {
	C.free(unsafe.Pointer(str))
}

// Global map to store step states
var stepStates = make(map[int64]*nn.StepState)
var stepStateNextID int64 = 1
var stepStateMu sync.RWMutex

//export LoomInitStepState
func LoomInitStepState(inputSize C.int) C.longlong {
	if currentNetwork == nil {
		return -1
	}

	state := currentNetwork.InitStepState(int(inputSize))

	stepStateMu.Lock()
	id := stepStateNextID
	stepStateNextID++
	stepStates[id] = state
	stepStateMu.Unlock()

	return C.longlong(id)
}

//export LoomSetInput
func LoomSetInput(handle C.longlong, input *C.float, length C.int) {
	stepStateMu.RLock()
	state, ok := stepStates[int64(handle)]
	stepStateMu.RUnlock()

	if !ok {
		return
	}

	// Convert C array to Go slice
	inputSlice := (*[1 << 30]float32)(unsafe.Pointer(input))[:length:length]
	goInputs := make([]float32, length)
	copy(goInputs, inputSlice)

	state.SetInput(goInputs)
}

//export LoomStepForward
func LoomStepForward(handle C.longlong) C.longlong {
	stepStateMu.RLock()
	state, ok := stepStates[int64(handle)]
	stepStateMu.RUnlock()

	if !ok || currentNetwork == nil {
		return -1
	}

	duration := currentNetwork.StepForward(state)
	return C.longlong(duration.Nanoseconds())
}

//export LoomGetOutput
func LoomGetOutput(handle C.longlong) *C.char {
	stepStateMu.RLock()
	state, ok := stepStates[int64(handle)]
	stepStateMu.RUnlock()

	if !ok {
		return C.CString(`{"error": "invalid handle"}`)
	}

	output := state.GetOutput()
	result, err := json.Marshal(output)
	if err != nil {
		return C.CString(fmt.Sprintf(`{"error": "%v"}`, err))
	}

	return C.CString(string(result))
}

//export LoomStepBackward
func LoomStepBackward(handle C.longlong, gradients *C.float, length C.int) *C.char {
	stepStateMu.RLock()
	state, ok := stepStates[int64(handle)]
	stepStateMu.RUnlock()

	if !ok || currentNetwork == nil {
		return C.CString(`{"error": "invalid handle or network"}`)
	}

	// Convert C array to Go slice
	gradSlice := (*[1 << 30]float32)(unsafe.Pointer(gradients))[:length:length]
	goGrads := make([]float32, length)
	copy(goGrads, gradSlice)

	gradInput, duration := currentNetwork.StepBackward(state, goGrads)

	response := map[string]interface{}{
		"grad_input": gradInput,
		"duration":   duration.Nanoseconds(),
	}

	result, err := json.Marshal(response)
	if err != nil {
		return C.CString(fmt.Sprintf(`{"error": "%v"}`, err))
	}

	return C.CString(string(result))
}

//export LoomApplyGradients
func LoomApplyGradients(learningRate C.float) {
	if currentNetwork != nil {
		currentNetwork.ApplyGradients(float32(learningRate))
	}
}

//export LoomApplyGradientsAdamW
func LoomApplyGradientsAdamW(learningRate, beta1, beta2, weightDecay C.float) {
	if currentNetwork != nil {
		currentNetwork.ApplyGradientsAdamW(float32(learningRate), float32(beta1), float32(beta2), float32(weightDecay))
	}
}

//export LoomApplyGradientsRMSprop
func LoomApplyGradientsRMSprop(learningRate, alpha, epsilon, momentum C.float) {
	if currentNetwork != nil {
		currentNetwork.ApplyGradientsRMSprop(float32(learningRate), float32(alpha), float32(epsilon), float32(momentum))
	}
}

//export LoomApplyGradientsSGDMomentum
func LoomApplyGradientsSGDMomentum(learningRate, momentum, dampening C.float, nesterov C.int) {
	if currentNetwork != nil {
		currentNetwork.ApplyGradientsSGDMomentum(float32(learningRate), float32(momentum), float32(dampening), nesterov != 0)
	}
}

//export LoomFreeStepState
func LoomFreeStepState(handle C.longlong) {
	stepStateMu.Lock()
	delete(stepStates, int64(handle))
	stepStateMu.Unlock()
}

func main() {}
