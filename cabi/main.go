package main

/*
#include <stdlib.h>
*/
import "C"

import (
	"encoding/json"
	"fmt"
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

//export FreeLoomString
func FreeLoomString(str *C.char) {
	C.free(unsafe.Pointer(str))
}

func main() {}
