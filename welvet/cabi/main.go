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

	"github.com/openfluke/loom/poly"
)

// Handle-based management maps
var networks = make(map[int64]*poly.VolumetricNetwork)
var networkNextID int64 = 1

var systolicStates = make(map[int64]*poly.SystolicState[float32])
var systolicNextID int64 = 1

var networkMu sync.RWMutex

// Helper: Error to C String
func errJSON(msg string) *C.char {
	return C.CString(fmt.Sprintf(`{"error": "%s"}`, msg))
}

// Helper: Convert Handle to Network
func getNetwork(handle int64) (*poly.VolumetricNetwork, bool) {
	networkMu.RLock()
	defer networkMu.RUnlock()
	n, ok := networks[handle]
	return n, ok
}

// Helper: Convert Handle to SystolicState
func getSystolicState(handle int64) (*poly.SystolicState[float32], bool) {
	networkMu.RLock()
	defer networkMu.RUnlock()
	s, ok := systolicStates[handle]
	return s, ok
}

//export LoomCreateNetwork
func LoomCreateNetwork(jsonConfig *C.char) C.longlong {
	configStr := C.GoString(jsonConfig)
	
	var n poly.VolumetricNetwork
	err := json.Unmarshal([]byte(configStr), &n)
	if err != nil {
		return -1
	}

	networkMu.Lock()
	id := networkNextID
	networkNextID++
	networks[id] = &n
	networkMu.Unlock()

	return C.longlong(id)
}

//export LoomFreeNetwork
func LoomFreeNetwork(handle C.longlong) {
	networkMu.Lock()
	delete(networks, int64(handle))
	networkMu.Unlock()
}

//export LoomCreateSystolicState
func LoomCreateSystolicState(networkHandle C.longlong) C.longlong {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return -1
	}

	s := poly.NewSystolicState[float32](n)

	networkMu.Lock()
	id := systolicNextID
	systolicNextID++
	systolicStates[id] = s
	networkMu.Unlock()

	return C.longlong(id)
}

//export LoomFreeSystolicState
func LoomFreeSystolicState(handle C.longlong) {
	networkMu.Lock()
	delete(systolicStates, int64(handle))
	networkMu.Unlock()
}

//export LoomSetInput
func LoomSetInput(stateHandle C.longlong, data *C.float, length C.int) {
	s, ok := getSystolicState(int64(stateHandle))
	if !ok {
		return
	}

	// Convert C array to Go slice
	ptr := unsafe.Pointer(data)
	slice := (*[1 << 30]float32)(ptr)[:length:length]
	
	// Create Tensor
	tensor := poly.NewTensorFromSlice(slice, int(length))
	s.SetInput(tensor)
}

//export LoomSystolicStep
func LoomSystolicStep(networkHandle C.longlong, stateHandle C.longlong, captureHistory C.int) C.longlong {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return -1
	}
	s, ok := getSystolicState(int64(stateHandle))
	if !ok {
		return -1
	}

	duration := poly.SystolicForward(n, s, captureHistory != 0)
	return C.longlong(duration.Nanoseconds())
}

//export LoomGetOutput
func LoomGetOutput(stateHandle C.longlong, layerIdx C.int) *C.char {
	s, ok := getSystolicState(int64(stateHandle))
	if !ok {
		return errJSON("invalid state handle")
	}

	networkMu.RLock()
	defer networkMu.RUnlock()
	
	if int(layerIdx) < 0 || int(layerIdx) >= len(s.LayerData) {
		return errJSON("layer index out of range")
	}
	
	output := s.LayerData[int(layerIdx)]
	if output == nil {
		return errJSON("no output for layer")
	}

	data, _ := json.Marshal(output.Data)
	return C.CString(string(data))
}

//export LoomApplyTargetProp
func LoomApplyTargetProp(networkHandle C.longlong, stateHandle C.longlong, targetData *C.float, targetLen C.int, learningRate C.float) {
	n, ok := getNetwork(int64(networkHandle))
	if !ok { return }
	s, ok := getSystolicState(int64(stateHandle))
	if !ok { return }

	// Convert C target to Tensor
	ptr := unsafe.Pointer(targetData)
	slice := (*[1 << 30]float32)(ptr)[:targetLen:targetLen]
	targetTensor := poly.NewTensorFromSlice(slice, int(targetLen))

	poly.SystolicApplyTargetProp(n, s, targetTensor, float32(learningRate))
}

//export LoomGetNetworkInfo
func LoomGetNetworkInfo(handle C.longlong) *C.char {
	n, ok := getNetwork(int64(handle))
	if !ok {
		return errJSON("invalid network handle")
	}

	info := map[string]interface{}{
		"total_layers": len(n.Layers),
		"grid": fmt.Sprintf("%dx%dx%d", n.Depth, n.Rows, n.Cols),
	}

	data, _ := json.Marshal(info)
	return C.CString(string(data))
}

//export LoomExtractDNA
func LoomExtractDNA(networkHandle C.longlong) *C.char {
	n, ok := getNetwork(int64(networkHandle))
	if !ok {
		return errJSON("invalid network handle")
	}

	dna := poly.ExtractDNA(n)
	data, _ := json.Marshal(dna)
	return C.CString(string(data))
}

//export LoomCompareDNA
func LoomCompareDNA(dna1JSON *C.char, dna2JSON *C.char) *C.char {
	var dna1, dna2 poly.NetworkDNA
	if err := json.Unmarshal([]byte(C.GoString(dna1JSON)), &dna1); err != nil {
		return errJSON("invalid DNA 1 JSON")
	}
	if err := json.Unmarshal([]byte(C.GoString(dna2JSON)), &dna2); err != nil {
		return errJSON("invalid DNA 2 JSON")
	}

	result := poly.CompareNetworks(dna1, dna2)
	data, _ := json.Marshal(result)
	return C.CString(string(data))
}

//export FreeLoomString
func FreeLoomString(ptr *C.char) {
	C.free(unsafe.Pointer(ptr))
}

func main() {}
