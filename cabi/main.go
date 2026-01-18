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
	"time"
	"unsafe"

	"github.com/openfluke/loom/gpu"
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
	output, _ := currentNetwork.Forward(goInputs)

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
		currentNetwork.ApplyGradients(float32(learningRate))
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

//export LoomSaveModelWithDType
func LoomSaveModelWithDType(modelID *C.char, dtype *C.char) *C.char {
	if currentNetwork == nil {
		return C.CString(`{"error": "no network created"}`)
	}

	id := C.GoString(modelID)
	dt := C.GoString(dtype)

	jsonStr, err := currentNetwork.SaveModelWithDType(id, dt)
	if err != nil {
		return C.CString(fmt.Sprintf(`{"error": "%v"}`, err))
	}

	return C.CString(jsonStr)
}

//export LoomLoadModelWithDType
func LoomLoadModelWithDType(jsonString *C.char, modelID *C.char, dtype *C.char) *C.char {
	jsonStr := C.GoString(jsonString)
	id := C.GoString(modelID)
	dt := C.GoString(dtype)

	network, storedType, err := nn.LoadModelWithDType(jsonStr, id, dt)
	if err != nil {
		return C.CString(fmt.Sprintf(`{"error": "%v"}`, err))
	}

	// Replace current network with loaded one
	currentNetwork = network

	return C.CString(fmt.Sprintf(`{"success": true, "stored_dtype": "%s"}`, storedType))
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

// ============================================================================
// TweenState C-ABI Exports
// ============================================================================

// Global map to store tween states
var tweenStates = make(map[int64]*nn.TweenState)
var tweenStateNextID int64 = 1
var tweenStateMu sync.RWMutex

//export LoomCreateTweenState
func LoomCreateTweenState(useChainRule C.int) C.longlong {
	if currentNetwork == nil {
		return -1
	}

	ts := nn.NewTweenState(currentNetwork, nil)
	if useChainRule != 0 {
		ts.Config.UseChainRule = true
	}

	tweenStateMu.Lock()
	id := tweenStateNextID
	tweenStateNextID++
	tweenStates[id] = ts
	tweenStateMu.Unlock()

	return C.longlong(id)
}

//export LoomTweenStep
func LoomTweenStep(handle C.longlong, input *C.float, inputLen C.int, targetClass C.int, outputSize C.int, learningRate C.float) C.float {
	tweenStateMu.RLock()
	ts, ok := tweenStates[int64(handle)]
	tweenStateMu.RUnlock()

	if !ok || currentNetwork == nil {
		return -1.0
	}

	// Convert C array to Go slice
	inputSlice := (*[1 << 30]float32)(unsafe.Pointer(input))[:inputLen:inputLen]
	goInput := make([]float32, inputLen)
	copy(goInput, inputSlice)

	gap := ts.TweenStep(currentNetwork, goInput, int(targetClass), int(outputSize), float32(learningRate))
	return C.float(gap)
}

//export LoomFreeTweenState
func LoomFreeTweenState(handle C.longlong) {
	tweenStateMu.Lock()
	delete(tweenStates, int64(handle))
	tweenStateMu.Unlock()
}

// ============================================================================
// AdaptationTracker C-ABI Exports
// ============================================================================

// Global map to store adaptation trackers
var adaptationTrackers = make(map[int64]*nn.AdaptationTracker)
var adaptationTrackerNextID int64 = 1
var adaptationTrackerMu sync.RWMutex

//export LoomCreateAdaptationTracker
func LoomCreateAdaptationTracker(windowDurationMs C.int, totalDurationMs C.int) C.longlong {
	windowDur := time.Duration(windowDurationMs) * time.Millisecond
	totalDur := time.Duration(totalDurationMs) * time.Millisecond

	tracker := nn.NewAdaptationTracker(windowDur, totalDur)

	adaptationTrackerMu.Lock()
	id := adaptationTrackerNextID
	adaptationTrackerNextID++
	adaptationTrackers[id] = tracker
	adaptationTrackerMu.Unlock()

	return C.longlong(id)
}

//export LoomTrackerSetModelInfo
func LoomTrackerSetModelInfo(handle C.longlong, modelName *C.char, modeName *C.char) {
	adaptationTrackerMu.RLock()
	tracker, ok := adaptationTrackers[int64(handle)]
	adaptationTrackerMu.RUnlock()

	if !ok {
		return
	}

	tracker.SetModelInfo(C.GoString(modelName), C.GoString(modeName))
}

//export LoomTrackerScheduleTaskChange
func LoomTrackerScheduleTaskChange(handle C.longlong, atOffsetMs C.int, taskID C.int, taskName *C.char) {
	adaptationTrackerMu.RLock()
	tracker, ok := adaptationTrackers[int64(handle)]
	adaptationTrackerMu.RUnlock()

	if !ok {
		return
	}

	offset := time.Duration(atOffsetMs) * time.Millisecond
	tracker.ScheduleTaskChange(offset, int(taskID), C.GoString(taskName))
}

//export LoomTrackerStart
func LoomTrackerStart(handle C.longlong, taskName *C.char, taskID C.int) {
	adaptationTrackerMu.RLock()
	tracker, ok := adaptationTrackers[int64(handle)]
	adaptationTrackerMu.RUnlock()

	if !ok {
		return
	}

	tracker.Start(C.GoString(taskName), int(taskID))
}

//export LoomTrackerRecordOutput
func LoomTrackerRecordOutput(handle C.longlong, isCorrect C.int) C.int {
	adaptationTrackerMu.RLock()
	tracker, ok := adaptationTrackers[int64(handle)]
	adaptationTrackerMu.RUnlock()

	if !ok {
		return -1
	}

	prevTask := tracker.RecordOutput(isCorrect != 0)
	return C.int(prevTask)
}

//export LoomTrackerGetCurrentTask
func LoomTrackerGetCurrentTask(handle C.longlong) C.int {
	adaptationTrackerMu.RLock()
	tracker, ok := adaptationTrackers[int64(handle)]
	adaptationTrackerMu.RUnlock()

	if !ok {
		return -1
	}

	return C.int(tracker.GetCurrentTask())
}

//export LoomTrackerFinalize
func LoomTrackerFinalize(handle C.longlong) *C.char {
	adaptationTrackerMu.RLock()
	tracker, ok := adaptationTrackers[int64(handle)]
	adaptationTrackerMu.RUnlock()

	if !ok {
		return C.CString(`{"error": "invalid tracker handle"}`)
	}

	result := tracker.Finalize()
	resultJSON, err := json.Marshal(result)
	if err != nil {
		return C.CString(fmt.Sprintf(`{"error": "%v"}`, err))
	}

	return C.CString(string(resultJSON))
}

//export LoomFreeTracker
func LoomFreeTracker(handle C.longlong) {
	adaptationTrackerMu.Lock()
	delete(adaptationTrackers, int64(handle))
	adaptationTrackerMu.Unlock()
}

// ============================================================================
// K-Means Clustering C-ABI Exports
// ============================================================================

//export LoomKMeansCluster
func LoomKMeansCluster(dataJSON *C.char, k C.int, maxIter C.int) *C.char {
	// Parse data (2D array of float32)
	var data [][]float32
	if err := json.Unmarshal([]byte(C.GoString(dataJSON)), &data); err != nil {
		return C.CString(fmt.Sprintf(`{"error": "invalid data JSON: %v"}`, err))
	}

	centroids, assignments := nn.KMeansCluster(data, int(k), int(maxIter), false)

	result := map[string]interface{}{
		"centroids":   centroids,
		"assignments": assignments,
	}

	resultJSON, _ := json.Marshal(result)
	return C.CString(string(resultJSON))
}

//export LoomSilhouetteScore
func LoomSilhouetteScore(dataJSON *C.char, assignmentsJSON *C.char) C.float {
	var data [][]float32
	if err := json.Unmarshal([]byte(C.GoString(dataJSON)), &data); err != nil {
		return -1.0
	}

	var assignments []int
	if err := json.Unmarshal([]byte(C.GoString(assignmentsJSON)), &assignments); err != nil {
		return -1.0
	}

	score := nn.ComputeSilhouetteScore(data, assignments)
	return C.float(score)
}

// ============================================================================
// Correlation Analysis C-ABI Exports
// ============================================================================

//export LoomComputeCorrelation
func LoomComputeCorrelation(dataJSON *C.char) *C.char {
	var data [][]float32
	if err := json.Unmarshal([]byte(C.GoString(dataJSON)), &data); err != nil {
		return C.CString(fmt.Sprintf(`{"error": "invalid data JSON: %v"}`, err))
	}

	result := nn.ComputeCorrelationMatrix(data, nil)
	if result == nil {
		return C.CString(`{"error": "failed to compute correlation"}`)
	}

	resultJSON, _ := json.Marshal(result)
	return C.CString(string(resultJSON))
}

// ============================================================================
// Network Grafting C-ABI Exports
// ============================================================================

// Global map to store networks for grafting
var graftNetworks = make(map[int64]*nn.Network)
var graftNetworkNextID int64 = 1
var graftNetworkMu sync.RWMutex

//export LoomCreateNetworkForGraft
func LoomCreateNetworkForGraft(jsonConfig *C.char) C.longlong {
	config := C.GoString(jsonConfig)

	network, err := nn.BuildNetworkFromJSON(config)
	if err != nil {
		return -1
	}

	network.InitializeWeights()

	graftNetworkMu.Lock()
	id := graftNetworkNextID
	graftNetworkNextID++
	graftNetworks[id] = network
	graftNetworkMu.Unlock()

	return C.longlong(id)
}

//export LoomGraftNetworks
func LoomGraftNetworks(networkIDsJSON *C.char, combineMode *C.char) *C.char {
	var networkIDs []int64
	if err := json.Unmarshal([]byte(C.GoString(networkIDsJSON)), &networkIDs); err != nil {
		return C.CString(fmt.Sprintf(`{"error": "invalid network IDs: %v"}`, err))
	}

	// Collect networks
	networks := make([]*nn.Network, 0, len(networkIDs))
	graftNetworkMu.RLock()
	for _, id := range networkIDs {
		if net, ok := graftNetworks[id]; ok {
			networks = append(networks, net)
		}
	}
	graftNetworkMu.RUnlock()

	if len(networks) < 2 {
		return C.CString(`{"error": "need at least 2 networks to graft"}`)
	}

	mode := C.GoString(combineMode)
	graftedConfig, err := nn.GraftNetworks(networks, mode)
	if err != nil {
		return C.CString(fmt.Sprintf(`{"error": "%v"}`, err))
	}

	result := map[string]interface{}{
		"success":      true,
		"type":         graftedConfig.Type,
		"num_branches": len(graftedConfig.ParallelBranches),
		"combine_mode": graftedConfig.CombineMode,
	}

	resultJSON, _ := json.Marshal(result)
	return C.CString(string(resultJSON))
}

//export LoomFreeGraftNetwork
func LoomFreeGraftNetwork(handle C.longlong) {
	graftNetworkMu.Lock()
	delete(graftNetworks, int64(handle))
	graftNetworkMu.Unlock()
}

// ============================================================================
// Learning Rate Scheduler C-ABI Exports
// ============================================================================

// Global map to store schedulers
var schedulers = make(map[int64]nn.LRScheduler)
var schedulerNextID int64 = 1
var schedulerMu sync.RWMutex

//export LoomCreateConstantScheduler
func LoomCreateConstantScheduler(baseLR C.float) C.longlong {
	sched := nn.NewConstantScheduler(float32(baseLR))

	schedulerMu.Lock()
	id := schedulerNextID
	schedulerNextID++
	schedulers[id] = sched
	schedulerMu.Unlock()

	return C.longlong(id)
}

//export LoomCreateLinearDecayScheduler
func LoomCreateLinearDecayScheduler(startLR, endLR C.float, totalSteps C.int) C.longlong {
	sched := nn.NewLinearDecayScheduler(float32(startLR), float32(endLR), int(totalSteps))

	schedulerMu.Lock()
	id := schedulerNextID
	schedulerNextID++
	schedulers[id] = sched
	schedulerMu.Unlock()

	return C.longlong(id)
}

//export LoomCreateCosineScheduler
func LoomCreateCosineScheduler(startLR, minLR C.float, totalSteps C.int) C.longlong {
	sched := nn.NewCosineAnnealingScheduler(float32(startLR), float32(minLR), int(totalSteps))

	schedulerMu.Lock()
	id := schedulerNextID
	schedulerNextID++
	schedulers[id] = sched
	schedulerMu.Unlock()

	return C.longlong(id)
}

//export LoomSchedulerGetLR
func LoomSchedulerGetLR(handle C.longlong, step C.int) C.float {
	schedulerMu.RLock()
	sched, ok := schedulers[int64(handle)]
	schedulerMu.RUnlock()

	if !ok {
		return -1.0
	}

	return C.float(sched.GetLR(int(step)))
}

//export LoomSchedulerName
func LoomSchedulerName(handle C.longlong) *C.char {
	schedulerMu.RLock()
	sched, ok := schedulers[int64(handle)]
	schedulerMu.RUnlock()

	if !ok {
		return C.CString("invalid")
	}

	return C.CString(sched.Name())
}

//export LoomFreeScheduler
func LoomFreeScheduler(handle C.longlong) {
	schedulerMu.Lock()
	delete(schedulers, int64(handle))
	schedulerMu.Unlock()
}

// ============================================================================
// Ensemble Features C-ABI Exports
// ============================================================================

//export LoomFindComplementaryMatches
func LoomFindComplementaryMatches(modelsJSON *C.char, minCoverage C.float) *C.char {
	var models []nn.ModelPerformance
	if err := json.Unmarshal([]byte(C.GoString(modelsJSON)), &models); err != nil {
		return C.CString(fmt.Sprintf(`{"error": "invalid models JSON: %v"}`, err))
	}

	matches := nn.FindComplementaryMatches(models, float64(minCoverage))

	result := map[string]interface{}{
		"matches":     matches,
		"num_matches": len(matches),
	}

	resultJSON, _ := json.Marshal(result)
	return C.CString(string(resultJSON))
}

// ============================================================================
// GPU Control C-ABI Exports
// ============================================================================

//export LoomSetGPUAdapterPreference
func LoomSetGPUAdapterPreference(adapter *C.char) {
	preference := C.GoString(adapter)
	gpu.SetAdapterPreference(preference)
}

//export LoomEnableGPU
func LoomEnableGPU(enable C.int) {
	if currentNetwork != nil {
		currentNetwork.GPU = (enable != 0)
	}
}

// ============================================================================
// Observer Pattern C-ABI Exports
// ============================================================================

// Global map to store recording observers
var recordingObservers = make(map[int64]*nn.RecordingObserver)
var recordingObserverNextID int64 = 1
var recordingObserverMu sync.RWMutex

//export LoomCreateRecordingObserver
func LoomCreateRecordingObserver(modelID *C.char) C.longlong {
	if currentNetwork == nil {
		return -1
	}

	obs := nn.NewRecordingObserver(C.GoString(modelID))

	// Attach to current network
	// Note: In a real scenario we might want to attach to specific layers,
	// but for the test we attach to all layers that support observation
	for i := range currentNetwork.Layers {
		currentNetwork.Layers[i].Observer = obs
	}

	recordingObserverMu.Lock()
	id := recordingObserverNextID
	recordingObserverNextID++
	recordingObservers[id] = obs
	recordingObserverMu.Unlock()

	return C.longlong(id)
}

//export LoomGetRecording
func LoomGetRecording(handle C.longlong) *C.char {
	recordingObserverMu.RLock()
	obs, ok := recordingObservers[int64(handle)]
	recordingObserverMu.RUnlock()

	if !ok {
		return C.CString(`{"error": "invalid observer handle"}`)
	}

	recording := obs.GetRecording()
	jsonBytes, err := json.Marshal(recording)
	if err != nil {
		return C.CString(fmt.Sprintf(`{"error": "%v"}`, err))
	}

	return C.CString(string(jsonBytes))
}

//export LoomFreeRecordingObserver
func LoomFreeRecordingObserver(handle C.longlong) {
	recordingObserverMu.Lock()
	delete(recordingObservers, int64(handle))
	recordingObserverMu.Unlock()

	// Detach from current network if it's still the same observer
	if currentNetwork != nil {
		for i := range currentNetwork.Layers {
			// exact check is hard without pointer comparison, but this is a test environment
			// verifying pointers in C-Go boundary is tricky.
			// We'll just clear it if we are freezing the observer currently attached.
			// Ideally we should track which observer is attached to which network.
			// For this "Simple" C ABI, we assume single threaded usage.
			currentNetwork.Layers[i].Observer = nil
		}
	}
}

// ============================================================================
// SafeTensors (WASM Memory) C-ABI Exports
// ============================================================================

// Global registry for loaded tensors
var safeTensorRegistry = make(map[int64]map[string][]float32)
var safeTensorNextID int64 = 1
var safeTensorMu sync.RWMutex

//export LoomSafeTensorRegistryNew
func LoomSafeTensorRegistryNew() C.longlong {
	safeTensorMu.Lock()
	id := safeTensorNextID
	safeTensorNextID++
	safeTensorRegistry[id] = make(map[string][]float32)
	safeTensorMu.Unlock()
	return C.longlong(id)
}

//export LoomSafeTensorRegistryLoadFromBuffer
func LoomSafeTensorRegistryLoadFromBuffer(handle C.longlong, data *C.uchar, length C.int) *C.char {
	safeTensorMu.RLock()
	registry, ok := safeTensorRegistry[int64(handle)]
	safeTensorMu.RUnlock()

	if !ok {
		return C.CString(`{"error": "invalid registry handle"}`)
	}

	// Convert C buffer to Go slice
	// Note: We use *[1 << 30]byte because uchar is 1 byte
	goData := C.GoBytes(unsafe.Pointer(data), length)

	tensors, err := nn.LoadSafetensorsFromBytes(goData)
	if err != nil {
		return C.CString(fmt.Sprintf(`{"error": "%v"}`, err))
	}

	// Merge into registry
	safeTensorMu.Lock()
	for k, v := range tensors {
		registry[k] = v
	}
	safeTensorMu.Unlock()

	return C.CString(fmt.Sprintf(`{"status": "success", "loaded_tensors": %d}`, len(tensors)))
}

//export LoomSafeTensorRegistryGet
func LoomSafeTensorRegistryGet(handle C.longlong, name *C.char) *C.char {
	safeTensorMu.RLock()
	registry, ok := safeTensorRegistry[int64(handle)]
	safeTensorMu.RUnlock()

	if !ok {
		return C.CString(`{"error": "invalid registry handle"}`)
	}

	tensorName := C.GoString(name)
	if data, exists := registry[tensorName]; exists {
		// Return as JSON array
		jsonBytes, err := json.Marshal(data)
		if err != nil {
			return C.CString(fmt.Sprintf(`{"error": "%v"}`, err))
		}
		return C.CString(string(jsonBytes))
	}

	return C.CString(`{"error": "tensor not found"}`)
}

//export LoomSafeTensorRegistryFree
func LoomSafeTensorRegistryFree(handle C.longlong) {
	safeTensorMu.Lock()
	delete(safeTensorRegistry, int64(handle))
	safeTensorMu.Unlock()
}

func main() {}
