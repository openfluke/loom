//go:build js && wasm
// +build js,wasm

package main

import (
	"encoding/json"
	"fmt"
	"reflect"
	"sync"
	"syscall/js"
	"time"

	"github.com/openfluke/loom/gpu"
	"github.com/openfluke/loom/nn"
)

// setupWebGPUWrapper performs the async navigator.gpu initialization entirely in Go
// and returns a Promise that resolves when ready.
func setupWebGPUWrapper(this js.Value, args []js.Value) interface{} {
	handler := js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		resolve := args[0]
		reject := args[1]

		go func() {
			nav := js.Global().Get("navigator")
			if nav.IsUndefined() || nav.Get("gpu").IsUndefined() {
				reject.Invoke("WebGPU not supported (navigator.gpu missing)")
				return
			}

			gpuObj := nav.Get("gpu")

			// 1. requestAdapter
			adapterPromise := gpuObj.Call("requestAdapter")
			adapterVal, err := awaitPromise(adapterPromise)
			if err != nil {
				reject.Invoke("requestAdapter failed: " + err.Error())
				return
			}
			if adapterVal.IsNull() {
				reject.Invoke("requestAdapter returned null")
				return
			}

			// 2. requestDevice
			devicePromise := adapterVal.Call("requestDevice")
			deviceVal, err := awaitPromise(devicePromise)
			if err != nil {
				reject.Invoke("requestDevice failed: " + err.Error())
				return
			}

			// 3. Expose to Window (mimicking user's old setup)
			js.Global().Set("webgpuAdapter", adapterVal)
			js.Global().Set("webgpuDevice", deviceVal)
			js.Global().Set("webgpuQueue", deviceVal.Get("queue"))

			fmt.Println("GOWASM: WebGPU initialized automatically!")
			resolve.Invoke("WebGPU Initialized via Go")
		}()

		return nil
	})

	promise := js.Global().Get("Promise").New(handler)
	return promise
}

// awaitPromise helper to wait for a JS Promise in Go
func awaitPromise(promise js.Value) (js.Value, error) {
	resultCh := make(chan js.Value)
	errCh := make(chan error)

	then := js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		resultCh <- args[0]
		return nil
	})
	catch := js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		errCh <- fmt.Errorf(args[0].String())
		return nil
	})

	promise.Call("then", then).Call("catch", catch)

	select {
	case res := <-resultCh:
		then.Release()
		catch.Release()
		return res, nil
	case err := <-errCh:
		then.Release()
		catch.Release()
		return js.Undefined(), err
	}
}

// methodWrapper dynamically wraps each method to expose it to JavaScript
func methodWrapper(network *nn.Network, methodName string) js.Func {
	return js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		method := reflect.ValueOf(network).MethodByName(methodName)
		if !method.IsValid() {
			return fmt.Sprintf("Method %s not found", methodName)
		}

		methodType := method.Type()

		// Handle no arguments case
		if len(args) == 0 || args[0].IsUndefined() || (args[0].Type() == js.TypeString && args[0].String() == "") {
			if methodType.NumIn() == 0 {
				inputs := []reflect.Value{}
				results := method.Call(inputs)
				return serializeResults(results)
			}
			return "No arguments provided"
		}

		var params []interface{}
		paramJSON := args[0].String()
		if err := json.Unmarshal([]byte(paramJSON), &params); err != nil {
			return fmt.Sprintf("Invalid JSON input: %v", err)
		}

		expectedParams := methodType.NumIn()
		if len(params) != expectedParams {
			return fmt.Sprintf("Expected %d parameters, got %d", expectedParams, len(params))
		}

		inputs := make([]reflect.Value, expectedParams)
		for i := 0; i < expectedParams; i++ {
			param := params[i]
			expectedType := methodType.In(i)

			converted, err := convertParameter(param, expectedType, i)
			if err != nil {
				return err.Error()
			}
			inputs[i] = converted
		}

		results := method.Call(inputs)
		return serializeResults(results)
	})
}

// convertParameter converts a JavaScript parameter to the expected Go type
func convertParameter(param interface{}, expectedType reflect.Type, paramIndex int) (reflect.Value, error) {
	switch expectedType.Kind() {
	case reflect.Slice:
		return convertSlice(param, expectedType, paramIndex)
	case reflect.Int:
		if val, ok := param.(float64); ok {
			return reflect.ValueOf(int(val)), nil
		}
		return reflect.Value{}, fmt.Errorf("parameter %d: expected int, got %T", paramIndex, param)
	case reflect.Float32:
		if val, ok := param.(float64); ok {
			return reflect.ValueOf(float32(val)), nil
		}
		return reflect.Value{}, fmt.Errorf("parameter %d: expected float32, got %T", paramIndex, param)
	case reflect.Float64:
		if val, ok := param.(float64); ok {
			return reflect.ValueOf(val), nil
		}
		return reflect.Value{}, fmt.Errorf("parameter %d: expected float64, got %T", paramIndex, param)
	case reflect.Bool:
		if val, ok := param.(bool); ok {
			return reflect.ValueOf(val), nil
		}
		return reflect.Value{}, fmt.Errorf("parameter %d: expected bool, got %T", paramIndex, param)
	case reflect.String:
		if val, ok := param.(string); ok {
			return reflect.ValueOf(val), nil
		}
		return reflect.Value{}, fmt.Errorf("parameter %d: expected string, got %T", paramIndex, param)
	case reflect.Ptr:
		// Handle pointer types
		if expectedType == reflect.TypeOf((*nn.TrainingConfig)(nil)) {
			// Parse TrainingConfig from JSON
			jsonBytes, err := json.Marshal(param)
			if err != nil {
				return reflect.Value{}, fmt.Errorf("parameter %d: failed to marshal config: %v", paramIndex, err)
			}
			config := &nn.TrainingConfig{}
			if err := json.Unmarshal(jsonBytes, config); err != nil {
				return reflect.Value{}, fmt.Errorf("parameter %d: failed to unmarshal config: %v", paramIndex, err)
			}
			return reflect.ValueOf(config), nil
		}
		return reflect.Value{}, fmt.Errorf("parameter %d: unsupported pointer type %s", paramIndex, expectedType.String())
	case reflect.Struct:
		// Handle struct types via JSON
		jsonBytes, err := json.Marshal(param)
		if err != nil {
			return reflect.Value{}, fmt.Errorf("parameter %d: failed to marshal struct: %v", paramIndex, err)
		}
		structVal := reflect.New(expectedType)
		if err := json.Unmarshal(jsonBytes, structVal.Interface()); err != nil {
			return reflect.Value{}, fmt.Errorf("parameter %d: failed to unmarshal struct: %v", paramIndex, err)
		}
		return structVal.Elem(), nil
	default:
		return reflect.Zero(expectedType), fmt.Errorf("parameter %d: unsupported type %s", paramIndex, expectedType.String())
	}
}

// convertSlice handles slice type conversions including multidimensional slices
func convertSlice(param interface{}, expectedType reflect.Type, paramIndex int) (reflect.Value, error) {
	val, ok := param.([]interface{})
	if !ok {
		return reflect.Value{}, fmt.Errorf("parameter %d: expected slice, got %T", paramIndex, param)
	}

	elemType := expectedType.Elem()
	slice := reflect.MakeSlice(expectedType, len(val), len(val))

	for j, v := range val {
		switch elemType.Kind() {
		case reflect.Int:
			if num, ok := v.(float64); ok {
				slice.Index(j).SetInt(int64(num))
			} else {
				return reflect.Value{}, fmt.Errorf("parameter %d: invalid slice element type %T at index %d", paramIndex, v, j)
			}
		case reflect.Float32:
			if num, ok := v.(float64); ok {
				slice.Index(j).SetFloat(num)
			} else {
				return reflect.Value{}, fmt.Errorf("parameter %d: invalid slice element type %T at index %d", paramIndex, v, j)
			}
		case reflect.Float64:
			if num, ok := v.(float64); ok {
				slice.Index(j).SetFloat(num)
			} else {
				return reflect.Value{}, fmt.Errorf("parameter %d: invalid slice element type %T at index %d", paramIndex, v, j)
			}
		case reflect.Bool:
			if b, ok := v.(bool); ok {
				slice.Index(j).SetBool(b)
			} else {
				return reflect.Value{}, fmt.Errorf("parameter %d: invalid slice element type %T at index %d", paramIndex, v, j)
			}
		case reflect.String:
			if s, ok := v.(string); ok {
				slice.Index(j).SetString(s)
			} else {
				return reflect.Value{}, fmt.Errorf("parameter %d: invalid slice element type %T at index %d", paramIndex, v, j)
			}
		case reflect.Slice:
			// Handle 2D/3D slices recursively
			converted, err := convertSlice(v, elemType, paramIndex)
			if err != nil {
				return reflect.Value{}, err
			}
			slice.Index(j).Set(converted)
		case reflect.Struct:
			// Handle struct slices (like TrainingBatch)
			jsonBytes, err := json.Marshal(v)
			if err != nil {
				return reflect.Value{}, fmt.Errorf("parameter %d: failed to marshal struct at index %d", paramIndex, j)
			}
			structVal := reflect.New(elemType)
			if err := json.Unmarshal(jsonBytes, structVal.Interface()); err != nil {
				return reflect.Value{}, fmt.Errorf("parameter %d: failed to unmarshal struct at index %d: %v", paramIndex, j, err)
			}
			slice.Index(j).Set(structVal.Elem())
		default:
			return reflect.Value{}, fmt.Errorf("parameter %d: unsupported slice element type %s", paramIndex, elemType.String())
		}
	}

	return slice, nil
}

// serializeResults converts method results to JSON
func serializeResults(results []reflect.Value) string {
	if len(results) == 0 {
		return "[]"
	}

	output := make([]interface{}, len(results))
	for i, result := range results {
		output[i] = result.Interface()
	}

	resultJSON, err := json.Marshal(output)
	if err != nil {
		return fmt.Sprintf("Failed to marshal results: %v", err)
	}
	return string(resultJSON)
}

// createNetworkFromJSON creates a network wrapper based on JSON config
func createNetworkFromJSON(this js.Value, args []js.Value) interface{} {
	if len(args) < 1 {
		errMsg := "Expected 1 argument: JSON configuration string"
		js.Global().Get("console").Call("error", errMsg)
		return errMsg
	}

	jsonConfig := args[0].String()

	// Build network from JSON
	network, err := nn.BuildNetworkFromJSON(jsonConfig)
	if err != nil {
		errMsg := fmt.Sprintf("Error: failed to create network: %v", err)
		js.Global().Get("console").Call("error", errMsg)
		return errMsg
	}

	// Initialize weights
	network.InitializeWeights()

	// Create JS object with all network methods
	obj := js.Global().Get("Object").New()

	// Use reflection to get all methods
	networkValue := reflect.ValueOf(network)
	networkType := networkValue.Type()

	for i := 0; i < networkType.NumMethod(); i++ {
		method := networkType.Method(i)
		// Only export public methods
		if method.Name[0] >= 'A' && method.Name[0] <= 'Z' {
			obj.Set(method.Name, methodWrapper(network, method.Name))
		}
	}

	// Add optimizer-specific gradient application methods
	obj.Set("ApplyGradientsSGDMomentum", methodWrapper(network, "ApplyGradientsSGDMomentum"))

	obj.Set("createStepState", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return "Expected 1 argument: inputSize"
		}
		inputSize := args[0].Int()
		state := network.InitStepState(inputSize)
		return createStepStateWrapper(state, network)
	}))

	// Add createTweenState method for neural tweening
	obj.Set("createTweenState", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		useChainRule := false
		if len(args) >= 1 {
			useChainRule = args[0].Bool()
		}

		config := nn.DefaultTweenConfig(network.TotalLayers())
		config.UseChainRule = useChainRule

		ts := nn.NewTweenState(network, config)
		return createTweenStateWrapper(ts, network)
	}))

	// Add getInputSize for convenience
	obj.Set("getInputSize", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return network.InputSize
	}))

	// Add Introspection methods manually if they are not on the struct or not exposed
	obj.Set("GetNetworkInfo", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		// Basic info
		info := map[string]interface{}{
			"InputSize":     network.InputSize,
			"TotalLayers":   network.TotalLayers(),
			"GridRows":      network.GridRows,
			"GridCols":      network.GridCols,
			"LayersPerCell": network.LayersPerCell,
		}
		b, _ := json.Marshal(info)
		return string(b)
	}))

	obj.Set("GetTotalParameters", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		total := 0
		for _, layer := range network.Layers {
			total += len(layer.Kernel) + len(layer.Bias)
			total += len(layer.EmbeddingWeights)
			total += len(layer.QWeights) + len(layer.KWeights) + len(layer.VWeights) + len(layer.OutputWeight)
			total += len(layer.WeightIH) + len(layer.WeightHH)
		}
		return fmt.Sprintf("[%d]", total)
	}))

	obj.Set("GetMemoryUsage", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		totalParams := 0
		for _, layer := range network.Layers {
			totalParams += len(layer.Kernel) + len(layer.Bias)
			totalParams += len(layer.EmbeddingWeights)
			totalParams += len(layer.QWeights) + len(layer.KWeights) + len(layer.VWeights) + len(layer.OutputWeight)
			totalParams += len(layer.WeightIH) + len(layer.WeightHH)
		}
		totalBytes := totalParams * 4
		return fmt.Sprintf("[%d]", totalBytes)
	}))

	// Add setGPU for controlling execution backend
	obj.Set("setGPU", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return "Expected 1 argument: useGPU (boolean)"
		}
		network.GPU = args[0].Bool()
		return nil
	}))

	return obj
}

// createStepStateWrapper creates a JS object for a StepState
func createStepStateWrapper(state *nn.StepState, network *nn.Network) js.Value {
	obj := js.Global().Get("Object").New()

	// setInput(data)
	obj.Set("setInput", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return "Expected input data"
		}

		// Handle JS Float32Array or Array
		var input []float32
		jsData := args[0]

		if jsData.Type() == js.TypeObject && jsData.Get("length").Type() == js.TypeNumber {
			length := jsData.Get("length").Int()
			input = make([]float32, length)

			// Check if it's a typed array (has buffer property)
			if !jsData.Get("buffer").IsUndefined() {
				// Copy directly from typed array memory if possible
				// For now, simple loop is safer across browsers/environments
				for i := 0; i < length; i++ {
					input[i] = float32(jsData.Index(i).Float())
				}
			} else {
				// Standard array
				for i := 0; i < length; i++ {
					input[i] = float32(jsData.Index(i).Float())
				}
			}
		} else {
			return "Invalid input data type"
		}

		state.SetInput(input)
		return nil
	}))

	// stepForward() -> duration (ms)
	obj.Set("stepForward", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		duration := network.StepForward(state)
		return float64(duration.Nanoseconds()) / 1000000.0 // Return ms
	}))

	// getOutput() -> Float32Array
	obj.Set("getOutput", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		output := state.GetOutput()

		// Create JS Float32Array
		arrayConstructor := js.Global().Get("Float32Array")
		jsArray := arrayConstructor.New(len(output))

		// Copy data
		for i, v := range output {
			jsArray.SetIndex(i, v)
		}

		return jsArray
	}))

	// stepBackward(gradients) -> gradInput
	obj.Set("stepBackward", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return "Expected gradients"
		}

		var grads []float32
		jsData := args[0]
		length := jsData.Get("length").Int()
		grads = make([]float32, length)

		for i := 0; i < length; i++ {
			grads[i] = float32(jsData.Index(i).Float())
		}

		gradInput, _ := network.StepBackward(state, grads)

		// Return gradInput as Float32Array
		arrayConstructor := js.Global().Get("Float32Array")
		jsArray := arrayConstructor.New(len(gradInput))
		for i, v := range gradInput {
			jsArray.SetIndex(i, v)
		}

		return jsArray
	}))

	return obj
}

// loadNetworkFromString loads a network from saved JSON string
func loadNetworkFromString(this js.Value, args []js.Value) interface{} {
	if len(args) < 2 {
		errMsg := "Expected 2 arguments: JSON string and model ID"
		js.Global().Get("console").Call("error", errMsg)
		return errMsg
	}

	jsonString := args[0].String()
	modelID := args[1].String()

	// Load network from JSON string
	network, err := nn.LoadModelFromString(jsonString, modelID)
	if err != nil {
		errMsg := fmt.Sprintf("Error: failed to load network: %v", err)
		js.Global().Get("console").Call("error", errMsg)
		return errMsg
	}

	// Create JS object with all network methods
	obj := js.Global().Get("Object").New()

	// Use reflection to get all methods
	networkValue := reflect.ValueOf(network)
	networkType := networkValue.Type()

	for i := 0; i < networkType.NumMethod(); i++ {
		method := networkType.Method(i)
		// Only export public methods
		if method.Name[0] >= 'A' && method.Name[0] <= 'Z' {
			obj.Set(method.Name, methodWrapper(network, method.Name))
		}
	}

	// Add createStepState method
	obj.Set("createStepState", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return "Expected 1 argument: inputSize"
		}
		inputSize := args[0].Int()
		state := network.InitStepState(inputSize)
		return createStepStateWrapper(state, network)
	}))

	// Add createTweenState method for neural tweening
	obj.Set("createTweenState", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		useChainRule := false
		if len(args) >= 1 {
			useChainRule = args[0].Bool()
		}

		config := nn.DefaultTweenConfig(network.TotalLayers())
		config.UseChainRule = useChainRule

		ts := nn.NewTweenState(network, config)
		return createTweenStateWrapper(ts, network)
	}))

	// Add getInputSize for convenience
	obj.Set("getInputSize", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return network.InputSize
	}))

	return obj
}

// ============================================================================
// TweenState Wrapper - Exposes neural tweening to JavaScript
// ============================================================================

// createTweenStateWrapper creates a JS object for a TweenState
func createTweenStateWrapper(ts *nn.TweenState, network *nn.Network) js.Value {
	obj := js.Global().Get("Object").New()

	// TweenStep(input, targetClass, outputSize, learningRate) -> loss
	obj.Set("TweenStep", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 4 {
			return "Expected 4 arguments: input, targetClass, outputSize, learningRate"
		}

		// Parse input array
		jsInput := args[0]
		inputLen := jsInput.Get("length").Int()
		input := make([]float32, inputLen)
		for i := 0; i < inputLen; i++ {
			input[i] = float32(jsInput.Index(i).Float())
		}

		targetClass := args[1].Int()
		outputSize := args[2].Int()
		learningRate := float32(args[3].Float())

		loss := ts.TweenStep(network, input, targetClass, outputSize, learningRate)
		return float64(loss)
	}))

	// setChainRule(enabled) - Enable/disable chain rule mode
	obj.Set("setChainRule", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return "Expected 1 argument: enabled (bool)"
		}
		ts.Config.UseChainRule = args[0].Bool()
		return nil
	}))

	// getChainRule() -> bool
	obj.Set("getChainRule", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return ts.Config.UseChainRule
	}))

	// getTweenSteps() -> number of tween steps performed
	obj.Set("getTweenSteps", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return ts.TweenSteps
	}))

	return obj
}

// createTweenState creates a TweenState for a network
func createTweenStateFromNetwork(this js.Value, args []js.Value) interface{} {
	// This function expects the network object to be passed
	// args[0] = network pointer (stored in closure)
	// args[1] = useChainRule (optional bool)

	return "Use network.createTweenState(useChainRule) instead"
}

// ============================================================================
// AdaptationTracker Wrapper - Tracks accuracy with task changes
// ============================================================================

// createAdaptationTrackerWrapper creates a JS object for an AdaptationTracker
func createAdaptationTrackerWrapper(tracker *nn.AdaptationTracker) js.Value {
	obj := js.Global().Get("Object").New()

	// setModelInfo(modelName, modeName)
	obj.Set("setModelInfo", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 2 {
			return "Expected 2 arguments: modelName, modeName"
		}
		tracker.SetModelInfo(args[0].String(), args[1].String())
		return nil
	}))

	// scheduleTaskChange(atOffsetMs, taskID, taskName)
	obj.Set("scheduleTaskChange", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 3 {
			return "Expected 3 arguments: atOffsetMs, taskID, taskName"
		}
		offsetMs := args[0].Int()
		taskID := args[1].Int()
		taskName := args[2].String()
		tracker.ScheduleTaskChange(time.Duration(offsetMs)*time.Millisecond, taskID, taskName)
		return nil
	}))

	// start(initialTask, initialTaskID)
	obj.Set("start", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 2 {
			return "Expected 2 arguments: initialTask, initialTaskID"
		}
		tracker.Start(args[0].String(), args[1].Int())
		return nil
	}))

	// recordOutput(isCorrect)
	obj.Set("recordOutput", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return "Expected 1 argument: isCorrect"
		}
		tracker.RecordOutput(args[0].Bool())
		return nil
	}))

	// getCurrentTask() -> taskID
	obj.Set("getCurrentTask", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return tracker.GetCurrentTask()
	}))

	// finalize() -> JSON result
	obj.Set("finalize", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		result := tracker.Finalize()
		jsonBytes, err := json.Marshal(result)
		if err != nil {
			return fmt.Sprintf("Error marshaling result: %v", err)
		}
		return string(jsonBytes)
	}))

	return obj
}

// createAdaptationTracker creates an AdaptationTracker
func createAdaptationTracker(this js.Value, args []js.Value) interface{} {
	if len(args) < 2 {
		return "Expected 2 arguments: windowDurationMs, totalDurationMs"
	}

	windowMs := args[0].Int()
	totalMs := args[1].Int()

	tracker := nn.NewAdaptationTracker(
		time.Duration(windowMs)*time.Millisecond,
		time.Duration(totalMs)*time.Millisecond,
	)

	return createAdaptationTrackerWrapper(tracker)
}

// ============================================================================
// Grafting & Stats Support
// ============================================================================

var graftNetworks = make(map[int64]*nn.Network)
var graftNetworkNextID int64 = 1
var graftNetworkMu sync.RWMutex

func createNetworkForGraft(this js.Value, args []js.Value) interface{} {
	if len(args) < 1 {
		return "Expected JSON config"
	}
	jsonConfig := args[0].String()
	network, err := nn.BuildNetworkFromJSON(jsonConfig)
	if err != nil {
		return -1
	}
	network.InitializeWeights()

	graftNetworkMu.Lock()
	id := graftNetworkNextID
	graftNetworkNextID++
	graftNetworks[id] = network
	graftNetworkMu.Unlock()

	return id
}

func graftNetworksWrapper(this js.Value, args []js.Value) interface{} {
	if len(args) < 2 {
		return `{"error": "Expected idsJSON and combineMode"}`
	}

	var networkIDs []int64
	if err := json.Unmarshal([]byte(args[0].String()), &networkIDs); err != nil {
		return fmt.Sprintf(`{"error": "Invalid IDs: %v"}`, err)
	}

	networks := make([]*nn.Network, 0, len(networkIDs))
	graftNetworkMu.RLock()
	for _, id := range networkIDs {
		if net, ok := graftNetworks[id]; ok {
			networks = append(networks, net)
		}
	}
	graftNetworkMu.RUnlock()

	if len(networks) < 2 {
		return `{"error": "Need at least 2 networks"}`
	}

	mode := args[1].String()
	res, err := nn.GraftNetworks(networks, mode)
	if err != nil {
		return fmt.Sprintf(`{"error": "%v"}`, err)
	}

	out := map[string]interface{}{
		"success":      true,
		"type":         res.Type,
		"num_branches": len(res.ParallelBranches),
		"combine_mode": res.CombineMode,
	}
	b, _ := json.Marshal(out)
	return string(b)
}

func kmeansClusterWrapper(this js.Value, args []js.Value) interface{} {
	if len(args) < 3 {
		return `{"error": "Expected dataJSON, k, iterations"}`
	}
	var data [][]float32
	json.Unmarshal([]byte(args[0].String()), &data)
	k := args[1].Int()
	iter := args[2].Int()

	centroids, assignments := nn.KMeansCluster(data, int(k), int(iter), false)

	res := map[string]interface{}{
		"centroids":  centroids,
		"assignment": assignments,
	}
	b, _ := json.Marshal(res)
	return string(b)
}

func computeCorrelationWrapper(this js.Value, args []js.Value) interface{} {
	if len(args) < 2 {
		return `{"error": "Expected matrixA, matrixB"}`
	}
	var A, B [][]float32
	json.Unmarshal([]byte(args[0].String()), &A)
	// args[1] can be null/undefined for auto-correlation?
	if !args[1].IsNull() && !args[1].IsUndefined() {
		json.Unmarshal([]byte(args[1].String()), &B)
	}

	res := nn.ComputeCorrelationMatrix(A, nil)
	b, _ := json.Marshal(res)
	return string(b)
}

func findComplementaryMatchesWrapper(this js.Value, args []js.Value) interface{} {
	if len(args) < 2 {
		return `{"error": "Expected modelsJSON, minCoverage"}`
	}
	var models []nn.ModelPerformance
	json.Unmarshal([]byte(args[0].String()), &models)
	minCov := args[1].Float()

	matches := nn.FindComplementaryMatches(models, minCov)
	b, _ := json.Marshal(matches)
	return string(b)
}

func silhouetteScoreWrapper(this js.Value, args []js.Value) interface{} {
	if len(args) < 2 {
		return -1.0
	}
	var data [][]float32
	json.Unmarshal([]byte(args[0].String()), &data)
	var assignments []int
	json.Unmarshal([]byte(args[1].String()), &assignments)

	return nn.ComputeSilhouetteScore(data, assignments)
}

// ============================================================================
// Learning Rate Scheduler Support
// ============================================================================

var schedulers = make(map[int64]nn.LRScheduler)
var schedulerNextID int64 = 1
var schedulerMu sync.RWMutex

func createSchedulerWrapper(sched nn.LRScheduler) js.Value {
	schedulerMu.Lock()
	id := schedulerNextID
	schedulerNextID++
	schedulers[id] = sched
	schedulerMu.Unlock()

	obj := js.Global().Get("Object").New()
	obj.Set("id", int(id))

	obj.Set("getLR", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return 0.0
		}
		return sched.GetLR(args[0].Int())
	}))

	obj.Set("name", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return sched.Name()
	}))

	obj.Set("free", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		schedulerMu.Lock()
		delete(schedulers, id)
		schedulerMu.Unlock()
		return nil
	}))

	return obj
}

func createConstantScheduler(this js.Value, args []js.Value) interface{} {
	if len(args) < 1 {
		return "Expected baseLR"
	}
	sched := nn.NewConstantScheduler(float32(args[0].Float()))
	return createSchedulerWrapper(sched)
}

func createLinearDecayScheduler(this js.Value, args []js.Value) interface{} {
	if len(args) < 3 {
		return "Expected startLR, endLR, totalSteps"
	}
	sched := nn.NewLinearDecayScheduler(float32(args[0].Float()), float32(args[1].Float()), args[2].Int())
	return createSchedulerWrapper(sched)
}

func createCosineScheduler(this js.Value, args []js.Value) interface{} {
	if len(args) < 3 {
		return "Expected startLR, minLR, totalSteps"
	}
	sched := nn.NewCosineAnnealingScheduler(float32(args[0].Float()), float32(args[1].Float()), args[2].Int())
	return createSchedulerWrapper(sched)
}

// setGpuDebug wrapper
func setGpuDebug(this js.Value, args []js.Value) interface{} {
	if len(args) < 1 {
		return "Expected 1 argument: enabled (boolean)"
	}
	enabled := args[0].Bool()
	gpu.SetDebug(enabled)
	return nil
}

func main() {
	fmt.Println("Loom WASM Framework Initialized")

	js.Global().Set("createLoomNetwork", js.FuncOf(createNetworkFromJSON))
	js.Global().Set("loadLoomNetwork", js.FuncOf(loadNetworkFromString))
	js.Global().Set("setupWebGPU", js.FuncOf(setupWebGPUWrapper))
	js.Global().Set("setGpuDebug", js.FuncOf(setGpuDebug))

	// Register AdaptationTracker for adaptation benchmarks
	js.Global().Set("createAdaptationTracker", js.FuncOf(createAdaptationTracker))

	// Register Grafting & Stats
	js.Global().Set("createNetworkForGraft", js.FuncOf(createNetworkForGraft))
	js.Global().Set("graftNetworks", js.FuncOf(graftNetworksWrapper))
	js.Global().Set("kmeansCluster", js.FuncOf(kmeansClusterWrapper))
	js.Global().Set("computeCorrelation", js.FuncOf(computeCorrelationWrapper))
	js.Global().Set("findComplementaryMatches", js.FuncOf(findComplementaryMatchesWrapper))
	js.Global().Set("computeSilhouetteScore", js.FuncOf(silhouetteScoreWrapper))

	// Register Schedulers
	js.Global().Set("createConstantScheduler", js.FuncOf(createConstantScheduler))
	js.Global().Set("createLinearDecayScheduler", js.FuncOf(createLinearDecayScheduler))
	js.Global().Set("createCosineScheduler", js.FuncOf(createCosineScheduler))

	fmt.Println("Available functions:")
	fmt.Println("  - createLoomNetwork(jsonConfig) - Create network from JSON")
	fmt.Println("  - loadLoomNetwork(jsonString, modelID) - Load network from saved JSON")
	fmt.Println("  - createAdaptationTracker(windowMs, totalMs) - Create tracker")
	fmt.Println("")
	fmt.Println("Network methods:")
	fmt.Println("  - network.createStepState(inputSize) - For stepping API")
	fmt.Println("  - network.createTweenState(useChainRule) - For tween learning")

	// Keep the program running
	select {}
}
