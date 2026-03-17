//go:build js && wasm
// +build js,wasm

// welvet WASM — M-POLY-VTD AI Engine for JavaScript/TypeScript (Loom v0.73.0)
//
// Exposes the poly.VolumetricNetwork API to JavaScript via WebAssembly.
// Supports 21 numerical types, systolic propagation, target propagation,
// weight morphing, and WebGPU acceleration.

package main

import (
	"encoding/json"
	"fmt"
	"sync"
	"syscall/js"

	"github.com/openfluke/loom/poly"
)

// ──────────────────────────────────────────────────────────────────────────────
// Global Registries
// ──────────────────────────────────────────────────────────────────────────────

var (
	networks      = make(map[int64]*poly.VolumetricNetwork)
	networkNextID int64 = 1

	systolicStates  = make(map[int64]*poly.SystolicState[float32])
	systolicNextID  int64 = 1

	targetPropStates = make(map[int64]*poly.TargetPropState[float32])
	targetPropNextID int64 = 1

	neatPopulations  = make(map[int64]*poly.NEATPopulation)
	neatPopNextID    int64 = 1

	mu sync.RWMutex
)

func storeNetwork(n *poly.VolumetricNetwork) int64 {
	mu.Lock()
	id := networkNextID
	networkNextID++
	networks[id] = n
	mu.Unlock()
	return id
}

func getNetwork(id int64) (*poly.VolumetricNetwork, bool) {
	mu.RLock()
	defer mu.RUnlock()
	n, ok := networks[id]
	return n, ok
}

func storeSystolicState(s *poly.SystolicState[float32]) int64 {
	mu.Lock()
	id := systolicNextID
	systolicNextID++
	systolicStates[id] = s
	mu.Unlock()
	return id
}

func getSystolicState(id int64) (*poly.SystolicState[float32], bool) {
	mu.RLock()
	defer mu.RUnlock()
	s, ok := systolicStates[id]
	return s, ok
}

func storeTargetPropState(s *poly.TargetPropState[float32]) int64 {
	mu.Lock()
	id := targetPropNextID
	targetPropNextID++
	targetPropStates[id] = s
	mu.Unlock()
	return id
}

func getTargetPropState(id int64) (*poly.TargetPropState[float32], bool) {
	mu.RLock()
	defer mu.RUnlock()
	s, ok := targetPropStates[id]
	return s, ok
}

func storeNEATPopulation(pop *poly.NEATPopulation) int64 {
	mu.Lock()
	id := neatPopNextID
	neatPopNextID++
	neatPopulations[id] = pop
	mu.Unlock()
	return id
}

func getNEATPopulation(id int64) (*poly.NEATPopulation, bool) {
	mu.RLock()
	defer mu.RUnlock()
	p, ok := neatPopulations[id]
	return p, ok
}

// ──────────────────────────────────────────────────────────────────────────────
// JS ↔ Go Helpers
// ──────────────────────────────────────────────────────────────────────────────

func errObj(msg string) js.Value {
	obj := js.Global().Get("Object").New()
	obj.Set("error", msg)
	return obj
}

func okObj() js.Value {
	obj := js.Global().Get("Object").New()
	obj.Set("status", "ok")
	return obj
}

func jsFloat32Array(data []float32) js.Value {
	arr := js.Global().Get("Float32Array").New(len(data))
	for i, v := range data {
		arr.SetIndex(i, float64(v))
	}
	return arr
}

func readFloat32Array(jsVal js.Value) []float32 {
	length := jsVal.Get("length").Int()
	out := make([]float32, length)
	for i := 0; i < length; i++ {
		out[i] = float32(jsVal.Index(i).Float())
	}
	return out
}

// ──────────────────────────────────────────────────────────────────────────────
// SystolicState Wrapper
// ──────────────────────────────────────────────────────────────────────────────

func createSystolicStateWrapper(n *poly.VolumetricNetwork, s *poly.SystolicState[float32]) js.Value {
	obj := js.Global().Get("Object").New()

	obj.Set("setInput", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return "Expected input data"
		}
		data := readFloat32Array(args[0])
		t := poly.NewTensorFromSlice(data, 1, len(data))
		s.SetInput(t)
		return nil
	}))

	obj.Set("step", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		capture := len(args) > 0 && args[0].Bool()
		dur := poly.SystolicForward(n, s, capture)
		return float64(dur.Nanoseconds()) / 1e6 // ms
	}))

	obj.Set("getOutput", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		// Return the last non-nil layer output
		layerIdx := -1
		if len(args) > 0 && !args[0].IsUndefined() {
			layerIdx = args[0].Int()
		}

		if layerIdx >= 0 && layerIdx < len(s.LayerData) {
			if s.LayerData[layerIdx] != nil {
				return jsFloat32Array(s.LayerData[layerIdx].Data)
			}
			return js.Global().Get("Float32Array").New(0)
		}

		// Default: last layer
		for i := len(s.LayerData) - 1; i >= 0; i-- {
			if s.LayerData[i] != nil {
				return jsFloat32Array(s.LayerData[i].Data)
			}
		}
		return js.Global().Get("Float32Array").New(0)
	}))

	obj.Set("backward", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return "Expected gradients"
		}
		data := readFloat32Array(args[0])
		t := poly.NewTensorFromSlice(data, 1, len(data))
		gradIn, _, err := poly.SystolicBackward(n, s, t)
		if err != nil {
			return errObj(err.Error())
		}
		if gradIn == nil {
			return js.Global().Get("Float32Array").New(0)
		}
		return jsFloat32Array(gradIn.Data)
	}))

	obj.Set("applyTargetProp", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 2 {
			return "Expected target data and learning rate"
		}
		data := readFloat32Array(args[0])
		lr := float32(args[1].Float())
		t := poly.NewTensorFromSlice(data, 1, len(data))
		poly.SystolicApplyTargetProp(n, s, t, lr)
		return nil
	}))

	obj.Set("stepCount", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return float64(s.StepCount)
	}))

	obj.Set("free", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return nil
	}))

	return obj
}

// ──────────────────────────────────────────────────────────────────────────────
// TargetPropState Wrapper
// ──────────────────────────────────────────────────────────────────────────────

func createTargetPropStateWrapper(n *poly.VolumetricNetwork, s *poly.TargetPropState[float32]) js.Value {
	obj := js.Global().Get("Object").New()

	obj.Set("forward", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return "Expected input data"
		}
		data := readFloat32Array(args[0])
		t := poly.NewTensorFromSlice(data, 1, len(data))
		out := poly.TargetPropForward(n, s, t)
		if out == nil {
			return js.Global().Get("Float32Array").New(0)
		}
		return jsFloat32Array(out.Data)
	}))

	obj.Set("backward", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return "Expected target data"
		}
		data := readFloat32Array(args[0])
		t := poly.NewTensorFromSlice(data, 1, len(data))
		poly.TargetPropBackward(n, s, t)
		return nil
	}))

	obj.Set("backwardChainRule", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return "Expected target data"
		}
		data := readFloat32Array(args[0])
		t := poly.NewTensorFromSlice(data, 1, len(data))
		poly.TargetPropBackwardChainRule(n, s, t)
		return nil
	}))

	obj.Set("applyGaps", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		lr := float32(0.001)
		if len(args) > 0 {
			lr = float32(args[0].Float())
		}
		poly.ApplyTargetPropGaps(n, s, lr)
		return nil
	}))

	obj.Set("free", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return nil
	}))

	return obj
}

// ──────────────────────────────────────────────────────────────────────────────
// Network Wrapper
// ──────────────────────────────────────────────────────────────────────────────

func createNetworkWrapper(n *poly.VolumetricNetwork) js.Value {
	id := storeNetwork(n)
	obj := js.Global().Get("Object").New()
	obj.Set("_id", float64(id))

	// sequentialForward(Float32Array | number[]) -> Float32Array
	obj.Set("sequentialForward", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return "Expected input data"
		}
		data := readFloat32Array(args[0])
		t := poly.NewTensorFromSlice(data, 1, len(data)) // shape: [batchSize=1, features]
		out, _, _ := poly.ForwardPolymorphic(n, t)
		if out == nil {
			return js.Global().Get("Float32Array").New(0)
		}
		return jsFloat32Array(out.Data)
	}))

	// getInfo() -> JSON string
	obj.Set("getInfo", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		info := map[string]interface{}{
			"depth":          n.Depth,
			"rows":           n.Rows,
			"cols":           n.Cols,
			"layers_per_cell": n.LayersPerCell,
			"total_layers":   len(n.Layers),
			"use_gpu":        n.UseGPU,
		}
		if len(n.Layers) > 0 {
			info["default_dtype"] = n.Layers[0].DType
		}
		b, _ := json.Marshal(info)
		return string(b)
	}))

	// extractDNA() -> JSON string
	obj.Set("extractDNA", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		dna := poly.ExtractDNA(n)
		b, _ := json.Marshal(dna)
		return string(b)
	}))

	// spliceDNA(otherNetworkID int, cfgJSON? string) -> Network JS object
	obj.Set("spliceDNA", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return `{"error": "Expected other network ID"}`
		}
		otherID := int64(args[0].Int())
		cfgJSON := "{}"
		if len(args) > 1 {
			cfgJSON = args[1].String()
		}
		other, ok := getNetwork(otherID)
		if !ok {
			return `{"error": "invalid other network ID"}`
		}
		var cfg poly.SpliceConfig
		if err := json.Unmarshal([]byte(cfgJSON), &cfg); err != nil {
			cfg = poly.DefaultSpliceConfig()
		}
		child := poly.SpliceDNA(n, other, cfg)
		return createNetworkWrapper(child)
	}))

	// neatMutate(cfgJSON? string) -> Network JS object
	obj.Set("neatMutate", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		cfgJSON := "{}"
		if len(args) > 0 {
			cfgJSON = args[0].String()
		}
		var cfg poly.NEATConfig
		if err := json.Unmarshal([]byte(cfgJSON), &cfg); err != nil {
			dModel := 64
			if len(n.Layers) > 0 {
				dModel = n.Layers[0].InputHeight
			}
			cfg = poly.DefaultNEATConfig(dModel)
		}
		child := poly.NEATMutate(n, cfg)
		return createNetworkWrapper(child)
	}))

	// extractBlueprint(modelID) -> JSON string
	obj.Set("extractBlueprint", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		modelID := "model"
		if len(args) > 0 {
			modelID = args[0].String()
		}
		bp := poly.ExtractNetworkBlueprint(n, modelID)
		b, _ := json.Marshal(bp)
		return string(b)
	}))

	// getLayerCount() -> number
	obj.Set("getLayerCount", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return len(n.Layers)
	}))

	// getLayerSpec(layerIdx) -> JSON string
	obj.Set("getLayerSpec", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return "Expected layerIdx"
		}
		idx := args[0].Int()
		if idx < 0 || idx >= len(n.Layers) {
			return `{"error": "layer index out of range"}`
		}
		l := &n.Layers[idx]
		spec := map[string]interface{}{
			"z": l.Z, "y": l.Y, "x": l.X, "l": l.L,
			"type":       l.Type,
			"dtype":      l.DType,
			"activation": l.Activation,
			"input_height": l.InputHeight,
			"output_height": l.OutputHeight,
		}
		b, _ := json.Marshal(spec)
		return string(b)
	}))

	// morphLayer(layerIdx, dtypeInt) -> ok/error JSON
	obj.Set("morphLayer", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 2 {
			return `{"error": "Expected layerIdx and dtype"}`
		}
		idx := args[0].Int()
		dtype := poly.DType(args[1].Int())
		if idx < 0 || idx >= len(n.Layers) {
			return `{"error": "layer index out of range"}`
		}
		if err := poly.MorphLayer(&n.Layers[idx], dtype); err != nil {
			b, _ := json.Marshal(map[string]string{"error": err.Error()})
			return string(b)
		}
		return `{"status": "ok"}`
	}))

	// initGPU() -> Promise<ok/error JSON>
	obj.Set("initGPU", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		handler := js.FuncOf(func(this js.Value, pArgs []js.Value) interface{} {
			resolve := pArgs[0]
			reject := pArgs[1]
			go func() {
				if err := n.InitWGPU(); err != nil {
					reject.Invoke(err.Error())
					return
				}
				resolve.Invoke(`{"status": "ok"}`)
			}()
			return nil
		})
		return js.Global().Get("Promise").New(handler)
	}))

	// syncToGPU() -> Promise<ok/error JSON>
	obj.Set("syncToGPU", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		handler := js.FuncOf(func(this js.Value, pArgs []js.Value) interface{} {
			resolve := pArgs[0]
			reject := pArgs[1]
			go func() {
				if err := n.SyncAllToGPU(); err != nil {
					reject.Invoke(err.Error())
					return
				}
				resolve.Invoke(`{"status": "ok"}`)
			}()
			return nil
		})
		return js.Global().Get("Promise").New(handler)
	}))

	// syncToCPU()
	obj.Set("syncToCPU", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		for i := range n.Layers {
			n.Layers[i].SyncToCPU()
		}
		n.UseGPU = false
		return nil
	}))

	// train(batchesJSON, epochs, lr) -> JSON result
	obj.Set("train", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 3 {
			return `{"error": "Expected batchesJSON, epochs, learningRate"}`
		}

		type tensorData struct {
			Shape []int     `json:"shape"`
			Data  []float32 `json:"data"`
		}
		type batchJSON struct {
			Input  tensorData `json:"input"`
			Target tensorData `json:"target"`
		}
		var rawBatches []batchJSON
		if err := json.Unmarshal([]byte(args[0].String()), &rawBatches); err != nil {
			return fmt.Sprintf(`{"error": "invalid batches JSON: %v"}`, err)
		}

		epochs := args[1].Int()
		lr := float32(args[2].Float())

		batches := make([]poly.TrainingBatch[float32], len(rawBatches))
		for i, b := range rawBatches {
			batches[i] = poly.TrainingBatch[float32]{
				Input:  poly.NewTensorFromSlice(b.Input.Data, b.Input.Shape...),
				Target: poly.NewTensorFromSlice(b.Target.Data, b.Target.Shape...),
			}
		}

		cfg := &poly.TrainingConfig{
			Epochs:       epochs,
			LearningRate: lr,
			LossType:     "mse",
			Verbose:      false,
		}

		trainResult, err := poly.Train(n, batches, cfg)
		if err != nil {
			return fmt.Sprintf(`{"error": "%v"}`, err)
		}

		result := map[string]interface{}{
			"final_loss":       trainResult.FinalLoss,
			"duration_ms":      float64(trainResult.TotalTime.Nanoseconds()) / 1e6,
			"epochs_completed": epochs,
			"loss_history":     trainResult.LossHistory,
		}
		b, _ := json.Marshal(result)
		return string(b)
	}))

	// createSystolicState() -> SystolicState wrapper object
	obj.Set("createSystolicState", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		s := poly.NewSystolicState[float32](n)
		return createSystolicStateWrapper(n, s)
	}))

	// createTargetPropState(useChainRule?) -> TargetPropState wrapper object
	obj.Set("createTargetPropState", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		cfg := poly.DefaultTargetPropConfig()
		if len(args) > 0 {
			cfg.UseChainRule = args[0].Bool()
		}
		s := poly.NewTargetPropState[float32](n, cfg)
		return createTargetPropStateWrapper(n, s)
	}))

	// free() - no-op in WASM (GC handles it), but provided for API compatibility
	obj.Set("free", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return nil
	}))

	return obj
}

// ──────────────────────────────────────────────────────────────────────────────
// NEAT Population Wrapper
// ──────────────────────────────────────────────────────────────────────────────

func createNEATPopulationWrapper(pop *poly.NEATPopulation) js.Value {
	id := storeNEATPopulation(pop)
	obj := js.Global().Get("Object").New()
	obj.Set("_id", float64(id))
	obj.Set("size", float64(len(pop.Networks)))

	// getNetwork(index) -> Network JS object (shared reference into population)
	obj.Set("getNetwork", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return nil
		}
		idx := args[0].Int()
		if idx < 0 || idx >= len(pop.Networks) {
			return nil
		}
		return createNetworkWrapper(pop.Networks[idx])
	}))

	// evolveWithFitnesses(Float64Array | number[]) -> status JSON
	obj.Set("evolveWithFitnesses", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return `{"error": "Expected fitnesses array"}`
		}
		length := args[0].Get("length").Int()
		fitnesses := make([]float64, length)
		for i := 0; i < length; i++ {
			fitnesses[i] = args[0].Index(i).Float()
		}
		i := 0
		pop.Evolve(func(_ *poly.VolumetricNetwork) float64 {
			if i < len(fitnesses) {
				f := fitnesses[i]
				i++
				return f
			}
			return 0
		})
		obj.Set("size", float64(len(pop.Networks)))
		return `{"status": "ok"}`
	}))

	// best() -> Network JS object
	obj.Set("best", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		best := pop.Best()
		if best == nil {
			return nil
		}
		return createNetworkWrapper(best)
	}))

	// bestFitness() -> float
	obj.Set("bestFitness", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		return pop.BestFitness()
	}))

	// summary(generation int) -> string
	obj.Set("summary", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		gen := 0
		if len(args) > 0 {
			gen = args[0].Int()
		}
		return pop.Summary(gen)
	}))

	// free()
	obj.Set("free", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		mu.Lock()
		delete(neatPopulations, id)
		mu.Unlock()
		return nil
	}))

	return obj
}

// ──────────────────────────────────────────────────────────────────────────────
// Top-Level WASM Exports
// ──────────────────────────────────────────────────────────────────────────────

// createLoomNetwork(jsonConfig string) -> Network JS object
func createLoomNetworkFn(this js.Value, args []js.Value) interface{} {
	if len(args) < 1 {
		return "Expected JSON config string"
	}
	n, err := poly.BuildNetworkFromJSON([]byte(args[0].String()))
	if err != nil {
		return fmt.Sprintf(`{"error": "failed to build network: %v"}`, err)
	}
	return createNetworkWrapper(n)
}

// loadLoomNetwork(safetensorsPath string) -> Network JS object
func loadLoomNetworkFn(this js.Value, args []js.Value) interface{} {
	if len(args) < 1 {
		return "Expected safetensors path"
	}
	n, err := poly.LoadUniversal(args[0].String())
	if err != nil {
		return fmt.Sprintf(`{"error": "failed to load network: %v"}`, err)
	}
	return createNetworkWrapper(n)
}

// compareDNA(dnaA JSON, dnaB JSON) -> similarity JSON
func compareDNAFn(this js.Value, args []js.Value) interface{} {
	if len(args) < 2 {
		return `{"error": "Expected two DNA JSON strings"}`
	}
	var dnaA, dnaB poly.NetworkDNA
	if err := json.Unmarshal([]byte(args[0].String()), &dnaA); err != nil {
		return fmt.Sprintf(`{"error": "invalid dnaA: %v"}`, err)
	}
	if err := json.Unmarshal([]byte(args[1].String()), &dnaB); err != nil {
		return fmt.Sprintf(`{"error": "invalid dnaB: %v"}`, err)
	}
	result := poly.CompareNetworks(dnaA, dnaB)
	b, _ := json.Marshal(result)
	return string(b)
}

// getDefaultTargetPropConfig() -> JSON config
func getDefaultTargetPropConfigFn(this js.Value, args []js.Value) interface{} {
	cfg := poly.DefaultTargetPropConfig()
	b, _ := json.Marshal(cfg)
	return string(b)
}

// defaultSpliceConfig() -> JSON
func defaultSpliceConfigFn(this js.Value, args []js.Value) interface{} {
	cfg := poly.DefaultSpliceConfig()
	b, _ := json.Marshal(cfg)
	return string(b)
}

// defaultNEATConfig(dModel int) -> JSON
func defaultNEATConfigFn(this js.Value, args []js.Value) interface{} {
	dModel := 64
	if len(args) > 0 {
		dModel = args[0].Int()
	}
	cfg := poly.DefaultNEATConfig(dModel)
	b, _ := json.Marshal(cfg)
	return string(b)
}

// createLoomNEATPopulation(networkID int, size int, cfgJSON? string) -> Population JS object
func createLoomNEATPopulationFn(this js.Value, args []js.Value) interface{} {
	if len(args) < 2 {
		return `{"error": "Expected networkID and size"}`
	}
	networkID := int64(args[0].Int())
	size := args[1].Int()
	cfgJSON := "{}"
	if len(args) > 2 {
		cfgJSON = args[2].String()
	}

	seed, ok := getNetwork(networkID)
	if !ok {
		return `{"error": "invalid network ID"}`
	}

	var cfg poly.NEATConfig
	if err := json.Unmarshal([]byte(cfgJSON), &cfg); err != nil {
		dModel := 64
		if len(seed.Layers) > 0 {
			dModel = seed.Layers[0].InputHeight
		}
		cfg = poly.DefaultNEATConfig(dModel)
	}

	pop := poly.NewNEATPopulation(seed, size, cfg)
	return createNEATPopulationWrapper(pop)
}

// ──────────────────────────────────────────────────────────────────────────────
// WebGPU Setup Helper
// ──────────────────────────────────────────────────────────────────────────────

func awaitPromise(promise js.Value) (js.Value, error) {
	resultCh := make(chan js.Value)
	errCh := make(chan error)

	then := js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		resultCh <- args[0]
		return nil
	})
	catch := js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		errCh <- fmt.Errorf("%s", args[0].String())
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

func setupWebGPUFn(this js.Value, args []js.Value) interface{} {
	handler := js.FuncOf(func(this js.Value, pArgs []js.Value) interface{} {
		resolve := pArgs[0]
		reject := pArgs[1]

		go func() {
			nav := js.Global().Get("navigator")
			if nav.IsUndefined() || nav.Get("gpu").IsUndefined() {
				reject.Invoke("WebGPU not supported")
				return
			}

			adapterPromise := nav.Get("gpu").Call("requestAdapter")
			adapterVal, err := awaitPromise(adapterPromise)
			if err != nil || adapterVal.IsNull() {
				reject.Invoke("requestAdapter failed")
				return
			}

			devicePromise := adapterVal.Call("requestDevice")
			deviceVal, err := awaitPromise(devicePromise)
			if err != nil {
				reject.Invoke("requestDevice failed")
				return
			}

			js.Global().Set("webgpuAdapter", adapterVal)
			js.Global().Set("webgpuDevice", deviceVal)
			js.Global().Set("webgpuQueue", deviceVal.Get("queue"))

			fmt.Println("welvet WASM: WebGPU initialized")
			resolve.Invoke("WebGPU ready")
		}()
		return nil
	})
	return js.Global().Get("Promise").New(handler)
}

// ──────────────────────────────────────────────────────────────────────────────
// main
// ──────────────────────────────────────────────────────────────────────────────

func main() {
	fmt.Println("welvet WASM — M-POLY-VTD Engine (Loom v0.74.0) initialized")

	js.Global().Set("createLoomNetwork", js.FuncOf(createLoomNetworkFn))
	js.Global().Set("loadLoomNetwork", js.FuncOf(loadLoomNetworkFn))
	js.Global().Set("setupWebGPU", js.FuncOf(setupWebGPUFn))
	js.Global().Set("compareLoomDNA", js.FuncOf(compareDNAFn))
	js.Global().Set("getDefaultTargetPropConfig", js.FuncOf(getDefaultTargetPropConfigFn))
	js.Global().Set("defaultSpliceConfig", js.FuncOf(defaultSpliceConfigFn))
	js.Global().Set("defaultNEATConfig", js.FuncOf(defaultNEATConfigFn))
	js.Global().Set("createLoomNEATPopulation", js.FuncOf(createLoomNEATPopulationFn))

	fmt.Println("Exposed globals:")
	fmt.Println("  createLoomNetwork(jsonConfig)          - Build network from JSON")
	fmt.Println("  loadLoomNetwork(path)                  - Load from SafeTensors")
	fmt.Println("  setupWebGPU()                          - Init WebGPU (Promise)")
	fmt.Println("  compareLoomDNA(dnaA, dnaB)             - Compare network DNA")
	fmt.Println("  getDefaultTargetPropConfig(n)          - Default TP config JSON")
	fmt.Println("  defaultSpliceConfig()                  - Default splice config JSON")
	fmt.Println("  defaultNEATConfig(dModel)              - Default NEAT config JSON")
	fmt.Println("  createLoomNEATPopulation(id, size, cfg)- Create NEAT population")
	fmt.Println("")
	fmt.Println("Network methods:")
	fmt.Println("  .sequentialForward(Float32Array)       - Full forward pass")
	fmt.Println("  .createSystolicState()                 - Stepping API")
	fmt.Println("  .createTargetPropState(chainRule)      - Target propagation")
	fmt.Println("  .morphLayer(idx, dtype)                - Switch numerical type")
	fmt.Println("  .initGPU()  .syncToGPU()               - WebGPU acceleration")
	fmt.Println("  .extractDNA()  .extractBlueprint()")
	fmt.Println("  .spliceDNA(otherID, cfgJSON)           - Genetic crossover")
	fmt.Println("  .neatMutate(cfgJSON)                   - NEAT mutation")
	fmt.Println("  .train(batchesJSON, epochs, lr)")
	fmt.Println("")
	fmt.Println("Population methods:")
	fmt.Println("  .getNetwork(index)                     - Get member network")
	fmt.Println("  .evolveWithFitnesses(float64[])        - Run one generation")
	fmt.Println("  .best()                                - Best network wrapper")
	fmt.Println("  .bestFitness()                         - Top fitness score")
	fmt.Println("  .summary(generation)                   - Diagnostic string")

	select {}
}
