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
	"time"

	"github.com/openfluke/loom/poly"
)

// ──────────────────────────────────────────────────────────────────────────────
// Global Registries
// ──────────────────────────────────────────────────────────────────────────────

var (
	networks     = make(map[int64]*poly.VolumetricNetwork)
	networkNextID int64 = 1

	systolicStates   = make(map[int64]*poly.SystolicState[float32])
	systolicNextID   int64 = 1

	targetPropStates  = make(map[int64]*poly.TargetPropState[float32])
	targetPropNextID  int64 = 1

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
		t := poly.NewTensorFromSlice(data, len(data))
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
		t := poly.NewTensorFromSlice(data, len(data))
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
		t := poly.NewTensorFromSlice(data, len(data))
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
		t := poly.NewTensorFromSlice(data, len(data))
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
		t := poly.NewTensorFromSlice(data, len(data))
		poly.TargetPropBackward(n, s, t)
		return nil
	}))

	obj.Set("backwardChainRule", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return "Expected target data"
		}
		data := readFloat32Array(args[0])
		t := poly.NewTensorFromSlice(data, len(data))
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
	obj := js.Global().Get("Object").New()

	// sequentialForward(Float32Array | number[]) -> Float32Array
	obj.Set("sequentialForward", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 1 {
			return "Expected input data"
		}
		data := readFloat32Array(args[0])
		t := poly.NewTensorFromSlice(data, len(data))
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

		type batch struct {
			Input  []float32 `json:"input"`
			Target []float32 `json:"target"`
		}
		var batches []batch
		if err := json.Unmarshal([]byte(args[0].String()), &batches); err != nil {
			return fmt.Sprintf(`{"error": "invalid batches JSON: %v"}`, err)
		}

		epochs := args[1].Int()
		lr := float32(args[2].Float())

		var finalLoss float64
		start := time.Now()

		s := poly.NewSystolicState[float32](n)

		for ep := 0; ep < epochs; ep++ {
			var epochLoss float64
			for _, b := range batches {
				in := poly.NewTensorFromSlice(b.Input, len(b.Input))
				tgt := poly.NewTensorFromSlice(b.Target, len(b.Target))

				s.SetInput(in)
				poly.SystolicForward(n, s, true)

				// Get output
				var out *poly.Tensor[float32]
				for i := len(s.LayerData) - 1; i >= 0; i-- {
					if s.LayerData[i] != nil {
						out = s.LayerData[i]
						break
					}
				}
				if out == nil {
					continue
				}

				// MSE loss gradient
				grad := poly.NewTensor[float32](len(out.Data))
				var loss float64
				for i := range out.Data {
					diff := out.Data[i] - tgt.Data[i]
					grad.Data[i] = 2.0 * diff / float32(len(out.Data))
					loss += float64(diff * diff)
				}
				epochLoss += loss / float64(len(out.Data))

				_, layerGrads, bErr := poly.SystolicBackward(n, s, grad)
				if bErr != nil || layerGrads == nil {
					continue
				}

				// Apply gradients to all layers
				for i := range n.Layers {
					if i < len(layerGrads) && layerGrads[i][1] != nil {
						poly.ApplyRecursiveGradients(&n.Layers[i], poly.ConvertTensor[float32, float32](layerGrads[i][1]), lr)
					}
				}
			}
			finalLoss = epochLoss / float64(len(batches))
		}

		result := map[string]interface{}{
			"final_loss":       finalLoss,
			"duration_ms":      float64(time.Since(start).Nanoseconds()) / 1e6,
			"epochs_completed": epochs,
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
	fmt.Println("welvet WASM — M-POLY-VTD Engine (Loom v0.73.0) initialized")

	js.Global().Set("createLoomNetwork", js.FuncOf(createLoomNetworkFn))
	js.Global().Set("loadLoomNetwork", js.FuncOf(loadLoomNetworkFn))
	js.Global().Set("setupWebGPU", js.FuncOf(setupWebGPUFn))
	js.Global().Set("compareLoomDNA", js.FuncOf(compareDNAFn))
	js.Global().Set("getDefaultTargetPropConfig", js.FuncOf(getDefaultTargetPropConfigFn))

	fmt.Println("Exposed globals:")
	fmt.Println("  createLoomNetwork(jsonConfig)     - Build network from JSON")
	fmt.Println("  loadLoomNetwork(path)             - Load from SafeTensors")
	fmt.Println("  setupWebGPU()                     - Init WebGPU (Promise)")
	fmt.Println("  compareLoomDNA(dnaA, dnaB)        - Compare network DNA")
	fmt.Println("  getDefaultTargetPropConfig(n)     - Default TP config JSON")
	fmt.Println("")
	fmt.Println("Network methods:")
	fmt.Println("  .sequentialForward(Float32Array)  - Full forward pass")
	fmt.Println("  .createSystolicState()            - Stepping API")
	fmt.Println("  .createTargetPropState(chainRule) - Target propagation")
	fmt.Println("  .morphLayer(idx, dtype)           - Switch numerical type")
	fmt.Println("  .initGPU()  .syncToGPU()          - WebGPU acceleration")
	fmt.Println("  .extractDNA()  .extractBlueprint()")
	fmt.Println("  .train(batchesJSON, epochs, lr)")

	select {}
}
