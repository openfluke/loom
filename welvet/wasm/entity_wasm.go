//go:build js && wasm

package main

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"runtime"
	"syscall/js"

	"github.com/openfluke/loom/poly"
)

var (
	entityTransformers      = make(map[int64]*poly.EntityTransformer)
	entityTransformerNextID int64 = 1
)

func readUint8Array(jsVal js.Value) []byte {
	length := jsVal.Get("length").Int()
	out := make([]byte, length)
	js.CopyBytesToGo(out, jsVal)
	return out
}

func jsUint8Array(data []byte) js.Value {
	arr := js.Global().Get("Uint8Array").New(len(data))
	js.CopyBytesToJS(arr, data)
	return arr
}

func entityLoadOptionsFromJSON(jsonStr string) *poly.EntityLoadOptions {
	if jsonStr == "" || jsonStr == "null" {
		return nil
	}
	var indices []int
	if err := json.Unmarshal([]byte(jsonStr), &indices); err != nil || len(indices) == 0 {
		return nil
	}
	return &poly.EntityLoadOptions{LayerIndices: indices}
}

func buildTransformerFromEntity(et *poly.EntityTransformer, dt poly.DType, template poly.Template) (interface{}, poly.DType) {
	poly.PrepareEntityTransformerInference(et)
	if dt == 0 {
		dt = poly.DTypeFloat32
	}
	switch dt {
	case poly.DTypeFloat64:
		return poly.BuildTransformerFromEntity[float64](et, template), dt
	case poly.DTypeFloat32, poly.DTypeFloat16, poly.DTypeBFloat16, poly.DTypeFP8E4M3, poly.DTypeFP8E5M2:
		return poly.BuildTransformerFromEntity[float32](et, template), poly.DTypeFloat32
	case poly.DTypeInt64:
		return poly.BuildTransformerFromEntity[int64](et, template), dt
	case poly.DTypeInt32:
		return poly.BuildTransformerFromEntity[int32](et, template), dt
	case poly.DTypeInt16:
		return poly.BuildTransformerFromEntity[int16](et, template), dt
	case poly.DTypeInt8, poly.DTypeInt4, poly.DTypeFP4, poly.DTypeInt2, poly.DTypeTernary, poly.DTypeBinary:
		return poly.BuildTransformerFromEntity[int8](et, template), poly.DTypeInt8
	case poly.DTypeUint64:
		return poly.BuildTransformerFromEntity[uint64](et, template), dt
	case poly.DTypeUint32:
		return poly.BuildTransformerFromEntity[uint32](et, template), dt
	case poly.DTypeUint16:
		return poly.BuildTransformerFromEntity[uint16](et, template), dt
	case poly.DTypeUint8, poly.DTypeUint4, poly.DTypeUint2:
		return poly.BuildTransformerFromEntity[uint8](et, template), poly.DTypeUint8
	default:
		return poly.BuildTransformerFromEntity[float32](et, template), poly.DTypeFloat32
	}
}

func storeEntityTransformer(et *poly.EntityTransformer) int64 {
	mu.Lock()
	id := entityTransformerNextID
	entityTransformerNextID++
	entityTransformers[id] = et
	mu.Unlock()
	return id
}

func getEntityTransformer(id int64) (*poly.EntityTransformer, bool) {
	mu.RLock()
	defer mu.RUnlock()
	et, ok := entityTransformers[id]
	return et, ok
}

func deserializeLoomEntityFn(this js.Value, args []js.Value) interface{} {
	if len(args) < 1 {
		return `{"error":"Expected Uint8Array entity wire"}`
	}
	n, err := poly.DeserializeEntity(readUint8Array(args[0]))
	if err != nil {
		return fmt.Sprintf(`{"error":"%v"}`, err)
	}
	return createNetworkWrapper(n)
}

func deserializeEntityLayerFn(this js.Value, args []js.Value) interface{} {
	if len(args) < 2 {
		return `{"error":"Expected entity bytes and layerIndex"}`
	}
	n, err := poly.DeserializeEntityLayer(readUint8Array(args[0]), args[1].Int())
	if err != nil {
		return fmt.Sprintf(`{"error":"%v"}`, err)
	}
	return createNetworkWrapper(n)
}

func deserializeEntityWithOptionsFn(this js.Value, args []js.Value) interface{} {
	if len(args) < 1 {
		return `{"error":"Expected Uint8Array entity wire"}`
	}
	optsJSON := ""
	if len(args) > 1 {
		optsJSON = args[1].String()
	}
	n, err := poly.DeserializeEntityWithOptions(readUint8Array(args[0]), entityLoadOptionsFromJSON(optsJSON))
	if err != nil {
		return fmt.Sprintf(`{"error":"%v"}`, err)
	}
	return createNetworkWrapper(n)
}

func layerPersistenceFromEntityFn(this js.Value, args []js.Value) interface{} {
	if len(args) < 2 {
		return `{"error":"Expected entity bytes and layerIndex"}`
	}
	raw, scale, native, err := poly.LayerPersistenceFromEntity(readUint8Array(args[0]), args[1].Int())
	if err != nil {
		return fmt.Sprintf(`{"error":"%v"}`, err)
	}
	out, _ := json.Marshal(map[string]interface{}{
		"weights": base64.StdEncoding.EncodeToString(raw),
		"scale":   scale,
		"native":  native,
	})
	return string(out)
}

func entityGPUWeightDTypeFn(this js.Value, args []js.Value) interface{} {
	if len(args) < 2 {
		return `{"error":"Expected storedDType and useGPU"}`
	}
	return poly.EntityGPUWeightDType(poly.DType(args[0].Int()), args[1].Int() != 0)
}

func packQ4_0GPUFn(this js.Value, args []js.Value) interface{} {
	if len(args) < 1 {
		return `{"error":"Expected weights JSON array"}`
	}
	var weights []float32
	if err := json.Unmarshal([]byte(args[0].String()), &weights); err != nil {
		return `{"error":"invalid weights JSON"}`
	}
	scales, packed := poly.PackQ4_0GPU(weights)
	out, _ := json.Marshal(map[string]interface{}{"scales": scales, "packed": packed})
	return string(out)
}

func deserializeEntityTransformerFn(this js.Value, args []js.Value) interface{} {
	if len(args) < 1 {
		return `{"error":"Expected Uint8Array entity wire"}`
	}
	et, err := poly.DeserializeEntityTransformer(readUint8Array(args[0]))
	if err != nil {
		return fmt.Sprintf(`{"error":"%v"}`, err)
	}
	return float64(storeEntityTransformer(et))
}

func buildTransformerFromEntityFn(this js.Value, args []js.Value) interface{} {
	if len(args) < 1 {
		return `{"error":"Expected entityTransformer handle"}`
	}
	et, ok := getEntityTransformer(int64(args[0].Float()))
	if !ok {
		return `{"error":"invalid entity transformer handle"}`
	}
	dt := poly.DTypeFloat32
	if len(args) > 1 {
		dt = poly.DType(args[1].Int())
	}
	tr, resolvedDT := buildTransformerFromEntity(et, dt, poly.Template{})
	switch t := tr.(type) {
	case *poly.Transformer[float32]:
		return float64(storeTransformer(t))
	default:
		_ = resolvedDT
		return fmt.Sprintf(`{"error":"transformer dtype %v not exposed in WASM wrapper yet"}`, resolvedDT)
	}
}

func freeEntityTransformerFn(this js.Value, args []js.Value) interface{} {
	if len(args) < 1 {
		return nil
	}
	mu.Lock()
	delete(entityTransformers, int64(args[0].Float()))
	mu.Unlock()
	return nil
}

func loomGCFn(this js.Value, args []js.Value) interface{} {
	runtime.GC()
	return nil
}

func registerEntityWasmGlobals() {
	js.Global().Set("loomGC", js.FuncOf(loomGCFn))
	js.Global().Set("deserializeLoomEntity", js.FuncOf(deserializeLoomEntityFn))
	js.Global().Set("deserializeEntityWithOptions", js.FuncOf(deserializeEntityWithOptionsFn))
	js.Global().Set("deserializeEntityLayer", js.FuncOf(deserializeEntityLayerFn))
	js.Global().Set("layerPersistenceFromEntity", js.FuncOf(layerPersistenceFromEntityFn))
	js.Global().Set("entityGPUWeightDType", js.FuncOf(entityGPUWeightDTypeFn))
	js.Global().Set("packQ4_0GPU", js.FuncOf(packQ4_0GPUFn))
	js.Global().Set("deserializeEntityTransformer", js.FuncOf(deserializeEntityTransformerFn))
	js.Global().Set("buildTransformerFromEntity", js.FuncOf(buildTransformerFromEntityFn))
	js.Global().Set("freeEntityTransformer", js.FuncOf(freeEntityTransformerFn))
}
