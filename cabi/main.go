package main

/*
#include <stdlib.h>
#include <stdbool.h>

// Ensure bool/true/false are available
#ifndef __cplusplus
#ifndef bool
#define bool _Bool
#define true 1
#define false 0
#endif
#endif
*/
import "C"

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strconv"
	"sync"
	"time"
	"unsafe"

	"github.com/openfluke/loom/nn"
)

var (
	mu      sync.Mutex
	nextID  int64 = 1
	objects       = map[int64]interface{}{}
)

func put(o interface{}) int64 {
	mu.Lock()
	defer mu.Unlock()
	id := nextID
	nextID++
	objects[id] = o
	return id
}

func get(id int64) (interface{}, bool) {
	mu.Lock()
	defer mu.Unlock()
	o, ok := objects[id]
	return o, ok
}

func del(id int64) {
	mu.Lock()
	defer mu.Unlock()
	delete(objects, id)
}

func cstr(s string) *C.char        { return C.CString(s) }
func asJSON(v interface{}) *C.char { b, _ := json.Marshal(v); return C.CString(string(b)) }
func errJSON(msg string) *C.char {
	return asJSON(map[string]string{"error": msg})
}

// Dynamic parameter conversion (adapted from WASM bridge)
func convertParameter(param interface{}, expectedType reflect.Type, paramIndex int) (reflect.Value, error) {
	// Handle nil
	if param == nil {
		return reflect.Zero(expectedType), nil
	}

	switch expectedType.Kind() {
	case reflect.Slice:
		return convertSlice(param, expectedType, paramIndex)
	case reflect.Map:
		return convertMap(param, expectedType, paramIndex)
	case reflect.Struct:
		return convertStructValue(param, expectedType, paramIndex)
	case reflect.Ptr:
		if expectedType.Elem().Kind() == reflect.Struct {
			return convertStruct(param, expectedType.Elem(), paramIndex)
		}
		return reflect.Value{}, fmt.Errorf("parameter %d: unsupported pointer type %s", paramIndex, expectedType.String())

	// Integers (including custom types like LayerType)
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		switch v := param.(type) {
		case float64:
			intVal := reflect.New(expectedType).Elem()
			intVal.SetInt(int64(v))
			return intVal, nil
		case float32:
			intVal := reflect.New(expectedType).Elem()
			intVal.SetInt(int64(v))
			return intVal, nil
		case int, int8, int16, int32, int64:
			val := reflect.ValueOf(v)
			return val.Convert(expectedType), nil
		default:
			return reflect.Value{}, fmt.Errorf("parameter %d: expected int, got %T", paramIndex, param)
		}

	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		switch v := param.(type) {
		case float64:
			uintVal := reflect.New(expectedType).Elem()
			uintVal.SetUint(uint64(v))
			return uintVal, nil
		case float32:
			uintVal := reflect.New(expectedType).Elem()
			uintVal.SetUint(uint64(v))
			return uintVal, nil
		case uint, uint8, uint16, uint32, uint64:
			val := reflect.ValueOf(v)
			return val.Convert(expectedType), nil
		default:
			return reflect.Value{}, fmt.Errorf("parameter %d: expected uint, got %T", paramIndex, param)
		}

	case reflect.Float32, reflect.Float64:
		switch v := param.(type) {
		case float64:
			return reflect.ValueOf(v).Convert(expectedType), nil
		case float32:
			return reflect.ValueOf(v).Convert(expectedType), nil
		default:
			return reflect.Value{}, fmt.Errorf("parameter %d: expected float, got %T", paramIndex, param)
		}

	case reflect.Bool:
		if v, ok := param.(bool); ok {
			return reflect.ValueOf(v), nil
		}
		return reflect.Value{}, fmt.Errorf("parameter %d: expected bool, got %T", paramIndex, param)

	case reflect.String:
		if v, ok := param.(string); ok {
			return reflect.ValueOf(v), nil
		}
		return reflect.Value{}, fmt.Errorf("parameter %d: expected string, got %T", paramIndex, param)

	default:
		// time.Duration special case
		if expectedType == reflect.TypeOf(time.Duration(0)) {
			switch v := param.(type) {
			case float64:
				return reflect.ValueOf(time.Duration(v)), nil
			case float32:
				return reflect.ValueOf(time.Duration(v)), nil
			}
			return reflect.Value{}, fmt.Errorf("parameter %d: expected duration number, got %T", paramIndex, param)
		}
		return reflect.Zero(expectedType), fmt.Errorf("parameter %d: unsupported type %s", paramIndex, expectedType.String())
	}
}

func convertStruct(param interface{}, structType reflect.Type, paramIndex int) (reflect.Value, error) {
	if param == nil {
		return reflect.Zero(reflect.PointerTo(structType)), nil
	}

	mapData, ok := param.(map[string]interface{})
	if !ok {
		return reflect.Value{}, fmt.Errorf("parameter %d: expected map for struct, got %T", paramIndex, param)
	}

	structPtr := reflect.New(structType)
	structVal := structPtr.Elem()

	for fieldName, fieldValue := range mapData {
		field := structVal.FieldByName(fieldName)
		if !field.IsValid() || !field.CanSet() {
			continue
		}

		converted, err := convertParameter(fieldValue, field.Type(), paramIndex)
		if err != nil {
			return reflect.Value{}, fmt.Errorf("field %s: %w", fieldName, err)
		}
		field.Set(converted)
	}

	return structPtr, nil
}

func convertStructValue(param interface{}, structType reflect.Type, paramIndex int) (reflect.Value, error) {
	if param == nil {
		return reflect.Zero(structType), nil
	}

	mapData, ok := param.(map[string]interface{})
	if !ok {
		return reflect.Value{}, fmt.Errorf("parameter %d: expected map for struct, got %T", paramIndex, param)
	}

	structVal := reflect.New(structType).Elem()

	for fieldName, fieldValue := range mapData {
		field := structVal.FieldByName(fieldName)
		if !field.IsValid() || !field.CanSet() {
			continue
		}

		converted, err := convertParameter(fieldValue, field.Type(), paramIndex)
		if err != nil {
			return reflect.Value{}, fmt.Errorf("field %s: %w", fieldName, err)
		}
		field.Set(converted)
	}

	return structVal, nil
}

func convertSlice(param interface{}, expectedType reflect.Type, paramIndex int) (reflect.Value, error) {
	if param == nil {
		return reflect.Zero(expectedType), nil
	}

	val, ok := param.([]interface{})
	if !ok {
		// Coerce a single number into a 1-length slice
		if n, ok := param.(float64); ok {
			s := reflect.MakeSlice(expectedType, 1, 1)
			elem, err := convertParameter(n, expectedType.Elem(), paramIndex)
			if err != nil {
				return reflect.Value{}, err
			}
			s.Index(0).Set(elem)
			return s, nil
		}
		return reflect.Value{}, fmt.Errorf("parameter %d: expected slice, got %T", paramIndex, param)
	}

	elemType := expectedType.Elem()
	out := reflect.MakeSlice(expectedType, len(val), len(val))
	for i, raw := range val {
		if elemType.Kind() == reflect.Slice {
			conv, err := convertSlice(raw, elemType, paramIndex)
			if err != nil {
				return reflect.Value{}, err
			}
			out.Index(i).Set(conv)
			continue
		}
		if elemType.Kind() == reflect.Struct {
			if m, ok := raw.(map[string]interface{}); ok {
				b, _ := json.Marshal(m)
				holder := reflect.New(elemType).Interface()
				if err := json.Unmarshal(b, holder); err != nil {
					return reflect.Value{}, fmt.Errorf("parameter %d: struct element decode: %v", paramIndex, err)
				}
				out.Index(i).Set(reflect.ValueOf(holder).Elem())
				continue
			}
			return reflect.Value{}, fmt.Errorf("parameter %d: invalid struct element %T", paramIndex, raw)
		}
		elem, err := convertParameter(raw, elemType, paramIndex)
		if err != nil {
			return reflect.Value{}, err
		}
		out.Index(i).Set(elem)
	}
	return out, nil
}

func convertMap(param interface{}, expectedType reflect.Type, paramIndex int) (reflect.Value, error) {
	if param == nil {
		return reflect.Zero(expectedType), nil
	}

	jm, ok := param.(map[string]interface{})
	if !ok {
		return reflect.Value{}, fmt.Errorf("parameter %d: expected map, got %T", paramIndex, param)
	}
	keyT := expectedType.Key()
	valT := expectedType.Elem()
	out := reflect.MakeMap(expectedType)

	for keyStr, raw := range jm {
		var keyV reflect.Value
		switch keyT.Kind() {
		case reflect.String:
			keyV = reflect.ValueOf(keyStr)
		case reflect.Int:
			i, err := strconv.Atoi(keyStr)
			if err != nil {
				return reflect.Value{}, fmt.Errorf("parameter %d: bad map key %q", paramIndex, keyStr)
			}
			keyV = reflect.ValueOf(i)
		default:
			return reflect.Value{}, fmt.Errorf("parameter %d: unsupported map key type %s", paramIndex, keyT)
		}
		valV, err := convertParameter(raw, valT, paramIndex)
		if err != nil {
			return reflect.Value{}, err
		}
		out.SetMapIndex(keyV, valV)
	}
	return out, nil
}

// Dynamic method calling with JSON arguments
func callMethodWithJSON(target reflect.Value, argsJSON string) *C.char {
	mt := target.Type()
	want := mt.NumIn()

	// Parse argsJSON as array of parameters
	var params []interface{}
	if argsJSON == "" || argsJSON == "[]" {
		params = nil
	} else if err := json.Unmarshal([]byte(argsJSON), &params); err != nil {
		// If not an array, try single element
		var single interface{}
		if err2 := json.Unmarshal([]byte(argsJSON), &single); err2 != nil {
			return errJSON("Invalid JSON input: " + err.Error())
		}
		params = []interface{}{single}
	}

	if len(params) != want {
		return errJSON(fmt.Sprintf("Expected %d parameters, got %d", want, len(params)))
	}

	in := make([]reflect.Value, want)
	for i := 0; i < want; i++ {
		exp := mt.In(i)
		val, err := convertParameter(params[i], exp, i)
		if err != nil {
			return errJSON(err.Error())
		}
		in[i] = val
	}

	defer func() {
		if r := recover(); r != nil {
			// Handle panics gracefully
		}
	}()

	out := target.Call(in)
	res := make([]interface{}, len(out))
	for i := range out {
		res[i] = out[i].Interface()
	}
	return asJSON(res)
}

// C ABI Exports

//export Loom_NewNetwork
func Loom_NewNetwork(
	inputSize, gridRows, gridCols, layersPerCell C.int,
	useGPU C.bool,
) *C.char {
	network := nn.NewNetwork(
		int(inputSize),
		int(gridRows),
		int(gridCols),
		int(layersPerCell),
	)

	var gpuInitOK bool
	var gpuInitMs int64

	if bool(useGPU) {
		startGPU := time.Now()
		if err := network.InitGPU(); err != nil {
			gpuInitOK = false
			gpuInitMs = time.Since(startGPU).Milliseconds()
		} else {
			gpuInitOK = true
			gpuInitMs = time.Since(startGPU).Milliseconds()
		}
	}

	id := put(network)
	totalLayers := int(gridRows) * int(gridCols) * int(layersPerCell)

	return asJSON(map[string]interface{}{
		"handle":       id,
		"type":         "Network",
		"input_size":   int(inputSize),
		"grid_rows":    int(gridRows),
		"grid_cols":    int(gridCols),
		"layers_cell":  int(layersPerCell),
		"total_layers": totalLayers,
		"gpu":          gpuInitOK,
		"gpu_init_ms":  gpuInitMs,
	})
}

//export Loom_InitDenseLayer
func Loom_InitDenseLayer(inputSize, outputSize, activation C.int) *C.char {
	config := nn.InitDenseLayer(
		int(inputSize),
		int(outputSize),
		nn.ActivationType(int(activation)),
	)

	data, err := json.Marshal(config)
	if err != nil {
		return errJSON("failed to serialize layer config: " + err.Error())
	}

	return C.CString(string(data))
}

//export Loom_CallLayerInit
func Loom_CallLayerInit(funcName *C.char, argsJSON *C.char) *C.char {
	name := C.GoString(funcName)

	// Get function from registry
	fn, ok := nn.GetLayerInitFunction(name)
	if !ok {
		return errJSON(fmt.Sprintf("layer init function %s not found", name))
	}

	// Parse arguments
	var args []interface{}
	if err := json.Unmarshal([]byte(C.GoString(argsJSON)), &args); err != nil {
		return errJSON("failed to parse arguments: " + err.Error())
	}

	// Call via reflection
	funcValue := reflect.ValueOf(fn)
	funcType := funcValue.Type()

	if len(args) != funcType.NumIn() {
		return errJSON(fmt.Sprintf("%s expects %d arguments, got %d", name, funcType.NumIn(), len(args)))
	}

	inputs := make([]reflect.Value, len(args))
	for i, arg := range args {
		expectedType := funcType.In(i)

		// Convert argument to expected type
		switch expectedType.Kind() {
		case reflect.Int:
			if num, ok := arg.(float64); ok {
				// Check if it's a named type (like ActivationType) or plain int
				if expectedType.String() != "int" {
					// It's a named int type (enum), convert via reflection
					inputs[i] = reflect.ValueOf(int(num)).Convert(expectedType)
				} else {
					inputs[i] = reflect.ValueOf(int(num))
				}
			} else {
				return errJSON(fmt.Sprintf("argument %d: expected int, got %T", i, arg))
			}
		case reflect.Float32:
			if num, ok := arg.(float64); ok {
				inputs[i] = reflect.ValueOf(float32(num))
			} else {
				return errJSON(fmt.Sprintf("argument %d: expected float32, got %T", i, arg))
			}
		default:
			return errJSON(fmt.Sprintf("argument %d: unsupported type %s", i, expectedType))
		}
	}

	// Call the function
	results := funcValue.Call(inputs)
	if len(results) != 1 {
		return errJSON("expected 1 return value")
	}

	// Marshal the LayerConfig result
	data, err := json.Marshal(results[0].Interface())
	if err != nil {
		return errJSON("failed to serialize result: " + err.Error())
	}

	return C.CString(string(data))
}

//export Loom_ListLayerInitFunctions
func Loom_ListLayerInitFunctions() *C.char {
	functions := nn.ListLayerInitFunctions()
	data, err := json.Marshal(functions)
	if err != nil {
		return errJSON("failed to serialize functions list: " + err.Error())
	}
	return C.CString(string(data))
}

//export Loom_SetLayer
func Loom_SetLayer(handle int64, row, col, layer C.int, configJSON *C.char) *C.char {
	obj, ok := get(handle)
	if !ok {
		return errJSON(fmt.Sprintf("invalid handle %d", handle))
	}

	network, ok := obj.(*nn.Network)
	if !ok {
		return errJSON("not a Network")
	}

	var config nn.LayerConfig
	if err := json.Unmarshal([]byte(C.GoString(configJSON)), &config); err != nil {
		return errJSON("failed to parse config JSON: " + err.Error())
	}

	network.SetLayer(int(row), int(col), int(layer), config)

	return asJSON(map[string]string{"status": "layer set"})
}

//export Loom_Call
func Loom_Call(handle int64, method *C.char, argsJSON *C.char) *C.char {
	obj, ok := get(handle)
	if !ok {
		return errJSON(fmt.Sprintf("invalid handle %d", handle))
	}

	methodName := C.GoString(method)
	m := reflect.ValueOf(obj).MethodByName(methodName)
	if !m.IsValid() {
		return errJSON("Method not found: " + methodName)
	}

	return callMethodWithJSON(m, C.GoString(argsJSON))
}

//export Loom_ListMethods
func Loom_ListMethods(handle int64) *C.char {
	obj, ok := get(handle)
	if !ok {
		return errJSON("invalid handle")
	}

	val := reflect.ValueOf(obj)
	typ := val.Type()
	methods := make([]map[string]interface{}, 0)

	for i := 0; i < typ.NumMethod(); i++ {
		method := typ.Method(i)
		if method.Name[0] >= 'A' && method.Name[0] <= 'Z' {
			methodType := method.Type
			params := make([]string, methodType.NumIn()-1) // -1 for receiver
			for j := 1; j < methodType.NumIn(); j++ {
				params[j-1] = methodType.In(j).String()
			}
			returns := make([]string, methodType.NumOut())
			for j := 0; j < methodType.NumOut(); j++ {
				returns[j] = methodType.Out(j).String()
			}

			methods = append(methods, map[string]interface{}{
				"name":       method.Name,
				"parameters": params,
				"returns":    returns,
			})
		}
	}

	return asJSON(map[string]interface{}{
		"methods": methods,
		"count":   len(methods),
	})
}

//export Loom_GetInfo
func Loom_GetInfo(handle int64) *C.char {
	obj, ok := get(handle)
	if !ok {
		return errJSON("invalid handle")
	}

	val := reflect.ValueOf(obj)
	typ := val.Type()

	info := map[string]interface{}{
		"type":    typ.String(),
		"kind":    typ.Kind().String(),
		"methods": typ.NumMethod(),
		"handle":  handle,
	}

	// Add network-specific info
	if network, ok := obj.(*nn.Network); ok {
		// Check GPU availability via reflection (deviceInfo is private)
		hasGPU := false
		netVal := reflect.ValueOf(network).Elem()
		if deviceField := netVal.FieldByName("deviceInfo"); deviceField.IsValid() {
			hasGPU = !deviceField.IsNil()
		}
		info["gpu_enabled"] = hasGPU
		info["grid_rows"] = network.GridRows
		info["grid_cols"] = network.GridCols
		info["layers_per_cell"] = network.LayersPerCell
		info["input_size"] = network.InputSize
		info["batch_size"] = network.BatchSize
		info["total_layers"] = network.TotalLayers()
	}

	return asJSON(info)
}

//export Loom_SaveModel
func Loom_SaveModel(handle int64, modelID *C.char) *C.char {
	obj, ok := get(handle)
	if !ok {
		return errJSON("invalid handle")
	}

	network, ok := obj.(*nn.Network)
	if !ok {
		return errJSON("not a Network")
	}

	jsonStr, err := network.SaveModelToString(C.GoString(modelID))
	if err != nil {
		return errJSON("failed to save model: " + err.Error())
	}

	return C.CString(jsonStr)
}

//export Loom_LoadModel
func Loom_LoadModel(jsonString *C.char, modelID *C.char) *C.char {
	network, err := nn.LoadModelFromString(C.GoString(jsonString), C.GoString(modelID))
	if err != nil {
		return errJSON("failed to load model: " + err.Error())
	}

	id := put(network)

	return asJSON(map[string]interface{}{
		"handle":       id,
		"type":         "Network",
		"grid_rows":    network.GridRows,
		"grid_cols":    network.GridCols,
		"layers_cell":  network.LayersPerCell,
		"total_layers": network.TotalLayers(),
	})
}

//export Loom_Free
func Loom_Free(handle int64) {
	// Clean up GPU resources if it's a network
	if obj, ok := get(handle); ok {
		if network, ok := obj.(*nn.Network); ok {
			network.ReleaseGPU()
		}
	}
	del(handle)
}

//export Loom_FreeCString
func Loom_FreeCString(p *C.char) {
	C.free(unsafe.Pointer(p))
}

//export Loom_GetVersion
func Loom_GetVersion() *C.char {
	return cstr("LOOM C ABI v1.0")
}

func main() {
	// This is a library, main() should be empty for CGO
}
