//go:build js && wasm
// +build js,wasm

package main

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strconv"
	"syscall/js"

	"github.com/openfluke/loom/nn"
)

// methodWrapper dynamically wraps each method to expose it to JavaScript
func methodWrapper(network *nn.Network, methodName string) js.Func {
	return js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		method := reflect.ValueOf(network).MethodByName(methodName)
		if !method.IsValid() {
			return fmt.Sprintf("Method %s not found", methodName)
		}

		methodType := method.Type()

		// Handle methods with no parameters
		if len(args) == 0 || args[0].IsUndefined() || args[0].String() == "" {
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
	case reflect.Map:
		return convertMap(param, expectedType, paramIndex)
	case reflect.Struct:
		// Handle struct types (like nn.LayerConfig)
		return convertStructValue(param, expectedType, paramIndex)
	case reflect.Ptr:
		// Handle pointer types (like *nn.TrainingConfig)
		if expectedType.Elem().Kind() == reflect.Struct {
			return convertStruct(param, expectedType.Elem(), paramIndex)
		}
		return reflect.Value{}, fmt.Errorf("parameter %d: unsupported pointer type %s", paramIndex, expectedType.String())
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		// Handle all integer types including custom types like LayerType
		if val, ok := param.(float64); ok {
			intVal := reflect.New(expectedType).Elem()
			intVal.SetInt(int64(val))
			return intVal, nil
		}
		return reflect.Value{}, fmt.Errorf("parameter %d: expected int, got %T", paramIndex, param)
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		if val, ok := param.(float64); ok {
			uintVal := reflect.New(expectedType).Elem()
			uintVal.SetUint(uint64(val))
			return uintVal, nil
		}
		return reflect.Value{}, fmt.Errorf("parameter %d: expected uint, got %T", paramIndex, param)
	case reflect.Float32:
		if val, ok := param.(float64); ok {
			return reflect.ValueOf(float32(val)).Convert(expectedType), nil
		}
		return reflect.Value{}, fmt.Errorf("parameter %d: expected float32, got %T", paramIndex, param)
	case reflect.Float64:
		if val, ok := param.(float64); ok {
			return reflect.ValueOf(val).Convert(expectedType), nil
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
	default:
		return reflect.Zero(expectedType), fmt.Errorf("parameter %d: unsupported type %s", paramIndex, expectedType.String())
	}
}

// convertStruct converts a map to a struct and returns a pointer
func convertStruct(param interface{}, structType reflect.Type, paramIndex int) (reflect.Value, error) {
	// Handle nil/null values - return nil pointer
	if param == nil {
		return reflect.Zero(reflect.PtrTo(structType)), nil
	}

	mapData, ok := param.(map[string]interface{})
	if !ok {
		return reflect.Value{}, fmt.Errorf("parameter %d: expected map for struct, got %T", paramIndex, param)
	}

	// Create a new instance of the struct
	structPtr := reflect.New(structType)
	structVal := structPtr.Elem()

	// Populate struct fields from map
	for fieldName, fieldValue := range mapData {
		field := structVal.FieldByName(fieldName)
		if !field.IsValid() || !field.CanSet() {
			continue // Skip unknown fields
		}

		// Convert the field value to the appropriate type
		converted, err := convertParameter(fieldValue, field.Type(), paramIndex)
		if err != nil {
			return reflect.Value{}, fmt.Errorf("field %s: %w", fieldName, err)
		}
		field.Set(converted)
	}

	return structPtr, nil
}

// convertStructValue converts a map to a struct and returns the value (not pointer)
func convertStructValue(param interface{}, structType reflect.Type, paramIndex int) (reflect.Value, error) {
	// Handle nil/null values - return zero value
	if param == nil {
		return reflect.Zero(structType), nil
	}

	mapData, ok := param.(map[string]interface{})
	if !ok {
		return reflect.Value{}, fmt.Errorf("parameter %d: expected map for struct, got %T", paramIndex, param)
	}

	// Create a new instance of the struct
	structVal := reflect.New(structType).Elem()

	// Populate struct fields from map
	for fieldName, fieldValue := range mapData {
		field := structVal.FieldByName(fieldName)
		if !field.IsValid() || !field.CanSet() {
			continue // Skip unknown or unexported fields
		}

		// Convert the field value to the appropriate type
		converted, err := convertParameter(fieldValue, field.Type(), paramIndex)
		if err != nil {
			return reflect.Value{}, fmt.Errorf("field %s: %w", fieldName, err)
		}
		field.Set(converted)
	}

	return structVal, nil
}

// convertSlice handles slice type conversions including multidimensional slices
func convertSlice(param interface{}, expectedType reflect.Type, paramIndex int) (reflect.Value, error) {
	// Handle nil/null values - return zero value (nil slice)
	if param == nil {
		return reflect.Zero(expectedType), nil
	}

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
			// Handle struct slices
			if structVal, ok := v.(map[string]interface{}); ok {
				structValue := reflect.New(elemType).Elem()
				for fieldName, fieldVal := range structVal {
					field := structValue.FieldByName(fieldName)
					if field.IsValid() && field.CanSet() {
						converted, err := convertParameter(fieldVal, field.Type(), paramIndex)
						if err == nil {
							field.Set(converted)
						}
					}
				}
				slice.Index(j).Set(structValue)
			}
		default:
			return reflect.Value{}, fmt.Errorf("parameter %d: unsupported slice element type %s", paramIndex, elemType.String())
		}
	}

	return slice, nil
}

// convertMap handles map type conversions
func convertMap(param interface{}, expectedType reflect.Type, paramIndex int) (reflect.Value, error) {
	// Handle nil/null values - return zero value (nil map)
	if param == nil {
		return reflect.Zero(expectedType), nil
	}

	jsonMap, ok := param.(map[string]interface{})
	if !ok {
		return reflect.Value{}, fmt.Errorf("parameter %d: expected map, got %T", paramIndex, param)
	}

	keyType := expectedType.Key()
	valueType := expectedType.Elem()
	mapValue := reflect.MakeMap(expectedType)

	for keyStr, val := range jsonMap {
		var keyValue reflect.Value
		var err error

		switch keyType.Kind() {
		case reflect.Int:
			key, parseErr := strconv.Atoi(keyStr)
			if parseErr != nil {
				return reflect.Value{}, fmt.Errorf("parameter %d: invalid map key %s", paramIndex, keyStr)
			}
			keyValue = reflect.ValueOf(key)
		case reflect.String:
			keyValue = reflect.ValueOf(keyStr)
		default:
			return reflect.Value{}, fmt.Errorf("parameter %d: unsupported map key type %s", paramIndex, keyType.String())
		}

		valueValue, err := convertParameter(val, valueType, paramIndex)
		if err != nil {
			return reflect.Value{}, err
		}

		mapValue.SetMapIndex(keyValue, valueValue)
	}

	return mapValue, nil
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

// newNetworkWrapper creates a wrapper for nn.NewNetwork
func newNetworkWrapper() js.Func {
	return js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		// Parse the NewNetwork arguments: inputSize, gridRows, gridCols, layersPerCell
		if len(args) < 4 {
			errMsg := "Expected 4 arguments: inputSize, gridRows, gridCols, layersPerCell"
			js.Global().Get("console").Call("error", errMsg)
			return errMsg
		}

		inputSize := args[0].Int()
		gridRows := args[1].Int()
		gridCols := args[2].Int()
		layersPerCell := args[3].Int()

		// Create the network
		network := nn.NewNetwork(inputSize, gridRows, gridCols, layersPerCell)

		// Create JavaScript object with all methods
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

		// Add introspection methods (using Network's built-in introspection)
		obj.Set("GetMethods", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
			methodsJSON, err := network.GetMethodsJSON()
			if err != nil {
				return fmt.Sprintf("Error getting methods: %v", err)
			}
			return methodsJSON
		}))

		obj.Set("ListMethods", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
			methods := network.ListMethods()
			data, _ := json.Marshal(methods)
			return string(data)
		}))

		obj.Set("GetMethodSignature", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
			if len(args) < 1 {
				return "Error: method name required"
			}
			methodName := args[0].String()
			signature, err := network.GetMethodSignature(methodName)
			if err != nil {
				return fmt.Sprintf("Error: %v", err)
			}
			return signature
		}))

		obj.Set("HasMethod", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
			if len(args) < 1 {
				return false
			}
			methodName := args[0].String()
			return network.HasMethod(methodName)
		}))

		return obj
	})
}

// loadModelFromStringWrapper creates a wrapper for nn.LoadModelFromString
func loadModelFromStringWrapper() js.Func {
	return js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 2 {
			errMsg := "Expected 2 arguments: jsonString, modelID"
			js.Global().Get("console").Call("error", errMsg)
			return errMsg
		}

		jsonString := args[0].String()
		modelID := args[1].String()

		// Load the network from JSON string
		network, err := nn.LoadModelFromString(jsonString, modelID)
		if err != nil {
			errMsg := fmt.Sprintf("Error loading model: %v", err)
			js.Global().Get("console").Call("error", errMsg)
			return errMsg
		}

		// Create JavaScript object with all methods
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

		// Add introspection methods (using Network's built-in introspection)
		obj.Set("GetMethods", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
			methodsJSON, err := network.GetMethodsJSON()
			if err != nil {
				return fmt.Sprintf("Error getting methods: %v", err)
			}
			return methodsJSON
		}))

		obj.Set("ListMethods", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
			methods := network.ListMethods()
			data, _ := json.Marshal(methods)
			return string(data)
		}))

		obj.Set("GetMethodSignature", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
			if len(args) < 1 {
				return "Error: method name required"
			}
			methodName := args[0].String()
			signature, err := network.GetMethodSignature(methodName)
			if err != nil {
				return fmt.Sprintf("Error: %v", err)
			}
			return signature
		}))

		obj.Set("HasMethod", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
			if len(args) < 1 {
				return false
			}
			methodName := args[0].String()
			return network.HasMethod(methodName)
		}))

		return obj
	})
}

func main() {
	fmt.Println("LOOM Neural Network WASM module initialized")

	// Register factory functions
	js.Global().Set("NewNetwork", newNetworkWrapper())
	js.Global().Set("LoadModelFromString", loadModelFromStringWrapper())

	// Helper functions
	js.Global().Set("InitDenseLayer", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 3 {
			return "Expected 3 arguments: inputSize, outputSize, activation"
		}
		inputSize := args[0].Int()
		outputSize := args[1].Int()
		activation := args[2].Int()

		config := nn.InitDenseLayer(inputSize, outputSize, nn.ActivationType(activation))

		// Serialize config to JSON
		data, err := json.Marshal(config)
		if err != nil {
			return fmt.Sprintf("Error: %v", err)
		}
		return string(data)
	}))

	js.Global().Set("InitMultiHeadAttentionLayer", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		if len(args) < 4 {
			return "Expected 4 arguments: dModel, numHeads, seqLength, activation"
		}
		dModel := args[0].Int()
		numHeads := args[1].Int()
		seqLength := args[2].Int()
		activation := args[3].Int()

		config := nn.InitMultiHeadAttentionLayer(dModel, numHeads, seqLength, nn.ActivationType(activation))

		// Serialize config to JSON
		data, err := json.Marshal(config)
		if err != nil {
			return fmt.Sprintf("Error: %v", err)
		}
		return string(data)
	}))

	fmt.Println("LOOM WASM API ready. Available functions:")
	fmt.Println("  - NewNetwork(inputSize, gridRows, gridCols, layersPerCell)")
	fmt.Println("  - LoadModelFromString(jsonString, modelID)")
	fmt.Println("  - InitDenseLayer(inputSize, outputSize, activation)")
	fmt.Println("  - InitMultiHeadAttentionLayer(dModel, numHeads, seqLength, activation)")

	// Keep the Go program running
	select {}
}
