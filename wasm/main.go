//go:build js && wasm
// +build js,wasm

package main

import (
	"encoding/json"
	"fmt"
	"reflect"
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

	return obj
}

func main() {
	fmt.Println("Loom WASM Framework Initialized")

	// Register the network creation function
	js.Global().Set("createLoomNetwork", js.FuncOf(createNetworkFromJSON))

	fmt.Println("Available functions:")
	fmt.Println("  - createLoomNetwork(jsonConfig) - Create network from JSON and get object with all methods")

	// Keep the program running
	select {}
}
