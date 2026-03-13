package nn

import (
	"fmt"
	"reflect"
)

// LayerInitFunction represents a layer initialization function
type LayerInitFunction struct {
	Name     string
	Function interface{} `json:"-"` // Omit from JSON serialization
	NumArgs  int
	ArgTypes []string
}

// layerInitRegistry is the global registry of layer init functions
var layerInitRegistry = map[string]interface{}{
	"InitDenseLayer":              InitDenseLayer,
	"InitConv2DLayer":             InitConv2DLayer,
	"InitMultiHeadAttentionLayer": InitMultiHeadAttentionLayer,
	"InitRNNLayer":                InitRNNLayer,
	"InitLSTMLayer":               InitLSTMLayer,
}

// GetLayerInitFunction returns a layer init function by name
func GetLayerInitFunction(name string) (interface{}, bool) {
	fn, ok := layerInitRegistry[name]
	return fn, ok
}

// ListLayerInitFunctions returns metadata about all available layer init functions
func ListLayerInitFunctions() []LayerInitFunction {
	var functions []LayerInitFunction

	for name, fn := range layerInitRegistry {
		fnType := reflect.TypeOf(fn)

		// Get argument types
		argTypes := make([]string, fnType.NumIn())
		for i := 0; i < fnType.NumIn(); i++ {
			argTypes[i] = fnType.In(i).String()
		}

		functions = append(functions, LayerInitFunction{
			Name:     name,
			Function: fn,
			NumArgs:  fnType.NumIn(),
			ArgTypes: argTypes,
		})
	}

	return functions
}

// CallLayerInitFunction calls a layer init function by name with the provided arguments
func CallLayerInitFunction(name string, args ...interface{}) (LayerConfig, error) {
	fn, ok := layerInitRegistry[name]
	if !ok {
		return LayerConfig{}, fmt.Errorf("layer init function %s not found", name)
	}

	fnValue := reflect.ValueOf(fn)
	fnType := fnValue.Type()

	// Validate number of arguments
	if len(args) != fnType.NumIn() {
		return LayerConfig{}, fmt.Errorf("%s expects %d arguments, got %d", name, fnType.NumIn(), len(args))
	}

	// Convert arguments to reflect.Value
	inputs := make([]reflect.Value, len(args))
	for i, arg := range args {
		inputs[i] = reflect.ValueOf(arg)
	}

	// Call the function
	results := fnValue.Call(inputs)

	// The result should be a LayerConfig
	if len(results) != 1 {
		return LayerConfig{}, fmt.Errorf("expected 1 return value, got %d", len(results))
	}

	config, ok := results[0].Interface().(LayerConfig)
	if !ok {
		return LayerConfig{}, fmt.Errorf("expected LayerConfig return type")
	}

	return config, nil
}
