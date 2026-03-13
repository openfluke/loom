package poly

import (
	"encoding/json"
	"fmt"
	"reflect"
)

// MethodInfo represents metadata about a method.
type MethodInfo struct {
	MethodName string          `json:"method_name"`
	Parameters []ParameterInfo `json:"parameters"`
	Returns    []string        `json:"returns"`
}

// ParameterInfo represents metadata about a parameter.
type ParameterInfo struct {
	Name string `json:"name"`
	Type string `json:"type"`
}

// GetMethodsJSON returns a JSON string containing all methods attached to the VolumetricNetwork struct.
func (n *VolumetricNetwork) GetMethodsJSON() (string, error) {
	methods, err := n.GetMethods()
	if err != nil {
		return "", fmt.Errorf("failed to retrieve methods: %w", err)
	}

	data, err := json.MarshalIndent(methods, "", "  ")
	if err != nil {
		return "", fmt.Errorf("failed to serialize methods to JSON: %w", err)
	}

	return string(data), nil
}

// GetMethods retrieves all public methods of the VolumetricNetwork struct.
func (n *VolumetricNetwork) GetMethods() ([]MethodInfo, error) {
	var methods []MethodInfo

	networkType := reflect.TypeOf(n)
	for i := 0; i < networkType.NumMethod(); i++ {
		method := networkType.Method(i)

		// Only include public methods
		if method.Name[0] < 'A' || method.Name[0] > 'Z' {
			continue
		}

		// Collect parameter information
		var params []ParameterInfo
		methodType := method.Type
		for j := 1; j < methodType.NumIn(); j++ { // Skip receiver
			paramType := methodType.In(j)
			params = append(params, ParameterInfo{
				Name: fmt.Sprintf("param%d", j-1),
				Type: paramType.String(),
			})
		}

		// Collect return type information
		var returns []string
		for j := 0; j < methodType.NumOut(); j++ {
			returns = append(returns, methodType.Out(j).String())
		}

		methods = append(methods, MethodInfo{
			MethodName: method.Name,
			Parameters: params,
			Returns:    returns,
		})
	}

	return methods, nil
}

// GetMethodSignature returns the signature of a specific method.
func (n *VolumetricNetwork) GetMethodSignature(methodName string) (string, error) {
	networkType := reflect.TypeOf(n)
	method, found := networkType.MethodByName(methodName)
	if !found {
		return "", fmt.Errorf("method %s not found", methodName)
	}

	methodType := method.Type
	params := []string{}
	for j := 1; j < methodType.NumIn(); j++ {
		params = append(params, methodType.In(j).String())
	}

	returns := []string{}
	for j := 0; j < methodType.NumOut(); j++ {
		returns = append(returns, methodType.Out(j).String())
	}

	signature := fmt.Sprintf("%s(%s)", methodName, joinStrings(params, ", "))
	if len(returns) > 0 {
		if len(returns) == 1 {
			signature += " " + returns[0]
		} else {
			signature += " (" + joinStrings(returns, ", ") + ")"
		}
	}

	return signature, nil
}

// ListMethods returns a simple list of all public method names.
func (n *VolumetricNetwork) ListMethods() []string {
	var methodNames []string
	networkType := reflect.TypeOf(n)
	for i := 0; i < networkType.NumMethod(); i++ {
		method := networkType.Method(i)
		if method.Name[0] >= 'A' && method.Name[0] <= 'Z' {
			methodNames = append(methodNames, method.Name)
		}
	}
	return methodNames
}

// HasMethod checks if a method exists on the VolumetricNetwork.
func (n *VolumetricNetwork) HasMethod(methodName string) bool {
	networkType := reflect.TypeOf(n)
	_, found := networkType.MethodByName(methodName)
	return found
}

// joinStrings is a helper to join string slices.
func joinStrings(strs []string, sep string) string {
	if len(strs) == 0 {
		return ""
	}
	result := strs[0]
	for i := 1; i < len(strs); i++ {
		result += sep + strs[i]
	}
	return result
}
