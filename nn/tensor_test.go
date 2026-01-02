package nn

import (
	"math"
	"testing"
)

// TestTensorCreation verifies basic tensor operations
func TestTensorCreation(t *testing.T) {
	// Test NewTensor
	tensor := NewTensor[float32](3, 4)
	if tensor.Size() != 12 {
		t.Errorf("Expected size 12, got %d", tensor.Size())
	}
	if len(tensor.Shape) != 2 || tensor.Shape[0] != 3 || tensor.Shape[1] != 4 {
		t.Errorf("Expected shape [3, 4], got %v", tensor.Shape)
	}

	// Test NewTensorFromSlice
	data := []float64{1, 2, 3, 4, 5, 6}
	tensor2 := NewTensorFromSlice(data, 2, 3)
	if tensor2.Size() != 6 {
		t.Errorf("Expected size 6, got %d", tensor2.Size())
	}
	if tensor2.Data[0] != 1 || tensor2.Data[5] != 6 {
		t.Errorf("Data not correctly initialized")
	}
}

// TestTensorClone verifies tensor cloning
func TestTensorClone(t *testing.T) {
	original := NewTensorFromSlice([]int32{1, 2, 3, 4}, 4)
	clone := original.Clone()

	// Modify original
	original.Data[0] = 100

	// Clone should be unchanged
	if clone.Data[0] != 1 {
		t.Errorf("Clone was modified when original changed")
	}
}

// TestTensorReshape verifies tensor reshaping
func TestTensorReshape(t *testing.T) {
	tensor := NewTensorFromSlice([]float32{1, 2, 3, 4, 5, 6}, 6)
	reshaped := tensor.Reshape(2, 3)

	if reshaped == nil {
		t.Fatal("Reshape returned nil")
	}
	if len(reshaped.Shape) != 2 || reshaped.Shape[0] != 2 || reshaped.Shape[1] != 3 {
		t.Errorf("Expected shape [2, 3], got %v", reshaped.Shape)
	}

	// Invalid reshape should return nil
	invalid := tensor.Reshape(2, 2)
	if invalid != nil {
		t.Error("Invalid reshape should return nil")
	}
}

// TestActivateGeneric verifies generic activation functions
func TestActivateGeneric(t *testing.T) {
	// Test with float32
	resultF32 := Activate[float32](0.5, ActivationSigmoid)
	expectedF32 := float32(1.0 / (1.0 + math.Exp(-0.5)))
	if math.Abs(float64(resultF32-expectedF32)) > 1e-6 {
		t.Errorf("Sigmoid float32: expected %f, got %f", expectedF32, resultF32)
	}

	// Test with float64
	resultF64 := Activate[float64](0.5, ActivationSigmoid)
	expectedF64 := 1.0 / (1.0 + math.Exp(-0.5))
	if math.Abs(resultF64-expectedF64) > 1e-10 {
		t.Errorf("Sigmoid float64: expected %f, got %f", expectedF64, resultF64)
	}

	// Test ReLU
	reluResult := Activate[float32](-1.0, ActivationScaledReLU)
	if reluResult != 0 {
		t.Errorf("ScaledReLU of negative should be 0, got %f", reluResult)
	}

	reluResultPos := Activate[float32](1.0, ActivationScaledReLU)
	if math.Abs(float64(reluResultPos-1.1)) > 1e-6 {
		t.Errorf("ScaledReLU of 1 should be 1.1, got %f", reluResultPos)
	}
}

// TestDenseForwardGeneric verifies generic dense layer
func TestDenseForwardGeneric(t *testing.T) {
	// Create simple 2x3 weight matrix and input
	weights := NewTensorFromSlice([]float32{
		1, 0, 0,
		0, 1, 0,
	}, 2*3) // 2 inputs, 3 outputs
	bias := NewTensorFromSlice([]float32{0.1, 0.2, 0.3}, 3)
	input := NewTensorFromSlice([]float32{1.0, 2.0}, 2)

	preAct, postAct := DenseForward(input, weights, bias, 2, 3, 1, ActivationLeakyReLU)

	// Expected: [1*1 + 2*0 + 0.1, 1*0 + 2*1 + 0.2, 1*0 + 2*0 + 0.3] = [1.1, 2.2, 0.3]
	if len(preAct.Data) != 3 {
		t.Fatalf("Expected 3 outputs, got %d", len(preAct.Data))
	}

	if math.Abs(float64(preAct.Data[0]-1.1)) > 1e-5 {
		t.Errorf("preAct[0]: expected 1.1, got %f", preAct.Data[0])
	}
	if math.Abs(float64(preAct.Data[1]-2.2)) > 1e-5 {
		t.Errorf("preAct[1]: expected 2.2, got %f", preAct.Data[1])
	}
	if math.Abs(float64(preAct.Data[2]-0.3)) > 1e-5 {
		t.Errorf("preAct[2]: expected 0.3, got %f", preAct.Data[2])
	}

	// With LeakyReLU, positive values pass through unchanged
	if math.Abs(float64(postAct.Data[0]-1.1)) > 1e-5 {
		t.Errorf("postAct[0] with LeakyReLU: expected 1.1, got %f", postAct.Data[0])
	}
}

// TestDenseFloat64 verifies dense layer works with float64
func TestDenseFloat64(t *testing.T) {
	weights := NewTensorFromSlice([]float64{1, 2}, 2)
	bias := NewTensorFromSlice([]float64{0.5}, 1)
	input := NewTensorFromSlice([]float64{3.0, 4.0}, 2)

	preAct, _ := DenseForward(input, weights, bias, 2, 1, 1, ActivationTanh)

	// Expected: 3*1 + 4*2 + 0.5 = 11.5
	expected := 11.5
	if math.Abs(preAct.Data[0]-expected) > 1e-10 {
		t.Errorf("Float64 dense: expected %f, got %f", expected, preAct.Data[0])
	}
}

// TestBackwardCompatibility verifies old float32 APIs still work
func TestBackwardCompatibility(t *testing.T) {
	// Test activateCPU (the old function)
	result := activateCPU(0.5, ActivationSigmoid)
	expected := float32(1.0 / (1.0 + math.Exp(-0.5)))
	if math.Abs(float64(result-expected)) > 1e-6 {
		t.Errorf("activateCPU backward compat: expected %f, got %f", expected, result)
	}
}
