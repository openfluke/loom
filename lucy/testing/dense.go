package testing

import (
	"github.com/openfluke/loom/poly"
)

var denseSpec = TestSpec{
	Name: "Dense",
	Layer: poly.PersistenceLayerSpec{
		Type:         "Dense",
		InputHeight:  1024,
		OutputHeight: 512,
		Activation:   "ReLU",
	},
	InputShape: []int{8, 1024},
}

func init() {
	RegisterTask(func() bool {
		return RunGenericLayerSuite(denseSpec, TestAll)
	})
}

// RunDenseL1Caching is defined in dense_forward.go (includes ASM SC/MC columns).
func RunDenseTraining()    { RunGenericLayerSuite(denseSpec, TestTraining|TestSaveLoad) }
func RunDenseGPUBackward() { RunGenericLayerSuite(denseSpec, TestBackward) }
// RunDenseGPUForward is in dense_forward.go (ASM timer columns).
