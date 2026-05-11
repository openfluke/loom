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

func RunDenseL1Caching()  { RunGenericLayerSuite(denseSpec, TestForward) }
func RunDenseTraining()   { RunGenericLayerSuite(denseSpec, TestTraining|TestSaveLoad) }
func RunDenseGPUForward() { RunGenericLayerSuite(denseSpec, TestForward) }
func RunDenseGPUBackward() { RunGenericLayerSuite(denseSpec, TestBackward) }
