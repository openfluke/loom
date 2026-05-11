package testing

import (
	"github.com/openfluke/loom/poly"
)

var swigluSpec = TestSpec{
	Name: "SwiGLU",
	Layer: poly.PersistenceLayerSpec{
		Type:         "SwiGLU",
		InputHeight:  128,
		OutputHeight: 256,
	},
	InputShape: []int{16, 128},
}

func init() {
	RegisterTask(func() bool {
		return RunGenericLayerSuite(swigluSpec, TestAll)
	})
}

func RunSwiGLUL1Caching()   { RunGenericLayerSuite(swigluSpec, TestForward) }
func RunSwiGLUTraining()    { RunGenericLayerSuite(swigluSpec, TestTraining|TestSaveLoad) }
func RunSwiGLUGPUForward()  { RunGenericLayerSuite(swigluSpec, TestForward) }
func RunSwiGLUGPUBackward() { RunGenericLayerSuite(swigluSpec, TestBackward) }
