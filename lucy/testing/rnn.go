package testing

import (
	"github.com/openfluke/loom/poly"
)

var rnnSpec = TestSpec{
	Name: "RNN",
	Layer: poly.PersistenceLayerSpec{
		Type:         "RNN",
		InputHeight:  16,
		OutputHeight: 32,
		SeqLength:    8,
		Activation:   "Tanh",
	},
	InputShape: []int{1, 8, 16},
}

func init() {
	RegisterTask(func() bool {
		return RunGenericLayerSuite(rnnSpec, TestAll)
	})
}

func RunRNNL1Caching()   { RunGenericLayerSuite(rnnSpec, TestForward) }
func RunRNNTraining()    { RunGenericLayerSuite(rnnSpec, TestTraining|TestSaveLoad) }
func RunRNNGPUForward()  { RunGenericLayerSuite(rnnSpec, TestForward) }
func RunRNNGPUBackward() { RunGenericLayerSuite(rnnSpec, TestBackward) }
