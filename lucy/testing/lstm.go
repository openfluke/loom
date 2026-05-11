package testing

import (
	"github.com/openfluke/loom/poly"
)

var lstmSpec = TestSpec{
	Name: "LSTM",
	Layer: poly.PersistenceLayerSpec{
		Type:         "LSTM",
		InputHeight:  16,
		OutputHeight: 32,
		SeqLength:    8,
	},
	InputShape: []int{1, 8, 16},
}

func init() {
	RegisterTask(func() bool {
		return RunGenericLayerSuite(lstmSpec, TestAll)
	})
}

func RunLSTML1Caching()   { RunGenericLayerSuite(lstmSpec, TestForward) }
func RunLSTMTraining()    { RunGenericLayerSuite(lstmSpec, TestTraining|TestSaveLoad) }
func RunLSTMGPUForward()  { RunGenericLayerSuite(lstmSpec, TestForward) }
func RunLSTMGPUBackward() { RunGenericLayerSuite(lstmSpec, TestBackward) }
