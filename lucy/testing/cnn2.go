package testing

import (
	"github.com/openfluke/loom/poly"
)

var cnn2Spec = TestSpec{
	Name: "CNN2",
	Layer: poly.PersistenceLayerSpec{
		Type:          "CNN2",
		Filters:       8,
		InputChannels: 3,
		InputHeight:   32,
		InputWidth:    32,
		OutputHeight:  32,
		OutputWidth:   32,
		KernelSize:    3,
		Stride:        1,
		Padding:       1,
		Activation:    "ReLU",
	},
	InputShape: []int{8, 3, 32, 32},
}

func init() {
	RegisterTask(func() bool {
		return RunGenericLayerSuite(cnn2Spec, TestAll)
	})
}

func RunCNN2L1Caching()  { RunGenericLayerSuite(cnn2Spec, TestForward) }
func RunCNN2Training()   { RunGenericLayerSuite(cnn2Spec, TestTraining|TestSaveLoad) }
func RunCNN2GPUForward() { RunGenericLayerSuite(cnn2Spec, TestForward) }
func RunCNN2GPUBackward() { RunGenericLayerSuite(cnn2Spec, TestBackward) }
