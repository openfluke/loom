package testing

import (
	"github.com/openfluke/loom/poly"
)

var cnn1Spec = TestSpec{
	Name: "CNN1",
	Layer: poly.PersistenceLayerSpec{
		Type:          "CNN1",
		Filters:       16,
		InputChannels: 3,
		InputHeight:   32,
		OutputHeight:  32,
		KernelSize:    3,
		Stride:        1,
		Padding:       1,
		Activation:    "ReLU",
	},
	InputShape: []int{8, 3, 32},
}

func init() {
	RegisterTask(func() bool {
		return RunGenericLayerSuite(cnn1Spec, TestAll)
	})
}

func RunCNN1L1Caching()  { RunGenericLayerSuite(cnn1Spec, TestForward) }
func RunCNN1Training()   { RunGenericLayerSuite(cnn1Spec, TestTraining|TestSaveLoad) }
func RunCNN1GPUForward() { RunGenericLayerSuite(cnn1Spec, TestForward) }
func RunCNN1GPUBackward() { RunGenericLayerSuite(cnn1Spec, TestBackward) }
