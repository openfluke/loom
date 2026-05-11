package testing

import (
	"github.com/openfluke/loom/poly"
)

var cnn3Spec = TestSpec{
	Name: "CNN3",
	Layer: poly.PersistenceLayerSpec{
		Type:          "CNN3",
		Filters:       4,
		InputChannels: 3,
		InputDepth:    16,
		InputHeight:   16,
		InputWidth:    16,
		OutputDepth:   16,
		OutputHeight:  16,
		OutputWidth:   16,
		KernelSize:    3,
		Stride:        1,
		Padding:       1,
		Activation:    "ReLU",
	},
	InputShape: []int{4, 3, 16, 16, 16},
}

func init() {
	RegisterTask(func() bool {
		return RunGenericLayerSuite(cnn3Spec, TestAll)
	})
}

func RunCNN3L1Caching()  { RunGenericLayerSuite(cnn3Spec, TestForward) }
func RunCNN3Training()   { RunGenericLayerSuite(cnn3Spec, TestTraining|TestSaveLoad) }
func RunCNN3GPUForward() { RunGenericLayerSuite(cnn3Spec, TestForward) }
func RunCNN3GPUBackward() { RunGenericLayerSuite(cnn3Spec, TestBackward) }
