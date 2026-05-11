package testing

import (
	"github.com/openfluke/loom/poly"
)

var mhaSpec = TestSpec{
	Name: "MHA",
	Layer: poly.PersistenceLayerSpec{
		Type:      "MHA",
		DModel:    128,
		NumHeads:  8,
		SeqLength: 16,
	},
	InputShape: []int{1, 16, 128},
}

func init() {
	RegisterTask(func() bool {
		return RunGenericLayerSuite(mhaSpec, TestAll)
	})
}

func RunMHAL1Caching() { RunGenericLayerSuite(mhaSpec, TestForward) }
func RunMHAForward()   { RunGenericLayerSuite(mhaSpec, TestForward) }
func RunMHABackward()  { RunGenericLayerSuite(mhaSpec, TestBackward) }
func RunMHATraining()  { RunGenericLayerSuite(mhaSpec, TestTraining|TestSaveLoad) }
