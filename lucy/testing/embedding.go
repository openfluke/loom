package testing

import (
	"github.com/openfluke/loom/poly"
)

var embeddingSpec = TestSpec{
	Name: "Embedding",
	Layer: poly.PersistenceLayerSpec{
		Type:         "Embedding",
		VocabSize:    100,
		EmbeddingDim: 64,
		InputHeight:  1,
		SeqLength:    1,
	},
	InputShape: []int{8, 1}, // Batch of 8 indices
}

func init() {
	RegisterTask(func() bool {
		return RunGenericLayerSuite(embeddingSpec, TestAll)
	})
}

func RunEmbeddingL1Caching()   { RunGenericLayerSuite(embeddingSpec, TestForward) }
func RunEmbeddingTraining()    { RunGenericLayerSuite(embeddingSpec, TestTraining|TestSaveLoad) }
func RunEmbeddingGPUForward()  { RunGenericLayerSuite(embeddingSpec, TestForward) }
func RunEmbeddingGPUBackward() { RunGenericLayerSuite(embeddingSpec, TestBackward) }
