// Package neural is a thin MNIST-oriented façade over poly.NeuralFountain.
//
// For arbitrary models / layers / datasets, call poly.NeuralFountain with your
// own NetworkFactory and []TrainingBatch[float32] directly.
package neural

import (
	"fmt"

	"github.com/openfluke/loom/poly"
)

// AssembleConfig is kept for MNIST demos; maps onto poly.NeuralFountainConfig.
type AssembleConfig struct {
	K            int
	Sizes        []int
	SpecialistEP int
	LR           float32
	LossRate     float64
	MaxOverhead  float64
	Verbose      bool
	Name         string
}

func DefaultAssembleConfig() AssembleConfig {
	return AssembleConfig{
		K:            16,
		Sizes:        []int{784, 128, 64, 10},
		SpecialistEP: 5,
		LR:           0.08,
		LossRate:     0.30,
		MaxOverhead:  5.0,
		Verbose:      true,
		Name:         "neural-fountain-master",
	}
}

// Sample is a labeled flat vector (MNIST-style). For general data use poly.TrainingBatch.
type Sample struct {
	X []float32
	Y int
}

// Master wraps poly.FountainMaster with MNIST Sample helpers.
type Master struct {
	*poly.FountainMaster
	Sizes []int
}

// AssembleMaster runs poly.NeuralFountain with a dense specialist factory.
func AssembleMaster(train []Sample, cfg AssembleConfig) (*Master, error) {
	if len(cfg.Sizes) < 2 {
		return nil, fmt.Errorf("neural: sizes required")
	}
	nClass := cfg.Sizes[len(cfg.Sizes)-1]
	batches := make([]poly.TrainingBatch[float32], len(train))
	for i, s := range train {
		tgt := make([]float32, nClass)
		if s.Y >= 0 && s.Y < nClass {
			tgt[s.Y] = 1
		}
		batches[i] = poly.TrainingBatch[float32]{
			Input:  poly.NewTensorFromSlice(s.X, 1, len(s.X)),
			Target: poly.NewTensorFromSlice(tgt, 1, nClass),
		}
	}
	pcfg := poly.NeuralFountainConfig{
		K:           cfg.K,
		Epochs:      cfg.SpecialistEP,
		LR:          cfg.LR,
		LossType:    "mse",
		Mode:        poly.TrainingModeCPUMC,
		LossRate:    cfg.LossRate,
		MaxOverhead: cfg.MaxOverhead,
		Verbose:     cfg.Verbose,
	}
	factory := poly.DenseSpecialistFactory(cfg.Name, cfg.Sizes, nil)
	fm, err := poly.NeuralFountain(factory, batches, pcfg)
	if err != nil {
		return nil, err
	}
	return &Master{FountainMaster: fm, Sizes: append([]int(nil), cfg.Sizes...)}, nil
}

// PredictArgmax averages expert logits then argmax.
func (m *Master) PredictArgmax(x []float32) (int, error) {
	if m == nil || m.FountainMaster == nil {
		return -1, fmt.Errorf("neural: empty master")
	}
	return m.FountainMaster.ForwardArgmax(poly.NewTensorFromSlice(x, 1, len(x)))
}

// OraclePredict uses the shard expert for train index i.
func (m *Master) OraclePredict(trainIdx int, x []float32) (int, error) {
	if m == nil || m.FountainMaster == nil {
		return -1, fmt.Errorf("neural: empty master")
	}
	return m.FountainMaster.OracleArgmax(trainIdx, poly.NewTensorFromSlice(x, 1, len(x)))
}

func EvalOracleTrainAccuracy(m *Master, train []Sample) float64 {
	if m == nil || len(train) == 0 {
		return 0
	}
	ok := 0
	for i, s := range train {
		pred, err := m.OraclePredict(i, s.X)
		if err == nil && pred == s.Y {
			ok++
		}
	}
	return float64(ok) / float64(len(train))
}

func EvalEnsembleAccuracy(m *Master, set []Sample) float64 {
	if m == nil || len(set) == 0 {
		return 0
	}
	ok := 0
	for _, s := range set {
		pred, err := m.PredictArgmax(s.X)
		if err == nil && pred == s.Y {
			ok++
		}
	}
	return float64(ok) / float64(len(set))
}

// EvalMeanNetAccuracy is retired (weight-mean master was removed); always 0.
func EvalMeanNetAccuracy(m *Master, set []Sample) float64 {
	_ = m
	_ = set
	return 0
}

// Re-exports for demos that pack/unpack nets directly.
var (
	PackNetworkWeights   = poly.PackNetworkWeights
	UnpackNetworkWeights = poly.UnpackNetworkWeights
)
