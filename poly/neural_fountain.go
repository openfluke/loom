package poly

import (
	"fmt"
	"math"
	"time"
)

// NetworkFactory builds a fresh trainable network for specialist index i.
// All specialists must share an identical weight layout so packed blobs match.
// Use any architecture (dense, CNN, residual, …) — Neural Fountain only sees
// TrainingBatch data + WeightStore masters.
type NetworkFactory func(specialistIdx int) (*VolumetricNetwork, error)

// NeuralFountainConfig controls shard specialize + LT weight recovery.
type NeuralFountainConfig struct {
	K            int     // specialist / shard count
	Epochs       int     // Train epochs per specialist
	LR           float32 // learning rate
	LossType     string  // "mse", "crossentropy", …
	Mode         TrainingMode
	LossRate     float64 // fountain erase probability
	MaxOverhead  float64 // spray budget as fraction over K
	Verbose      bool
	Seed         uint64 // fountain SeedRNG family (0 → derived)
	UseExactDType bool  // native-dtype train/forward (any layer.DType)
	UniformDType  DType // if non-zero, ApplyUniformDType on every specialist before train
}

// DefaultNeuralFountainConfig returns sane defaults for CPU multi-core train.
func DefaultNeuralFountainConfig() NeuralFountainConfig {
	return NeuralFountainConfig{
		K:           16,
		Epochs:      5,
		LR:          0.08,
		LossType:    "mse",
		Mode:        TrainingModeCPUMC,
		LossRate:    0.30,
		MaxOverhead: 5.0,
		Verbose:     true,
	}
}

// FountainMaster is the assembled Neural Fountain model: recovered specialists + ensemble.
type FountainMaster struct {
	Experts   []*VolumetricNetwork
	ShardOf   []int // batch / sample index → shard
	K         int
	Recovered int
	Received  int
	Sprayed   int
	// Wall-clock phases (microseconds); filled by NeuralFountain.
	SpecializeUs int64
	FountainUs   int64
}

// NeuralFountain trains K specialists on shards of arbitrary TrainingBatch data,
// ships their FP32 weight blobs through an LT fountain, peels them byte-exact,
// and returns a FountainMaster ensemble.
//
// factory may build dense, CNN, or any other VolumetricNetwork — only layout
// equality across specialists is required.
func NeuralFountain(factory NetworkFactory, batches []TrainingBatch[float32], cfg NeuralFountainConfig) (*FountainMaster, error) {
	if factory == nil {
		return nil, fmt.Errorf("poly: NeuralFountain nil factory")
	}
	if len(batches) == 0 {
		return nil, fmt.Errorf("poly: NeuralFountain empty batches")
	}
	if cfg.K < 2 {
		cfg.K = 2
	}
	if cfg.Epochs < 1 {
		cfg.Epochs = 1
	}
	if cfg.Mode == 0 {
		cfg.Mode = TrainingModeCPUMC
	}
	if cfg.LossType == "" {
		cfg.LossType = "mse"
	}
	if cfg.MaxOverhead <= 0 {
		cfg.MaxOverhead = 2
	}

	shards := partitionShards(len(batches), cfg.K)
	shardOf := make([]int, len(batches))
	for s, idxs := range shards {
		for _, i := range idxs {
			shardOf[i] = s
		}
	}

	if cfg.Verbose {
		fmt.Printf("── Neural Fountain · specialize ──\n")
		fmt.Printf("  batches=%d  K=%d specialists  epochs/shard=%d  loss=%s\n",
			len(batches), cfg.K, cfg.Epochs, cfg.LossType)
	}

	experts := make([]*VolumetricNetwork, cfg.K)
	blobs := make([][]byte, cfg.K)
	specStart := time.Now()
	for s := 0; s < cfg.K; s++ {
		net, err := factory(s)
		if err != nil {
			return nil, fmt.Errorf("poly: specialist %d factory: %w", s, err)
		}
		if net == nil {
			return nil, fmt.Errorf("poly: specialist %d factory returned nil", s)
		}
		prepareSpecialist(net, cfg)
		shardBatches := pickBatches(batches, shards[s])
		if len(shardBatches) == 0 {
			return nil, fmt.Errorf("poly: empty shard %d", s)
		}
		if err := trainSpecialistNet(net, shardBatches, cfg); err != nil {
			return nil, fmt.Errorf("poly: specialize %d: %w", s, err)
		}
		blob, err := PackNetworkWeights(net)
		if err != nil {
			return nil, err
		}
		experts[s] = net
		blobs[s] = blob
		if cfg.Verbose {
			fmt.Printf("  specialist %d/%d  shard_n=%d  weights=%d bytes  exact=%v\n",
				s+1, cfg.K, len(shards[s]), len(blob), net.UseExactDType)
		}
	}
	specializeUs := time.Since(specStart).Microseconds()

	if cfg.Verbose {
		fmt.Printf("\n── Neural Fountain · LT spray/peel weight blobs ──\n")
		fmt.Printf("  K=%d blocks · each %d bytes · loss=%.0f%%\n",
			cfg.K, len(blobs[0]), cfg.LossRate*100)
	}

	seed := cfg.Seed
	if seed == 0 {
		seed = SeedFrom("loom-poly-neural-fountain", uint64(cfg.K), uint64(len(blobs[0])))
	}
	fountainStart := time.Now()
	recovered, recv, sprayed, err := recoverWeightBlobs(blobs, seed, cfg.LossRate, cfg.MaxOverhead)
	if err != nil {
		return nil, err
	}
	if cfg.Verbose {
		fmt.Printf("  ✓ recovered %d/%d specialists  recv=%d (%.3f×K)  sprayed=%d\n",
			cfg.K, cfg.K, recv, float64(recv)/float64(cfg.K), sprayed)
	}

	out := make([]*VolumetricNetwork, cfg.K)
	for s := 0; s < cfg.K; s++ {
		net, err := factory(s)
		if err != nil {
			return nil, fmt.Errorf("poly: recover factory %d: %w", s, err)
		}
		prepareSpecialist(net, cfg)
		if err := UnpackNetworkWeights(net, recovered[s]); err != nil {
			return nil, fmt.Errorf("poly: unpack %d: %w", s, err)
		}
		out[s] = net
	}
	fountainUs := time.Since(fountainStart).Microseconds()

	return &FountainMaster{
		Experts:      out,
		ShardOf:      shardOf,
		K:            cfg.K,
		Recovered:    cfg.K,
		Received:     recv,
		Sprayed:      sprayed,
		SpecializeUs: specializeUs,
		FountainUs:   fountainUs,
	}, nil
}

// DenseSpecialistFactory builds He-initialized dense MLPs (last layer linear).
// dtypes are per-layer storage types (e.g. "float32", "float16", "int8"); empty → all float32.
// For CNN/residual/MHA/… supply your own NetworkFactory — pack/unpack walks any layer tree.
func DenseSpecialistFactory(name string, sizes []int, dtypes []string) NetworkFactory {
	if len(dtypes) == 0 && len(sizes) > 1 {
		dtypes = make([]string, len(sizes)-1)
		for i := range dtypes {
			dtypes[i] = "float32"
		}
	}
	return func(idx int) (*VolumetricNetwork, error) {
		topo := DenseTopologySeed(name, sizes)
		topo ^= uint64(idx+1) * 0x9e3779b97f4a7c15
		manifest, err := BuildDenseManifest(topo, sizes, dtypes)
		if err != nil {
			return nil, err
		}
		net, err := BuildDenseVolumetricFromManifest(manifest)
		if err != nil {
			return nil, err
		}
		if len(manifest.Layers) > 0 {
			last := net.GetLayer(0, 0, 0, len(manifest.Layers)-1)
			last.Activation = ActivationLinear
		}
		WireNetworkLayers(net)
		net.ReleaseFP32MasterWhenIdle = false
		_ = ConfigureNetworkForMode(net, TrainingModeCPUMC)
		net.EnsureTrainingWeights()
		MorphNetworkToLayerDTypes(net)
		return net, nil
	}
}

func prepareSpecialist(net *VolumetricNetwork, cfg NeuralFountainConfig) {
	WireNetworkLayers(net)
	net.ReleaseFP32MasterWhenIdle = false
	_ = ConfigureNetworkForMode(net, cfg.Mode)
	if cfg.UniformDType != 0 {
		ApplyUniformDType(net, cfg.UniformDType)
	}
	SetNetworkUseExactDType(net, cfg.UseExactDType)
	net.EnsureTrainingWeights()
	MorphNetworkToLayerDTypes(net)
}

// Forward averages specialist outputs (deployable Master). input shape must match experts.
func (m *FountainMaster) Forward(input *Tensor[float32]) (*Tensor[float32], error) {
	if m == nil || len(m.Experts) == 0 {
		return nil, fmt.Errorf("poly: FountainMaster empty")
	}
	if input == nil {
		return nil, fmt.Errorf("poly: FountainMaster nil input")
	}
	var acc []float32
	var shape []int
	n := 0
	for _, e := range m.Experts {
		if e == nil {
			continue
		}
		out, _, _ := ForwardPolymorphic(e, input)
		if out == nil || len(out.Data) == 0 {
			continue
		}
		if acc == nil {
			acc = make([]float32, len(out.Data))
			shape = append([]int(nil), out.Shape...)
		}
		if len(out.Data) != len(acc) {
			return nil, fmt.Errorf("poly: FountainMaster expert output size mismatch")
		}
		for j, v := range out.Data {
			acc[j] += v
		}
		n++
	}
	if n == 0 {
		return nil, fmt.Errorf("poly: FountainMaster no expert output")
	}
	inv := float32(1) / float32(n)
	for j := range acc {
		acc[j] *= inv
	}
	return NewTensorFromSlice(acc, shape...), nil
}

// ForwardArgmax returns argmax of the ensemble output (classification helper).
func (m *FountainMaster) ForwardArgmax(input *Tensor[float32]) (int, error) {
	out, err := m.Forward(input)
	if err != nil {
		return -1, err
	}
	best := 0
	for j := 1; j < len(out.Data); j++ {
		if out.Data[j] > out.Data[best] {
			best = j
		}
	}
	return best, nil
}

// OracleForward uses the specialist that owns sampleIdx (coverage / train check).
func (m *FountainMaster) OracleForward(sampleIdx int, input *Tensor[float32]) (*Tensor[float32], error) {
	if m == nil || len(m.Experts) == 0 {
		return nil, fmt.Errorf("poly: FountainMaster empty")
	}
	if sampleIdx < 0 || sampleIdx >= len(m.ShardOf) {
		return m.Forward(input)
	}
	si := m.ShardOf[sampleIdx]
	if si < 0 || si >= len(m.Experts) || m.Experts[si] == nil {
		return m.Forward(input)
	}
	out, _, _ := ForwardPolymorphic(m.Experts[si], input)
	if out == nil {
		return nil, fmt.Errorf("poly: oracle expert empty")
	}
	return out, nil
}

// OracleArgmax is OracleForward + argmax.
func (m *FountainMaster) OracleArgmax(sampleIdx int, input *Tensor[float32]) (int, error) {
	out, err := m.OracleForward(sampleIdx, input)
	if err != nil {
		return -1, err
	}
	best := 0
	for j := 1; j < len(out.Data); j++ {
		if out.Data[j] > out.Data[best] {
			best = j
		}
	}
	return best, nil
}

func partitionShards(n, k int) [][]int {
	out := make([][]int, k)
	for i := 0; i < n; i++ {
		s := i % k
		out[s] = append(out[s], i)
	}
	return out
}

func pickBatches(all []TrainingBatch[float32], idxs []int) []TrainingBatch[float32] {
	out := make([]TrainingBatch[float32], len(idxs))
	for i, j := range idxs {
		out[i] = all[j]
	}
	return out
}

func trainSpecialistNet(net *VolumetricNetwork, batches []TrainingBatch[float32], cfg NeuralFountainConfig) error {
	prepareSpecialist(net, cfg)
	tcfg := &TrainingConfig{
		Epochs:       cfg.Epochs,
		LearningRate: cfg.LR,
		LossType:     cfg.LossType,
		Mode:         cfg.Mode,
		Verbose:      false,
	}
	_, err := Train[float32](net, batches, tcfg)
	if err != nil {
		return err
	}
	// Refresh FP32 Masters from natives before fountain pack (exact-dtype path).
	net.EnsureTrainingWeights()
	net.SyncToCPU()
	return nil
}

func recoverWeightBlobs(blobs [][]byte, seed uint64, loss, maxOverhead float64) (recovered [][]byte, received, sprayed int, err error) {
	enc, err := NewLTEncoder(blobs, seed)
	if err != nil {
		return nil, 0, 0, err
	}
	dec := NewLTDecoder(len(blobs), len(blobs[0]))
	lossRng := NewSeedRNG(SeedFrom("loom-poly-neural-fountain-loss", seed))
	if maxOverhead < 1 {
		maxOverhead = 1
	}
	maxSpray := int(math.Ceil(float64(len(blobs)) * (1 + maxOverhead)))
	// Hard floor: at least several ×K sprays so lossy channels don't starve peel.
	if floor := len(blobs) * 8; maxSpray < floor {
		maxSpray = floor
	}
	last := 0
	stalls := 0
	for !dec.Done() && sprayed < maxSpray {
		drop := enc.Spray()
		sprayed++
		if lossRng.Float64() < loss {
			continue
		}
		received++
		dec.Catch(drop)
		known := dec.KnownCount()
		if known == last {
			stalls++
			if stalls%50 == 0 {
				dec.TryResidualGE(maxGEUnknowns)
			}
		} else {
			stalls = 0
			last = known
		}
	}
	if !dec.Done() {
		dec.TryResidualGE(maxGEUnknowns)
	}
	// Last resort: keep spraying a bit more with GE attempts (rateless).
	extra := 0
	for !dec.Done() && extra < len(blobs)*20 {
		drop := enc.Spray()
		sprayed++
		extra++
		if lossRng.Float64() < loss {
			continue
		}
		received++
		dec.Catch(drop)
		dec.TryResidualGE(maxGEUnknowns)
	}
	if !dec.Done() {
		return nil, received, sprayed, fmt.Errorf("poly: fountain stalled at %d/%d (recv=%d sprayed=%d)",
			dec.KnownCount(), len(blobs), received, sprayed)
	}
	if !BlocksEqual(blobs, dec.Recovered) {
		return nil, received, sprayed, fmt.Errorf("poly: recovered weights != packed specialists")
	}
	return dec.Recovered, received, sprayed, nil
}
