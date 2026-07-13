package poly

import (
	"encoding/binary"
	"hash/fnv"
	"math"
)

const seedGoldenRatio = 0x9e3779b97f4a7c15

// SeedFrom mixes arbitrary inputs into a single uint64 seed.
func SeedFrom(parts ...any) uint64 {
	var h uint64 = seedGoldenRatio
	for i, p := range parts {
		h = seedMix(h, uint64(i))
		h = seedMixValue(h, p)
	}
	return seedSplitmix64(h)
}

// DeriveLayerSeed derives a per-layer weight seed from init seed, index, and path.
func DeriveLayerSeed(initSeed uint64, layerIndex int, path string) uint64 {
	return SeedFrom(initSeed, layerIndex, path)
}

// SeedRNG is a deterministic xorshift64* PRNG (no math/rand).
type SeedRNG struct {
	state uint64
}

// NewSeedRNG returns an RNG initialized from seed.
func NewSeedRNG(seed uint64) *SeedRNG {
	if seed == 0 {
		seed = 0xdeadbeefcafebabe
	}
	return &SeedRNG{state: seedSplitmix64(seed)}
}

func (r *SeedRNG) Uint64() uint64 {
	x := r.state
	x ^= x >> 12
	x ^= x << 25
	x ^= x >> 27
	r.state = x
	return x * 0x2545F4914F6CDD1D
}

func (r *SeedRNG) Float64() float64 {
	return float64(r.Uint64()>>11) / (1 << 53)
}

func (r *SeedRNG) NormFloat64() float64 {
	for {
		u1 := r.Float64()*2 - 1
		u2 := r.Float64()*2 - 1
		s := u1*u1 + u2*u2
		if s > 0 && s < 1 {
			mul := math.Sqrt(-2 * math.Log(s) / s)
			return u1 * mul
		}
	}
}

func seedMix(h, v uint64) uint64 {
	h ^= v + seedGoldenRatio + (h << 6) + (h >> 2)
	return h
}

func seedMixValue(h uint64, v any) uint64 {
	switch x := v.(type) {
	case string:
		return seedMixBytes(h, []byte(x))
	case []byte:
		return seedMixBytes(h, x)
	case bool:
		if x {
			return seedMix(h, 1)
		}
		return seedMix(h, 0)
	case int:
		return seedMix(h, uint64(x))
	case int8:
		return seedMix(h, uint64(uint8(x)))
	case int16:
		return seedMix(h, uint64(uint16(x)))
	case int32:
		return seedMix(h, uint64(uint32(x)))
	case int64:
		return seedMix(h, uint64(x))
	case uint:
		return seedMix(h, uint64(x))
	case uint8:
		return seedMix(h, uint64(x))
	case uint16:
		return seedMix(h, uint64(x))
	case uint32:
		return seedMix(h, uint64(x))
	case uint64:
		return seedMix(h, x)
	case float32:
		return seedMix(h, uint64(math.Float32bits(x)))
	case float64:
		return seedMix(h, math.Float64bits(x))
	default:
		panic("poly: SeedFrom unsupported type")
	}
}

func seedMixBytes(h uint64, b []byte) uint64 {
	hasher := fnv.New64a()
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], h)
	_, _ = hasher.Write(buf[:])
	_, _ = hasher.Write(b)
	return hasher.Sum64()
}

func seedSplitmix64(x uint64) uint64 {
	x += 0x9e3779b97f4a7c15
	z := x
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

// InitFloat32HeSeeded fills weights with He-init from seed.
func InitFloat32HeSeeded(weights []float32, inputSize int, seed uint64) {
	if len(weights) == 0 {
		return
	}
	if inputSize <= 0 {
		inputSize = 1
	}
	rng := NewSeedRNG(seed)
	stddev := float32(math.Sqrt(2.0 / float64(inputSize)))
	for i := range weights {
		weights[i] = float32(rng.NormFloat64()) * stddev
	}
}

// InitWeightStoreHeSeeded He-inits a WeightStore master slice from seed.
func InitWeightStoreHeSeeded(ws *WeightStore, inputSize int, seed uint64) {
	if ws == nil || len(ws.Master) == 0 {
		return
	}
	InitFloat32HeSeeded(ws.Master, inputSize, seed)
	ws.Versions = make(map[DType]any)
	ws.CPUPacked = make(map[DType]any)
	ws.GPUWeights = make(map[DType]any)
}
