package poly

import (
	"fmt"
	"math"
)

const maxGEUnknowns = 384 // residual GE only when leftover unknowns are tiny

// Drop is one fountain spray: XOR of selected source blocks + neighbor list.
type Drop struct {
	ID        uint64
	Neighbors []int
	Payload   []byte
}

// LTEncoder is a Luby Transform (rateless) fountain over byte blocks.
type LTEncoder struct {
	Sources [][]byte
	K       int
	rng     *SeedRNG
	nextID  uint64
	cdf     []float64
}

// NewLTEncoder builds an LT fountain from equal-sized source blocks.
func NewLTEncoder(sources [][]byte, seed uint64) (*LTEncoder, error) {
	if len(sources) == 0 {
		return nil, fmt.Errorf("fountain: no source blocks")
	}
	sz := len(sources[0])
	for i, b := range sources {
		if len(b) != sz {
			return nil, fmt.Errorf("fountain: block %d size %d != %d", i, len(b), sz)
		}
	}
	k := len(sources)
	return &LTEncoder{
		Sources: sources,
		K:       k,
		rng:     NewSeedRNG(SeedFrom("loom-fountain-lt", seed, k, sz)),
		nextID:  1,
		cdf:     robustSolitonCDF(k),
	}, nil
}

// Spray generates the next encoded drop (endless).
func (e *LTEncoder) Spray() Drop {
	d := sampleDegree(e.cdf, e.rng)
	if d > e.K {
		d = e.K
	}
	if d < 1 {
		d = 1
	}
	neighbors := sampleUnique(e.K, d, e.rng)
	payload := make([]byte, len(e.Sources[0]))
	for _, n := range neighbors {
		xorBytes(payload, e.Sources[n])
	}
	id := e.nextID
	e.nextID++
	return Drop{ID: id, Neighbors: neighbors, Payload: payload}
}

type equation struct {
	alive     bool
	neighbors []int
	payload   []byte
}

// LTDecoder peels recovered source blocks via a ripple; residual GE only for tiny leftovers.
type LTDecoder struct {
	K         int
	BlockSize int
	Recovered [][]byte
	known     []bool
	knownN    int
	eqs       []equation
	srcToEq   [][]int // source index → equation indices mentioning it
	ripple    []int
}

// NewLTDecoder prepares an empty bucket for K source blocks.
func NewLTDecoder(k, blockSize int) *LTDecoder {
	return &LTDecoder{
		K:         k,
		BlockSize: blockSize,
		Recovered: make([][]byte, k),
		known:     make([]bool, k),
		srcToEq:   make([][]int, k),
	}
}

// Catch adds one drop and peels with a degree-1 ripple (no heavy residual GE).
func (d *LTDecoder) Catch(drop Drop) []int {
	if len(drop.Payload) != d.BlockSize {
		return nil
	}
	neighbors := append([]int(nil), drop.Neighbors...)
	payload := append([]byte(nil), drop.Payload...)
	neighbors, payload = d.reduce(neighbors, payload)
	if len(neighbors) == 0 {
		return nil
	}
	ei := len(d.eqs)
	d.eqs = append(d.eqs, equation{alive: true, neighbors: neighbors, payload: payload})
	for _, n := range neighbors {
		d.srcToEq[n] = append(d.srcToEq[n], ei)
	}
	if len(neighbors) == 1 {
		d.ripple = append(d.ripple, neighbors[0])
	}
	return d.drainRipple()
}

// TryResidualGE runs GF(2) GE only when unknowns ≤ maxUnknowns (default maxGEUnknowns).
func (d *LTDecoder) TryResidualGE(maxUnknowns int) []int {
	if d.Done() {
		return nil
	}
	if maxUnknowns <= 0 {
		maxUnknowns = maxGEUnknowns
	}
	if d.K-d.knownN > maxUnknowns {
		return nil
	}
	newly := d.gaussianResidual()
	if len(newly) > 0 {
		newly = append(newly, d.drainRipple()...)
	}
	return newly
}

func (d *LTDecoder) reduce(neighbors []int, payload []byte) ([]int, []byte) {
	out := neighbors[:0]
	for _, n := range neighbors {
		if d.known[n] {
			xorBytes(payload, d.Recovered[n])
			continue
		}
		out = append(out, n)
	}
	return out, payload
}

func (d *LTDecoder) drainRipple() []int {
	var newly []int
	for len(d.ripple) > 0 {
		src := d.ripple[len(d.ripple)-1]
		d.ripple = d.ripple[:len(d.ripple)-1]

		if !d.known[src] {
			payload, ok := d.findDeg1Payload(src)
			if !ok {
				continue
			}
			d.Recovered[src] = payload
			d.known[src] = true
			d.knownN++
			newly = append(newly, src)
		}

		// Always propagate from known src (covers GE-seeded recoveries too).
		for _, ei := range d.srcToEq[src] {
			eq := &d.eqs[ei]
			if !eq.alive {
				continue
			}
			nb, pl := d.reduce(eq.neighbors, append([]byte(nil), eq.payload...))
			eq.neighbors = nb
			eq.payload = pl
			switch len(nb) {
			case 0:
				eq.alive = false
			case 1:
				d.ripple = append(d.ripple, nb[0])
			}
		}
	}
	return newly
}

func (d *LTDecoder) findDeg1Payload(src int) ([]byte, bool) {
	for _, ei := range d.srcToEq[src] {
		eq := &d.eqs[ei]
		if !eq.alive {
			continue
		}
		nb, pl := d.reduce(eq.neighbors, append([]byte(nil), eq.payload...))
		eq.neighbors = nb
		eq.payload = pl
		if len(nb) == 0 {
			eq.alive = false
			continue
		}
		if len(nb) == 1 && nb[0] == src {
			return append([]byte(nil), pl...), true
		}
	}
	return nil, false
}

func (d *LTDecoder) gaussianResidual() []int {
	var unknowns []int
	idxOf := map[int]int{}
	for i := 0; i < d.K; i++ {
		if !d.known[i] {
			idxOf[i] = len(unknowns)
			unknowns = append(unknowns, i)
		}
	}
	u := len(unknowns)
	if u == 0 {
		return nil
	}

	type row struct {
		bits    []bool
		payload []byte
	}
	rows := make([]row, 0, len(d.eqs))
	for i := range d.eqs {
		eq := &d.eqs[i]
		if !eq.alive {
			continue
		}
		nb, pl := d.reduce(eq.neighbors, append([]byte(nil), eq.payload...))
		eq.neighbors = nb
		eq.payload = pl
		if len(nb) == 0 {
			eq.alive = false
			continue
		}
		bits := make([]bool, u)
		ok := true
		for _, n := range nb {
			j, exists := idxOf[n]
			if !exists {
				ok = false
				break
			}
			bits[j] = !bits[j]
		}
		if !ok || !anyTrue(bits) {
			continue
		}
		rows = append(rows, row{bits: bits, payload: pl})
	}
	if len(rows) < u {
		return nil
	}

	pivotOf := make([]int, u)
	for i := range pivotOf {
		pivotOf[i] = -1
	}
	r := 0
	for c := 0; c < u && r < len(rows); c++ {
		piv := -1
		for i := r; i < len(rows); i++ {
			if rows[i].bits[c] {
				piv = i
				break
			}
		}
		if piv < 0 {
			continue
		}
		rows[r], rows[piv] = rows[piv], rows[r]
		for i := 0; i < len(rows); i++ {
			if i != r && rows[i].bits[c] {
				xorBool(rows[i].bits, rows[r].bits)
				xorBytes(rows[i].payload, rows[r].payload)
			}
		}
		pivotOf[c] = r
		r++
	}

	var newly []int
	for c := 0; c < u; c++ {
		pr := pivotOf[c]
		if pr < 0 {
			continue
		}
		alone := true
		for j := 0; j < u; j++ {
			if j != c && rows[pr].bits[j] {
				alone = false
				break
			}
		}
		if !alone || !rows[pr].bits[c] {
			continue
		}
		src := unknowns[c]
		if d.known[src] {
			continue
		}
		d.Recovered[src] = append([]byte(nil), rows[pr].payload...)
		d.known[src] = true
		d.knownN++
		newly = append(newly, src)
		d.ripple = append(d.ripple, src)
	}
	return newly
}

func (d *LTDecoder) Done() bool { return d.knownN >= d.K }

func (d *LTDecoder) Progress() float64 {
	if d.K == 0 {
		return 0
	}
	return float64(d.knownN) / float64(d.K)
}

func (d *LTDecoder) KnownCount() int { return d.knownN }

func (d *LTDecoder) IsKnown(i int) bool {
	return i >= 0 && i < d.K && d.known[i]
}

func robustSolitonCDF(k int) []float64 {
	if k <= 1 {
		return []float64{1}
	}
	c := 0.1
	delta := 0.05
	R := c * math.Sqrt(float64(k)) * math.Log(float64(k)/delta)
	if R < 1 {
		R = 1
	}
	mu := make([]float64, k+1)
	var sum float64
	for d := 1; d <= k; d++ {
		var rho float64
		if d == 1 {
			rho = 1.0 / float64(k)
		} else {
			rho = 1.0 / (float64(d) * float64(d-1))
		}
		var tau float64
		bound := int(math.Floor(float64(k) / R))
		if d >= 1 && d <= bound-1 {
			tau = R / (float64(d) * float64(k))
		} else if d == int(math.Round(float64(k)/R)) {
			tau = R * math.Log(R/delta) / float64(k)
		}
		mu[d] = rho + tau
		sum += mu[d]
	}
	cdf := make([]float64, k+1)
	var cum float64
	for d := 1; d <= k; d++ {
		cum += mu[d] / sum
		cdf[d] = cum
	}
	cdf[k] = 1
	return cdf
}

func sampleDegree(cdf []float64, rng *SeedRNG) int {
	u := rng.Float64()
	for d := 1; d < len(cdf); d++ {
		if u <= cdf[d] {
			return d
		}
	}
	return len(cdf) - 1
}

func sampleUnique(n, d int, rng *SeedRNG) []int {
	if d > n {
		d = n
	}
	if d <= 0 {
		return nil
	}
	// Rejection sampling is fine while d ≪ n (typical LT degrees).
	if d*d < 4*n || d < 64 {
		out := make([]int, 0, d)
		seen := make(map[int]struct{}, d)
		for len(out) < d {
			x := int(rng.Uint64() % uint64(n))
			if _, ok := seen[x]; ok {
				continue
			}
			seen[x] = struct{}{}
			out = append(out, x)
		}
		return out
	}
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	for i := 0; i < d; i++ {
		j := i + int(rng.Uint64()%uint64(n-i))
		idx[i], idx[j] = idx[j], idx[i]
	}
	return append([]int(nil), idx[:d]...)
}

func xorBytes(dst, src []byte) {
	n := len(dst)
	if len(src) < n {
		n = len(src)
	}
	for i := 0; i < n; i++ {
		dst[i] ^= src[i]
	}
}

func xorBool(dst, src []bool) {
	for i := range dst {
		dst[i] = dst[i] != src[i]
	}
}

func anyTrue(b []bool) bool {
	for _, v := range b {
		if v {
			return true
		}
	}
	return false
}

// BlocksEqual reports byte-exact match of two block slices.
func BlocksEqual(a, b [][]byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] == nil || b[i] == nil || len(a[i]) != len(b[i]) {
			return false
		}
		for j := range a[i] {
			if a[i][j] != b[i][j] {
				return false
			}
		}
	}
	return true
}
