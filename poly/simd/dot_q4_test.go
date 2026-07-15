package simd

import (
	"math"
	"testing"
)

func TestDotQ4_0Rows4MatchesRow(t *testing.T) {
	const rows, cols = 8, 256
	w := make([]float32, rows*cols)
	in := make([]float32, cols)
	for i := range w {
		w[i] = float32((i%17)-8) * 0.02
	}
	for i := range in {
		in[i] = float32((i%11)-5) * 0.03
	}
	scales, packed := packQ4ForTest(w)
	out := make([]float32, 4)
	DotQ4_0Rows4(in, scales, packed, 0, cols, out)
	for r := 0; r < 4; r++ {
		want := float32(DotQ4_0Row(in, scales, packed, r*cols, cols, 0))
		if math.Abs(float64(want-out[r])) > 1e-4 {
			t.Fatalf("row %d: want %g got %g", r, want, out[r])
		}
	}
}

func TestDotQ4_0RowMatchesGo(t *testing.T) {
	const rows, cols = 4, 256
	w := make([]float32, rows*cols)
	in := make([]float32, cols)
	for i := range w {
		w[i] = float32((i%17)-8) * 0.02
	}
	for i := range in {
		in[i] = float32((i%11)-5) * 0.03
	}
	// Pack via poly — duplicated light pack here to avoid import cycle.
	scales, packed := packQ4ForTest(w)

	for r := 0; r < rows; r++ {
		base := r * cols
		want := dotQ4_0RowGo(in, scales, packed, base, cols, 0.5)
		got := DotQ4_0Row(in, scales, packed, base, cols, 0.5)
		if math.Abs(want-got) > 1e-4 {
			t.Fatalf("row %d: go %g simd %g", r, want, got)
		}
	}
}

func packQ4ForTest(data []float32) (scales []float32, packed []uint32) {
	n := len(data)
	numBlocks := (n + 31) / 32
	scales = make([]float32, numBlocks)
	packedSize := numBlocks * 4
	aligned := (packedSize + 63) &^ 63
	if aligned < 512 {
		aligned = 512
	}
	packed = make([]uint32, aligned)
	for b := 0; b < numBlocks; b++ {
		start := b * 32
		end := start + 32
		if end > n {
			end = n
		}
		maxAbs := float32(0)
		for j := start; j < end; j++ {
			a := data[j]
			if a < 0 {
				a = -a
			}
			if a > maxAbs {
				maxAbs = a
			}
		}
		scale := maxAbs / 7
		if scale == 0 {
			scale = 1
		}
		scales[b] = scale
		var bytes [16]byte
		for j := 0; j < 16; j++ {
			idx1 := start + j*2
			idx2 := start + j*2 + 1
			v1, v2 := float32(0), float32(0)
			if idx1 < n {
				v1 = data[idx1]
			}
			if idx2 < n {
				v2 = data[idx2]
			}
			q1 := int8(math.Round(float64(v1 / scale)))
			q2 := int8(math.Round(float64(v2 / scale)))
			if q1 > 7 {
				q1 = 7
			}
			if q1 < -8 {
				q1 = -8
			}
			if q2 > 7 {
				q2 = 7
			}
			if q2 < -8 {
				q2 = -8
			}
			bytes[j] = byte(q1&0xF) | (byte(q2&0xF) << 4)
		}
		for j := 0; j < 4; j++ {
			packed[b*4+j] = uint32(bytes[j*4]) |
				(uint32(bytes[j*4+1]) << 8) |
				(uint32(bytes[j*4+2]) << 16) |
				(uint32(bytes[j*4+3]) << 24)
		}
	}
	return scales, packed
}

func BenchmarkDotQ4_0Row(b *testing.B) {
	const cols = 1024
	w := make([]float32, cols)
	in := make([]float32, cols)
	for i := range w {
		w[i] = float32((i%17)-8) * 0.01
	}
	for i := range in {
		in[i] = float32((i%9)-4) * 0.05
	}
	scales, packed := packQ4ForTest(w)
	b.ResetTimer()
	var sink float64
	for i := 0; i < b.N; i++ {
		sink = DotQ4_0Row(in, scales, packed, 0, cols, 0)
	}
	_ = sink
}

func BenchmarkDotQ4_0RowGo(b *testing.B) {
	const cols = 1024
	w := make([]float32, cols)
	in := make([]float32, cols)
	for i := range w {
		w[i] = float32((i%17)-8) * 0.01
	}
	for i := range in {
		in[i] = float32((i%9)-4) * 0.05
	}
	scales, packed := packQ4ForTest(w)
	b.ResetTimer()
	var sink float64
	for i := 0; i < b.N; i++ {
		sink = dotQ4_0RowGo(in, scales, packed, 0, cols, 0)
	}
	_ = sink
}

func BenchmarkDotQ4_0Rows4(b *testing.B) {
	const cols = 1024
	const rows = 64
	w := make([]float32, rows*cols)
	in := make([]float32, cols)
	for i := range w {
		w[i] = float32((i%17)-8) * 0.01
	}
	for i := range in {
		in[i] = float32((i%9)-4) * 0.05
	}
	scales, packed := packQ4ForTest(w)
	out := make([]float32, 4)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for r := 0; r < rows; r += 4 {
			DotQ4_0Rows4(in, scales, packed, r*cols, cols, out)
		}
	}
}
