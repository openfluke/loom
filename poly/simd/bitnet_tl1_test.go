package simd

import (
	"math/rand"
	"testing"
)

func TestTL1RowDotMatchesMAD(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	for _, cols := range []int{32, 64, 70, 256, 2560, 6912} {
		pairCount := (cols + 1) / 2
		stride := (pairCount + 1) / 2
		nibbles := make([]uint8, stride)
		for i := range nibbles {
			nibbles[i] = uint8(rng.Intn(256))
		}
		xq := make([]int8, cols)
		codes := make([]uint8, cols)
		for i := range xq {
			xq[i] = int8(rng.Intn(256) - 128)
			codes[i] = uint8(rng.Intn(3))
		}
		// Build nibbles from codes for a fair compare
		for c := 0; c+1 < cols; c += 2 {
			idx := TL1IndexFromCodes(codes[c], codes[c+1])
			pair := c / 2
			if pair&1 == 0 {
				nibbles[pair/2] = idx << 4
			} else {
				nibbles[pair/2] |= idx
			}
		}
		var tailCode uint8 = 1
		var tailAct int8
		if cols&1 == 1 {
			tailCode = codes[cols-1]
			tailAct = xq[cols-1]
		}

		qlut := make([]int16, pairCount*16)
		BuildBitNetTL1QLUT(xq, cols, pairCount, qlut)

		want := madDotFromCodes(codes, xq, cols)
		got := BitNetTL1RowDotGo(nibbles, qlut, pairCount, tailCode, tailAct)
		if got != want {
			t.Fatalf("cols=%d: tl1=%d mad=%d", cols, got, want)
		}
	}
}

func madDotFromCodes(codes []uint8, xq []int8, cols int) int32 {
	var sum int32
	for i := 0; i < cols; i++ {
		w := int32(codes[i]) - 1
		sum += w * int32(xq[i])
	}
	return sum
}

func BenchmarkTL1RowDot2560(b *testing.B) {
	const cols = 2560
	pairCount := (cols + 1) / 2
	stride := (pairCount + 1) / 2
	nibbles := make([]uint8, stride)
	xq := make([]int8, cols)
	for i := range xq {
		xq[i] = int8(i%17 - 8)
	}
	for c := 0; c+1 < cols; c += 2 {
		idx := TL1IndexFromCodes(uint8(c%3), uint8((c+1)%3))
		pair := c / 2
		if pair&1 == 0 {
			nibbles[pair/2] = idx << 4
		} else {
			nibbles[pair/2] |= idx
		}
	}
	qlut := make([]int16, pairCount*16)
	BuildBitNetTL1QLUT(xq, cols, pairCount, qlut)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sink32 = BitNetTL1RowDot(nibbles, qlut, pairCount, 1, 0)
	}
}

func BenchmarkTL1RowDotGo2560(b *testing.B) {
	const cols = 2560
	pairCount := (cols + 1) / 2
	stride := (pairCount + 1) / 2
	nibbles := make([]uint8, stride)
	xq := make([]int8, cols)
	for i := range xq {
		xq[i] = int8(i%17 - 8)
	}
	qlut := make([]int16, pairCount*16)
	BuildBitNetTL1QLUT(xq, cols, pairCount, qlut)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sink32 = BitNetTL1RowDotGo(nibbles, qlut, pairCount, 1, 0)
	}
}

var sink32 int32
