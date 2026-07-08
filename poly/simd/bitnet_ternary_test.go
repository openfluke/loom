package simd

import (
	"math/rand"
	"testing"
)

// TestBitNetTernaryKernelMatchesGo proves the SIMD MAD kernel returns exactly the
// same int32 as the scalar Go reference for every dtype-realistic input. Because
// the arithmetic is integer, "close" is not good enough — it must be identical.
func TestBitNetTernaryKernelMatchesGo(t *testing.T) {
	SetBitNetTernarySimdForward(true)
	defer SetBitNetTernarySimdForward(false)
	if !ternarySimdEnabled() {
		t.Skip("ternary SIMD not enabled on this build/arch")
	}

	rng := rand.New(rand.NewSource(42))
	// BitNet MAD contract: nBytes is a multiple of 32. Cover several row widths
	// plus a padded tail case.
	for _, nBytes := range []int{32, 64, 96, 512, 1536, 4096} {
		for trial := 0; trial < 200; trial++ {
			codes := make([]uint8, nBytes)
			acts := make([]int8, nBytes)
			for i := range codes {
				codes[i] = uint8(rng.Intn(3)) // {0,1,2} — real ternary codes
				acts[i] = int8(rng.Intn(256) - 128)
			}
			want := bitNetTernaryCodeRowDotGo(codes, acts, nBytes)
			got := BitNetTernaryCodeRowDot(codes, acts, nBytes)
			if got != want {
				t.Fatalf("nBytes=%d trial=%d: simd=%d want=%d", nBytes, trial, got, want)
			}
		}
	}
}

func benchTernaryVec(n int) ([]uint8, []int8) {
	rng := rand.New(rand.NewSource(1))
	codes := make([]uint8, n)
	acts := make([]int8, n)
	for i := range codes {
		codes[i] = uint8(rng.Intn(3))
		acts[i] = int8(rng.Intn(256) - 128)
	}
	return codes, acts
}

var ternarySink int32

func BenchmarkBitNetTernarySimd1536(b *testing.B) {
	SetBitNetTernarySimdForward(true)
	defer SetBitNetTernarySimdForward(false)
	codes, acts := benchTernaryVec(1536)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ternarySink = BitNetTernaryCodeRowDot(codes, acts, 1536)
	}
}

func BenchmarkBitNetTernaryScalar1536(b *testing.B) {
	codes, acts := benchTernaryVec(1536)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ternarySink = bitNetTernaryCodeRowDotGo(codes, acts, 1536)
	}
}
