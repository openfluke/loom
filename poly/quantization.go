package poly

import (
	"math"
)

// Q4_0Block represents a block of 32 quantized 4-bit weights.
// Total size: 4 (f32 scale) + 16 (32 nibbles) = 20 bytes.
// Bandwidth: 0.625 bytes per weight.
type Q4_0Block struct {
	Scale   float32
	Weights [16]byte // 32 nibbles
}

// QuantizeQ4_0 converts a slice of f32 weights into Q4_0 blocks.
func QuantizeQ4_0(weights []float32) []Q4_0Block {
	n := len(weights)
	blockCount := (n + 31) / 32
	blocks := make([]Q4_0Block, blockCount)

	for i := 0; i < blockCount; i++ {
		start := i * 32
		end := start + 32
		if end > n {
			end = n
		}

		// Find max absolute value in block
		maxAbs := float32(0)
		for j := start; j < end; j++ {
			abs := float32(math.Abs(float64(weights[j])))
			if abs > maxAbs {
				maxAbs = abs
			}
		}

		scale := maxAbs / 7.0
		blocks[i].Scale = scale

		for j := 0; j < 16; j++ {
			idx1 := start + j*2
			idx2 := start + j*2 + 1

			v1 := float32(0)
			if idx1 < n {
				v1 = weights[idx1]
			}
			v2 := float32(0)
			if idx2 < n {
				v2 = weights[idx2]
			}

			q1 := int8(math.Round(float64(v1 / scale)))
			if q1 > 7 { q1 = 7 }
			if q1 < -8 { q1 = -8 }

			q2 := int8(math.Round(float64(v2 / scale)))
			if q2 > 7 { q2 = 7 }
			if q2 < -8 { q2 = -8 }

			// Pack into byte (lower 4 bits, upper 4 bits)
			// We store 4-bit signed as offset or raw? 
			// Q4_0 usually uses (x - 8) or similar.
			// Let's use simple signed 4-bit: -8 to 7.
			// To store in a byte: (q1 & 0xF) | ((q2 & 0xF) << 4)
			blocks[i].Weights[j] = byte(q1&0xF) | (byte(q2&0xF) << 4)
		}
	}

	return blocks
}

// DequantizeQ4_0 converts Q4_0 blocks back to f32.
func DequantizeQ4_0(blocks []Q4_0Block, n int) []float32 {
	res := make([]float32, n)
	for i, b := range blocks {
		scale := b.Scale
		for j := 0; j < 16; j++ {
			w := b.Weights[j]
			q1 := int8(w & 0xF)
			if q1 > 7 { q1 -= 16 } // Sign extend
			
			q2 := int8(w >> 4)
			if q2 > 7 { q2 -= 16 } // Sign extend

			idx1 := i*32 + j*2
			idx2 := i*32 + j*2 + 1
			if idx1 < n {
				res[idx1] = float32(q1) * scale
			}
			if idx2 < n {
				res[idx2] = float32(q2) * scale
			}
		}
	}
	return res
}
