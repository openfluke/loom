package poly

import (
	"fmt"
	"math"
	"runtime"
	"sync"
	"time"

	"github.com/openfluke/loom/poly/simd"
)

// Q4 LM-head cache: PackQ4_0 of logits matrix (vocab × hidden). Built at SyncInferenceCPU
// when UsePackedQ4CPU — frees untied FP32 LMHead afterward so decode streams ~4× less bytes.

func (t *Transformer[T]) usePackedQ4LMHead() bool {
	if t == nil || t.Network == nil || t.Network.UseGPU || !t.Network.UsePackedQ4CPU {
		return false
	}
	return len(t.lmHeadQ4Packed) > 0 && t.lmHeadQ4Rows == t.VocabSize && t.lmHeadQ4Cols == t.HiddenSize
}

// EnsurePackedQ4LMHead packs the FP32 LM head (or tied embeddings) into Q4 for CPU logits.
// Skips work when a baked transformer.lm_head.q4_0 blob was already loaded.
// When the head is untied, releases the FP32 LMHead slice afterward.
func (t *Transformer[T]) EnsurePackedQ4LMHead() {
	if t == nil || t.Network == nil || t.Network.UseGPU || !t.Network.UsePackedQ4CPU {
		return
	}
	if t.VocabSize <= 0 || t.HiddenSize <= 0 {
		return
	}
	if t.usePackedQ4LMHead() {
		// Baked Q4 already present — still drop untied FP32 head to free RAM.
		if !t.lmHeadTiedToEmbeddings() && len(t.LMHead) > 0 {
			freed := len(t.LMHead) * 4
			t.LMHead = nil
			fmt.Printf("🧮 LM head: using baked Q4 logits; freed %s FP32 LMHead\n", formatByteSize(int64(freed)))
			AggressiveReleaseMemoryToOS()
		}
		return
	}
	src := t.LMHead
	if len(src) < t.VocabSize*t.HiddenSize {
		src = t.Embeddings
	}
	if len(src) < t.VocabSize*t.HiddenSize {
		return
	}
	src = src[:t.VocabSize*t.HiddenSize]

	start := time.Now()
	scales, packed := PackQ4_0GPUParallel(src)
	if len(scales) == 0 || len(packed) == 0 {
		return
	}
	t.lmHeadQ4Scales = scales
	t.lmHeadQ4Packed = packed
	t.lmHeadQ4Rows = t.VocabSize
	t.lmHeadQ4Cols = t.HiddenSize
	t.lmHeadLogitsF32 = make([]float32, t.VocabSize)

	freed := 0
	if !t.lmHeadTiedToEmbeddings() && len(t.LMHead) > 0 {
		freed = len(t.LMHead) * 4
		t.LMHead = nil
		runtime.GC()
	}
	elapsed := time.Since(start).Round(time.Millisecond)
	q4Bytes := len(scales)*4 + len(packed)*4
	msg := fmt.Sprintf("🧮 LM head → packed Q4 for logits (%d×%d, %s → %s packed in %s)",
		t.VocabSize, t.HiddenSize,
		formatByteSize(int64(len(src)*4)),
		formatByteSize(int64(q4Bytes)),
		elapsed)
	if freed > 0 {
		msg += fmt.Sprintf("; freed %s FP32 LMHead", formatByteSize(int64(freed)))
	}
	fmt.Println(msg)
	AggressiveReleaseMemoryToOS()
}

func (t *Transformer[T]) applyPackedQ4LMHead(hidden []T) []float32 {
	if !t.usePackedQ4LMHead() {
		return nil
	}
	hf, ok := any(hidden).([]float32)
	if !ok {
		hf = make([]float32, len(hidden))
		for i := range hidden {
			hf[i] = float32(hidden[i])
		}
	}
	if len(hf) < t.HiddenSize {
		return nil
	}
	hf = hf[:t.HiddenSize]

	out32 := t.lmHeadLogitsF32
	if len(out32) < t.VocabSize {
		out32 = make([]float32, t.VocabSize)
		t.lmHeadLogitsF32 = out32
	}

	useSimd := simd.SimdEnabled()
	gemvQ4_0PackedParallelF32(t.lmHeadQ4Scales, t.lmHeadQ4Packed, hf, out32[:t.VocabSize], t.VocabSize, t.HiddenSize, useSimd)
	return out32[:t.VocabSize]
}

// formatByteSize is a tiny local helper (avoids importing lucy helpers into poly).
func formatByteSize(n int64) string {
	if n < 1024 {
		return fmt.Sprintf("%d B", n)
	}
	f := float64(n)
	units := []string{"KB", "MB", "GB"}
	u := -1
	for f >= 1024 && u < len(units)-1 {
		f /= 1024
		u++
	}
	if u < 0 {
		return fmt.Sprintf("%d B", n)
	}
	return fmt.Sprintf("%.2f %s", f, units[u])
}

// PackQ4_0GPUParallel is PackQ4_0GPU with parallel block quantize (large LM heads).
func PackQ4_0GPUParallel(data []float32) (scales []float32, packed []uint32) {
	n := len(data)
	if n == 0 {
		return nil, nil
	}
	blockCount := (n + 31) / 32
	packedSize := blockCount * 4
	alignedSize := (packedSize + 63) &^ 63
	if alignedSize < 512 {
		alignedSize = 512
	}
	scales = make([]float32, blockCount)
	packed = make([]uint32, alignedSize)

	workers := runtime.GOMAXPROCS(0)
	if workers < 1 {
		workers = 1
	}
	if blockCount < workers*8 {
		workers = 1
	}
	var wg sync.WaitGroup
	chunk := (blockCount + workers - 1) / workers
	for w := 0; w < workers; w++ {
		b0 := w * chunk
		b1 := b0 + chunk
		if b1 > blockCount {
			b1 = blockCount
		}
		if b0 >= b1 {
			break
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			quantizeQ4_0BlocksInto(data, scales, packed, lo, hi)
		}(b0, b1)
	}
	wg.Wait()
	return scales, packed
}

func quantizeQ4_0BlocksInto(data, scales []float32, packed []uint32, blockLo, blockHi int) {
	n := len(data)
	for i := blockLo; i < blockHi; i++ {
		start := i * 32
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
		scale := maxAbs / 7.0
		scales[i] = scale
		var bytes [16]byte
		if scale != 0 {
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
				if q1 > 7 {
					q1 = 7
				}
				if q1 < -8 {
					q1 = -8
				}
				q2 := int8(math.Round(float64(v2 / scale)))
				if q2 > 7 {
					q2 = 7
				}
				if q2 < -8 {
					q2 = -8
				}
				bytes[j] = byte(q1&0xF) | (byte(q2&0xF) << 4)
			}
		}
		base := i * 4
		for j := 0; j < 4; j++ {
			packed[base+j] = uint32(bytes[j*4]) |
				(uint32(bytes[j*4+1]) << 8) |
				(uint32(bytes[j*4+2]) << 16) |
				(uint32(bytes[j*4+3]) << 24)
		}
	}
}
