package nn

// =============================================================================
// NVFP4 (E2M1) — Pure-Go CPU Emulation
//
// Format: 4-bit mini-float  [S | E1 E0 | M0]
//   bit 3   = sign      (1 = negative)
//   bits 2:1 = exponent  (biased by 1, so values 0-3 → actual exp -1..2)
//   bit 0   = mantissa  (implicit leading 1 when exp≠0; subnormal when exp=0)
//
// Value table (positive half):
//   0 0 0 0  → 0.0
//   0 0 0 1  → 0.5   (subnormal: 0.mantissa × 2^(1-bias) = 0.1 × 2^0)
//   0 0 1 0  → 1.0   (exp=1→actual=0: 1.0 × 2^0)
//   0 0 1 1  → 1.5
//   0 1 0 0  → 2.0   (exp=2→actual=1: 1.0 × 2^1)
//   0 1 0 1  → 3.0
//   0 1 1 0  → 4.0   (exp=3→actual=2: 1.0 × 2^2)
//   0 1 1 1  → 6.0
//   Negatives are the sign-mirrored versions.
//
// Micro-scaling (MXFP4 / Blackwell style):
//   Every group of MicroScaleGroup weights shares one float32 scale factor.
//   result = sum_of_fp4_products × scale   (done at accumulation time)
// =============================================================================

import (
	"math"
	"math/bits"
	"unsafe"
)

// MicroScaleGroup is the number of E2M1 element-products that share one
// float32 scale factor.  Blackwell hardware uses 16.
const MicroScaleGroup = 16

// fp4Bias is the exponent bias for E2M1 (bias = 2^(expBits-1) - 1 = 1).
const fp4Bias = 1

// fp4Mag is a precomputed lookup table mapping a 3-bit unsigned nibble
// (sign stripped) to its float32 magnitude.  Index = bits[2:0] of the nibble.
//
// Indexed by (nibble & 0x7):
//
//	0 → 0.0, 1 → 0.5, 2 → 1.0, 3 → 1.5, 4 → 2.0, 5 → 3.0, 6 → 4.0, 7 → 6.0
var fp4Mag = [8]float32{0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}

// =============================================================================
// PackedWeights — storage type
// =============================================================================

// PackedWeights stores two 4-bit E2M1 weights in a single uint8.
//   - Low nibble  (bits 3:0) = first weight
//   - High nibble (bits 7:4) = second weight
//
// This halves memory bandwidth vs. float32, mirroring how Blackwell
// stores NVFP4 weight tiles before feeding them to Tensor Cores.
type PackedWeights struct {
	// Data holds pairs of E2M1 weights.  len(Data) == ceil(numWeights/2).
	Data []uint8
	// Rows × Cols describe the logical weight matrix dimensions.
	Rows, Cols int
	// Scales holds the micro-scale factors in per-column, per-row-group layout.
	// Index: Scales[col * NumRowGroups + rowGroup].
	// This ensures each output neuron's input-dimension group gets its own scale.
	Scales       []float32
	NumRowGroups int // = ceil(Rows / MicroScaleGroup)
}

// NewPackedWeights creates a PackedWeights from a []float32 weight matrix,
// quantising every value to E2M1.  Scales are computed per-group (max-abs
// normalisation so the largest value in the group maps to the representable
// maximum, fp4Max = 6.0).
// NewPackedWeights creates a PackedWeights from a []float32 weight matrix,
// quantising every value to E2M1.  Scales are computed per-column and per
// row-group so that each output neuron's input-dimension groups are scaled
// independently — matching MXFP4 micro-scaling semantics.
//
// Weight layout (row-major): weights[i*cols + o] = weight(row=i, col=o).
func NewPackedWeights(weights []float32, rows, cols int) *PackedWeights {
	const fp4Max = 6.0
	n := rows * cols

	numBytes := (n + 1) / 2
	numRowGroups := (rows + MicroScaleGroup - 1) / MicroScaleGroup

	pw := &PackedWeights{
		Data:         make([]uint8, numBytes),
		Rows:         rows,
		Cols:         cols,
		Scales:       make([]float32, cols*numRowGroups),
		NumRowGroups: numRowGroups,
	}

	for o := 0; o < cols; o++ {
		for g := 0; g < numRowGroups; g++ {
			rowStart := g * MicroScaleGroup
			rowEnd := rowStart + MicroScaleGroup
			if rowEnd > rows {
				rowEnd = rows
			}

			// Compute per-column, per-row-group scale (max-abs normalisation).
			maxAbs := float32(0)
			for i := rowStart; i < rowEnd; i++ {
				k := i*cols + o
				if a := float32(math.Abs(float64(weights[k]))); a > maxAbs {
					maxAbs = a
				}
			}
			scale := float32(1.0)
			if maxAbs > 0 {
				scale = maxAbs / fp4Max
			}
			pw.Scales[o*numRowGroups+g] = scale

			invScale := float32(1)
			if scale > 0 {
				invScale = 1.0 / scale
			}
			for i := rowStart; i < rowEnd; i++ {
				k := i*cols + o
				nibble := quantiseToE2M1(weights[k] * invScale)
				byteIdx := k >> 1
				if k&1 == 0 {
					pw.Data[byteIdx] = nibble & 0x0F
				} else {
					pw.Data[byteIdx] |= (nibble & 0x0F) << 4
				}
			}
		}
	}
	return pw
}

// Get extracts the i-th E2M1 nibble from PackedWeights.
// Uses unsafe pointer arithmetic to avoid bounds-check on the hot path.
//
//go:nosplit
func (pw *PackedWeights) Get(i int) uint8 {
	byteIdx := i >> 1
	b := *(*uint8)(unsafe.Pointer(
		uintptr(unsafe.Pointer(&pw.Data[0])) + uintptr(byteIdx),
	))
	if i&1 == 0 {
		return b & 0x0F
	}
	return b >> 4
}

// =============================================================================
// E2M1 bitwise multiply — the "gate" emulation
// =============================================================================

// MultiplyE2M1 multiplies two 4-bit E2M1 nibbles using pure integer logic,
// emulating what a Tensor Core does in hardware.
//
// Algorithm (mirrors hardware gate-level multiply):
//  1. Extract sign, exponent, mantissa from both nibbles.
//  2. Result sign  = sA XOR sB.
//  3. Result exp   = eA + eB (handle subnormals specially).
//  4. Result mant  = product of significands (with implicit leading 1).
//  5. Normalise back to E2M1 and return the integer nibble.
//
// Returns the 4-bit result as a uint8 (high nibble is zero).
//
// NOTE: overflow/+inf saturates to the largest representable value (6.0 / -6.0).
func MultiplyE2M1(a, b uint8) uint8 {
	// ── Unpack a ──────────────────────────────────────────────────────────────
	sA := (a >> 3) & 1   // sign bit
	eA := (a >> 1) & 0x3 // 2-bit exponent
	mA := a & 1          // mantissa bit

	// ── Unpack b ──────────────────────────────────────────────────────────────
	sB := (b >> 3) & 1
	eB := (b >> 1) & 0x3
	mB := b & 1

	// ── Special case: either operand is zero ──────────────────────────────────
	// Zero in E2M1 = both exponent and mantissa bits are 0.
	if (eA == 0 && mA == 0) || (eB == 0 && mB == 0) {
		return 0
	}

	// ── Result sign ───────────────────────────────────────────────────────────
	sR := sA ^ sB

	// ── Build significands (Q1.1 fixed point: implicit 1 when exp≠0) ─────────
	// sigA = implicit_bit . mA  (2-bit fixed point, value in [1.0, 1.5] × 2^0)
	// For subnormals (eA=0): sigA = 0.mA (no implicit leading 1)
	var sigA, sigB uint32
	if eA == 0 {
		sigA = uint32(mA) // 0.mA × 2^(1-bias) = mA × 2^0  = 0 or 0.5
	} else {
		sigA = 2 | uint32(mA) // 1.mA in Q1.1: value = (2 + mA) × 0.5
	}
	if eB == 0 {
		sigB = uint32(mB)
	} else {
		sigB = 2 | uint32(mB)
	}

	// Multiply significands: result is at most Q2.2 (4 bits), fits in uint32.
	// sigA,sigB ∈ {0,1,2,3}  → product ∈ {0..9}, 4 bits.
	sigR := sigA * sigB // Q2.2 fixed-point product

	// ── Compute result exponent ───────────────────────────────────────────────
	// Actual exponent of A = eA - bias (with subnormal treated as exp=1-bias).
	var actA, actB int
	if eA == 0 {
		actA = 1 - fp4Bias // subnormal effective exponent
	} else {
		actA = int(eA) - fp4Bias
	}
	if eB == 0 {
		actB = 1 - fp4Bias
	} else {
		actB = int(eB) - fp4Bias
	}

	// sigA and sigB are integers where the actual significand = value/2.
	// E.g. sigA=2 → 1.0, sigA=3 → 1.5.  So:
	//   actual product significand = (sigA/2) × (sigB/2) = sigR/4
	//   actual product value       = sigR/4 × 2^(actA+actB)
	//
	// After normalising sigR by shifting right by (leadBit-1) to get
	// sigR_norm ∈ {2,3} (also a Q1.1 significand), we need:
	//   sigR_norm/2 × 2^actR  =  sigR/4 × 2^(actA+actB)
	//   actR = actA + actB - 1 + shift    (where shift = leadBit-1)
	actR := actA + actB - 1

	// Normalise sigR: shift until the leading bit is in position 1 (Q1.1 = 1.m).
	if sigR == 0 {
		return sR << 3 // product is zero
	}
	leadBit := 31 - bits.LeadingZeros32(sigR) // position of highest set bit
	// We want leadBit == 1 so that sigR ∈ {2,3} representing 1.0 or 1.5.
	shift := leadBit - 1
	actR += shift // each right-shift of significand decrements exp by 1

	if shift >= 0 {
		sigR >>= uint(shift)
	} else {
		sigR <<= uint(-shift)
	}

	// ── Encode result exponent back to biased 2-bit form ─────────────────────
	eR := actR + fp4Bias

	// ── Clamp / saturate ─────────────────────────────────────────────────────
	if eR >= 4 {
		// Overflow → saturate to ±6.0  (0b x111)
		return (sR << 3) | 0x7
	}
	if eR <= 0 {
		// Underflow → zero
		return sR << 3
	}

	mR := uint8(sigR & 1) // keep only the mantissa bit

	return (sR << 3) | (uint8(eR) << 1) | mR
}

// E2M1ToFloat32 decodes a single 4-bit E2M1 nibble to float32.
// Uses the precomputed magnitude table for zero-branch decoding.
//
//go:nosplit
func E2M1ToFloat32(nibble uint8) float32 {
	mag := fp4Mag[nibble&0x7]
	if nibble>>3 != 0 {
		return -mag
	}
	return mag
}

// quantiseToE2M1 maps a float32 value to the nearest E2M1 nibble.
// The input should already be prescaled to the fp4 representable range.
func quantiseToE2M1(v float32) uint8 {
	sign := uint8(0)
	if v < 0 {
		sign = 1
		v = -v
	}
	// Find nearest magnitude in the table.
	best := uint8(0)
	bestDist := math.MaxFloat32
	for i, mag := range fp4Mag {
		d := math.Abs(float64(v - mag))
		if d < bestDist {
			bestDist = d
			best = uint8(i)
		}
	}
	return (sign << 3) | (best & 0x7)
}

// =============================================================================
// ForwardRowFP4 — forward.go–compatible row-level multiply-accumulate
// =============================================================================

// FP4RowResult holds the raw integer accumulator and the group scales,
// separated so callers can defer the final float32 multiply to a fused step.
type FP4RowResult struct {
	// IntAcc holds one int32 accumulator per output neuron.
	// Each entry is the sum of E2M1 integer products within a micro-scale group.
	IntAcc []int32
	// Scales is a [outputNeurons][numGroups] flat array: Scales[o*numGroups+g]
	// is the scale for output neuron `o`, micro-group `g`.
	// In practice both the input row and the weight column share the same group
	// index, so we store only the weight scales here and the caller supplies the
	// activation scale.
	WeightScales []float32
	// NumGroups is the number of micro-scale groups per output neuron.
	NumGroups int
}

// ForwardRowFP4 performs a single-row (one input vector) forward pass
// against a PackedWeights matrix using E2M1 bitwise multiply-accumulate.
//
// # Algorithm
//
//  1. Divide the inner-product accumulation into groups of MicroScaleGroup.
//  2. Within each group, call MultiplyE2M1 in integer domain — no float cast.
//  3. After the group, multiply the integer partial sum by (weightScale)
//     and accumulate into a float32 result for the neuron.
//
// This mirrors how Blackwell Tensor Cores execute MXFP4:
//   - Integer multiply inside the 4-bit window.
//   - Scale applied once per micro-tile before adding to the float32 accumulator.
//
// Parameters:
//   - inputNibbles: E2M1-encoded input row  (one nibble per element, NOT packed)
//   - inputScales:  per–micro-group scales for the input row
//     (len == ceil(inputSize/MicroScaleGroup))
//   - pw:           packed weight matrix  (rows=inputSize, cols=outputSize)
//   - bias:         float32 bias vector   (len == outputSize, may be nil)
//
// Returns a []float32 of length pw.Cols (one accumulated float per neuron).
func ForwardRowFP4(
	inputNibbles []uint8,
	inputScales []float32,
	pw *PackedWeights,
	bias []float32,
) []float32 {
	inputSize := pw.Rows
	outputSize := pw.Cols
	numGroups := pw.NumRowGroups

	out := make([]float32, outputSize)

	for o := 0; o < outputSize; o++ {
		var acc float32
		for g := 0; g < numGroups; g++ {
			start := g * MicroScaleGroup
			end := start + MicroScaleGroup
			if end > inputSize {
				end = inputSize
			}

			// Integer accumulator for this micro-group.
			var intAcc int32

			for i := start; i < end; i++ {
				// weight[i][o] = index i*outputSize + o
				wIdx := i*outputSize + o
				wNibble := pw.Get(wIdx)
				aNibble := inputNibbles[i]

				// Multiply the two nibble magnitudes in integer form (×2 each →
				// product is in units of ×4).  Track sign separately to avoid a
				// float conversion on every element — the float multiply with the
				// group scale happens only once per group outside this loop.
				aMagInt := fp4MagInt[aNibble&0x7]
				wMagInt := fp4MagInt[wNibble&0x7]
				aNeg := (aNibble >> 3) != 0
				wNeg := (wNibble >> 3) != 0
				p := aMagInt * wMagInt
				if aNeg != wNeg {
					intAcc -= p
				} else {
					intAcc += p
				}
			}

			// Apply micro-scaling once per group: one float32 multiply.
			// intAcc units are mag_a×2 × mag_w×2 = actual_product × 4,
			// so multiply by 0.25 to recover the sum of float products,
			// then by wScale and aScale to dequantise both axes.
			wScale := float32(1.0)
			if scaleIdx := o*pw.NumRowGroups + g; scaleIdx < len(pw.Scales) {
				wScale = pw.Scales[scaleIdx]
			}
			aScale := float32(1.0)
			if g < len(inputScales) {
				aScale = inputScales[g]
			}
			acc += float32(intAcc) * 0.25 * wScale * aScale
		}

		if bias != nil && o < len(bias) {
			acc += bias[o]
		}
		out[o] = acc
	}
	return out
}

// fp4MagInt is the magnitude lookup scaled by ×2 (so all entries are integers),
// matching fp4Mag × 2.  This lets ForwardRowFP4 stay in integer arithmetic
// for the inner loop.
//
// Index → fp4Mag × 2:
//
//	0→0, 1→1, 2→2, 3→3, 4→4, 5→6, 6→8, 7→12
var fp4MagInt = [8]int32{0, 1, 2, 3, 4, 6, 8, 12}

// =============================================================================
// QuantiseInputRowFP4  — helper to prepare an activation row
// =============================================================================

// QuantiseInputRowFP4 converts a []float32 activation row into the
// (nibbles, scales) pair expected by ForwardRowFP4.
//
// The output nibbles are NOT packed — one uint8 per element — which avoids
// a pack/unpack on the activation side where packing rarely saves bandwidth
// (activations are not stored for long, unlike weights).
func QuantiseInputRowFP4(row []float32) (nibbles []uint8, scales []float32) {
	const fp4Max = 6.0
	n := len(row)
	numGroups := (n + MicroScaleGroup - 1) / MicroScaleGroup

	nibbles = make([]uint8, n)
	scales = make([]float32, numGroups)

	for g := 0; g < numGroups; g++ {
		start := g * MicroScaleGroup
		end := start + MicroScaleGroup
		if end > n {
			end = n
		}

		// Compute group scale.
		maxAbs := float32(0)
		for k := start; k < end; k++ {
			if a := float32(math.Abs(float64(row[k]))); a > maxAbs {
				maxAbs = a
			}
		}
		scale := float32(1.0)
		if maxAbs > 0 {
			scale = maxAbs / fp4Max
		}
		scales[g] = scale

		invScale := float32(1)
		if scale > 0 {
			invScale = 1.0 / scale
		}
		for k := start; k < end; k++ {
			nibbles[k] = quantiseToE2M1(row[k] * invScale)
		}
	}
	return nibbles, scales
}

// =============================================================================
// DenseForwardFP4 — drop-in for denseForwardCPU with FP4 weights
// =============================================================================

// DenseForwardFP4 is a forward.go–compatible dense layer forward pass that
// keeps weights in E2M1 format throughout the inner loop.
//
// It wraps ForwardRowFP4 and handles the batchSize dimension, returning
// (preAct, postAct) slices identical in shape to denseForwardCPU.
//
// Usage alongside GenericForwardPass / ForwardCPU:
//
//	pw := NewPackedWeights(config.Kernel, config.InputHeight, config.OutputHeight)
//	preAct, postAct := DenseForwardFP4(data, pw, config.Bias, batchSize, config.Activation)
func DenseForwardFP4(
	input []float32,
	pw *PackedWeights,
	bias []float32,
	batchSize int,
	activation ActivationType,
) (preAct, postAct []float32) {
	inputSize := pw.Rows
	outputSize := pw.Cols

	preAct = make([]float32, batchSize*outputSize)
	postAct = make([]float32, batchSize*outputSize)

	for b := 0; b < batchSize; b++ {
		rowStart := b * inputSize
		rowEnd := rowStart + inputSize
		if rowEnd > len(input) {
			rowEnd = len(input)
		}
		row := input[rowStart:rowEnd]

		// Quantise the activation row to E2M1:
		aNibbles, aScales := QuantiseInputRowFP4(row)

		// Run the bitwise multiply-accumulate:
		result := ForwardRowFP4(aNibbles, aScales, pw, bias)

		// Copy result and apply activation:
		outOffset := b * outputSize
		for o := 0; o < outputSize; o++ {
			preAct[outOffset+o] = result[o]
			postAct[outOffset+o] = activateCPU(result[o], activation)
		}
	}
	return preAct, postAct
}
