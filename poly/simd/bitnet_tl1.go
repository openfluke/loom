package simd

// TL1 maps pairs of BitNet ternary weights {-1,0,+1} (codes {0,1,2}) to a 4-bit
// LUT index, matching microsoft/BitNet TL1_1 packing (Table 2 in bitnet.cpp docs).
var tl1IndexFromCodes = [3][3]uint8{
	{0, 1, 2}, // w0=-1
	{3, 4, 5}, // w0=0
	{6, 7, 8}, // w0=+1
}

// TL1IndexFromCodes returns the 4-bit TL1 index for ternary codes c0,c1 in {0,1,2}.
func TL1IndexFromCodes(c0, c1 uint8) uint8 {
	if c0 > 2 {
		c0 = 1
	}
	if c1 > 2 {
		c1 = 1
	}
	return tl1IndexFromCodes[c0][c1]
}

// tl1LUTEntry returns the int16 partial dot for weight-pair index idx with
// activations (b0, b1). idx 0..8 are the nine ternary pairs; 9..15 are zero.
func tl1LUTEntry(b0, b1 int8, idx uint8) int16 {
	switch idx {
	case 0:
		return int16(-(int32(b0) + int32(b1)))
	case 1:
		return int16(-int32(b0))
	case 2:
		return int16(-int32(b0) + int32(b1))
	case 3:
		return int16(-int32(b1))
	case 4:
		return 0
	case 5:
		return int16(b1)
	case 6:
		return int16(int32(b0) - int32(b1))
	case 7:
		return int16(b0)
	case 8:
		return int16(int32(b0) + int32(b1))
	default:
		return 0
	}
}

// BuildBitNetTL1QLUT builds per-pair int16 lookup tables from quantized int8
// activations. out has length pairCount*16; pair p uses out[p*16 : p*16+16].
// cols is the real column count (activations may be zero-padded to pairCount*2).
func BuildBitNetTL1QLUT(xq []int8, cols, pairCount int, out []int16) {
	if pairCount <= 0 || len(out) < pairCount*16 {
		return
	}
	for p := 0; p < pairCount; p++ {
		c0 := 2 * p
		c1 := c0 + 1
		var b0, b1 int8
		if c0 < cols {
			b0 = xq[c0]
		}
		if c1 < cols {
			b1 = xq[c1]
		}
		base := p * 16
		for idx := uint8(0); idx < 16; idx++ {
			out[base+int(idx)] = tl1LUTEntry(b0, b1, idx)
		}
	}
}

// BitNetTL1RowDotGo is the portable TL1 matvec inner loop for one output row.
// nibbles holds two 4-bit TL1 indices per byte (high nibble first pair).
// qlut is pairCount*16 int16 tables from BuildBitNetTL1QLUT. tailCode/tailAct
// handle an odd final column (code in {0,1,2}, act int8); pass tailCode=1 act=0
// when cols is even.
func BitNetTL1RowDotGo(nibbles []uint8, qlut []int16, pairCount int, tailCode uint8, tailAct int8) int32 {
	var sum int32
	for p := 0; p < pairCount; p++ {
		idx := nibbleAt(nibbles, p)
		sum += int32(qlut[p*16+int(idx)])
	}
	if tailCode <= 2 && tailCode != 1 {
		w := int32(tailCode) - 1
		sum += w * int32(tailAct)
	}
	return sum
}

func nibbleAt(nibbles []uint8, pair int) uint8 {
	b := nibbles[pair>>1]
	if pair&1 == 0 {
		return b >> 4
	}
	return b & 0x0f
}
