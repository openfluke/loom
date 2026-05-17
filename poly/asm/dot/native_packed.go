package dot

// Packed low-bit weight row dots (weights in []uint32 bitstreams). Input is []uint8
// with one signed quant byte per element (poly morph storage). No FP decode.

// NibblePackedRowNativeI64: int4/uint4/fp4 codes packed 8 per uint32.
func NibblePackedRowNativeI64(x []uint8, packed []uint32, rowOff, n int) int64 {
	if n <= 0 {
		return 0
	}
	return nibblePackedRowNativeI64(x, packed, rowOff, n)
}

func NibblePackedRowNativeI64Go(x []uint8, packed []uint32, rowOff, n int) int64 {
	var sum int64
	for i := 0; i < n; i++ {
		idx := rowOff + i
		word := packed[idx/8]
		shift := uint((idx % 8) * 4)
		code := int8((word >> shift) & 0x0F)
		if code > 7 {
			code -= 16
		}
		sum += int64(int8(x[i])) * int64(code)
	}
	return sum
}

// TwoBitPackedRowNativeI64: int2/uint2/ternary packed 16 per uint32.
func TwoBitPackedRowNativeI64(x []uint8, packed []uint32, rowOff, n int, ternary bool) int64 {
	if n <= 0 {
		return 0
	}
	return twoBitPackedRowNativeI64(x, packed, rowOff, n, ternary)
}

func TwoBitPackedRowNativeI64Go(x []uint8, packed []uint32, rowOff, n int, ternary bool) int64 {
	var sum int64
	for i := 0; i < n; i++ {
		idx := rowOff + i
		word := packed[idx/16]
		shift := uint((idx % 16) * 2)
		code := int8((word >> shift) & 0x03)
		if ternary {
			switch code {
			case 0:
				code = -1
			case 2:
				code = 1
			default:
				code = 0
			}
		} else if code > 1 {
			code -= 4
		}
		sum += int64(int8(x[i])) * int64(code)
	}
	return sum
}

// BinaryPackedRowNativeI64: 32 binary weights per uint32; input ±1 in uint8.
func BinaryPackedRowNativeI64(x []uint8, packed []uint32, rowOff, n int) int64 {
	if n <= 0 {
		return 0
	}
	return binaryPackedRowNativeI64(x, packed, rowOff, n)
}

func BinaryPackedRowNativeI64Go(x []uint8, packed []uint32, rowOff, n int) int64 {
	var sum int64
	for i := 0; i < n; i++ {
		idx := rowOff + i
		word := packed[idx/32]
		shift := uint(idx % 32)
		w := int8(-1)
		if (word>>shift)&1 != 0 {
			w = 1
		}
		sum += int64(int8(x[i])) * int64(w)
	}
	return sum
}
