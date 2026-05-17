package matmul

import "github.com/openfluke/loom/poly/asm/dot"

// Native tiled forward: integer multiply + int64 accumulate, cast to output (no FP).

func ForwardNativeU8(
	preAct, input, weights []uint8,
	batch, inDim, outDim int,
	mc bool, tileSize int,
	signedLanes bool,
) {
	dotTile := dotTileU8Native
	if signedLanes {
		dotTile = dotTileU8SignedNative
	}
	forwardTiledU8Native(preAct, input, weights, batch, inDim, outDim, mc, tileSize, dotTile)
}

func ForwardNativeI8(preAct, input, weights []int8, batch, inDim, outDim int, mc bool, tileSize int) {
	forwardTiledI8Native(preAct, input, weights, batch, inDim, outDim, mc, tileSize, dotTileI8Native)
}

func ForwardNativeI16(preAct, input, weights []int16, batch, inDim, outDim int, mc bool, tileSize int) {
	forwardTiledI16Native(preAct, input, weights, batch, inDim, outDim, mc, tileSize, dotTileI16Native)
}

func ForwardNativeI32(preAct, input, weights []int32, batch, inDim, outDim int, mc bool, tileSize int) {
	forwardTiledI32Native(preAct, input, weights, batch, inDim, outDim, mc, tileSize, dotTileI32Native)
}

func ForwardNativeU16(preAct, input, weights []uint16, batch, inDim, outDim int, mc bool, tileSize int) {
	forwardTiledU16Native(preAct, input, weights, batch, inDim, outDim, mc, tileSize, dotTileU16Native)
}

func ForwardNativeU32(preAct, input, weights []uint32, batch, inDim, outDim int, mc bool, tileSize int) {
	forwardTiledU32Native(preAct, input, weights, batch, inDim, outDim, mc, tileSize, dotTileU32Native)
}

func ForwardNativeI64(preAct, input, weights []int64, batch, inDim, outDim int, mc bool, tileSize int) {
	forwardTiledI64Native(preAct, input, weights, batch, inDim, outDim, mc, tileSize, dotTileI64Native)
}

func ForwardNativeU64(preAct, input, weights []uint64, batch, inDim, outDim int, mc bool, tileSize int) {
	forwardTiledU64Native(preAct, input, weights, batch, inDim, outDim, mc, tileSize, dotTileU64Native)
}

// ForwardNativePackedNibble uses packed []uint32 weights (int4/uint4/fp4) and uint8 activations.
func ForwardNativePackedNibble(
	preAct, input []uint8,
	packed []uint32,
	batch, inDim, outDim int,
	mc bool, tileSize int,
) {
	forwardTiledPackedNative(preAct, input, packed, batch, inDim, outDim, mc, tileSize, false, false)
}

func ForwardNativePackedTwoBit(
	preAct, input []uint8,
	packed []uint32,
	batch, inDim, outDim int,
	mc bool, tileSize int,
	ternary bool,
) {
	forwardTiledPackedNative(preAct, input, packed, batch, inDim, outDim, mc, tileSize, true, ternary)
}

func ForwardNativePackedBinary(
	preAct, input []uint8,
	packed []uint32,
	batch, inDim, outDim int,
	mc bool, tileSize int,
) {
	if tileSize <= 0 {
		tileSize = 32
	}
	run := func(oTile, oEnd int) {
		for b := 0; b < batch; b++ {
			inRow := input[b*inDim : (b+1)*inDim]
			outRow := preAct[b*outDim : (b+1)*outDim]
			for o := oTile; o < oEnd; o++ {
				var sum int64
				for iTile := 0; iTile < inDim; iTile += tileSize {
					iEnd := iTile + tileSize
					if iEnd > inDim {
						iEnd = inDim
					}
					if iTile > 0 {
						sum = int64(int8(outRow[o]))
					}
					sum += dot.BinaryPackedRowNativeI64(inRow[iTile:iEnd], packed, o*inDim+iTile, iEnd-iTile)
				}
				outRow[o] = uint8(clampInt8(sum))
			}
		}
	}
	if mc {
		OverOutputTiles(outDim, tileSize, run)
	} else {
		run(0, outDim)
	}
}

type dotTileU8NativeFn func(inRow, wRow []uint8, i0, i1 int, prev int64) int64
type dotTileI8NativeFn func(inRow, wRow []int8, i0, i1 int, prev int64) int64
type dotTileI16NativeFn func(inRow, wRow []int16, i0, i1 int, prev int64) int64
type dotTileI32NativeFn func(inRow, wRow []int32, i0, i1 int, prev int64) int64
type dotTileU16NativeFn func(inRow, wRow []uint16, i0, i1 int, prev int64) int64
type dotTileU32NativeFn func(inRow, wRow []uint32, i0, i1 int, prev int64) int64
type dotTileI64NativeFn func(inRow, wRow []int64, i0, i1 int, prev int64) int64
type dotTileU64NativeFn func(inRow, wRow []uint64, i0, i1 int, prev int64) int64

var (
	dotTileU8Native       = dotTileU8NativeGo
	dotTileU8SignedNative = dotTileU8SignedNativeGo
	dotTileI8Native       = dotTileI8NativeGo
	dotTileI16Native      = dotTileI16NativeGo
	dotTileI32Native      = dotTileI32NativeGo
	dotTileI64Native      = dotTileI64NativeGo
	dotTileU16Native      = dotTileU16NativeGo
	dotTileU32Native      = dotTileU32NativeGo
	dotTileU64Native      = dotTileU64NativeGo
)

func dotTileU8NativeGo(inRow, wRow []uint8, i0, i1 int, prev int64) int64 {
	return prev + dot.U8TileNativeI64(inRow[i0:i1], wRow[i0:i1], i1-i0)
}

func dotTileU8SignedNativeGo(inRow, wRow []uint8, i0, i1 int, prev int64) int64 {
	return prev + dot.U8BytesTileNativeI64(inRow[i0:i1], wRow[i0:i1], i1-i0)
}

func dotTileI8NativeGo(inRow, wRow []int8, i0, i1 int, prev int64) int64 {
	return prev + dot.I8TileNativeI64(inRow[i0:i1], wRow[i0:i1], i1-i0)
}

func dotTileI16NativeGo(inRow, wRow []int16, i0, i1 int, prev int64) int64 {
	return prev + dot.I16TileNativeI64(inRow[i0:i1], wRow[i0:i1], i1-i0)
}

func dotTileI32NativeGo(inRow, wRow []int32, i0, i1 int, prev int64) int64 {
	return prev + dot.I32TileNativeI64(inRow[i0:i1], wRow[i0:i1], i1-i0)
}

func dotTileU16NativeGo(inRow, wRow []uint16, i0, i1 int, prev int64) int64 {
	return prev + dot.U16TileNativeI64(inRow[i0:i1], wRow[i0:i1], i1-i0)
}

func dotTileU32NativeGo(inRow, wRow []uint32, i0, i1 int, prev int64) int64 {
	return prev + dot.U32TileNativeI64(inRow[i0:i1], wRow[i0:i1], i1-i0)
}

func dotTileI64NativeGo(inRow, wRow []int64, i0, i1 int, prev int64) int64 {
	return prev + dot.I64TileNativeI64(inRow[i0:i1], wRow[i0:i1], i1-i0)
}

func dotTileU64NativeGo(inRow, wRow []uint64, i0, i1 int, prev int64) int64 {
	return prev + dot.U64TileNativeI64(inRow[i0:i1], wRow[i0:i1], i1-i0)
}

func forwardTiledU8Native(
	preAct, input, weights []uint8,
	batch, inDim, outDim int,
	mc bool, tileSize int,
	dotTile dotTileU8NativeFn,
) {
	if tileSize <= 0 {
		tileSize = 32
	}
	run := func(oTile, oEnd int) {
		for iTile := 0; iTile < inDim; iTile += tileSize {
			iEnd := iTile + tileSize
			if iEnd > inDim {
				iEnd = inDim
			}
			for b := 0; b < batch; b++ {
				inRow := input[b*inDim : (b+1)*inDim]
				outRow := preAct[b*outDim : (b+1)*outDim]
				for o := oTile; o < oEnd; o++ {
					prev := int64(0)
					if iTile > 0 {
						prev = int64(outRow[o])
					}
					wRow := weights[o*inDim : (o+1)*inDim]
					outRow[o] = uint8(clampInt8(dotTile(inRow, wRow, iTile, iEnd, prev)))
				}
			}
		}
	}
	if mc {
		OverOutputTiles(outDim, tileSize, run)
	} else {
		run(0, outDim)
	}
}

func forwardTiledI8Native(
	preAct, input, weights []int8,
	batch, inDim, outDim int,
	mc bool, tileSize int,
	dotTile dotTileI8NativeFn,
) {
	if tileSize <= 0 {
		tileSize = 32
	}
	run := func(oTile, oEnd int) {
		for iTile := 0; iTile < inDim; iTile += tileSize {
			iEnd := iTile + tileSize
			if iEnd > inDim {
				iEnd = inDim
			}
			for b := 0; b < batch; b++ {
				inRow := input[b*inDim : (b+1)*inDim]
				outRow := preAct[b*outDim : (b+1)*outDim]
				for o := oTile; o < oEnd; o++ {
					prev := int64(0)
					if iTile > 0 {
						prev = int64(outRow[o])
					}
					wRow := weights[o*inDim : (o+1)*inDim]
					outRow[o] = int8(clampInt8(dotTile(inRow, wRow, iTile, iEnd, prev)))
				}
			}
		}
	}
	if mc {
		OverOutputTiles(outDim, tileSize, run)
	} else {
		run(0, outDim)
	}
}

func forwardTiledI16Native(
	preAct, input, weights []int16,
	batch, inDim, outDim int,
	mc bool, tileSize int,
	dotTile dotTileI16NativeFn,
) {
	if tileSize <= 0 {
		tileSize = 32
	}
	run := func(oTile, oEnd int) {
		for iTile := 0; iTile < inDim; iTile += tileSize {
			iEnd := iTile + tileSize
			if iEnd > inDim {
				iEnd = inDim
			}
			for b := 0; b < batch; b++ {
				inRow := input[b*inDim : (b+1)*inDim]
				outRow := preAct[b*outDim : (b+1)*outDim]
				for o := oTile; o < oEnd; o++ {
					prev := int64(0)
					if iTile > 0 {
						prev = int64(outRow[o])
					}
					wRow := weights[o*inDim : (o+1)*inDim]
					outRow[o] = int16(clampInt16(dotTile(inRow, wRow, iTile, iEnd, prev)))
				}
			}
		}
	}
	if mc {
		OverOutputTiles(outDim, tileSize, run)
	} else {
		run(0, outDim)
	}
}

func forwardTiledI32Native(
	preAct, input, weights []int32,
	batch, inDim, outDim int,
	mc bool, tileSize int,
	dotTile dotTileI32NativeFn,
) {
	if tileSize <= 0 {
		tileSize = 32
	}
	run := func(oTile, oEnd int) {
		for iTile := 0; iTile < inDim; iTile += tileSize {
			iEnd := iTile + tileSize
			if iEnd > inDim {
				iEnd = inDim
			}
			for b := 0; b < batch; b++ {
				inRow := input[b*inDim : (b+1)*inDim]
				outRow := preAct[b*outDim : (b+1)*outDim]
				for o := oTile; o < oEnd; o++ {
					prev := int64(0)
					if iTile > 0 {
						prev = int64(outRow[o])
					}
					wRow := weights[o*inDim : (o+1)*inDim]
					outRow[o] = int32(clampInt32(dotTile(inRow, wRow, iTile, iEnd, prev)))
				}
			}
		}
	}
	if mc {
		OverOutputTiles(outDim, tileSize, run)
	} else {
		run(0, outDim)
	}
}

func forwardTiledU16Native(
	preAct, input, weights []uint16,
	batch, inDim, outDim int,
	mc bool, tileSize int,
	dotTile dotTileU16NativeFn,
) {
	if tileSize <= 0 {
		tileSize = 32
	}
	run := func(oTile, oEnd int) {
		for iTile := 0; iTile < inDim; iTile += tileSize {
			iEnd := iTile + tileSize
			if iEnd > inDim {
				iEnd = inDim
			}
			for b := 0; b < batch; b++ {
				inRow := input[b*inDim : (b+1)*inDim]
				outRow := preAct[b*outDim : (b+1)*outDim]
				for o := oTile; o < oEnd; o++ {
					prev := int64(0)
					if iTile > 0 {
						prev = int64(outRow[o])
					}
					wRow := weights[o*inDim : (o+1)*inDim]
					outRow[o] = uint16(clampUint16(dotTile(inRow, wRow, iTile, iEnd, prev)))
				}
			}
		}
	}
	if mc {
		OverOutputTiles(outDim, tileSize, run)
	} else {
		run(0, outDim)
	}
}

func forwardTiledI64Native(
	preAct, input, weights []int64,
	batch, inDim, outDim int,
	mc bool, tileSize int,
	dotTile dotTileI64NativeFn,
) {
	if tileSize <= 0 {
		tileSize = 32
	}
	run := func(oTile, oEnd int) {
		for iTile := 0; iTile < inDim; iTile += tileSize {
			iEnd := iTile + tileSize
			if iEnd > inDim {
				iEnd = inDim
			}
			for b := 0; b < batch; b++ {
				inRow := input[b*inDim : (b+1)*inDim]
				outRow := preAct[b*outDim : (b+1)*outDim]
				for o := oTile; o < oEnd; o++ {
					prev := int64(0)
					if iTile > 0 {
						prev = outRow[o]
					}
					wRow := weights[o*inDim : (o+1)*inDim]
					outRow[o] = clampInt64(dotTile(inRow, wRow, iTile, iEnd, prev))
				}
			}
		}
	}
	if mc {
		OverOutputTiles(outDim, tileSize, run)
	} else {
		run(0, outDim)
	}
}

func forwardTiledU64Native(
	preAct, input, weights []uint64,
	batch, inDim, outDim int,
	mc bool, tileSize int,
	dotTile dotTileU64NativeFn,
) {
	if tileSize <= 0 {
		tileSize = 32
	}
	run := func(oTile, oEnd int) {
		for iTile := 0; iTile < inDim; iTile += tileSize {
			iEnd := iTile + tileSize
			if iEnd > inDim {
				iEnd = inDim
			}
			for b := 0; b < batch; b++ {
				inRow := input[b*inDim : (b+1)*inDim]
				outRow := preAct[b*outDim : (b+1)*outDim]
				for o := oTile; o < oEnd; o++ {
					prev := int64(0)
					if iTile > 0 {
						prev = int64(outRow[o])
					}
					wRow := weights[o*inDim : (o+1)*inDim]
					outRow[o] = clampUint64(dotTile(inRow, wRow, iTile, iEnd, prev))
				}
			}
		}
	}
	if mc {
		OverOutputTiles(outDim, tileSize, run)
	} else {
		run(0, outDim)
	}
}

func forwardTiledU32Native(
	preAct, input, weights []uint32,
	batch, inDim, outDim int,
	mc bool, tileSize int,
	dotTile dotTileU32NativeFn,
) {
	if tileSize <= 0 {
		tileSize = 32
	}
	run := func(oTile, oEnd int) {
		for iTile := 0; iTile < inDim; iTile += tileSize {
			iEnd := iTile + tileSize
			if iEnd > inDim {
				iEnd = inDim
			}
			for b := 0; b < batch; b++ {
				inRow := input[b*inDim : (b+1)*inDim]
				outRow := preAct[b*outDim : (b+1)*outDim]
				for o := oTile; o < oEnd; o++ {
					prev := int64(0)
					if iTile > 0 {
						prev = int64(outRow[o])
					}
					wRow := weights[o*inDim : (o+1)*inDim]
					outRow[o] = uint32(clampUint32(dotTile(inRow, wRow, iTile, iEnd, prev)))
				}
			}
		}
	}
	if mc {
		OverOutputTiles(outDim, tileSize, run)
	} else {
		run(0, outDim)
	}
}

func forwardTiledPackedNative(
	preAct, input []uint8,
	packed []uint32,
	batch, inDim, outDim int,
	mc bool, tileSize int,
	twoBit, ternary bool,
) {
	if tileSize <= 0 {
		tileSize = 32
	}
	run := func(oTile, oEnd int) {
		for b := 0; b < batch; b++ {
			inRow := input[b*inDim : (b+1)*inDim]
			outRow := preAct[b*outDim : (b+1)*outDim]
			for o := oTile; o < oEnd; o++ {
				var sum int64
				rowOff := o * inDim
				for iTile := 0; iTile < inDim; iTile += tileSize {
					iEnd := iTile + tileSize
					if iEnd > inDim {
						iEnd = inDim
					}
					if iTile > 0 {
						sum = int64(int8(outRow[o]))
					}
					chunk := iEnd - iTile
					if twoBit {
						sum += dot.TwoBitPackedRowNativeI64(inRow[iTile:iEnd], packed, rowOff+iTile, chunk, ternary)
					} else {
						sum += dot.NibblePackedRowNativeI64(inRow[iTile:iEnd], packed, rowOff+iTile, chunk)
					}
				}
				outRow[o] = uint8(clampInt8(sum))
			}
		}
	}
	if mc {
		OverOutputTiles(outDim, tileSize, run)
	} else {
		run(0, outDim)
	}
}

func clampInt8(v int64) int8 {
	if v > 127 {
		return 127
	}
	if v < -128 {
		return -128
	}
	return int8(v)
}

func clampInt16(v int64) int16 {
	if v > 32767 {
		return 32767
	}
	if v < -32768 {
		return -32768
	}
	return int16(v)
}

func clampInt32(v int64) int32 {
	if v > 2147483647 {
		return 2147483647
	}
	if v < -2147483648 {
		return -2147483648
	}
	return int32(v)
}

func clampUint16(v int64) uint16 {
	if v < 0 {
		return 0
	}
	if v > 65535 {
		return 65535
	}
	return uint16(v)
}

func clampUint32(v int64) uint32 {
	if v < 0 {
		return 0
	}
	if v > 4294967295 {
		return 4294967295
	}
	return uint32(v)
}

func clampInt64(v int64) int64 {
	return v
}

func clampUint64(v int64) uint64 {
	if v < 0 {
		return 0
	}
	return uint64(v)
}
