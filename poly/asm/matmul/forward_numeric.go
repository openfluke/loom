package matmul

// ForwardNumeric runs tiled forward for integer element types (asm dot on amd64/arm64).
func ForwardNumeric(preAct, input, weights any, batch, inDim, outDim int, mc bool, tileSize int) {
	switch w := weights.(type) {
	case []int:
		p, in := preAct.([]int), input.([]int)
		ForwardTiledNumeric(p, in, w, batch, inDim, outDim, mc, tileSize)
	case []int8:
		p, in := preAct.([]int8), input.([]int8)
		ForwardTiledI8(p, in, w, batch, inDim, outDim, mc, tileSize)
	case []int16:
		p, in := preAct.([]int16), input.([]int16)
		ForwardTiledI16(p, in, w, batch, inDim, outDim, mc, tileSize)
	case []int32:
		p, in := preAct.([]int32), input.([]int32)
		ForwardTiledI32(p, in, w, batch, inDim, outDim, mc, tileSize)
	case []int64:
		p, in := preAct.([]int64), input.([]int64)
		ForwardTiledI64(p, in, w, batch, inDim, outDim, mc, tileSize)
	case []uint:
		p, in := preAct.([]uint), input.([]uint)
		ForwardTiledNumeric(p, in, w, batch, inDim, outDim, mc, tileSize)
	case []uint8:
		p, in := preAct.([]uint8), input.([]uint8)
		ForwardTiledU8(p, in, w, batch, inDim, outDim, mc, tileSize)
	case []uint16:
		p, in := preAct.([]uint16), input.([]uint16)
		ForwardTiledU16(p, in, w, batch, inDim, outDim, mc, tileSize)
	case []uint32:
		p, in := preAct.([]uint32), input.([]uint32)
		ForwardTiledU32(p, in, w, batch, inDim, outDim, mc, tileSize)
	case []uint64:
		p, in := preAct.([]uint64), input.([]uint64)
		ForwardTiledU64(p, in, w, batch, inDim, outDim, mc, tileSize)
	}
}
