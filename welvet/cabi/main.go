package main

/*
#include <stdlib.h>
#include <stdint.h>
*/
import "C"

import (
	"unsafe"
)

/*
POLY IS MULTI-TYPE
type DType int

const (
	DTypeFloat64  DType = 0  // 64-bit double
	DTypeFloat32  DType = 1  // Standard 32-bit float
	DTypeFloat16  DType = 2  // 16-bit float
	DTypeBFloat16 DType = 3  // 16-bit Brain Float
	DTypeFP8E4M3  DType = 4  // 8-bit FP8 (E4M3)
	DTypeFP8E5M2  DType = 5  // 8-bit FP8 (E5M2)
	DTypeInt64    DType = 6  // 64-bit integer
	DTypeInt32    DType = 7  // 32-bit integer
	DTypeInt16    DType = 8  // 16-bit integer
	DTypeInt8     DType = 9  // 8-bit integer
	DTypeUint64   DType = 10 // 64-bit unsigned
	DTypeUint32   DType = 11 // 32-bit unsigned
	DTypeUint16   DType = 12 // 16-bit unsigned
	DTypeUint8    DType = 13 // 8-bit unsigned
	DTypeInt4     DType = 14 // 4-bit integer
	DTypeUint4    DType = 15 // 4-bit unsigned
	DTypeFP4      DType = 16 // 4-bit E2M1
	DTypeInt2     DType = 17 // 2-bit integer
	DTypeUint2    DType = 18 // 2-bit unsigned
	DTypeTernary  DType = 19 // 2-bit (Ternary: -1, 0, 1)
	DTypeBinary   DType = 20 // 1-bit (XNOR-Net)
)
*/

// FreeLoomString releases memory for strings allocated by the Go bridge.
//
//export FreeLoomString
func FreeLoomString(ptr *C.char) {
	C.free(unsafe.Pointer(ptr))
}

func main() {}
