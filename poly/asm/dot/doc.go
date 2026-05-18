// Package dot provides scalar dot products for matmul inner loops.
//
// Plan 9 assembly (amd64 + arm64): dotF32, dotF64, dotF32AccF64, dot*NativeI64,
// dot*AccF64, and packed-row kernels. Sources are checked-in .s files per arch.
//
// Other GOARCH builds use Go fallbacks.
package dot
