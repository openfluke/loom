// Package dot provides scalar dot products for matmul inner loops.
// amd64 and arm64 use Plan 9 assembly; other platforms use Go fallbacks.
package dot
