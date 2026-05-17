//go:build !amd64 && !arm64

package asm

// Enabled is false on platforms without assembly dot kernels (wasm, 386, etc.).
func Enabled() bool { return false }
