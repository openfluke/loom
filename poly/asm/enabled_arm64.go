//go:build arm64

package asm

// Enabled reports whether Plan 9 assembly dot kernels are linked for this process.
func Enabled() bool { return true }
