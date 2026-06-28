//go:build !linux || !cgo

package accel

func PrepareRuntime() error { return nil }

func runtimeHint(err error) string { return "" }

func runtimeLDLibraryPath() string { return "" }
