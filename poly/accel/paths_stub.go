//go:build !linux || !cgo

package accel

func defaultIntelPluginPath() string { return "" }

func intelDepsSearchDirs() []string { return nil }

func loomRoot() string { return "" }
