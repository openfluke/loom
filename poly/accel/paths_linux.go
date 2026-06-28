//go:build linux && cgo

package accel

import (
	"os"
	"path/filepath"
)

const intelPluginName = "libloom_accel_intel.so"

// defaultIntelPluginPath resolves the Intel C ABI shared library.
// Order: LOOM_ACCEL_INTEL_SO → LOOM_ROOT → walk up from cwd for accel/intel/build/.
func defaultIntelPluginPath() string {
	if v := os.Getenv("LOOM_ACCEL_INTEL_SO"); v != "" {
		return v
	}
	if root := loomRoot(); root != "" {
		if p := filepath.Join(root, "accel", "intel", "build", intelPluginName); fileExists(p) {
			return p
		}
	}
	if p := findUpward("accel", "intel", "build", intelPluginName); p != "" {
		return p
	}
	return ""
}

func intelDepsSearchDirs() []string {
	var dirs []string
	if root := loomRoot(); root != "" {
		dirs = append(dirs, filepath.Join(root, "accel", "intel", "deps"))
	}
	if p := findUpward("accel", "intel", "deps"); p != "" {
		dirs = append(dirs, p)
	}
	return dedupePaths(dirs)
}

// loomRoot is the Loom module root (directory containing go.mod for github.com/openfluke/loom).
func loomRoot() string {
	if v := os.Getenv("LOOM_ROOT"); v != "" {
		return v
	}
	if p := findUpward("go.mod"); p != "" {
		return filepath.Dir(p)
	}
	return ""
}

func findUpward(parts ...string) string {
	cwd, err := os.Getwd()
	if err != nil {
		return ""
	}
	for dir := cwd; ; dir = filepath.Dir(dir) {
		candidate := filepath.Join(append([]string{dir}, parts...)...)
		if fileExists(candidate) {
			return candidate
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
	}
	return ""
}

func fileExists(path string) bool {
	if path == "" {
		return false
	}
	_, err := os.Stat(path)
	return err == nil
}
