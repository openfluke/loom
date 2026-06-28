//go:build linux && cgo

package accel

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// PrepareRuntime preloads OpenVINO / NPU driver .so dependencies so dlopen(loom_accel_intel)
// works without manually sourcing accel/intel/setup_env.sh.
func PrepareRuntime() error {
	required, optional := discoverPreloadLibs()
	if len(required) == 0 && openvinoDir() == "" {
		return fmt.Errorf("OpenVINO not found — run accel/intel/install_openvino.sh and source setup_env.sh, or set INTEL_OPENVINO_DIR")
	}
	for _, lib := range required {
		if err := preloadSharedLib(lib); err != nil {
			return fmt.Errorf("preload %s: %w", lib, err)
		}
	}
	for _, lib := range optional {
		_ = preloadSharedLib(lib)
	}
	return nil
}

func discoverPreloadLibs() (required, optional []string) {

	ovDir := openvinoDir()
	if ovDir != "" {
		tbbLib := filepath.Join(ovDir, "runtime", "3rdparty", "tbb", "lib")
		required = append(required, pickExisting(
			filepath.Join(tbbLib, "libtbb.so.12"),
			filepath.Join(tbbLib, "libtbb.so"),
		)...)
		required = append(required, globOne(filepath.Join(ovDir, "runtime", "lib", "intel64", "libopenvino.so*"))...)
	}

	npuDir := npuDriverLibDir()
	if npuDir != "" {
		optional = append(optional, globOne(filepath.Join(npuDir, "libnpu_driver_compiler.so"))...)
	}

	augmentLDLibraryPath(ovDir, npuDir)
	return dedupePaths(required), dedupePaths(optional)
}

func dedupePaths(paths []string) []string {
	seen := make(map[string]struct{}, len(paths))
	out := make([]string, 0, len(paths))
	for _, p := range paths {
		if p == "" {
			continue
		}
		if _, ok := seen[p]; ok {
			continue
		}
		seen[p] = struct{}{}
		out = append(out, p)
	}
	return out
}

func pickExisting(paths ...string) []string {
	for _, p := range paths {
		if _, err := os.Stat(p); err == nil {
			return []string{p}
		}
	}
	return nil
}

func runtimeLDLibraryPath() string {
	ovDir := openvinoDir()
	npuDir := npuDriverLibDir()
	dirs := make([]string, 0, 4)
	if ovDir != "" {
		dirs = append(dirs,
			filepath.Join(ovDir, "runtime", "3rdparty", "tbb", "lib"),
			filepath.Join(ovDir, "runtime", "lib", "intel64"),
		)
	}
	if npuDir != "" {
		dirs = append(dirs, npuDir)
	}
	if len(dirs) == 0 {
		return ""
	}
	return strings.Join(dirs, ":")
}

func augmentLDLibraryPath(ovDir, npuDir string) {
	ld := runtimeLDLibraryPath()
	if ld == "" {
		return
	}
	cur := os.Getenv("LD_LIBRARY_PATH")
	if cur != "" {
		ld = ld + ":" + cur
	}
	_ = os.Setenv("LD_LIBRARY_PATH", ld)
}

func openvinoDir() string {
	if v := os.Getenv("INTEL_OPENVINO_DIR"); v != "" {
		return v
	}
	for _, base := range npuExampleDepsBases() {
		matches, _ := filepath.Glob(filepath.Join(base, "openvino_toolkit_*"))
		if len(matches) > 0 {
			sort.Strings(matches)
			return matches[len(matches)-1]
		}
	}
	return ""
}

func npuDriverLibDir() string {
	if v := os.Getenv("INTEL_NPU_LIBDIR"); v != "" {
		return v
	}
	for _, base := range npuExampleDepsBases() {
		p := filepath.Join(base, "npu-driver", "root", "usr", "lib", "x86_64-linux-gnu")
		if st, err := os.Stat(p); err == nil && st.IsDir() {
			return p
		}
	}
	return ""
}

func npuExampleDepsBases() []string {
	return intelDepsSearchDirs()
}

func globOne(pattern string) []string {
	matches, err := filepath.Glob(pattern)
	if err != nil || len(matches) == 0 {
		return nil
	}
	// Prefer highest-versioned soname (e.g. libopenvino.so.2540).
	sort.Strings(matches)
	return []string{matches[len(matches)-1]}
}

func runtimeHint(err error) string {
	if err == nil {
		return ""
	}
	msg := err.Error()
	if strings.Contains(msg, "cannot open shared object") || strings.Contains(msg, "No such file") {
		return "\nHint: cd accel/intel && source ./setup_env.sh\n" +
			"      or set INTEL_OPENVINO_DIR + INTEL_NPU_LIBDIR"
	}
	return ""
}
