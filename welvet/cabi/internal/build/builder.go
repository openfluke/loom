package main

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

// Platform defines a target OS/Arch combination
type Platform struct {
	OS   string
	Arch string
}

func (p Platform) String() string {
	return fmt.Sprintf("%s_%s", p.OS, p.Arch)
}

func main() {
	targetOS := flag.String("os", runtime.GOOS, "Target OS (windows, linux, darwin)")
	targetArch := flag.String("arch", runtime.GOARCH, "Target Architecture (amd64, arm64, 386)")
	outDir := flag.String("out", "dist", "Output directory")
	clean := flag.Bool("clean", false, "Clean output directory before building")
	flag.Parse()

	// 1. Setup paths
	platforms := []Platform{{OS: *targetOS, Arch: *targetArch}}
	if *targetOS == "all" {
		platforms = []Platform{
			{"windows", "amd64"},
			{"windows", "arm64"},
			{"linux", "amd64"},
			{"linux", "arm64"},
			{"darwin", "amd64"},
			{"darwin", "arm64"},
		}
	}

	for _, p := range platforms {
		buildPlatform(p, *outDir, *clean)
	}
}

func buildPlatform(p Platform, outBase string, clean bool) {
	fmt.Printf("\n--- Building for %s ---\n", p)

	// Determine file extension
	ext := ".so"
	if p.OS == "windows" {
		ext = ".dll"
	} else if p.OS == "darwin" {
		ext = ".dylib"
	}

	outputPath := filepath.Join(outBase, p.String())
	outputFile := filepath.Join(outputPath, "welvet"+ext)

	// 2. Clean if requested
	if clean {
		fmt.Printf("Cleaning %s...\n", outputPath)
		os.RemoveAll(outputPath)
	}

	if err := os.MkdirAll(outputPath, 0755); err != nil {
		fmt.Printf("Error creating directory: %v\n", err)
		return
	}

	// 3. Run Build
	fmt.Printf("Compiling %s...\n", outputFile)
	
	cmd := exec.Command("go", "build", 
		"-buildmode=c-shared", 
		"-o", outputFile, 
		"../../main.go",
	)

	// Set Environment Variables for Cross-Compilation
	cmd.Env = append(os.Environ(),
		"GOOS="+p.OS,
		"GOARCH="+p.Arch,
		"CGO_ENABLED=1",
	)

	// Note: Cross-compiling CGO requires a cross-compiler (CC)
	// We'll rely on the host having the appropriate CC in PATH if doing true cross-builds.
	
	output, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Printf("Build FAILED:\n%s\n", string(output))
		return
	}

	fmt.Printf("Build SUCCESS: %s\n", outputFile)
	
	// Copy header to a flat includes or keep with binary?
	// Legacy kept it with binary, which is easier for relative includes in C projects.
	fmt.Printf("Header generated: %s\n", strings.Replace(outputFile, ext, ".h", 1))
}
