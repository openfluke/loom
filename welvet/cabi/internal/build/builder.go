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
	test := flag.Bool("test", false, "Run verification tests after building")
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
		if *test {
			verifyPlatform(p, *outDir)
		}
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
		"../../", // Point to cabi/main.go
	)

	// Set Environment Variables for Cross-Compilation
	cmd.Env = append(os.Environ(),
		"GOOS="+p.OS,
		"GOARCH="+p.Arch,
		"CGO_ENABLED=1",
	)

	output, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Printf("Build FAILED:\n%s\n", string(output))
		return
	}

	fmt.Printf("Build SUCCESS: %s\n", outputFile)
	fmt.Printf("Header generated: %s\n", strings.Replace(outputFile, ext, ".h", 1))
}

func verifyPlatform(p Platform, outBase string) {
	if p.OS != runtime.GOOS || p.Arch != runtime.GOARCH {
		fmt.Printf("\n--- Skipping Verification for %s (Cross-compiled) ---\n", p)
		return
	}

	fmt.Printf("\n--- Verifying for %s ---\n", p)

	ext := ".so"
	if p.OS == "windows" {
		ext = ".dll"
	} else if p.OS == "darwin" {
		ext = ".dylib"
	}

	binPath := filepath.Join(outBase, p.String(), "welvet"+ext)
	absBinPath, _ := filepath.Abs(binPath)

	testDir := "../../test"
	verifySrc := filepath.Join(testDir, "cabi_verify.c")
	verifyExe := filepath.Join(outBase, p.String(), "cabi_verify")
	if p.OS == "windows" {
		verifyExe += ".exe"
	}

	// Compile C Runner
	fmt.Printf("Compiling C-ABI Verification Runner...\n")
	var cmd *exec.Cmd
	if p.OS == "windows" {
		// Try gcc first (MinGW), then cl (MSVC)
		if _, err := exec.LookPath("gcc"); err == nil {
			cmd = exec.Command("gcc", "-I"+testDir, verifySrc, "-o", verifyExe)
		} else {
			cmd = exec.Command("cl", "/I"+testDir, "/Fe:"+verifyExe, verifySrc)
		}
	} else {
		cmd = exec.Command("gcc", "-I"+testDir, verifySrc, "-o", verifyExe, "-ldl")
	}

	output, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Printf("Compilation FAILED:\n%s\n", string(output))
		return
	}

	// Run Verification
	fmt.Printf("Running Verification...\n")
	runCmd := exec.Command(verifyExe, absBinPath)
	output, err = runCmd.CombinedOutput()
	fmt.Printf("%s\n", string(output))
	if err != nil {
		fmt.Printf("Verification FAILED!\n")
	} else {
		fmt.Printf("Verification PASSED!\n")
	}
}
