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

// ──────────────────────────────────────────────────────────────────────────────
// Platform definitions
// ──────────────────────────────────────────────────────────────────────────────

type Platform struct {
	GOOS      string // go env value: windows, linux, darwin, android, ios
	GOARCH    string // go env value: amd64, arm64, arm, 386
	GOARM     string // optional: "7" for armv7
	DirName   string // dist subdirectory name
	LibName   string // output filename without path
	BuildMode string // c-shared or c-archive
}

func (p Platform) Ext() string {
	switch p.GOOS {
	case "windows":
		return ".dll"
	case "darwin":
		return ".dylib"
	case "ios":
		return ".a"
	default: // linux, android
		return ".so"
	}
}

// allPlatforms is the canonical full target list.
var allPlatforms = []Platform{
	// ── Desktop / Server ──────────────────────────────────────────────────────
	{GOOS: "linux", GOARCH: "amd64", DirName: "linux_amd64", BuildMode: "c-shared"},
	{GOOS: "linux", GOARCH: "arm64", DirName: "linux_arm64", BuildMode: "c-shared"},
	{GOOS: "linux", GOARCH: "arm", GOARM: "7", DirName: "linux_armv7", BuildMode: "c-shared"},
	{GOOS: "linux", GOARCH: "386", DirName: "linux_x86", BuildMode: "c-shared"},
	{GOOS: "darwin", GOARCH: "amd64", DirName: "macos_amd64", BuildMode: "c-shared"},
	{GOOS: "darwin", GOARCH: "arm64", DirName: "macos_arm64", BuildMode: "c-shared"},
	{GOOS: "windows", GOARCH: "amd64", DirName: "windows_amd64", BuildMode: "c-shared"},
	{GOOS: "windows", GOARCH: "arm64", DirName: "windows_arm64", BuildMode: "c-shared"},
	{GOOS: "windows", GOARCH: "386", DirName: "windows_x86", BuildMode: "c-shared"},
	// ── Android ───────────────────────────────────────────────────────────────
	{GOOS: "android", GOARCH: "arm64", DirName: "android_arm64", BuildMode: "c-shared"},
	{GOOS: "android", GOARCH: "arm", GOARM: "7", DirName: "android_armv7", BuildMode: "c-shared"},
	{GOOS: "android", GOARCH: "amd64", DirName: "android_x86_64", BuildMode: "c-shared"},
	{GOOS: "android", GOARCH: "386", DirName: "android_x86", BuildMode: "c-shared"},
	// ── iOS (static archives) ─────────────────────────────────────────────────
	{GOOS: "ios", GOARCH: "arm64", DirName: "ios_arm64", BuildMode: "c-archive"},
	{GOOS: "ios", GOARCH: "amd64", DirName: "ios_sim_amd64", BuildMode: "c-archive"},
	// arm64 simulator uses GOOS=ios GOARCH=arm64 + SDK=iphonesimulator;
	// handled separately in buildIOS.
}

// ──────────────────────────────────────────────────────────────────────────────
// CLI entry point
// ──────────────────────────────────────────────────────────────────────────────

func main() {
	targetOS := flag.String("os", runtime.GOOS, "Target OS: windows|linux|darwin|android|ios|all")
	targetArch := flag.String("arch", runtime.GOARCH, "Target arch: amd64|arm64|arm|386|armv7|universal")
	outDir := flag.String("out", "dist", "Output directory")
	clean := flag.Bool("clean", false, "Remove output directory before building")
	test := flag.Bool("test", false, "Run C verification after building (native targets only)")
	flag.Parse()

	if *clean {
		fmt.Printf("Cleaning %s...\n", *outDir)
		os.RemoveAll(*outDir)
	}

	var targets []Platform

	switch *targetOS {
	case "all":
		targets = allPlatforms
		// macOS universal and iOS xcframework are handled separately below
	default:
		targets = selectPlatforms(*targetOS, *targetArch)
	}

	successes := []string{}
	failures := []string{}

	for _, p := range targets {
		if p.GOOS == "ios" {
			if err := buildIOS(p, *outDir); err != nil {
				fmt.Printf("FAILED %s: %v\n", p.DirName, err)
				failures = append(failures, p.DirName)
			} else {
				successes = append(successes, p.DirName)
			}
			continue
		}
		if err := buildPlatform(p, *outDir); err != nil {
			fmt.Printf("FAILED %s: %v\n", p.DirName, err)
			failures = append(failures, p.DirName)
		} else {
			successes = append(successes, p.DirName)
			if *test {
				verifyPlatform(p, *outDir)
			}
		}
	}

	// macOS universal fat binary (only when building all or explicitly requested)
	if *targetOS == "all" || (*targetOS == "darwin" && *targetArch == "universal") {
		if err := buildMacUniversal(*outDir); err != nil {
			fmt.Printf("FAILED macos_universal: %v\n", err)
			failures = append(failures, "macos_universal")
		} else {
			successes = append(successes, "macos_universal")
		}
	}

	// iOS XCFramework (only when building all ios or xcframework)
	if *targetOS == "all" || (*targetOS == "ios" && *targetArch == "xcframework") {
		if err := buildXCFramework(*outDir); err != nil {
			fmt.Printf("FAILED ios_xcframework: %v\n", err)
			failures = append(failures, "ios_xcframework")
		} else {
			successes = append(successes, "ios_xcframework")
		}
	}

	printSummary(successes, failures)
}

// ──────────────────────────────────────────────────────────────────────────────
// Platform selection
// ──────────────────────────────────────────────────────────────────────────────

func selectPlatforms(goos, arch string) []Platform {
	// Normalise arch aliases
	switch arch {
	case "x86_64":
		arch = "amd64"
	case "aarch64":
		arch = "arm64"
	case "armv7":
		arch = "arm"
	case "x86":
		arch = "386"
	case "universal", "xcframework":
		// handled outside this loop
		return nil
	}

	for _, p := range allPlatforms {
		goarm := ""
		if arch == "arm" {
			goarm = "7"
		}
		if p.GOOS == goos && p.GOARCH == arch && p.GOARM == goarm {
			return []Platform{p}
		}
	}

	// Fallback: build native
	p := Platform{
		GOOS:      goos,
		GOARCH:    arch,
		DirName:   goos + "_" + arch,
		BuildMode: "c-shared",
	}
	if goos == "darwin" {
		p.BuildMode = "c-shared"
	} else if goos == "ios" {
		p.BuildMode = "c-archive"
	}
	return []Platform{p}
}

// ──────────────────────────────────────────────────────────────────────────────
// Standard build (c-shared / c-archive via go build)
// ──────────────────────────────────────────────────────────────────────────────

func buildPlatform(p Platform, outBase string) error {
	fmt.Printf("\n--- Building %s ---\n", p.DirName)

	outPath := filepath.Join(outBase, p.DirName)
	if err := os.MkdirAll(outPath, 0755); err != nil {
		return err
	}

	libName := "welvet" + p.Ext()
	outFile := filepath.Join(outPath, libName)

	cc := crossCC(p)

	cmd := exec.Command("go", "build",
		"-buildmode="+p.BuildMode,
		"-o", outFile,
		"../../",
	)

	env := append(os.Environ(),
		"GOOS="+p.GOOS,
		"GOARCH="+p.GOARCH,
		"CGO_ENABLED=1",
	)
	if p.GOARM != "" {
		env = append(env, "GOARM="+p.GOARM)
	}
	if cc != "" {
		env = append(env, "CC="+cc)
	}

	cmd.Env = env
	out, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Printf("%s\n", string(out))
		return fmt.Errorf("go build: %w", err)
	}

	hFile := strings.TrimSuffix(outFile, p.Ext()) + ".h"
	fmt.Printf("  ✓  %s\n", outFile)
	fmt.Printf("  ✓  %s\n", hFile)

	// Always compile cabi_verify alongside the library so it ships in dist/.
	compileVerify(p, outPath, cc)
	return nil
}

// compileVerify builds cabi_verify(.exe) into outPath using the same compiler
// as the library build. For cross-compiled targets the binary runs on the
// target device; for native targets it can be executed with --test.
func compileVerify(p Platform, outPath string, cc string) {
	testDir := "../../test"
	verifySrc := filepath.Join(testDir, "cabi_verify.c")

	verifyExe := filepath.Join(outPath, "cabi_verify")
	if p.GOOS == "windows" {
		verifyExe += ".exe"
	}

	var args []string
	var compilerBin string

	switch p.GOOS {
	case "windows":
		if cc == "" {
			if _, err := exec.LookPath("gcc"); err == nil {
				cc = "gcc"
			} else {
				fmt.Printf("  ⚠  cabi_verify: no gcc found for Windows, skipping\n")
				return
			}
		}
		// mingw: no -ldl needed, link against the DLL
		libFile := filepath.Join(outPath, "welvet.dll")
		compilerBin = cc
		args = []string{"-I" + testDir, "-I" + outPath, verifySrc, "-o", verifyExe, libFile}

	case "android":
		if cc == "" {
			fmt.Printf("  ⚠  cabi_verify: no NDK CC available, skipping\n")
			return
		}
		// Android: PIE required, link the .so
		libFile := filepath.Join(outPath, "welvet.so")
		absLib, _ := filepath.Abs(libFile)
		absOut, _ := filepath.Abs(outPath)
		compilerBin = cc
		args = []string{"-I" + testDir, "-I" + absOut, verifySrc, "-o", verifyExe,
			"-L" + absOut, absLib, "-lm", "-pie"}

	default: // linux, darwin
		if cc == "" {
			cc = "gcc"
		}
		compilerBin = cc
		args = []string{"-I" + testDir, "-I" + outPath, verifySrc, "-o", verifyExe, "-ldl"}
	}

	fmt.Printf("  compiling cabi_verify...\n")
	out, err := exec.Command(compilerBin, args...).CombinedOutput()
	if err != nil {
		fmt.Printf("  ⚠  cabi_verify compile failed: %s\n", strings.TrimSpace(string(out)))
		return
	}
	fmt.Printf("  ✓  %s\n", verifyExe)
}

// crossCC returns the C compiler needed to cross-compile for p.
// Returns "" to use the default CC already in the environment.
func crossCC(p Platform) string {
	host := runtime.GOOS

	switch p.GOOS {
	case "linux":
		switch p.GOARCH {
		case "arm64":
			if host != "linux" || runtime.GOARCH != "arm64" {
				return "aarch64-linux-gnu-gcc"
			}
		case "arm":
			return "arm-linux-gnueabihf-gcc"
		case "386":
			if host != "linux" || runtime.GOARCH != "386" {
				return "i686-linux-gnu-gcc"
			}
		}
	case "windows":
		switch p.GOARCH {
		case "amd64":
			return "x86_64-w64-mingw32-gcc"
		case "386":
			return "i686-w64-mingw32-gcc"
		case "arm64":
			return "aarch64-w64-mingw32-gcc"
		}
	case "android":
		return androidCC(p)
	}
	return ""
}

// ──────────────────────────────────────────────────────────────────────────────
// Android NDK helpers
// ──────────────────────────────────────────────────────────────────────────────

const androidAPILevel = "21"

func androidCC(p Platform) string {
	ndkPath := os.Getenv("ANDROID_NDK_HOME")
	if ndkPath == "" {
		ndkPath = os.Getenv("NDK_HOME")
	}
	if ndkPath == "" {
		fmt.Println("  ⚠  ANDROID_NDK_HOME not set — skipping Android CC")
		return ""
	}

	prebuilt := ndkPrebuilt(ndkPath)
	if prebuilt == "" {
		return ""
	}

	binDir := filepath.Join(ndkPath, "toolchains", "llvm", "prebuilt", prebuilt, "bin")

	var triple string
	switch p.GOARCH {
	case "arm64":
		triple = "aarch64-linux-android"
	case "arm":
		triple = "armv7a-linux-androideabi"
	case "amd64":
		triple = "x86_64-linux-android"
	case "386":
		triple = "i686-linux-android"
	default:
		return ""
	}

	return filepath.Join(binDir, triple+androidAPILevel+"-clang")
}

func ndkPrebuilt(ndkPath string) string {
	candidates := []string{
		"linux-x86_64",
		"darwin-arm64",
		"darwin-x86_64",
		"windows-x86_64",
	}
	for _, c := range candidates {
		if _, err := os.Stat(filepath.Join(ndkPath, "toolchains", "llvm", "prebuilt", c)); err == nil {
			return c
		}
	}
	return ""
}

// ──────────────────────────────────────────────────────────────────────────────
// iOS build (c-archive + xcrun)
// ──────────────────────────────────────────────────────────────────────────────

func buildIOS(p Platform, outBase string) error {
	if runtime.GOOS != "darwin" {
		return fmt.Errorf("iOS builds require macOS")
	}

	fmt.Printf("\n--- Building %s ---\n", p.DirName)

	// Determine SDK and clang arch
	sdk := "iphoneos"
	isSimARM := strings.Contains(p.DirName, "sim")
	if isSimARM || p.GOARCH == "amd64" {
		sdk = "iphonesimulator"
	}

	clangArch := p.GOARCH
	if clangArch == "amd64" {
		clangArch = "x86_64"
	}

	sdkPath, err := runOutput("xcrun", "--sdk", sdk, "--show-sdk-path")
	if err != nil {
		return fmt.Errorf("xcrun sdk path: %w", err)
	}
	clangBin, err := runOutput("xcrun", "--sdk", sdk, "--find", "clang")
	if err != nil {
		return fmt.Errorf("xcrun clang: %w", err)
	}

	minFlag := "-mios-version-min=13.0"
	if sdk == "iphonesimulator" {
		minFlag = "-mios-simulator-version-min=13.0"
	}

	cc := strings.TrimSpace(clangBin) + " -isysroot " + strings.TrimSpace(sdkPath) +
		" " + minFlag + " -arch " + clangArch

	outPath := filepath.Join(outBase, p.DirName)
	if err := os.MkdirAll(outPath, 0755); err != nil {
		return err
	}
	outFile := filepath.Join(outPath, "welvet.a")

	cmd := exec.Command("go", "build",
		"-buildmode=c-archive",
		"-o", outFile,
		"../../",
	)
	cmd.Env = append(os.Environ(),
		"GOOS=ios",
		"GOARCH="+p.GOARCH,
		"CGO_ENABLED=1",
		"CC="+cc,
	)
	if isSimARM {
		cmd.Env = append(cmd.Env, "GOFLAGS=-tags=ios_simulator")
	}

	out, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Printf("%s\n", string(out))
		return fmt.Errorf("go build ios: %w", err)
	}

	fmt.Printf("  ✓  %s\n", outFile)
	return nil
}

// ──────────────────────────────────────────────────────────────────────────────
// iOS XCFramework
// ──────────────────────────────────────────────────────────────────────────────

func buildXCFramework(outBase string) error {
	if runtime.GOOS != "darwin" {
		return fmt.Errorf("XCFramework requires macOS")
	}
	fmt.Println("\n--- Building ios_xcframework ---")

	deviceLib := filepath.Join(outBase, "ios_arm64", "welvet.a")
	simAMDLib := filepath.Join(outBase, "ios_sim_amd64", "welvet.a")
	simARMLib := filepath.Join(outBase, "ios_sim_arm64", "welvet.a")
	deviceHeaders := filepath.Join(outBase, "ios_arm64")
	xcfOut := filepath.Join(outBase, "ios_xcframework", "Welvet.xcframework")

	os.MkdirAll(filepath.Dir(xcfOut), 0755)
	os.RemoveAll(xcfOut)

	// Create fat simulator lib if both sim slices exist
	simLib := ""
	simHeaders := ""
	hasFatSim := false
	if _, err := os.Stat(simARMLib); err == nil {
		if _, err := os.Stat(simAMDLib); err == nil {
			fatPath := filepath.Join(outBase, "ios_xcframework", "sim_fat", "welvet.a")
			os.MkdirAll(filepath.Dir(fatPath), 0755)
			if err := run("lipo", "-create", simAMDLib, simARMLib, "-output", fatPath); err == nil {
				// copy header
				hSrc := filepath.Join(outBase, "ios_sim_arm64", "welvet.h")
				hDst := filepath.Join(filepath.Dir(fatPath), "welvet.h")
				copyFile(hSrc, hDst)
				simLib = fatPath
				simHeaders = filepath.Dir(fatPath)
				hasFatSim = true
			}
		}
		if !hasFatSim {
			simLib = simARMLib
			simHeaders = filepath.Join(outBase, "ios_sim_arm64")
		}
	} else if _, err := os.Stat(simAMDLib); err == nil {
		simLib = simAMDLib
		simHeaders = filepath.Join(outBase, "ios_sim_amd64")
	}

	if _, err := os.Stat(deviceLib); err != nil {
		return fmt.Errorf("device library not found: %s", deviceLib)
	}
	if simLib == "" {
		return fmt.Errorf("no simulator library found")
	}

	args := []string{
		"-create-xcframework",
		"-library", deviceLib, "-headers", deviceHeaders,
		"-library", simLib, "-headers", simHeaders,
		"-output", xcfOut,
	}
	if err := run("xcodebuild", args...); err != nil {
		return err
	}
	fmt.Printf("  ✓  %s\n", xcfOut)
	return nil
}

// ──────────────────────────────────────────────────────────────────────────────
// macOS Universal fat binary
// ──────────────────────────────────────────────────────────────────────────────

func buildMacUniversal(outBase string) error {
	if runtime.GOOS != "darwin" {
		return fmt.Errorf("universal binary requires macOS (lipo)")
	}
	fmt.Println("\n--- Building macos_universal ---")

	amd64Lib := filepath.Join(outBase, "macos_amd64", "welvet.dylib")
	arm64Lib := filepath.Join(outBase, "macos_arm64", "welvet.dylib")

	for _, f := range []string{amd64Lib, arm64Lib} {
		if _, err := os.Stat(f); err != nil {
			return fmt.Errorf("required slice missing: %s", f)
		}
	}

	outPath := filepath.Join(outBase, "macos_universal")
	os.MkdirAll(outPath, 0755)
	fatLib := filepath.Join(outPath, "welvet.dylib")

	if err := run("lipo", "-create", amd64Lib, arm64Lib, "-output", fatLib); err != nil {
		return err
	}

	// Copy header from arm64 build (same regardless of arch)
	hSrc := filepath.Join(outBase, "macos_arm64", "welvet.h")
	hDst := filepath.Join(outPath, "welvet.h")
	copyFile(hSrc, hDst)

	fmt.Printf("  ✓  %s\n", fatLib)
	return nil
}

// ──────────────────────────────────────────────────────────────────────────────
// Verification (native only)
// ──────────────────────────────────────────────────────────────────────────────

func verifyPlatform(p Platform, outBase string) {
	if p.GOOS != runtime.GOOS || p.GOARCH != runtime.GOARCH {
		fmt.Printf("  (skipping --test run — cross-compiled target)\n")
		return
	}

	fmt.Printf("\n--- Verifying %s ---\n", p.DirName)

	verifyExe := filepath.Join(outBase, p.DirName, "cabi_verify")
	if p.GOOS == "windows" {
		verifyExe += ".exe"
	}
	if _, err := os.Stat(verifyExe); err != nil {
		fmt.Printf("  ⚠  cabi_verify not found — was it compiled?\n")
		return
	}

	absBin, _ := filepath.Abs(filepath.Join(outBase, p.DirName, "welvet"+p.Ext()))
	out, err := exec.Command(verifyExe, absBin).CombinedOutput()
	fmt.Printf("%s\n", string(out))
	if err != nil {
		fmt.Printf("  ✗ Verification FAILED\n")
	} else {
		fmt.Printf("  ✓ Verification PASSED\n")
	}
}

// ──────────────────────────────────────────────────────────────────────────────
// Summary
// ──────────────────────────────────────────────────────────────────────────────

func printSummary(successes, failures []string) {
	fmt.Println()
	fmt.Println("╔════════════════════════════════════════════════════╗")
	fmt.Println("║                  Build Summary                     ║")
	fmt.Println("╚════════════════════════════════════════════════════╝")

	if len(successes) > 0 {
		fmt.Printf("\n✅ Success (%d):\n", len(successes))
		for _, s := range successes {
			fmt.Printf("   ✓ %s\n", s)
		}
	}

	if len(failures) > 0 {
		fmt.Printf("\n❌ Failed (%d):\n", len(failures))
		for _, f := range failures {
			fmt.Printf("   ✗ %s\n", f)
		}
		fmt.Println()
		fmt.Println("💡 Missing cross-compilers — install with:")
		fmt.Println("   Linux/WSL:")
		fmt.Println("     sudo apt install gcc-aarch64-linux-gnu gcc-arm-linux-gnueabihf \\")
		fmt.Println("       gcc-i686-linux-gnu mingw-w64 gcc-mingw-w64-aarch64")
		fmt.Println("     sudo apt install android-ndk   (or set ANDROID_NDK_HOME manually)")
		fmt.Println("   macOS:")
		fmt.Println("     brew install mingw-w64 aarch64-unknown-linux-gnu")
		fmt.Println("     brew install --cask android-ndk")
		fmt.Println("     export ANDROID_NDK_HOME=/opt/homebrew/share/android-ndk")
	}

	fmt.Println()
}

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────

func run(name string, args ...string) error {
	cmd := exec.Command(name, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func runOutput(name string, args ...string) (string, error) {
	out, err := exec.Command(name, args...).Output()
	return string(out), err
}

func copyFile(src, dst string) {
	data, err := os.ReadFile(src)
	if err != nil {
		return
	}
	os.WriteFile(dst, data, 0644)
}
