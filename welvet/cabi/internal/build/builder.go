package main

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
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

// allPlatforms is the canonical full target list (64-bit only; no iOS simulator slices).
var allPlatforms = []Platform{
	// ── Desktop / Server ──────────────────────────────────────────────────────
	{GOOS: "linux", GOARCH: "amd64", DirName: "linux_amd64", BuildMode: "c-shared"},
	{GOOS: "linux", GOARCH: "arm64", DirName: "linux_arm64", BuildMode: "c-shared"},
	{GOOS: "darwin", GOARCH: "amd64", DirName: "macos_amd64", BuildMode: "c-shared"},
	{GOOS: "darwin", GOARCH: "arm64", DirName: "macos_arm64", BuildMode: "c-shared"},
	{GOOS: "windows", GOARCH: "amd64", DirName: "windows_amd64", BuildMode: "c-shared"},
	{GOOS: "windows", GOARCH: "arm64", DirName: "windows_arm64", BuildMode: "c-shared"},
	// ── Android (64-bit) ───────────────────────────────────────────────────────
	{GOOS: "android", GOARCH: "arm64", DirName: "android_arm64", BuildMode: "c-shared"},
	{GOOS: "android", GOARCH: "amd64", DirName: "android_x86_64", BuildMode: "c-shared"},
	// ── iOS device (static archive) ─────────────────────────────────────────────
	{GOOS: "ios", GOARCH: "arm64", DirName: "ios_arm64", BuildMode: "c-archive"},
}

// ──────────────────────────────────────────────────────────────────────────────
// CLI entry point
// ──────────────────────────────────────────────────────────────────────────────

func main() {
	targetOS := flag.String("os", runtime.GOOS, "Target OS: windows|linux|darwin|android|ios|all")
	targetArch := flag.String("arch", runtime.GOARCH, "Target arch: amd64|arm64|universal|xcframework")
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
// Lucy (CLI) — same GOOS/GOARCH/CC as the C-ABI slice; output next to welvet.*
// ──────────────────────────────────────────────────────────────────────────────

func lucyModuleDir() (string, error) {
	return filepath.Abs(filepath.Join("..", "..", "..", "..", "lucy"))
}

func lucyExeName(goos string) string {
	if goos == "windows" {
		return "lucy.exe"
	}
	return "lucy"
}

// buildLucyInto builds the lucy main package into outPath (e.g. dist/macos_arm64/lucy).
// cc is the C compiler for CGO (webgpu); empty uses the default CC for the host.
func buildLucyInto(p Platform, outPath, cc string) error {
	lucyDir, err := lucyModuleDir()
	if err != nil {
		return err
	}
	if _, err := os.Stat(filepath.Join(lucyDir, "go.mod")); err != nil {
		return fmt.Errorf("lucy module: %w", err)
	}
	absOut, err := filepath.Abs(outPath)
	if err != nil {
		return err
	}
	// -o is resolved relative to cmd.Dir (lucy module root); use an absolute path
	// so binaries land in dist/<slice>/ next to welvet, not under lucy/dist/.
	outExe := filepath.Join(absOut, lucyExeName(p.GOOS))
	fmt.Printf("  compiling lucy → %s...\n", filepath.Base(outExe))
	cmd := exec.Command("go", "build", "-o", outExe, ".")
	cmd.Dir = lucyDir
	env := append(cleanBuildEnv(),
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
	if p.GOOS == "windows" && p.GOARCH == "arm64" {
		env = append(env, "CGO_LDFLAGS=-loleaut32 -lole32 -luuid")
	}
	cmd.Env = env
	out, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Printf("%s\n", string(out))
		return fmt.Errorf("lucy: %w", err)
	}
	fmt.Printf("  ✓  %s\n", filepath.Join(outPath, lucyExeName(p.GOOS)))
	return nil
}

// ──────────────────────────────────────────────────────────────────────────────
// Platform selection
// ──────────────────────────────────────────────────────────────────────────────

func selectPlatforms(goos, arch string) []Platform {
	if goos == "ios" {
		switch arch {
		case "xcframework", "universal":
			return nil // handled at call site
		case "arm64", "":
			return []Platform{{GOOS: "ios", GOARCH: "arm64", DirName: "ios_arm64", BuildMode: "c-archive"}}
		}
	}

	// Normalise arch aliases
	switch arch {
	case "x86_64":
		arch = "amd64"
	case "aarch64":
		arch = "arm64"
	case "universal", "xcframework":
		return nil // handled at call site
	}

	for _, p := range allPlatforms {
		if p.GOOS == goos && p.GOARCH == arch && p.GOARM == "" {
			return []Platform{p}
		}
	}

	// Fallback
	p := Platform{
		GOOS:      goos,
		GOARCH:    arch,
		DirName:   goos + "_" + arch,
		BuildMode: "c-shared",
	}
	if goos == "ios" {
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
	if p.GOOS == "windows" && cc == "" {
		return fmt.Errorf("windows/%s: no mingw cross-compiler (install llvm-mingw to ~/llvm-mingw and set LLVM_MINGW_HOME in .build_env)", p.GOARCH)
	}

	cmd := exec.Command("go", "build",
		"-buildmode="+p.BuildMode,
		"-o", outFile,
		"../../",
	)

	env := append(cleanBuildEnv(),
		"GOOS="+p.GOOS,
		"GOARCH="+p.GOARCH,
		"CGO_ENABLED=1",
	)
	if p.GOARM != "" {
		env = append(env, "GOARM="+p.GOARM)
	}
	if cc != "" {
		// CC may contain flags (e.g. "clang -arch x86_64"); pass as-is — the
		// Go toolchain splits it correctly when it invokes the C compiler.
		env = append(env, "CC="+cc)
	}
	if p.GOOS == "windows" && p.GOARCH == "arm64" {
		env = append(env, "CGO_LDFLAGS=-loleaut32 -lole32 -luuid")
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
	if err := buildLucyInto(p, outPath, cc); err != nil {
		return err
	}
	bundleWindowsRuntime(p, outPath, cc)
	return nil
}

// bundleWindowsRuntime copies MinGW/UCRT DLLs required at load time (e.g. libunwind.dll
// from llvm-mingw). Without these, lucy.exe / welvet.dll exit immediately on Windows
// with no console output when the loader cannot resolve imports.
func bundleWindowsRuntime(p Platform, outPath, cc string) {
	if p.GOOS != "windows" {
		return
	}
	for _, name := range []string{"libunwind.dll"} {
		src := findWindowsRuntimeDLL(p, cc, name)
		if src == "" {
			fmt.Printf("  ⚠  %s not found — Windows builds need it beside lucy.exe (llvm-mingw …/bin)\n", name)
			continue
		}
		dst := filepath.Join(outPath, name)
		if err := copyFileErr(src, dst); err != nil {
			fmt.Printf("  ⚠  copy %s: %v\n", name, err)
			continue
		}
		fmt.Printf("  ✓  %s (Windows runtime)\n", name)
	}
}

func findWindowsRuntimeDLL(p Platform, cc, name string) string {
	try := func(path string) string {
		path = filepath.Clean(path)
		if st, err := os.Stat(path); err == nil && !st.IsDir() {
			return path
		}
		return ""
	}

	// llvm-mingw layout: $ROOT/<triple>/bin/libunwind.dll
	triple := "x86_64-w64-mingw32"
	if p.GOARCH == "arm64" {
		triple = "aarch64-w64-mingw32"
	}
	var llvmRoots []string
	if v := strings.TrimSpace(os.Getenv("LLVM_MINGW_HOME")); v != "" {
		llvmRoots = append(llvmRoots, v)
	}
	if h, err := os.UserHomeDir(); err == nil {
		llvmRoots = append(llvmRoots, filepath.Join(h, "llvm-mingw"))
	}
	seen := map[string]struct{}{}
	for _, root := range llvmRoots {
		root = filepath.Clean(root)
		if root == "" {
			continue
		}
		if _, ok := seen[root]; ok {
			continue
		}
		seen[root] = struct{}{}
		if p := try(filepath.Join(root, triple, "bin", name)); p != "" {
			return p
		}
		if p := try(filepath.Join(root, "bin", name)); p != "" {
			return p
		}
	}

	if cc != "" {
		compilerBin, _ := splitCC(cc)
		if compilerBin != "" {
			if p := try(filepath.Join(filepath.Dir(compilerBin), name)); p != "" {
				return p
			}
			if out, err := exec.Command(compilerBin, "-print-file-name="+name).Output(); err == nil {
				if p := try(strings.TrimSpace(string(out))); p != "" {
					return p
				}
			}
		}
	}
	return ""
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
	var compilerFlags []string

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
		compilerBin, compilerFlags = splitCC(cc)
		libFile := filepath.Join(outPath, "welvet.dll")
		args = []string{"-I" + testDir, "-I" + outPath, verifySrc, "-o", verifyExe, libFile}

	case "android":
		if cc == "" {
			fmt.Printf("  ⚠  cabi_verify: no NDK CC available, skipping\n")
			return
		}
		compilerBin, compilerFlags = splitCC(cc)
		libFile := filepath.Join(outPath, "welvet.so")
		absLib, _ := filepath.Abs(libFile)
		absOut, _ := filepath.Abs(outPath)
		args = []string{"-I" + testDir, "-I" + absOut, verifySrc, "-o", verifyExe,
			"-L" + absOut, absLib, "-lm", "-pie"}

	default: // linux, darwin (including cross-arch darwin builds)
		if cc == "" {
			cc = "clang"
			if _, err := exec.LookPath("clang"); err != nil {
				cc = "gcc"
			}
		}
		compilerBin, compilerFlags = splitCC(cc)
		args = []string{"-I" + testDir, "-I" + outPath, verifySrc, "-o", verifyExe, "-ldl"}
	}

	fmt.Printf("  compiling cabi_verify...\n")
	cmdArgs := append(compilerFlags, args...)
	out, err := exec.Command(compilerBin, cmdArgs...).CombinedOutput()
	if err != nil {
		fmt.Printf("  ⚠  cabi_verify compile failed: %s\n", strings.TrimSpace(string(out)))
		return
	}
	fmt.Printf("  ✓  %s\n", verifyExe)
}

// firstPathCC returns the first name found on PATH (full path), or "".
func firstPathCC(names ...string) string {
	for _, n := range names {
		if p, err := exec.LookPath(n); err == nil {
			return p
		}
	}
	return ""
}

// firstLLVMMinGWCC finds a Windows cross-compiler under LLVM_MINGW_HOME (not global PATH).
func firstLLVMMinGWCC(names ...string) string {
	var roots []string
	if v := strings.TrimSpace(os.Getenv("LLVM_MINGW_HOME")); v != "" {
		roots = append(roots, v)
	}
	if h, err := os.UserHomeDir(); err == nil {
		roots = append(roots, filepath.Join(h, "llvm-mingw"))
	}
	seen := map[string]struct{}{}
	for _, root := range roots {
		root = filepath.Clean(root)
		if root == "" {
			continue
		}
		if _, ok := seen[root]; ok {
			continue
		}
		seen[root] = struct{}{}
		bin := filepath.Join(root, "bin")
		for _, name := range names {
			p := filepath.Join(bin, name)
			if st, err := os.Stat(p); err == nil && !st.IsDir() {
				return p
			}
		}
	}
	return firstPathCC(names...)
}

func cleanBuildEnv() []string {
	out := make([]string, 0, len(os.Environ()))
	for _, e := range os.Environ() {
		if strings.HasPrefix(e, "CC=") || strings.HasPrefix(e, "CGO_LDFLAGS=") {
			continue
		}
		out = append(out, e)
	}
	return out
}

// crossCC returns the C compiler (and any required flags) needed to
// cross-compile for p.  Returns "" to use the default CC in the environment.
// NOTE: the returned string may contain spaces (e.g. "clang -arch x86_64");
// callers must split it with splitCC() before passing to exec.Command.
func crossCC(p Platform) string {
	host := runtime.GOOS
	hostArch := runtime.GOARCH

	switch p.GOOS {
	case "linux":
		switch p.GOARCH {
		case "amd64":
			if host == "linux" && hostArch == "amd64" {
				return ""
			}
			if cc := firstPathCC("x86_64-linux-gnu-gcc", "x86_64-unknown-linux-gnu-gcc"); cc != "" {
				return cc
			}
			return "x86_64-linux-gnu-gcc"
		case "arm64":
			if host == "linux" && hostArch == "arm64" {
				return ""
			}
			if cc := firstPathCC("aarch64-linux-gnu-gcc", "aarch64-unknown-linux-gnu-gcc"); cc != "" {
				return cc
			}
			return "aarch64-linux-gnu-gcc"
		}
	case "darwin":
		if host == "darwin" {
			return darwinCC(p, hostArch)
		}
	case "windows":
		switch p.GOARCH {
		case "amd64":
			if cc := firstPathCC("x86_64-w64-mingw32-gcc"); cc != "" {
				return cc
			}
			return "x86_64-w64-mingw32-gcc"
		case "arm64":
			if cc := firstLLVMMinGWCC(
				"aarch64-w64-mingw32-clang",
				"aarch64-w64-mingw32-gcc",
			); cc != "" {
				return cc
			}
			return ""
		}
	case "android":
		return androidCC(p)
	}
	return ""
}

// darwinCC returns macOS SDK clang with -isysroot (and -arch when cross-compiling on Apple Silicon/Intel).
func darwinCC(p Platform, hostArch string) string {
	sdkPath, err := runOutput("xcrun", "--show-sdk-path")
	if err != nil {
		sdkPath = ""
	}
	clangBin, err := runOutput("xcrun", "--find", "clang")
	if err != nil {
		clangBin = "/usr/bin/clang"
	}
	cc := strings.TrimSpace(clangBin)
	if sdkPath != "" {
		cc += " -isysroot " + strings.TrimSpace(sdkPath)
	}
	if p.GOARCH != hostArch {
		clangArch := p.GOARCH
		if clangArch == "amd64" {
			clangArch = "x86_64"
		}
		cc += " -arch " + clangArch
	}
	return cc
}

// splitCC breaks a CC string like "clang -arch x86_64" into ("clang", ["-arch","x86_64"]).
func splitCC(cc string) (string, []string) {
	parts := strings.Fields(cc)
	if len(parts) == 0 {
		return "", nil
	}
	return parts[0], parts[1:]
}

// ──────────────────────────────────────────────────────────────────────────────
// Android NDK helpers
// ──────────────────────────────────────────────────────────────────────────────

const androidAPILevel = "21"

func isValidNDKRoot(dir string) bool {
	if dir == "" {
		return false
	}
	st, err := os.Stat(filepath.Join(dir, "toolchains", "llvm", "prebuilt"))
	return err == nil && st.IsDir()
}

// resolveAndroidNDKRoot finds an NDK install: explicit env, then
// ANDROID_HOME / ANDROID_SDK_ROOT (ndk/<ver>, ndk-bundle), default SDK dirs per OS, Homebrew.
func resolveAndroidNDKRoot() string {
	for _, key := range []string{"ANDROID_NDK_HOME", "NDK_HOME", "ANDROID_NDK_ROOT"} {
		if v := strings.TrimSpace(os.Getenv(key)); v != "" && isValidNDKRoot(v) {
			return v
		}
	}
	seen := map[string]struct{}{}
	sdkRoots := []string{
		strings.TrimSpace(os.Getenv("ANDROID_HOME")),
		strings.TrimSpace(os.Getenv("ANDROID_SDK_ROOT")),
	}
	if h, err := os.UserHomeDir(); err == nil && h != "" {
		switch runtime.GOOS {
		case "darwin":
			sdkRoots = append(sdkRoots, filepath.Join(h, "Library", "Android", "sdk"))
		case "windows":
			sdkRoots = append(sdkRoots, filepath.Join(h, "AppData", "Local", "Android", "sdk"))
		default: // linux, freebsd, …
			sdkRoots = append(sdkRoots, filepath.Join(h, "Android", "sdk"))
		}
	}
	for _, root := range sdkRoots {
		if root == "" {
			continue
		}
		root = filepath.Clean(root)
		if _, dup := seen[root]; dup {
			continue
		}
		seen[root] = struct{}{}
		if isValidNDKRoot(filepath.Join(root, "ndk-bundle")) {
			return filepath.Join(root, "ndk-bundle")
		}
		ndks, _ := filepath.Glob(filepath.Join(root, "ndk", "*"))
		if best := newestValidNDKDir(ndks); best != "" {
			return best
		}
	}
	homebrew := strings.TrimSpace(os.Getenv("HOMEBREW_PREFIX"))
	candidates := []string{
		"/opt/homebrew/share/android-ndk",
		"/usr/local/share/android-ndk",
	}
	if homebrew != "" {
		candidates = append(candidates,
			filepath.Join(homebrew, "share", "android-ndk"),
			filepath.Join(homebrew, "opt", "android-ndk"),
		)
	}
	for _, p := range candidates {
		if isValidNDKRoot(p) {
			return p
		}
	}
	return ""
}

func newestValidNDKDir(paths []string) string {
	var good []string
	for _, p := range paths {
		if isValidNDKRoot(p) {
			good = append(good, p)
		}
	}
	if len(good) == 0 {
		return ""
	}
	sort.Slice(good, func(i, j int) bool {
		return compareNDKVersion(filepath.Base(good[i]), filepath.Base(good[j])) > 0
	})
	return good[0]
}

func compareNDKVersion(a, b string) int {
	pa := strings.Split(a, ".")
	pb := strings.Split(b, ".")
	n := len(pa)
	if len(pb) > n {
		n = len(pb)
	}
	for i := 0; i < n; i++ {
		var na, nb int
		if i < len(pa) {
			na, _ = strconv.Atoi(strings.TrimLeft(pa[i], "v"))
		}
		if i < len(pb) {
			nb, _ = strconv.Atoi(strings.TrimLeft(pb[i], "v"))
		}
		if na != nb {
			return na - nb
		}
	}
	return strings.Compare(a, b)
}

func androidCC(p Platform) string {
	ndkPath := resolveAndroidNDKRoot()
	if ndkPath == "" {
		fmt.Println("  ⚠  Android NDK not found (set ANDROID_NDK_HOME or install NDK under ANDROID_HOME/ndk/<ver>)")
		return ""
	}

	prebuilt := ndkPrebuilt(ndkPath)
	if prebuilt == "" {
		fmt.Printf("  ⚠  NDK has no llvm prebuilt for this host under %s\n", ndkPath)
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
	prebuiltDir := filepath.Join(ndkPath, "toolchains", "llvm", "prebuilt")
	ents, err := os.ReadDir(prebuiltDir)
	if err != nil {
		return ""
	}
	names := make([]string, 0, len(ents))
	for _, e := range ents {
		if e.IsDir() {
			names = append(names, e.Name())
		}
	}
	if len(names) == 0 {
		return ""
	}
	// Prefer a stable order for common hosts; otherwise use any available prebuilt.
	preferred := []string{
		"darwin-arm64", "darwin-x86_64",
		"linux-aarch64", "linux-x86_64",
		"windows-x86_64",
	}
	for _, want := range preferred {
		for _, got := range names {
			if got == want {
				return got
			}
		}
	}
	return names[0]
}

// ──────────────────────────────────────────────────────────────────────────────
// iOS build (c-archive + xcrun)
// ──────────────────────────────────────────────────────────────────────────────

func buildIOS(p Platform, outBase string) error {
	if runtime.GOOS != "darwin" {
		return fmt.Errorf("iOS builds require macOS")
	}

	fmt.Printf("\n--- Building %s ---\n", p.DirName)

	// Device-only (iphoneos); simulator slices are not part of this matrix.
	sdk := "iphoneos"
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

	clangTarget := clangArch + "-apple-ios13.0"
	cc := strings.TrimSpace(clangBin) + " -isysroot " + strings.TrimSpace(sdkPath) +
		" -target " + clangTarget

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

	out, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Printf("%s\n", string(out))
		return fmt.Errorf("go build ios: %w", err)
	}

	fmt.Printf("  ✓  %s\n", outFile)
	if err := buildLucyInto(p, outPath, cc); err != nil {
		return err
	}
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

	device := Platform{GOOS: "ios", GOARCH: "arm64", DirName: "ios_arm64", BuildMode: "c-archive"}
	deviceLib := filepath.Join(outBase, device.DirName, "welvet.a")
	if _, err := os.Stat(deviceLib); err != nil {
		if err2 := buildIOS(device, outBase); err2 != nil {
			return fmt.Errorf("ios_arm64: %w", err2)
		}
	}

	deviceHeaders := filepath.Join(outBase, "ios_arm64")
	xcfOut := filepath.Join(outBase, "ios_xcframework", "Welvet.xcframework")

	if _, err := os.Stat(deviceLib); err != nil {
		return fmt.Errorf("device library not found: %s", deviceLib)
	}

	os.MkdirAll(filepath.Dir(xcfOut), 0755)
	os.RemoveAll(xcfOut)

	args := []string{
		"-create-xcframework",
		"-library", deviceLib, "-headers", deviceHeaders,
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

	// Auto-build any missing slice
	slices := []Platform{
		{GOOS: "darwin", GOARCH: "amd64", DirName: "macos_amd64", BuildMode: "c-shared"},
		{GOOS: "darwin", GOARCH: "arm64", DirName: "macos_arm64", BuildMode: "c-shared"},
	}
	for _, p := range slices {
		lib := filepath.Join(outBase, p.DirName, "welvet.dylib")
		lucyBin := filepath.Join(outBase, p.DirName, lucyExeName("darwin"))
		if _, err := os.Stat(lib); err != nil {
			if err2 := buildPlatform(p, outBase); err2 != nil {
				return fmt.Errorf("building %s slice: %w", p.DirName, err2)
			}
		} else if _, err := os.Stat(lucyBin); err != nil {
			sliceOut := filepath.Join(outBase, p.DirName)
			if err := buildLucyInto(p, sliceOut, crossCC(p)); err != nil {
				return fmt.Errorf("lucy %s: %w", p.DirName, err)
			}
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

	lucyAMD := filepath.Join(outBase, "macos_amd64", lucyExeName("darwin"))
	lucyARM := filepath.Join(outBase, "macos_arm64", lucyExeName("darwin"))
	fatLucy := filepath.Join(outPath, lucyExeName("darwin"))
	if _, err1 := os.Stat(lucyAMD); err1 == nil {
		if _, err2 := os.Stat(lucyARM); err2 == nil {
			if err := run("lipo", "-create", lucyAMD, lucyARM, "-output", fatLucy); err != nil {
				return err
			}
			fmt.Printf("  ✓  %s\n", fatLucy)
		}
	}

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
		fmt.Println("💡 Missing cross-compilers / NDK — install or set:")
		switch runtime.GOOS {
		case "darwin":
			fmt.Println("   Linux amd64 + arm64 (cross from macOS):")
			fmt.Println("     brew tap messense/macos-cross-toolchains")
			fmt.Println("     brew install x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu")
			fmt.Println("     # adds x86_64-linux-gnu-gcc and aarch64-linux-gnu-gcc to PATH")
			fmt.Println("   Windows amd64:")
			fmt.Println("     brew install mingw-w64")
			fmt.Println("   Windows arm64 (mingw-w64 formula has no aarch64 GCC — use llvm-mingw):")
			fmt.Println("     https://github.com/mstorsjo/llvm-mingw/releases")
			fmt.Println("     Download *-macos-* (or *-macos-universal-*), unpack, set LLVM_MINGW_HOME=/path/to/llvm-mingw")
			fmt.Println("     (builder copies libunwind.dll into dist/windows_* for lucy.exe / welvet.dll)")
		default:
			fmt.Println("   Debian/Ubuntu cross packages:")
			fmt.Println("     sudo apt install gcc-aarch64-linux-gnu gcc-x86-64-linux-gnu \\")
			fmt.Println("       mingw-w64 gcc-mingw-w64-aarch64")
		}
		fmt.Println("   Android (Studio): export ANDROID_HOME=…  # builder uses newest $ANDROID_HOME/ndk/<ver> if unset")
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
	_ = copyFileErr(src, dst)
}

func copyFileErr(src, dst string) error {
	data, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	return os.WriteFile(dst, data, 0755)
}
