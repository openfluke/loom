package main

import (
	"fmt"
	"net/http"
	"os"
	"path/filepath"
)

func main() {
	// Resolve paths: server/ → wasm/ → typescript/dist/
	wd, _ := os.Getwd()
	wasmDir := filepath.Join(wd, "..")
	distDir := filepath.Join(wd, "..", "..", "typescript", "dist")

	port := "3000"
	if p := os.Getenv("PORT"); p != "" {
		port = p
	}

	mux := http.NewServeMux()

	// Index
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			// Fall through to static file serving
			http.ServeFile(w, r, filepath.Join(wasmDir, filepath.Clean(r.URL.Path)))
			return
		}

		_, wasmMissing := os.Stat(filepath.Join(distDir, "main.wasm"))

		w.Header().Set("Content-Type", "text/html")
		fmt.Fprint(w, `<!DOCTYPE html><html><head>
<meta charset="UTF-8">
<title>Loom WASM Test Suite</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0d0d0d; color: #c8c8c8; font-family: 'Courier New', monospace;
       font-size: 14px; padding: 40px; }
h1 { color: #7ec8e3; margin-bottom: 8px; font-size: 18px; }
.sub { color: #555; margin-bottom: 32px; font-size: 11px; }
a { display: block; color: #7ec8e3; text-decoration: none; padding: 10px 16px;
    border: 1px solid #1e3a4a; margin-bottom: 8px; max-width: 520px; }
a:hover { background: #1e3a4a; }
.warn { color: #e0a020; margin-top: 24px; font-size: 11px; padding: 10px;
        border: 1px solid #5a4010; max-width: 520px; }
</style></head><body>
<h1>=== Loom WASM Test Suite ===</h1>
<div class="sub">Build main.wasm first: <strong>cd ..  &amp;&amp;  build.bat</strong> (Windows) or <strong>./build.sh</strong> (Linux/Mac)</div>
<a href="/cabi_verify.html">C-ABI Verify — all WASM exports + smoke tests</a>
<a href="/benchmark_tiling.html">Benchmark: Tiling — forward pass (11 layer types)</a>
<a href="/benchmark_training.html">Benchmark: Training — forward + train (9 layer types)</a>
<a href="/benchmark_training_comparison.html">Benchmark: Training Comparison — 6 architectures, 20 epochs</a>
<a href="/dna_evo_benchmark.html">Benchmark: DNA &amp; Evolution — splice, NEAT, population</a>
`)
		if os.IsNotExist(wasmMissing) {
			fmt.Fprint(w, `<div class="warn">⚠ main.wasm not found in typescript/dist/ — run build.bat or build.sh first.</div>`)
		}
		fmt.Fprint(w, `</body></html>`)
	})

	// main.wasm with correct MIME type
	mux.HandleFunc("/main.wasm", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/wasm")
		http.ServeFile(w, r, filepath.Join(distDir, "main.wasm"))
	})

	// wasm_exec.js — prefer dist, fall back to wasm dir
	mux.HandleFunc("/wasm_exec.js", func(w http.ResponseWriter, r *http.Request) {
		path := filepath.Join(distDir, "wasm_exec.js")
		if _, err := os.Stat(path); os.IsNotExist(err) {
			path = filepath.Join(wasmDir, "wasm_exec.js")
		}
		w.Header().Set("Content-Type", "application/javascript")
		http.ServeFile(w, r, path)
	})

	fmt.Printf("\n  Loom WASM Test Suite  →  http://localhost:%s\n", port)
	fmt.Printf("  HTML : %s\n", wasmDir)
	fmt.Printf("  WASM : %s\n\n", distDir)

	if err := http.ListenAndServe(":"+port, mux); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}
