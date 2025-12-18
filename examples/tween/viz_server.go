package main

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"time"
)

const port = ":8000"

func main() {
	// Get the directory of the server
	vizDir, err := os.Getwd()
	if err != nil {
		log.Fatal(err)
	}

	// Data file path - now in the same directory
	resultsPath := filepath.Join(vizDir, "test18_results.json")

	// Check if results exist, if not run benchmark
	if _, err := os.Stat(resultsPath); os.IsNotExist(err) {
		fmt.Println("🚀 Results not found. Running Test 18 Adaptation Benchmark...")
		fmt.Println("This may take a few seconds as it runs multiple architectures in parallel.")

		cmd := exec.Command("go", "run", "test18_architecture_adaptation.go")
		cmd.Dir = vizDir
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			log.Fatalf("❌ Error running benchmark: %v", err)
		}

		if _, err := os.Stat(resultsPath); err == nil {
			fmt.Println("✅ Benchmark complete. Results saved to test18_results.json")
		} else {
			log.Fatal("❌ Benchmark failed or test18_results.json was not created.")
		}
	}

	// Serve static files
	fs := http.FileServer(http.Dir(vizDir))
	http.Handle("/", fs)

	// Serve the JSON file as well
	http.HandleFunc("/test18_results.json", func(w http.ResponseWriter, r *http.Request) {
		http.ServeFile(w, r, resultsPath)
	})

	fmt.Printf("🌐 Starting visualization server at http://localhost%s\n", port)
	fmt.Println("📊 Dashboard is live! Press Ctrl+C to stop.")

	// Open browser
	go func() {
		time.Sleep(1 * time.Second)
		openBrowser("http://localhost" + port)
	}()

	log.Fatal(http.ListenAndServe(port, nil))
}

func openBrowser(url string) {
	var err error
	switch runtime.GOOS {
	case "linux":
		err = exec.Command("xdg-open", url).Start()
	case "windows":
		err = exec.Command("rundll32", "url.dll,FileProtocolHandler", url).Start()
	case "darwin":
		err = exec.Command("open", url).Start()
	default:
		err = fmt.Errorf("unsupported platform")
	}
	if err != nil {
		fmt.Printf("Please open %s in your browser.\n", url)
	}
}
