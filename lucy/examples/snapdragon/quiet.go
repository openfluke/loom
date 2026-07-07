package snapdragon

import (
	"io"
	"os"
	"strings"
)

// quietQnnOutput hides QNN/HTP chatter on the terminal unless LOOM_QNN_VERBOSE is set.
// Bench output still lands in lucy_testing_output/snapdragon.txt via BeginSession.
func quietQnnOutput() func() {
	if os.Getenv("LOOM_QNN_VERBOSE") != "" {
		return func() {}
	}
	orig := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		return func() {}
	}
	os.Stdout = w
	done := make(chan struct{})
	go func() {
		defer close(done)
		buf := make([]byte, 8192)
		for {
			n, e := r.Read(buf)
			if n > 0 {
				for _, line := range strings.Split(string(buf[:n]), "\n") {
					if line == "" {
						continue
					}
					if qnnNoiseLine(line) {
						continue
					}
					_, _ = io.WriteString(orig, line+"\n")
				}
			}
			if e != nil {
				break
			}
		}
	}()
	return func() {
		_ = w.Close()
		<-done
		_ = r.Close()
		os.Stdout = orig
	}
}

func qnnNoiseLine(line string) bool {
	s := strings.TrimSpace(line)
	if s == "" {
		return true
	}
	prefixes := []string{
		"<W>", "<E>",
		"Starting stage:",
		"Completed stage:",
		"DspTransport",
		"HtpProvider",
		"IDspTransport",
		"Failed to load skel",
		"Traditional path not available",
		"Incompatible profiling event",
		"HTP user driver is loaded",
		"PrepareLibLoader",
		"Hexagon option:",
		"Cannot find HTP_USR_DRV_GRAPH_CONFIG",
		"Hardware info config array",
		"Sanitizing the value for hvx_threads",
		"Initializing HtpProvider",
		"PrepareLibLoader",
	}
	for _, p := range prefixes {
		if strings.Contains(s, p) {
			return true
		}
	}
	return false
}
