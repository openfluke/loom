package snapdragon

import "github.com/openfluke/loom/poly/accel"

// ensureQnnEnv puts the QNN backend dirs on the process DLL search path so the
// plugin's LoadLibrary("QnnHtp.dll"/"QnnCpu.dll") resolves. Unlike the Intel/Linux
// LD_LIBRARY_PATH case, Windows consults PATH at LoadLibrary time, so no re-exec
// is needed — PrepareQualcommRuntime mutates the in-process PATH directly.
func ensureQnnEnv() {
	_ = accel.PrepareQualcommRuntime()
	// LOOM_QNN_VERBOSE=1 shows QNN/HTP warnings on the terminal (off by default).
}
