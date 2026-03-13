package pods

import "errors"

// Single canonical error used across CPU/GPU builds.
var ErrNoGPU = errors.New("gpu unavailable (build with -tags=gpu to enable)")
