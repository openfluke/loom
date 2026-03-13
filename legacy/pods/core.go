package pods

import (
	"context"
	"time"

	"github.com/openfluke/loom/detector"
)

// Pod is a unit of work (scan, GEMM, culling, STFT, …).
type Pod interface {
	Name() string
	Run(ctx *ExecContext, in any) (out any, err error)
}

// ExecContext carries execution choices and capabilities.
type ExecContext struct {
	Ctx      context.Context
	UseGPU   bool             // high-level knob; pods may override per-op
	Report   *detector.Report // detector output (limits, features, recs)
	GPU      GPUHooks         // nil unless -tags=gpu and initialized
	TempPool *Pool            // scratch buffers (optional)
	Now      time.Time
}

// Pool can hand out scratch slices to avoid GC churn (optional).
type Pool struct{}

func NewContext(rep *detector.Report) *ExecContext {
	return &ExecContext{
		Ctx:    context.Background(),
		UseGPU: false,
		Report: rep,
		Now:    time.Now(),
	}
}

func (ec *ExecContext) WithGPU(g GPUHooks) *ExecContext {
	ec.GPU = g
	ec.UseGPU = g != nil
	return ec
}
