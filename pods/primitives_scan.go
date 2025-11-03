package pods

import "errors"

type ScanIn struct {
	In        []uint32
	Inclusive bool
}
type ScanOut struct {
	Out []uint32
}

type ScanPod struct{}

func (ScanPod) Name() string { return "primitives/scan" }

func (ScanPod) Run(x *ExecContext, in any) (any, error) {
	args, ok := in.(ScanIn)
	if !ok {
		return nil, errors.New("ScanIn expected")
	}
	out := make([]uint32, len(args.In))
	if x.UseGPU && x.GPU != nil {
		// TODO: upload buffer + x.GPU.DispatchScanU32(...)
		// Fallback to CPU until kernels are wired:
	}
	// Blelloch-like two-pass or simple inclusive scan
	if args.Inclusive {
		var acc uint32
		for i, v := range args.In {
			acc += v
			out[i] = acc
		}
	} else {
		var acc uint32
		for i, v := range args.In {
			out[i] = acc
			acc += v
		}
	}
	return ScanOut{Out: out}, nil
}
