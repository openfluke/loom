package pods

import "errors"

type GEMMIn struct {
	M, N, K int
	A, B    []float32 // row-major A[M,K], B[K,N]
	Alpha   float32   // C = alpha*A*B + beta*C (beta=0 here for baseline)
}
type GEMMOut struct {
	C []float32 // row-major C[M,N]
}

type GEMMPod struct{}

func (GEMMPod) Name() string { return "ml/gemm" }

func (GEMMPod) Run(x *ExecContext, in any) (any, error) {
	args, ok := in.(GEMMIn)
	if !ok {
		return nil, errors.New("GEMMIn expected")
	}
	M, N, K := args.M, args.N, args.K
	if len(args.A) != M*K || len(args.B) != K*N {
		return nil, errors.New("bad shapes")
	}
	C := make([]float32, M*N)
	// TODO GPU: x.GPU.DispatchGEMM(M,N,K, ptrA, ptrB, ptrC, args.Alpha, 0)
	// CPU tiled (simple cache-friendly baseline)
	const TS = 64
	for i0 := 0; i0 < M; i0 += TS {
		for k0 := 0; k0 < K; k0 += TS {
			for j0 := 0; j0 < N; j0 += TS {
				iMax := min(i0+TS, M)
				kMax := min(k0+TS, K)
				jMax := min(j0+TS, N)
				for i := i0; i < iMax; i++ {
					for k := k0; k < kMax; k++ {
						ai := args.A[i*K+k] * args.Alpha
						rowC := i * N
						rowB := k * N
						for j := j0; j < jMax; j++ {
							C[rowC+j] += ai * args.B[rowB+j]
						}
					}
				}
			}
		}
	}
	return GEMMOut{C: C}, nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
