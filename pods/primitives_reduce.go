package pods

import "errors"

type ReduceIn struct {
	In   []float32
	Kind string // "sum"|"min"|"max"
}
type ReduceOut struct {
	Value float32
}

type ReducePod struct{}

func (ReducePod) Name() string { return "primitives/reduce" }

func (ReducePod) Run(x *ExecContext, in any) (any, error) {
	args, ok := in.(ReduceIn)
	if !ok {
		return nil, errors.New("ReduceIn expected")
	}
	if len(args.In) == 0 {
		return ReduceOut{0}, nil
	}
	// GPU hook placeholder
	// if x.UseGPU && x.GPU != nil { return ReduceOut{v}, nil }
	switch args.Kind {
	case "sum":
		var s float32
		for _, v := range args.In {
			s += v
		}
		return ReduceOut{Value: s}, nil
	case "min":
		m := args.In[0]
		for _, v := range args.In[1:] {
			if v < m {
				m = v
			}
		}
		return ReduceOut{Value: m}, nil
	case "max":
		m := args.In[0]
		for _, v := range args.In[1:] {
			if v > m {
				m = v
			}
		}
		return ReduceOut{Value: m}, nil
	default:
		return nil, errors.New("unknown kind")
	}
}
