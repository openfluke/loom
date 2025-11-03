package pods

// Keep it lean; you can grow these later.

type Tensor[T ~float32 | ~float64] struct {
	Data    []T
	Shape   []int // row-major
	Strides []int
}

func NewTensor[T ~float32 | ~float64](shape ...int) Tensor[T] {
	n := 1
	for _, d := range shape {
		n *= d
	}
	strides := make([]int, len(shape))
	s := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = s
		s *= shape[i]
	}
	return Tensor[T]{Data: make([]T, n), Shape: shape, Strides: strides}
}

type Mesh struct {
	Positions []float32 // xyzxyz…
	Normals   []float32 // optional
	Indices   []uint32
}

type ImageFrame struct {
	W, H   int
	Pixels []float32 // RGB or RGBA planar/interleaved; keep it simple first
	Stride int
	Format string // "RGB", "RGBA", "YUV420", …
}

type AudioBuffer struct {
	SampleRate int
	Channels   int
	Samples    [][]float32 // [ch][n]
}
