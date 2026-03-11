package pods

type RGBToYUVIn struct {
	Frame ImageFrame // Format "RGB"
}
type RGBToYUVOut struct {
	Y, U, V []float32 // planar 4:4:4 for simplicity
}

type RGBToYUVPod struct{}

func (RGBToYUVPod) Name() string { return "video/rgb_to_yuv444" }

func (RGBToYUVPod) Run(_ *ExecContext, in any) (any, error) {
	a := in.(RGBToYUVIn)
	n := a.Frame.W * a.Frame.H
	Y := make([]float32, n)
	U := make([]float32, n)
	V := make([]float32, n)
	p := a.Frame.Pixels
	for i := 0; i < n; i++ {
		r := p[3*i+0]
		g := p[3*i+1]
		b := p[3*i+2]
		Y[i] = 0.2126*r + 0.7152*g + 0.0722*b
		U[i] = -0.1146*r - 0.3854*g + 0.5000*b
		V[i] = 0.5000*r - 0.4542*g - 0.0458*b
	}
	return RGBToYUVOut{Y: Y, U: U, V: V}, nil
}
