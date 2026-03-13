package pods

import "math"

type Frustum struct{ Planes [6][4]float32 } // ax+by+cz+d >= 0
type CullingIn struct {
	Frustum Frustum
	Bounds  [][6]float32 // AABB: xmin,ymin,zmin,xmax,ymax,zmax
}
type CullingOut struct{ Visible []bool }

type CullingPod struct{}

func (CullingPod) Name() string { return "render/cull_frustum" }

func (CullingPod) Run(_ *ExecContext, in any) (any, error) {
	a, _ := in.(CullingIn)
	out := make([]bool, len(a.Bounds))
	for i, b := range a.Bounds {
		// AABB vs frustum: for each plane, test the most positive vertex.
		xmin, ymin, zmin, xmax, ymax, zmax := b[0], b[1], b[2], b[3], b[4], b[5]
		vis := true
		for _, p := range a.Frustum.Planes {
			ax, by, cz, d := p[0], p[1], p[2], p[3]
			px := xmax
			if ax < 0 {
				px = xmin
			}
			py := ymax
			if by < 0 {
				py = ymin
			}
			pz := zmax
			if cz < 0 {
				pz = zmin
			}
			if ax*px+by*py+cz*pz+d < 0 {
				vis = false
				break
			}
		}
		out[i] = vis
	}
	_ = math.E // silence import in case you extend
	return CullingOut{Visible: out}, nil
}
