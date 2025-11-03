package pods

import "container/heap"

type GridMap struct {
	W, H  int
	Solid func(x, y int) bool
	Cost  func(x, y int) float32 // optional; nil -> 1
}
type AStarIn struct {
	Map         GridMap
	Start, Goal [2]int
}
type AStarOut struct{ Path [][2]int }

type node struct {
	x, y   int
	g, h   float32
	parent *node
	idx    int
}
type pq []*node

func (p pq) Len() int           { return len(p) }
func (p pq) Less(i, j int) bool { return p[i].g+p[i].h < p[j].g+p[j].h }
func (p *pq) Swap(i, j int)     { (*p)[i], (*p)[j] = (*p)[j], (*p)[i]; (*p)[i].idx = i; (*p)[j].idx = j }
func (p *pq) Push(x any)        { *p = append(*p, x.(*node)) }
func (p *pq) Pop() any          { old := *p; x := old[len(old)-1]; *p = old[:len(old)-1]; return x }

type AStarPod struct{}

func (AStarPod) Name() string { return "ai/astar" }

func (AStarPod) Run(_ *ExecContext, in any) (any, error) {
	a := in.(AStarIn)
	W, H := a.Map.W, a.Map.H
	sx, sy := a.Start[0], a.Start[1]
	gx, gy := a.Goal[0], a.Goal[1]
	open := &pq{}
	heap.Init(open)
	start := &node{x: sx, y: sy}
	heap.Push(open, start)
	vis := make(map[[2]int]*node)
	vis[[2]int{sx, sy}] = start
	neighbors := [][2]int{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}
	hfun := func(x, y int) float32 {
		dx := x - gx
		dy := y - gy
		if dx < 0 {
			dx = -dx
		}
		if dy < 0 {
			dy = -dy
		}
		return float32(dx + dy) // manhattan
	}
	for open.Len() > 0 {
		cur := heap.Pop(open).(*node)
		if cur.x == gx && cur.y == gy {
			var path [][2]int
			for n := cur; n != nil; n = n.parent {
				path = append(path, [2]int{n.x, n.y})
			}
			for i, j := 0, len(path)-1; i < j; i, j = i+1, j-1 {
				path[i], path[j] = path[j], path[i]
			}
			return AStarOut{Path: path}, nil
		}
		for _, d := range neighbors {
			nx, ny := cur.x+d[0], cur.y+d[1]
			if nx < 0 || ny < 0 || nx >= W || ny >= H {
				continue
			}
			if a.Map.Solid != nil && a.Map.Solid(nx, ny) {
				continue
			}
			step := float32(1.0)
			if a.Map.Cost != nil {
				step = a.Map.Cost(nx, ny)
			}
			g := cur.g + step
			key := [2]int{nx, ny}
			if v, ok := vis[key]; ok && g >= v.g {
				continue
			}
			n := &node{x: nx, y: ny, g: g, h: hfun(nx, ny), parent: cur}
			vis[key] = n
			heap.Push(open, n)
		}
	}
	return AStarOut{Path: nil}, nil
}
