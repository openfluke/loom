package poly

import (
	"math"
	"math/rand"
	"runtime"
	"sync"
)

// KMeansCluster performs K-means clustering on a set of tensors.
func KMeansCluster[T Numeric](data []*Tensor[T], k int, maxIter int, parallel bool) (centroids [][]float32, assignments []int) {
	if len(data) == 0 || k <= 0 { return nil, nil }
	if k > len(data) { k = len(data) }

	dim := len(data[0].Data)
	centroids = make([][]float32, k)
	assignments = make([]int, len(data))

	// 1. Initialize centroids randomly
	perm := rand.Perm(len(data))
	for i := 0; i < k; i++ {
		centroids[i] = make([]float32, dim)
		for j, v := range data[perm[i]].Data {
			centroids[i][j] = float32(v)
		}
	}

	for iter := 0; iter < maxIter; iter++ {
		changes := 0
		var mu sync.Mutex

		// 2. Assign points to nearest centroid
		work := func(start, end int) {
			localChanges := 0
			for i := start; i < end; i++ {
				minDist := float32(math.MaxFloat32)
				best := 0
				for cIdx, c := range centroids {
					d := EuclideanDistance(data[i].Data, c)
					if d < minDist {
						minDist = d
						best = cIdx
					}
				}
				if assignments[i] != best {
					assignments[i] = best
					localChanges++
				}
			}
			mu.Lock()
			changes += localChanges
			mu.Unlock()
		}

		if parallel {
			var wg sync.WaitGroup
			n := runtime.NumCPU()
			chunk := (len(data) + n - 1) / n
			for i := 0; i < n; i++ {
				s := i * chunk
				e := s + chunk
				if e > len(data) { e = len(data) }
				if s >= e { break }
				wg.Add(1)
				go func(s, e int) { defer wg.Done(); work(s, e) }(s, e)
			}
			wg.Wait()
		} else {
			work(0, len(data))
		}

		if changes == 0 && iter > 0 { break }

		// 3. Update centroids
		counts := make([]int, k)
		newCentroids := make([][]float32, k)
		for i := range newCentroids { newCentroids[i] = make([]float32, dim) }

		for i, cIdx := range assignments {
			counts[cIdx]++
			for j, v := range data[i].Data {
				newCentroids[cIdx][j] += float32(v)
			}
		}

		for i := 0; i < k; i++ {
			if counts[i] > 0 {
				scale := 1.0 / float32(counts[i])
				for j := range centroids[i] {
					centroids[i][j] = newCentroids[i][j] * scale
				}
			} else {
				ridx := rand.Intn(len(data))
				for j, v := range data[ridx].Data {
					centroids[i][j] = float32(v)
				}
			}
		}
	}

	return centroids, assignments
}

// EuclideanDistance computes the distance between a Numeric slice and a float32 centroid.
func EuclideanDistance[T Numeric](a []T, b []float32) float32 {
	var sum float32
	n := len(a)
	if len(b) < n { n = len(b) }
	for i := 0; i < n; i++ {
		d := float32(a[i]) - b[i]
		sum += d * d
	}
	return float32(math.Sqrt(float64(sum)))
}

// ComputeSilhouetteScore calculates the mean Silhouette Coefficient of all samples.
func ComputeSilhouetteScore[T Numeric](data []*Tensor[T], assignments []int) float32 {
	if len(data) < 2 { return 0 }
	n := len(data)
	totalScore := float32(0)

	for i := 0; i < n; i++ {
		a := float32(0)
		b := float32(math.MaxFloat32)
		myCluster := assignments[i]
		sameCount := 0
		
		dists := make(map[int]float32)
		counts := make(map[int]int)

		for j := 0; j < n; j++ {
			if i == j { continue }
			d := EuclideanDistanceT(data[i].Data, data[j].Data)
			if assignments[j] == myCluster {
				a += d
				sameCount++
			} else {
				dists[assignments[j]] += d
				counts[assignments[j]]++
			}
		}

		if sameCount > 0 { a /= float32(sameCount) }
		for id, sum := range dists {
			mean := sum / float32(counts[id])
			if mean < b { b = mean }
		}
		if len(dists) == 0 { b = 0 }

		maxAB := a
		if b > a { maxAB = b }
		if maxAB > 0 { totalScore += (b - a) / maxAB }
	}

	return totalScore / float32(n)
}

// CosineDistance computes the semantic distance (1 - cosine similarity) between vectors.
func CosineDistance[T Numeric](a []T, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 { return 1.0 }
	
	dot := 0.0
	normA := 0.0
	normB := 0.0
	
	for i := range a {
		va := float64(a[i])
		vb := float64(b[i])
		dot += va * vb
		normA += va * va
		normB += vb * vb
	}
	
	if normA == 0 || normB == 0 { return 1.0 }
	sim := dot / (math.Sqrt(normA) * math.Sqrt(normB))
	return float32(1.0 - sim)
}

// HierarchicalGroup performs a simple agglomerative grouping until a distance threshold is met.
func HierarchicalGroup[T Numeric](data []*Tensor[T], threshold float32) []int {
	n := len(data)
	assignments := make([]int, n)
	for i := range assignments { assignments[i] = i }
	
	for {
		minDist := float32(math.MaxFloat32)
		u, v := -1, -1
		
		for i := 0; i < n; i++ {
			for j := i + 1; j < n; j++ {
				if assignments[i] == assignments[j] { continue }
				d := EuclideanDistanceT(data[i].Data, data[j].Data)
				if d < minDist {
					minDist = d
					u, v = i, j
				}
			}
		}
		
		if u == -1 || minDist > threshold { break }
		
		// Merge cluster of v into u
		oldV := assignments[v]
		newU := assignments[u]
		for i := range assignments {
			if assignments[i] == oldV { assignments[i] = newU }
		}
	}
	return assignments
}

// EuclideanDistanceT computes distance between two Numeric slices.
func EuclideanDistanceT[T Numeric](a, b []T) float32 {
	var sum float32
	n := len(a)
	if len(b) < n { n = len(b) }
	for i := 0; i < n; i++ {
		d := float32(a[i]) - float32(b[i])
		sum += d * d
	}
	return float32(math.Sqrt(float64(sum)))
}
