package nn

import (
	"math"
	"math/rand"
	"runtime"
	"sync"
)

// KMeansCluster performs K-means clustering on the provided data vectors.
// data: Slice of feature vectors (e.g., performance masks converted to floats)
// k: Number of clusters
// maxIter: Maximum number of iterations
// parallel: If true, uses all available CPUs for the assignment step
// Returns:
// - centroids: The final cluster centers
// - assignments: Cluster index for each data point
func KMeansCluster(data [][]float32, k int, maxIter int, parallel bool) (centroids [][]float32, assignments []int) {
	if len(data) == 0 || k <= 0 {
		return nil, nil
	}
	if k > len(data) {
		k = len(data)
	}

	dim := len(data[0])
	centroids = make([][]float32, k)
	assignments = make([]int, len(data))

	// 1. Initialize centroids randomly (Forgy method)
	perm := rand.Perm(len(data))
	for i := 0; i < k; i++ {
		centroids[i] = make([]float32, dim)
		copy(centroids[i], data[perm[i]])
	}

	for iter := 0; iter < maxIter; iter++ {
		changes := 0
		var changesMux sync.Mutex

		// 2. Assign points to nearest centroid
		if parallel {
			var wg sync.WaitGroup
			numWorkers := runtime.NumCPU()
			chunkSize := (len(data) + numWorkers - 1) / numWorkers

			for w := 0; w < numWorkers; w++ {
				start := w * chunkSize
				end := start + chunkSize
				if end > len(data) {
					end = len(data)
				}
				if start >= end {
					break
				}

				wg.Add(1)
				go func(start, end int) {
					defer wg.Done()
					localChanges := 0
					for i := start; i < end; i++ {
						minDist := float32(math.MaxFloat32)
						bestCluster := 0

						for cIdx, centroid := range centroids {
							dist := EuclideanDistance(data[i], centroid)
							if dist < minDist {
								minDist = dist
								bestCluster = cIdx
							}
						}

						if assignments[i] != bestCluster {
							assignments[i] = bestCluster
							localChanges++
						}
					}
					changesMux.Lock()
					changes += localChanges
					changesMux.Unlock()
				}(start, end)
			}
			wg.Wait()
		} else {
			// Sequential execution
			for i, point := range data {
				minDist := float32(math.MaxFloat32)
				bestCluster := 0

				for cIdx, centroid := range centroids {
					dist := EuclideanDistance(point, centroid)
					if dist < minDist {
						minDist = dist
						bestCluster = cIdx
					}
				}

				if assignments[i] != bestCluster {
					assignments[i] = bestCluster
					changes++
				}
			}
		}

		// Converged?
		if changes == 0 && iter > 0 {
			break
		}

		// 3. Update centroids
		// Typically fast enough to do sequentially unless K is huge
		counts := make([]int, k)
		newCentroids := make([][]float32, k)
		for i := range newCentroids {
			newCentroids[i] = make([]float32, dim)
		}

		for i, clusterIdx := range assignments {
			counts[clusterIdx]++
			for j, val := range data[i] {
				newCentroids[clusterIdx][j] += val
			}
		}

		for i := 0; i < k; i++ {
			if counts[i] > 0 {
				scale := 1.0 / float32(counts[i])
				for j := range centroids[i] {
					centroids[i][j] = newCentroids[i][j] * scale
				}
			} else {
				// Re-initialize empty cluster to a random point
				// (to avoid getting stuck with fewer clusters)
				ridx := rand.Intn(len(data))
				copy(centroids[i], data[ridx])
			}
		}
	}

	return centroids, assignments
}

// EuclideanDistance computes the distance between two vectors.
func EuclideanDistance(a, b []float32) float32 {
	var sum float32
	// Handle different lengths gracefully (though typically they should match)
	minLen := len(a)
	if len(b) < minLen {
		minLen = len(b)
	}

	for i := 0; i < minLen; i++ {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return float32(math.Sqrt(float64(sum)))
}

// ComputeSilhouetteScore calculates the mean Silhouette Coefficient of all samples.
// Returns a value between -1 and 1. High value indicates well-separated clusters.
func ComputeSilhouetteScore(data [][]float32, assignments []int) float32 {
	if len(data) < 2 {
		return 0
	}

	totalScore := float32(0)
	n := len(data)

	// Precompute distances
	// Note: For very large datasets, this O(N^2) approach might be slow.
	// Given typical ensemble sizes (e.g., < 100 architectures), this is fine.

	for i := 0; i < n; i++ {
		a := float32(0) // Mean distance to other points in same cluster
		b := float32(math.MaxFloat32) // Mean distance to points in nearest neighbor cluster

		myCluster := assignments[i]
		sameClusterCount := 0
		
		// Map of other_cluster_id -> sum_distance, count
		otherClusterDists := make(map[int]float32)
		otherClusterCounts := make(map[int]int)

		for j := 0; j < n; j++ {
			if i == j {
				continue
			}
			dist := EuclideanDistance(data[i], data[j])
			
			if assignments[j] == myCluster {
				a += dist
				sameClusterCount++
			} else {
				otherClusterDists[assignments[j]] += dist
				otherClusterCounts[assignments[j]]++
			}
		}

		if sameClusterCount > 0 {
			a /= float32(sameClusterCount)
		} else {
			a = 0 // Single element cluster
		}

		// Find nearest neighbor cluster distance (b)
		for cId, sumDist := range otherClusterDists {
			meanDist := sumDist / float32(otherClusterCounts[cId])
			if meanDist < b {
				b = meanDist
			}
		}
		
		if len(otherClusterDists) == 0 {
			b = 0 // No other clusters
		}

		// Silhouette for point i
		s := float32(0)
		maxAB := a
		if b > a {
			maxAB = b
		}
		
		if maxAB > 0 {
			s = (b - a) / maxAB
		}
		
		totalScore += s
	}

	return totalScore / float32(n)
}
