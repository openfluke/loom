# Clustering Utilities

`clustering.go` provides standard unsupervised clustering primitives used by Loom's NAS and ensemble analysis pipelines. These are pure Go, parallelism-aware, and operate on `[]float32` feature vectors.

---

## KMeansCluster

Standard K-means clustering with optional parallel assignment:

```go
centroids, assignments := nn.KMeansCluster(data, k, maxIter, parallel)
```

| Argument | Type | Description |
|---|---|---|
| `data` | `[][]float32` | Feature vectors to cluster |
| `k` | `int` | Number of clusters |
| `maxIter` | `int` | Maximum iterations (stops early if converged) |
| `parallel` | `bool` | Use all CPUs for the assignment step |

Returns:
- `centroids` — `[][]float32` of length `k`, the final cluster centres
- `assignments` — `[]int` of length `len(data)`, cluster index for each point

### Algorithm

1. **Initialise**: Randomly select `k` data points as initial centroids (Forgy method)
2. **Assign**: Each point → nearest centroid (Euclidean distance)
3. **Update**: Move each centroid to the mean of its assigned points
4. **Repeat** until convergence (no reassignments) or `maxIter` reached
5. **Empty cluster rescue**: If a cluster becomes empty, re-seed from a random data point

```go
// Group 200 performance vectors into 5 clusters, parallel
data := collectPerformanceVectors(nets) // [][]float32
centroids, assignments := nn.KMeansCluster(data, 5, 100, true)

for i, assignment := range assignments {
    fmt.Printf("Net %d → Cluster %d\n", i, assignment)
}
```

### Parallel Mode

When `parallel == true`, the assignment step (the O(N×K) bottleneck) is split across all available CPU cores via goroutines. The update step (centroid recalculation) always runs sequentially since K is typically small.

For typical NAS ensemble sizes (< 200 architectures), the speedup is modest. For larger sweeps (thousands of architectures), enable parallel mode.

---

## EuclideanDistance

Computes L2 distance between two float32 vectors:

```go
dist := nn.EuclideanDistance(a, b)
```

Handles mismatched lengths gracefully (uses `min(len(a), len(b))`). Returns `float32`.

```go
v1 := []float32{1.0, 2.0, 3.0}
v2 := []float32{4.0, 6.0, 3.0}
d := nn.EuclideanDistance(v1, v2) // sqrt(9 + 16 + 0) = 5.0
```

---

## ComputeSilhouetteScore

Measures clustering quality using the **Silhouette Coefficient**:

```go
score := nn.ComputeSilhouetteScore(data, assignments)
```

Returns a `float32` in [-1, 1]:

| Range | Interpretation |
|---|---|
| ~1.0 | Points are well inside their cluster, far from others — excellent separation |
| ~0.0 | Points are near cluster boundaries — ambiguous assignment |
| ~-1.0 | Points are closer to a neighbouring cluster — wrong assignment |

### How It Works

For each point `i`:
- **a(i)** = mean distance to all other points in the **same** cluster
- **b(i)** = mean distance to all points in the **nearest other** cluster
- **s(i)** = `(b(i) - a(i)) / max(a(i), b(i))`

Final score = mean of `s(i)` across all `N` points.

> [!NOTE]
> This is O(N²) in distance computations. For NAS ensemble sizes (< 200), this is fast. For very large datasets, consider approximate methods.

```go
score := nn.ComputeSilhouetteScore(data, assignments)
fmt.Printf("Silhouette score: %.3f\n", score)
// > 0.5: good separation
// 0.2-0.5: reasonable
// < 0.2: overlapping clusters, consider fewer K
```

---

## Usage in NAS Pipeline

Clustering is used to group architectures by their performance profile and select diverse representatives:

```go
// 1. Train N architectures, collect performance vectors
var perfVectors [][]float32
for _, net := range trainedNets {
    vec := getPerformanceVector(net) // e.g. accuracy per task
    perfVectors = append(perfVectors, vec)
}

// 2. Cluster into K groups
k := 5
centroids, assignments := nn.KMeansCluster(perfVectors, k, 100, true)

// 3. Evaluate cluster quality
score := nn.ComputeSilhouetteScore(perfVectors, assignments)
fmt.Printf("Cluster quality: %.3f (K=%d)\n", score, k)

// 4. Pick the best representative from each cluster
bestPerCluster := make([]*nn.Network, k)
for clusterIdx := 0; clusterIdx < k; clusterIdx++ {
    bestPerCluster[clusterIdx] = pickBest(trainedNets, assignments, clusterIdx)
}

// 5. Graft the best representatives together
superHive, _ := nn.GraftNetworks(bestPerCluster, "filter")
```

This ensures the grafted model contains diverse expertise rather than N copies of the same strategy.

---

## See Also

- [architecture.md](architecture.md) — Random architecture generation and `BuildDiverseNetwork`
- [grafting.md](grafting.md) — Combining selected representatives into a Super-Hive
- [kmeans.md](kmeans.md) — The differentiable KMeans **layer** (distinct from this utility — this file provides the classical K-means algorithm used for NAS analysis)
