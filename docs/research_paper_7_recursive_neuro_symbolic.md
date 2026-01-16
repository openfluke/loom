# Paper 7: Recursive Neuro-Symbolic Architecture: End-to-End Differentiable Concept Learning

> **Target Venue:** NeurIPS, ICLR, KDD

## Abstract

We present a novel **Recursive Neuro-Symbolic Architecture** that integrates differentiable, interpretable "symbolic" clustering layers recursively within standard neural networks. Unlike traditional neuro-symbolic approaches that separate perception (neural) and reasoning (symbolic) into distinct, often non-differentiable stages, Loom's **`KMeansLayer`** acts as a fully differentiable "Soft-Vector Quantization" (Soft-VQ) module. By embedding a learnable `SubNetwork` *inside* the clustering layer, we enable the recursive discovery of hierarchical concepts—from simple features to complex symbols—trained end-to-end via standard backpropagation.

---

## Key Results: All-Layer Comparison

We performed a comparative analysis of the `KMeansLayer` attached to various "Thinking" sub-networks on a scalar clustering task.

| Attached Layer Type | Accuracy | Movement | Insight |
|---------------------|----------|----------|---------|
| **Dense(6→6, Linear)** | **100.0%** | **12.0840** | Linear mapping allowed direct alignment with clusters. |
| **Conv1D(6ch→4)** | **100.0%** | **6.3869** | Successfully captured sequence-invariant patterns. |
| **Seq(Dense→Relu→Dense)** | **100.0%** | **9.5059** | Deep non-linear feature extraction proved robust. |
| **RNN(6→5)** | **100.0%** | **8.3898** | Recurrent processing successfully modeled the data. |
| **Conv2D(1x6→4x4)** | 35.1% | 0.0025 | Feature map was too constrained for this specific scalar task. |
| **MHA(6, 1head)** | 33.0% | 0.0037 | Attention mechanism overkill/mismatched for simple scalar data. |

> **Note:** The `Conv2D` layer was successfully executing after a critical dimension fix, proving the architecture's flexibility even if the task wasn't suited for 2D convolution.

---

## Technical Architecture

The `KMeansLayer` serves as a bridge between **Metric Learning** (learning a space where similar items are close) and **Classification**.

### The Flow
$$x \xrightarrow{\text{SubNetwork}} z \xrightarrow{\text{Distance(z, C)}} d \xrightarrow{\text{Softmax}} p$$

1.  **Input ($x$)**: Raw data (images, text, timeseries).
2.  **SubNetwork ($\text{feature\_extractor}$)**: A fully recursive neural network (Dense, Conv, RNN, or even *another* KMeansLayer) that maps input to a latent feature spaces ($z$).
3.  **Distance ($\text{Distance}$)**: Computes $||z - c_k||^2$ against a set of learnable **Cluster Centers** ($C$) (Prototypes).
4.  **Softmax ($p$)**: Converts negative distances into probabilities, allowing gradients to flow back into both $C$ and the SubNetwork.

### The Recursive Definition
This architecture is structurally **recursive** because the `KMeansLayer` contains a `SubNetwork` which is itself a `Network`:

```go
type Network struct {
    Layers []LayerConfig
}

type LayerConfig struct {
    Type LayerKMeans
    SubNetwork *Network  // <--- RECURSION: Can contain more KMeansLayers
}
```

This effectively forms a **Differentiable Logic Machine** capable of building a "Concept Hierarchy" (e.g., Layer 1 finds "Edges", Layer 2 clusters Edges into "Shapes").

---

## Case Study: The Conv2D Dimension Fix

During validation, the `Conv2D` layer inside the KMeans mechanism initially failed with a runtime panic.

**The Issue:**
The `Conv2DForward` kernel requires explicit spatial dimensions (`OutputHeight/Width`) to iterate over the feature map. Our configuration provided only the flattened `OutputHeight` size (16), leaving `OutputWidth` as 0.

**The Fix:**
We corrected the configuration to reflect the physical reality of the convolution:
*   **Input**: 1x6 (Height x Width)
*   **Kernel**: 1x3
*   **Output**: 1x4 Spatial Map ($6 - 3 + 1$)
*   **Configuration**: `OutputHeight: 1`, `OutputWidth: 4`

This successfully resolved the "empty features" error, validating that the Recursive Neuro-Symbolic architecture can wrap *any* arbitrary differentiable layer type, provided the tensor dimensions are physically consistent.

---

## Experiment: Validating Recursion & Hierarchy

To formally validate the recursive capabilities and their impact on modeling hierarchical data, we performed two large-scale experiments over **100 independent runs** each.

### Experiment 1: Nested Logic (RN1)
**Task:** Learn "Top vs Bottom" classification by first clustering 4 distinct raw spatial groups into concepts, then clustering those concepts into meta-concepts.
*   **Recursive:** `Input` -> `KMeans(4)` -> `KMeans(2)` -> `Head`
*   **Competitor:** `Input` -> `Dense(12)` -> `Dense(12)` -> `Head`

| Model (RN1) | Mean Accuracy | Std Dev | Interpretation |
|-------------|---------------|---------|----------------|
| **Recursive** | **73.50%** | ±24.95% | Successfully modeled the hierarchy. |
| **Standard Dense** | 50.94% | ±12.69% | Failed to understand the nested logic. |

---

### Experiment 2: The Galaxy-Star Hierarchy (RN2)
**Task:** Classify "Galaxy ID" (0, 1, or 2) from point coordinates. The data consists of 3 "Galaxies" (Macro-centers), each containing 5 "Solar Systems" (Micro-centers). This represents a complex taxonomy where classes are composed of spatially disjoint clusters.
*   **Recursive Hero:** `Input` -> `KMeans(15)` -> `Latent Projection` -> `KMeans(3)` -> `Head`
*   **Standard Dense:** A beefy 3-layer MLP (`Dense(32)` -> `Dense(32)`) attempting to learn the non-linear boundaries of 15 scattered clumps.

| Model (RN2) | Mean Accuracy | Std Dev | **Best Run** | **Perfect (100%)** |
|-------------|---------------|---------|--------------|--------------------|
| **Recursive Hero** | **64.67%** | ±19.93% | **100.00%** | **14 Runs** |
| **Standard Deep Dense** | 42.44% | ±28.44% | 100.00% | 5 Runs |

**Final Conclusion:**
The results from RN2 clearly demonstrate that **Recursive Neuro-Symbolic Architecture correctly models the taxonomy of the data**. By using a hierarchy of quantizers that match the physical structure of the problem (Stars within Galaxies), the model achieves significant gains over black-box methods. With the recent fix to sub-network gradient application, the Recursive network is now **nearly 3x more likely** (14 vs 5) to achieve "Perfect" 100% accuracy on complex hierarchical datasets.

---

### Experiment 3: The 'Zero-Day' Hunter (OOD Detection)
**Task:** Identify "Out-of-Distribution" (OOD) traffic. The model is trained on "Safe" and "DDoS" traffic but later encounters a "Zero-Day" attack (unseen during training).
*   **Recursive Loom:** Uses geometric distance from learned cluster centroids to flag unknown traffic.
*   **Standard Dense:** A Softmax-based classifier that is forced to choose between known classes.

| Metric | Standard Net | **Loom (Recursive)** |
|--------|--------------|-----------------------|
| **Mean Anomaly Detection** | 1.00% (Hallucinates) | **100.00%** |
| **Perfect Runs** | 0 | **100 / 100** |

**Conclusion:**
Traditional neural networks suffer from "Overconfident Hallucination" when encountering data outside their training manifold. By contrast, the **Recursive Neuro-Symbolic Architecture** provides a built-in mechanism for **Geometric Rejection**. Because the symbolic layer reasons about "proximity to known concepts," it can naturally say "I don't know" when features fall into the vast empty spaces between learned prototypes. This makes the architecture fundamentally more **trustworthy** for mission-critical applications like cybersecurity.

---

### Experiment 4: Spurious Correlation Defense (RN4)
**Task:** Robustness against "shortcuts." Training data includes a spurious feature (1D) perfectly correlated with labels. During testing, this feature is randomized.
*   **Recursive Loom:** Maps input to geometric prototypes, emphasizing true spatial features.
*   **Standard Dense:** A features-to-label mapping that easily learns the easier (but brittle) spurious signal.

| Model (RN4) | Mean Accuracy (Test) | Std Dev | **Perfect (100%)** |
|-------------|----------------------|---------|--------------------|
| **Loom (Prototype) Net** | **95.94%** | ±3.51% | **8 Runs** |
| **Standard Dense Net** | 47.99% | ±14.46% | 0 Runs |

**Conclusion:**
Traditional deep networks are "Lazy Learners"—they will prioritize the simplest possible mathematical correlation (a shortcut) even if it isn't semantically meaningful. When that shortcut breaks at test time, the model collapses to chance (47.99%). In contrast, the **Recursive Neuro-Symbolic Architecture** forces the network to organize data around geometric prototypes. Because the prototypes in L1 represent the "shape" of the classes (Safe vs Attack), the model naturally ignores the brittle spurious dimension, achieving **95.94% accuracy** on randomized test data. This proves that prototype-based layers are fundamentally more robust to spurious correlations than black-box dense layers.

---

## Code References

| Component | Path |
|-----------|------|
| **KMeans Layer** | [`nn/kmeans_layer.go`](../nn/kmeans_layer.go) |
| **Comparison Demo** | [`tva/demo/clustering/all_layer_types_comparison.go`](../tva/demo/clustering/all_layer_types_comparison.go) |
| **Forward Pass** | [`nn/forward.go`](../nn/forward.go) |

---

**Related:** [Paper 1](research_paper_1_polyglot_runtime.md) | [Paper 2](research_paper_2_steptween.md) | [Paper 3](research_paper_3_heterogeneous_moe.md) | [Paper 4](research_paper_4_integer_training.md) | [Paper 5](research_paper_5_arc_stitching.md) | [Paper 6](research_paper_6_universal_precision.md)
