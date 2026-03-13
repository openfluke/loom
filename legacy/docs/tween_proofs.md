# NeuralTween: Mathematical Proofs and Manifold Theory

This document outlines the mathematical foundation and manifold topology underlying the `NeuralTween` implementation found in `nn/tween.go`.

## 1. The Activation Manifold $\mathcal{M}$

Traditional backpropagation formulates learning as traversing the loss landscape with respect to the network's parameters $\Theta$. In contrast, **NeuralTween** operates directly on the **Activation Manifold** $\mathcal{M}$. 

Let $\mathcal{M}_i$ be the manifold of all possible activation states at layer $i$. For an input $\mathbf{x}$, the forward pass yields an actual activation vector $\mathbf{a}_i \in \mathcal{M}_i$. Concurrently, a backward projection (from the expected output) yields a target activation vector $\mathbf{t}_i \in \mathcal{M}_i$. 

The core premise of NeuralTween is not to minimize a solitary scalar loss $L(\Theta)$, but to geometrically close the topological gap between $\mathbf{a}_i$ and $\mathbf{t}_i$ within $\mathcal{M}_i$ at every layer simultaneously.

## 2. Gap (Distance Metric)

The primary local objective at any layer is the **Gap**, measured as the scaled Euclidean distance (Root Mean Square Error) between the forward activation $\mathbf{a}$ and the backward target $\mathbf{t}$ for a layer with dimension $N$:

$$
\text{Gap}(\mathbf{a}, \mathbf{t}) = \sqrt{\frac{1}{N+1} \sum_{j=1}^N (a_j - t_j)^2}
$$

By minimizing $\text{Gap}(\mathbf{a}_i, \mathbf{t}_i) \to 0$ iteratively across all subsets of the network, the forward manifold aligns with the backward expectation manifold.

## 3. Link Budget (Information Preservation)

To quantify how much information survives the layer transformations, NeuralTween uses the **Link Budget**. It captures the directional alignment of the actual and target vectors using Cosine Similarity:

$$
\cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{t}}{\|\mathbf{a}\|_2 \|\mathbf{t}\|_2} = \frac{\sum_{j=1}^N a_j t_j}{\sqrt{\sum_{j=1}^N a_j^2} \sqrt{\sum_{j=1}^N t_j^2}}
$$

The Link Budget normalizes this span to a probabilistic bound $[0, 1]$:

$$
\text{LinkBudget} = \frac{\cos(\theta) + 1}{2}
$$

- A Link Budget $\to 1$ indicates near-perfect constructive interference (ideal information preservation).
- A Link Budget $\to 0.5$ implies orthogonality (no meaningful relationship).
- A Link Budget $\to 0$ implies destructive interference.

## 4. Local Chain Rule Gradients

Unlike global backpropagation, NeuralTween generates gradients based on the local error $\mathbf{e} = \mathbf{t} - \mathbf{a}$. 
For a dense layer outputting $a_{out} = \sigma(\mathbf{W} \cdot a_{in} + \mathbf{b})$, the gradient with respect to output $j$ is scaled by the derivative of the activation function $\sigma'$:

$$
\delta_j = e_j \cdot \sigma'(a_{out, j}) \cdot D_{\text{scale}}
$$

The weight update gradient $\Delta W_{inp, j}$ is subsequently formed by the local chain rule:

$$
\Delta W_{inp, j} = \eta \cdot a_{in, inp} \cdot \delta_j
$$

where $\eta$ is the dense learning rate.

## 5. Depth Scaling Factor $D_{\text{scale}}$

To prevent gradient vanishing and explosion as targets propagate backward (or equivalently as error propagates), gradients are modulated by a structural depth scale component:

$$
D_{\text{scale}} = \gamma^{(L - 1 - i)}
$$

where $\gamma$ is the `DepthScaleFactor` constant (e.g., $1.2$), $L$ is the total number of layers, and $i$ is the current layer index. This guarantees that deeper layers maintain a mathematically balanced influence velocity compared to shallower layers.

## 6. Momentum and Weight Velocity

Stable manifold traversal is assured via smoothed velocity vectors. For a given weight $w$, its velocity $v$ acts as an accumulator for the deltas:

$$
v^{(t)} = m \cdot v^{(t-1)} + (1 - m) \cdot \Delta w
$$

$$
w^{(t+1)} = w^{(t)} + v^{(t)}
$$

where $m$ is the tunable momentum constant. This acts as a low-pass filter on the activation manifold adjustments, averting erratic oscillations when the forward vector $\mathbf{a}_i$ and target vector $\mathbf{t}_i$ converge in high-curvature regions of the space.
