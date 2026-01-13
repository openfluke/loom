# Understanding Layer Types

This guide explains what each layer type actually does—not just the math, but the intuition behind why these layers exist and when you'd use them.

---

## Dense Layers: The Universal Connector

A Dense layer (also called "fully connected" or "linear") is the simplest and most fundamental building block. Every input connects to every output.

### What It Actually Does

Imagine you have 4 inputs and want 3 outputs:

```
    Inputs                    Outputs
    
    x₁ ─────┬────┬────┬────▶ y₁
            │    │    │
    x₂ ─────┼────┼────┼────▶ y₂
            │    │    │
    x₃ ─────┼────┼────┼────▶ y₃
            │    │    │
    x₄ ─────┴────┴────┴────▶ (every input connects to every output)
    
    Total connections: 4 × 3 = 12 weights
    Plus 3 biases (one per output)
```

Each output is computed as:
```
y₁ = activation( w₁₁×x₁ + w₁₂×x₂ + w₁₃×x₃ + w₁₄×x₄ + b₁ )
y₂ = activation( w₂₁×x₁ + w₂₂×x₂ + w₂₃×x₃ + w₂₄×x₄ + b₂ )
y₃ = activation( w₃₁×x₁ + w₃₂×x₂ + w₃₃×x₃ + w₃₄×x₄ + b₃ )
```

### Why It's Called "Dense"

Because the weight matrix is *dense*—every possible connection exists. This is the opposite of sparse connections (like Conv2D) where only local neighborhoods connect.

### The Weight Matrix Visualized

```
               Input features (1024)
              ┌─────────────────────────┐
              │                         │
              ▼                         ▼
           ┌──────────────────────────────┐
Row 0:     │ w₀,₀  w₀,₁  w₀,₂  ...  w₀,₁₀₂₃│ → Output 0
Row 1:     │ w₁,₀  w₁,₁  w₁,₂  ...  w₁,₁₀₂₃│ → Output 1
Row 2:     │ w₂,₀  w₂,₁  w₂,₂  ...  w₂,₁₀₂₃│ → Output 2
  ...      │  ...   ...   ...        ...   │
Row 511:   │ w₅₁₁,₀ ...             w₅₁₁,₁₀₂₃│ → Output 511
           └──────────────────────────────┘
           
Matrix shape: [512 outputs × 1024 inputs] = 524,288 weights

Each row computes one output neuron.
Each column represents how much one input affects all outputs.
```

### When to Use Dense Layers

- **Classification heads**: Map features to class probabilities
- **Fully connected networks**: Simple stacked architectures
- **Dimensionality changes**: Go from 1024 features to 256, or vice versa
- **After flattening**: Following Conv2D layers in CNNs

---

## Conv2D Layers: Finding Patterns in Images

Convolutional layers look for *local patterns* that can appear anywhere in an image. Instead of connecting everything to everything, they slide a small "kernel" (also called "filter") across the image.

### The Key Insight

Consider detecting a vertical edge. This pattern:
```
Dark | Light | Dark
  0  |   1   |  0
  0  |   1   |  0
  0  |   1   |  0
```

Can appear **anywhere** in an image. A Dense layer would need to learn separate weights for each position. A Conv2D layer learns the pattern **once** and slides it across the entire image.

### How Convolution Works

```
Input Image (4×4):                  Kernel (3×3):
┌─────┬─────┬─────┬─────┐          ┌─────┬─────┬─────┐
│  1  │  2  │  3  │  4  │          │ a   │ b   │ c   │
├─────┼─────┼─────┼─────┤          ├─────┼─────┼─────┤
│  5  │  6  │  7  │  8  │          │ d   │ e   │ f   │
├─────┼─────┼─────┼─────┤          ├─────┼─────┼─────┤
│  9  │ 10  │ 11  │ 12  │          │ g   │ h   │ i   │
├─────┼─────┼─────┼─────┤          └─────┴─────┴─────┘
│ 13  │ 14  │ 15  │ 16  │
└─────┴─────┴─────┴─────┘

Step 1: Place kernel at top-left

┌─────┬─────┬─────┐─────┐
│  1  │  2  │  3  │  4  │     Output[0,0] = 
├─────┼─────┼─────┤─────┤       1×a + 2×b + 3×c +
│  5  │  6  │  7  │  8  │       5×d + 6×e + 7×f +
├─────┼─────┼─────┤─────┤       9×g + 10×h + 11×i
│  9  │ 10  │ 11  │ 12  │
└─────┴─────┴─────┘─────┘

Step 2: Slide kernel one position right

┌─────┬─────┬─────┬─────┐
│  1  │  2  │  3  │  4  │     Output[0,1] = 
├─────┼─────┼─────┼─────┤       2×a + 3×b + 4×c +
│  5  │  6  │  7  │  8  │       6×d + 7×e + 8×f +
├─────┴─────┼─────┼─────┤       10×g + 11×h + 12×i
            │     │     │
            └─────┴─────┘

Continue until covering entire image...

Output (2×2):
┌─────┬─────┐
│ o₀₀ │ o₀₁ │
├─────┼─────┤
│ o₁₀ │ o₁₁ │
└─────┴─────┘
```

### Stride and Padding

**Stride** controls how far the kernel moves each step:
```
Stride = 1: Kernel moves 1 pixel at a time (detailed output)
Stride = 2: Kernel moves 2 pixels at a time (smaller output)

Stride 1:                    Stride 2:
Step: 0 1 2 3               Step: 0   2
      ▼ ▼ ▼ ▼                     ▼   ▼
      ■□□□□                       ■□□□□
       ■□□□                        ■
        ■□□                         
         ■□
```

**Padding** adds pixels around the edges:
```
Without Padding:              With Padding (1 pixel):
Input: 4×4                   Input: 4×4 → Padded: 6×6
Kernel: 3×3                  Kernel: 3×3
Output: 2×2                  Output: 4×4 (same as input!)

                             ┌───┬───┬───┬───┬───┬───┐
                             │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │
                             ├───┼───┼───┼───┼───┼───┤
                             │ 0 │   │   │   │   │ 0 │
                             ├───┤   Input   │───┤
                             │ 0 │    4×4    │ 0 │
                             ├───┤           │───┤
                             │ 0 │   │   │   │   │ 0 │
                             ├───┼───┼───┼───┼───┼───┤
                             │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │
                             └───┴───┴───┴───┴───┴───┘
```

### Multiple Filters = Multiple Feature Maps

Real Conv2D layers have multiple filters, each detecting a different pattern:

```
Input Image                 Filters                    Feature Maps
(28×28×1)                  (3×3×1 each)               (26×26×32)
                                                       
    │                    ┌──────────┐                   Output
    │                    │ Filter 0 │───▶ Edge detector
    │                    ├──────────┤                   
    └───────────────────▶│ Filter 1 │───▶ Blob detector   32 different
                         ├──────────┤                     feature maps
                         │ Filter 2 │───▶ Corner detector
                         ├──────────┤
                         │   ...    │
                         ├──────────┤
                         │Filter 31 │───▶ Some pattern
                         └──────────┘
```

---

## Conv1D Layers: Local Patterns in Sequences

Conv1D is the 1D version of convolution. Instead of sliding a 2D kernel across an image, it slides a **1D kernel along a sequence** (audio, time series, token embeddings). The key idea is the same: learn a local pattern once and reuse it everywhere.

```
Input (channels × seqLen)          Kernel (channels × kernelSize)
┌───────────────┐                 ┌───────────────┐
│ c0: 1 2 3 4 5 │   ─────▶         │ c0: a b c     │
│ c1: 6 7 8 9 0 │                 │ c1: d e f     │
└───────────────┘                 └───────────────┘
```

**Output length**:
```
outLen = (seqLen + 2*padding - kernelSize) / stride + 1
```

Use Conv1D when:
- Sequences have **local structure** (phonemes, n-grams, sensor spikes)
- You want **translation invariance** along time
- You need efficient feature extraction before RNN/Attention

---

## Embedding Layers: Token Lookup Tables

Embeddings convert discrete IDs (tokens) into dense vectors. Think of it as a **learned dictionary**:

```
Embedding table: [vocabSize × embeddingDim]
Token ID 42 → row 42 → vector of length embeddingDim
```

Only the rows used in the current batch receive gradients. This makes embedding updates efficient even with huge vocabularies.

Use embeddings when:
- Inputs are **categorical** (tokens, IDs, symbols)
- You need a learned representation before sequence models
- You want compact, trainable vector spaces

---

## SwiGLU: Gated MLP Blocks

SwiGLU is a modern gated feedforward block used in LLMs. It uses **two projections** plus a gating nonlinearity:

```
gate = SiLU(x · W_gate + b_gate)
up   = x · W_up + b_up
out  = (gate ⊙ up) · W_down + b_down
```

This gating tends to train better than a plain ReLU MLP, especially in deep transformer stacks.

Use SwiGLU when:
- Building transformer-style feedforward blocks
- You want stronger expressivity than a single dense layer

---

## Multi-Head Attention: Learning What to Focus On

Attention is the mechanism that powers Transformers. It lets the network decide which parts of the input are relevant for each output.

### The Core Idea

Imagine reading the sentence: "The cat sat on the mat because it was tired."

When interpreting "it", you need to figure out what "it" refers to. Attention learns to **look back at "cat"** when processing "it".

```
Input sequence: [The] [cat] [sat] [on] [the] [mat] [because] [it] [was] [tired]
                  ↑     ↑                                      |
                  └─────┴──────────────────────────────────────┘
                        "it" attends strongly to "cat"
```

### Query, Key, Value: The Mechanism

Attention uses three projections of each token:
- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What do I contain?"
- **Value (V)**: "What information do I provide?"

```
Each token gets transformed into Q, K, and V vectors:

Token "it"           Token "cat"
    │                    │
    ▼                    ▼
┌───────┐            ┌───────┐
│ Q: "?" │            │ K: "animal"│
│ K: "pronoun"│      │ V: cat info │
│ V: "it"│            └───────┘
└───────┘
    │                    │
    │    Attention       │
    │    Computation:    │
    │                    │
    │   score = Q · K    │
    │   weight = softmax(score)
    │   output = weight × V
    │                    │
    ▼                    ▼
"it" looks at "cat" with high weight,
retrieves cat's information
```

### Multi-Head: Multiple Perspectives

"Multi-head" means running multiple attention computations in parallel, each learning different relationships:

```
Input: [batch, sequence, 512]
        │
        │ Split into 8 heads (512 ÷ 8 = 64 dims each)
        ▼
┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
│Head 0│Head 1│Head 2│Head 3│Head 4│Head 5│Head 6│Head 7│
│64d   │64d   │64d   │64d   │64d   │64d   │64d   │64d   │
│      │      │      │      │      │      │      │      │
│syntax│coref │topic │entity│tense │...   │...   │...   │
└──┬───┴──┬───┴──┬───┴──┬───┴──┬───┴──┬───┴──┬───┴──┬───┘
   │      │      │      │      │      │      │      │
   └──────┴──────┴──────┼──────┴──────┴──────┴──────┘
                        │
                        ▼ Concatenate
                   [batch, seq, 512]
                        │
                        ▼ Output projection
                   [batch, seq, 512]
```

Each head can specialize: one might track grammatical relationships, another coreference, another topic similarity.

### The Attention Formula

```
Attention(Q, K, V) = softmax(Q × Kᵀ / √d) × V

Where:
- Q × Kᵀ produces [seq, seq] matrix of "how much does each token attend to each other"
- √d is a scaling factor to prevent dot products from getting too large
- softmax normalizes each row to sum to 1 (probabilistic attention)
- × V retrieves the actual information based on attention weights
```

Visual:
```
        Keys (all tokens)
        ┌───┬───┬───┬───┐
Queries │0.1│0.7│0.1│0.1│ "it" → mostly attends to "cat"
(all    ├───┼───┼───┼───┤
tokens) │0.3│0.3│0.2│0.2│ "cat" → attends somewhat evenly
        ├───┼───┼───┼───┤
        │0.2│0.2│0.3│0.3│ "sat" → attends to later tokens
        └───┴───┴───┴───┘
        
        Each row sums to 1.0 (softmax normalization)
```

---

## RNN: Remembering the Past

Recurrent Neural Networks process sequences by maintaining a "hidden state" that summarizes everything seen so far.

### The Key Insight

Dense layers have no memory—they process each input independently. RNNs carry information forward through time.

```
Without RNN (Dense):
    
    Token 1 → [Dense] → Output 1    (no connection)
    Token 2 → [Dense] → Output 2    (no connection)
    Token 3 → [Dense] → Output 3    (can't see tokens 1 or 2!)


With RNN:
    
    Token 1 → [RNN] → Output 1
                │
                ▼ hidden state
    Token 2 → [RNN] → Output 2
                │
                ▼ hidden state carries info from tokens 1-2
    Token 3 → [RNN] → Output 3  ← can "remember" earlier tokens!
```

### How the Hidden State Works

At each time step, the RNN combines:
1. Current input
2. Previous hidden state

```
Time step t:

                Previous hidden state
                h_{t-1}
                    │
                    ▼
              ┌───────────────────┐
Current  ───▶ │                   │ ───▶ Output y_t
input x_t     │   h_t = tanh(     │
              │     W_ih × x_t +  │
              │     W_hh × h_{t-1}│
              │     + bias )      │
              └─────────┬─────────┘
                        │
                        ▼
                New hidden state h_t
                (passed to next step)
```

### Unrolled Through Time

```
Sequence: [x₁, x₂, x₃, x₄]

     x₁        x₂        x₃        x₄
      │         │         │         │
      ▼         ▼         ▼         ▼
   ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
   │ RNN  │→│ RNN  │→│ RNN  │→│ RNN  │
   │ Cell │ │ Cell │ │ Cell │ │ Cell │
   └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘
      │         │         │         │
      ▼         ▼         ▼         ▼
     y₁        y₂        y₃        y₄

   h₀→      h₁→      h₂→      h₃→      h₄
   (init)                              (final)
```

The same weights are used at every time step—the network "time-shares" its parameters.

---

## LSTM: Solving the Vanishing Gradient Problem

Standard RNNs struggle with long sequences. Gradients either vanish (become tiny) or explode (become huge) when backpropagating through many time steps.

LSTM (Long Short-Term Memory) solves this with **gates**—learned mechanisms that control information flow.

### The Three Gates

```
LSTM Cell:

                      ┌─────────────────────────────────────────┐
                      │                                         │
    ┌─────────────────┼──────────────────────────────────────┐  │
    │                 ▼                                      │  │
    │   ┌──────────────────────────────────────────────┐     │  │
    │   │                 Cell State (cₜ)              │     │  │
    │   │  "The highway for information"               │     │  │
    │   └───────┬──────────┬──────────┬────────────────┘     │  │
    │           │          │          │                      │  │
    │           │          ▼          │                      │  │
    │     ┌─────▼─────┐  ┌───┐  ┌─────▼─────┐               │  │
    │     │  Forget   │  │ + │  │  Input    │               │  │
    │     │   Gate    │  │   │  │   Gate    │               │  │
    │     │  (what to │  └─┬─┘  │  (what to │               │  │
    │     │   forget) │    │    │   add)    │               │  │
    │     └─────┬─────┘    │    └─────┬─────┘               │  │
    │           │          │          │                      │  │
    │           │    ┌─────┴─────┐    │                      │  │
    │           │    │  tanh     │    │                      │  │
    │           ▼    │ (new info)│    ▼                      │  │
    │     ┌───────┐  └─────┬─────┘  ┌───────┐               │  │
    │     │ ×     │        │        │  ×    │               │  │
    │     └───┬───┘        │        └───┬───┘               │  │
    │         │            │            │                    │  │
    │         └────────────┼────────────┘                    │  │
    │                      │                                 │  │
    │                      ▼                                 │  │
    │                ┌───────────┐                           │  │
    │                │  Output   │                           │  │
    │                │   Gate    │◀──────────────────────────┘  │
    │                │ (what to  │                              │
    │                │  output)  │                              │
    │                └─────┬─────┘                              │
    │                      │                                    │
    │                      ▼                                    │
    │                 Hidden State (hₜ)                         │
    │                      │                                    │
    └──────────────────────┼────────────────────────────────────┘
                           │
                           ▼
                       Output
```

### What Each Gate Does

**Forget Gate**: "What should I throw away from the cell state?"
```
f_t = σ(W_f × [h_{t-1}, x_t] + b_f)

Output: values between 0 and 1
- 0 means "forget completely"
- 1 means "remember completely"

Example: Processing a period "." might signal to forget the subject
of the previous sentence.
```

**Input Gate**: "What new information should I store?"
```
i_t = σ(W_i × [h_{t-1}, x_t] + b_i)     ← how much to add
g_t = tanh(W_g × [h_{t-1}, x_t] + b_g)  ← what to add

Example: Seeing a new subject noun might store that in the cell state.
```

**Output Gate**: "What should I output based on the cell state?"
```
o_t = σ(W_o × [h_{t-1}, x_t] + b_o)
h_t = o_t × tanh(c_t)

Example: When generating a verb, output information about the subject
(for agreement) but not necessarily everything in the cell.
```

### Why This Solves Vanishing Gradients

The cell state has a "highway" path through time:
```
c₀ ──×f₁──+──×f₂──+──×f₃──+──×f₄──+── c₄

The gradient can flow through the + operations almost unchanged!
Unlike vanilla RNN where gradients must pass through tanh repeatedly.
```

---

## Softmax: Turning Numbers into Probabilities

Softmax converts a vector of arbitrary real numbers into a probability distribution (values between 0 and 1 that sum to 1).

### The Basic Transformation

```
Input (logits):  [2.0, 1.0, 0.1]
                   │
                   ▼ exp(each value)
             [7.39, 2.72, 1.11]
                   │
                   ▼ divide by sum (11.22)
Output:      [0.66, 0.24, 0.10]
             ─────────────────
               sums to 1.0 ✓
```

The largest input gets the largest probability, but all outputs are positive and normalized.

### Why Not Just Normalize Directly?

The exponential has important properties:
1. **Always positive**: Even negative inputs become positive
2. **Amplifies differences**: Large inputs dominate
3. **Smooth gradients**: Differentiable everywhere

```
Without exp:          With exp (softmax):
[2, 1, -1] → ?       [2, 1, -1] → [7.39, 2.72, 0.37]
Can't normalize      All positive! Can normalize.
when there are       
negatives.           

[10, 10, 10] → [.33, .33, .33]   Equal values → equal probs ✓
[100, 0, 0] → [.99, .004, .004]  Large diff → confident prediction
```

### Loom's 10 Softmax Variants

Loom treats Softmax as a first-class layer with multiple variants:

```
Standard Softmax:
[logits] → [probabilities that sum to 1]

Grid Softmax (Native Mixture of Experts!):
┌──────────────────────────────────────┐
│  Expert 0: [0.1, 0.2, 0.3, 0.4] = 1  │
│  Expert 1: [0.5, 0.2, 0.2, 0.1] = 1  │  Each ROW sums to 1
│  Expert 2: [0.1, 0.1, 0.1, 0.7] = 1  │  independently
└──────────────────────────────────────┘

Temperature Softmax:
- Low temperature (0.1): Sharp, confident, picks one option
- High temperature (2.0): Smooth, uncertain, spreads probability

Masked Softmax:
[logits] + [mask: True, False, True, True]
         → [0.33, 0.00, 0.33, 0.34]
           Masked positions get zero probability
           (Useful for: legal moves in games)

Sparsemax:
Like softmax, but can produce exact zeros
[logits] → [0.6, 0.4, 0.0, 0.0]
            Interpretable! Only a few options selected.
```

---

## Normalization Layers: Keeping Activations Stable

As data flows through many layers, values can drift—becoming very large or very small. Normalization layers re-center and re-scale activations.

### Layer Normalization

Normalizes across the feature dimension for each sample independently:

```
Input: [batch, features]
        ┌────────────────────────────┐
Sample 0│ 100 │ -50 │  25 │  75 │   │ ← Normalize this row
        ├────────────────────────────┤
Sample 1│  10 │  20 │  30 │  40 │   │ ← Normalize this row
        ├────────────────────────────┤
Sample 2│ 0.1 │ 0.2 │ 0.3 │ 0.4 │   │ ← Normalize this row
        └────────────────────────────┘

For each sample:
1. Compute mean: μ = (100 + -50 + 25 + 75) / 4 = 37.5
2. Compute std:  σ = sqrt(variance) = ~54.5
3. Normalize:    (x - μ) / σ

Output: values with mean≈0, std≈1

Plus learnable parameters:
output = γ × normalized + β
(γ and β are learned per feature)
```

### RMS Normalization (Llama-style)

Like LayerNorm but simpler—only divides by root-mean-square, no mean subtraction:

```
rms = sqrt(mean(x²))
output = x / rms × γ

Why use it?
- Slightly faster (no mean computation)
- Works well empirically for modern LLMs
- Used in Llama, Mistral, etc.
```

### Why Normalization Matters

```
Without Normalization:         With Normalization:

Layer 1 output: ~100           Layer 1 output: ~100 → norm → ~0
Layer 2 output: ~10000         Layer 2 output: ~0 → x → ~0 → norm → ~0
Layer 3 output: ~1000000       Layer 3 output: ~0 → x → ~0 → norm → ~0
...                            ...
Values explode!                Values stay controlled throughout
Training becomes unstable      Training is stable
```

---

## Structural Layers: Composing Complex Architectures

### Sequential Layer

Chains sub-layers one after another:

```
Sequential([Dense(512), ReLU(), Dense(256), ReLU(), Dense(10)])

    Input
      │
      ▼
  ┌────────────┐
  │Dense(512)  │
  └─────┬──────┘
        │
        ▼
  ┌────────────┐
  │   ReLU     │
  └─────┬──────┘
        │
        ▼
  ┌────────────┐
  │Dense(256)  │
  └─────┬──────┘
        │
        ▼
  ┌────────────┐
  │   ReLU     │
  └─────┬──────┘
        │
        ▼
  ┌────────────┐
  │Dense(10)   │
  └─────┬──────┘
        │
        ▼
    Output
```

### Parallel Layer

Runs multiple branches simultaneously, then combines results:

```
          Input
            │
     ┌──────┼──────┐
     │      │      │
     ▼      ▼      ▼
  ┌─────┐┌─────┐┌─────┐
  │LSTM ││Dense││Conv │   Three different "experts"
  └──┬──┘└──┬──┘└──┬──┘   process the same input
     │      │      │
     └──────┼──────┘
            │
            ▼
       ┌─────────┐
       │ Combine │
       └────┬────┘
            │
            ▼
         Output

Combine modes:
- concat: Concatenate all outputs [lstm_out, dense_out, conv_out]
- add:    Element-wise sum
- avg:    Element-wise average
- filter: Softmax-weighted combination (learned gating)
```

The **filter** mode is particularly interesting—it's a learned routing mechanism:

```
Filter mode (Soft Mixture of Experts):

     Input
       │
       ├──────────────────────────────┐
       │                              │
  ┌────▼────┐                    ┌────▼────┐
  │ Branch 0│                    │ Gating  │
  │ (expert)│                    │ Network │
  └────┬────┘                    └────┬────┘
       │                              │
  ┌────▼────┐                         │
  │ Branch 1│                    ┌────▼────┐
  │ (expert)│                    │ Softmax │
  └────┬────┘                    │ [0.6,   │
       │                         │  0.3,   │
  ┌────▼────┐                    │  0.1]   │
  │ Branch 2│                    └────┬────┘
  │ (expert)│                         │
  └────┬────┘                         │
       │                              │
       └───────────┬──────────────────┘
                   │
                   ▼
         0.6×branch0 + 0.3×branch1 + 0.1×branch2
                   │
                   ▼
                Output
```

### Residual Layer

Adds the input directly to the output (skip connection):

```
        Input
          │
     ┌────┴────┐
     │         │
     ▼         │
 ┌────────┐    │
 │  Sub   │    │ Skip
 │ Layers │    │ Connection
 └───┬────┘    │
     │         │
     ▼         │
   ┌───┐       │
   │ + │◀──────┘
   └─┬─┘
     │
     ▼
   Output = SubLayers(Input) + Input
```

Why this matters:
- Gradients can flow directly through the skip connection
- Makes very deep networks trainable
- If sublayers learn identity (do nothing), the layer still passes input through

---

## Summary: Choosing the Right Layer

| Task | Layer | Why |
|------|-------|-----|
| General feature transformation | Dense | Universal approximator |
| Image features | Conv2D | Locality, translation invariance |
| Sequence relationships | Attention | Long-range dependencies |
| Sequential memory | LSTM | Handles long sequences |
| Classification output | Softmax | Probabilities that sum to 1 |
| Training stability | LayerNorm/RMSNorm | Prevents value drift |
| Multiple experts | Parallel + Filter | Learned routing |
| Deep networks | Residual | Skip connections for gradient flow |
