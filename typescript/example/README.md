# Loom TypeScript Examples

This directory contains examples demonstrating the Loom TypeScript/WASM neural network framework.

## Examples

### 1. Command Line Demo (`grid-scatter.ts`)

Multi-agent grid scatter training demo showing 3 heterogeneous agents collaborating.

**Run with:**

```bash
bun install
bun run grid-scatter.ts
```

**Features:**

- 3 heterogeneous agents (Feature Extractor, LSTM, RNN)
- Grid scatter architecture
- Binary classification task
- 99.46% improvement in 0.47 seconds

---

### 2. Browser Interactive Demo (`index.html`)

Full-featured web interface for testing all Loom capabilities.

**Run with:**

```bash
# Start a web server
python3 -m http.server 8081

# Or use any other HTTP server
# npx http-server -p 8081
# php -S localhost:8081
```

Then open: **http://localhost:8081**

**Features:**

#### 📝 Network Configuration

- Create networks from JSON config
- Customizable layer architectures
- Support for all layer types (Dense, LSTM, RNN, GRU, CNN, Parallel)

#### 🎯 Forward Pass

- Interactive input testing
- Real-time output display
- Network information viewing

#### 🤖 Grid Scatter Demo

- Multi-agent training demonstration
- 3 agent collaboration (Feature Extractor + LSTM + RNN)
- Live training metrics
- Prediction testing

#### 🏋️ Training

- Configurable epochs and learning rate
- Pattern recognition task
- Training progress display
- Performance metrics

#### 💾 Model Persistence

- Save trained models to memory
- Load saved models
- Model serialization

#### 📊 Real-time Logs

- Color-coded output (success/error/info)
- Timestamped operations
- Performance timing

---

## Architecture Examples

### Simple Feedforward Network

```json
{
  "batch_size": 1,
  "layers": [
    {
      "type": "dense",
      "input_size": 4,
      "output_size": 8,
      "activation": "relu"
    },
    {
      "type": "dense",
      "input_size": 8,
      "output_size": 2,
      "activation": "softmax"
    }
  ]
}
```

### Grid Scatter Multi-Agent Network

```json
{
  "batch_size": 1,
  "grid_rows": 1,
  "grid_cols": 3,
  "layers_per_cell": 1,
  "layers": [
    {
      "type": "dense",
      "input_size": 8,
      "output_size": 16,
      "activation": "relu"
    },
    {
      "type": "parallel",
      "combine_mode": "grid_scatter",
      "grid_output_rows": 3,
      "grid_output_cols": 1,
      "grid_output_layers": 1,
      "grid_positions": [
        {
          "branch_index": 0,
          "target_row": 0,
          "target_col": 0,
          "target_layer": 0
        },
        {
          "branch_index": 1,
          "target_row": 1,
          "target_col": 0,
          "target_layer": 0
        },
        {
          "branch_index": 2,
          "target_row": 2,
          "target_col": 0,
          "target_layer": 0
        }
      ],
      "branches": [
        {
          "type": "parallel",
          "combine_mode": "add",
          "branches": [
            {
              "type": "dense",
              "input_size": 16,
              "output_size": 8,
              "activation": "relu"
            },
            {
              "type": "dense",
              "input_size": 16,
              "output_size": 8,
              "activation": "gelu"
            }
          ]
        },
        { "type": "lstm", "input_size": 16, "hidden_size": 8, "seq_length": 1 },
        { "type": "rnn", "input_size": 16, "hidden_size": 8, "seq_length": 1 }
      ]
    },
    {
      "type": "dense",
      "input_size": 24,
      "output_size": 2,
      "activation": "sigmoid"
    }
  ]
}
```

---

## Usage Patterns

### TypeScript/JavaScript (Node/Bun)

```typescript
import Loom from "@openfluke/welvet";

const loom = new Loom();
await loom.init();

const network = loom.createNetwork({
  batch_size: 1,
  layers: [
    { type: "dense", input_size: 4, output_size: 8, activation: "relu" },
    { type: "dense", input_size: 8, output_size: 2, activation: "sigmoid" },
  ],
});

// Forward pass
const output = network.ForwardCPU(JSON.stringify([[0.1, 0.2, 0.3, 0.4]]));
console.log(JSON.parse(output));

// Training
const batches = [
  { Input: [0.1, 0.2, 0.3, 0.4], Target: [1.0, 0.0] },
  { Input: [0.4, 0.3, 0.2, 0.1], Target: [0.0, 1.0] },
];

const config = {
  Epochs: 1000,
  LearningRate: 0.01,
  LossType: "mse",
  Verbose: false,
};

const result = network.Train(JSON.stringify([batches, config]));
const trainingData = JSON.parse(result)[0];
console.log(`Final Loss: ${trainingData.FinalLoss}`);
```

### Browser (ES Modules)

```html
<script type="module">
  import Loom from "./dist/index.js";

  const loom = new Loom();
  await loom.initBrowser();

  const network = loom.createNetwork({
    batch_size: 1,
    layers: [
      { type: "dense", input_size: 10, output_size: 8, activation: "relu" },
      { type: "dense", input_size: 8, output_size: 2, activation: "softmax" },
    ],
  });

  const result = network.ForwardCPU(
    JSON.stringify([
      [
        /* 10 inputs */
      ],
    ])
  );
  console.log(JSON.parse(result));
</script>
```

---

## Expected Output

### Grid Scatter Demo

```
🤖 Running Grid Scatter Multi-Agent Training...
✅ Agent network created!
Training for 800 epochs with learning rate 0.150
✅ Training complete!
Training time: 0.47 seconds
Initial Loss: 0.252249
Final Loss: 0.001374
Improvement: 99.46%
Total Epochs: 800

Final predictions:
Sample 0: [0.979, 0.021] → Class 0 (expected 0) ✓
Sample 1: [0.014, 0.986] → Class 1 (expected 1) ✓
Sample 2: [0.052, 0.947] → Class 1 (expected 1) ✓
Sample 3: [0.954, 0.046] → Class 0 (expected 0) ✓

✅ Multi-agent training complete!
```

---

## Troubleshooting

### CORS Issues in Browser

If you see CORS errors, make sure you're serving the files through an HTTP server, not opening the HTML file directly.

### WASM Module Not Found

Ensure `main.wasm` and `wasm_exec.js` are in the same directory as your HTML file or properly referenced.

### Module Import Errors

For browser usage, make sure to use ES module imports (`type="module"` in script tags).

---

## Learn More

- [Parent README](../README.md) - Full API documentation
- [WASM README](../../wasm/README.md) - WASM module details
- [Go Examples](../../examples/) - More network architectures
