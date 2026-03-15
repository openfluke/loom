/**
 * welvet — Type Definitions for the M-POLY-VTD AI Engine
 *
 * Wraps the Loom v0.73.0 WASM module which supports 21 numerical types,
 * systolic grid propagation, target propagation, and WebGPU acceleration.
 */

// ──────────────────────────────────────────────────────────────────────────────
// Numerical Type Constants (matches poly.DType)
// ──────────────────────────────────────────────────────────────────────────────

export const DType = {
  FLOAT64:  0,
  FLOAT32:  1,
  FLOAT16:  2,
  BFLOAT16: 3,
  FP8_E4M3: 4,
  FP8_E5M2: 5,
  INT64:    6,
  INT32:    7,
  INT16:    8,
  INT8:     9,
  UINT64:   10,
  UINT32:   11,
  UINT16:   12,
  UINT8:    13,
  INT4:     14,
  UINT4:    15,
  FP4:      16,
  INT2:     17,
  UINT2:    18,
  TERNARY:  19,
  BINARY:   20,
} as const;
export type DTypeValue = typeof DType[keyof typeof DType];

// ──────────────────────────────────────────────────────────────────────────────
// Layer Type Constants (matches poly.LayerType)
// ──────────────────────────────────────────────────────────────────────────────

export const LayerType = {
  DENSE:               0,
  RMS_NORM:            1,
  LAYER_NORM:          2,
  MHA:                 3,
  SOFTMAX:             4,
  SWIGLU:              5,
  EMBEDDING:           6,
  RESIDUAL:            7,
  KMEANS:              8,
  RNN:                 9,
  LSTM:                10,
  CNN1:                11,
  CNN2:                12,
  CNN3:                13,
  CONV_TRANSPOSED_1D:  14,
  CONV_TRANSPOSED_2D:  15,
  CONV_TRANSPOSED_3D:  16,
} as const;
export type LayerTypeValue = typeof LayerType[keyof typeof LayerType];

// ──────────────────────────────────────────────────────────────────────────────
// Activation Type Constants (matches poly.ActivationType)
// ──────────────────────────────────────────────────────────────────────────────

export const Activation = {
  RELU:    0,
  SILU:    1,
  GELU:    2,
  TANH:    3,
  SIGMOID: 4,
  LINEAR:  5,
} as const;
export type ActivationValue = typeof Activation[keyof typeof Activation];

// ──────────────────────────────────────────────────────────────────────────────
// Network Configuration
// ──────────────────────────────────────────────────────────────────────────────

export interface LayerSpec {
  /** Layer type (use LayerType constants) */
  type: string | number;
  /** Numerical precision (use DType constants, default: FLOAT32) */
  dtype?: number;
  /** Activation function (use Activation constants) */
  activation?: string | number;

  // Dimensions
  input_height?: number;
  input_width?: number;
  input_depth?: number;
  output_height?: number;
  output_width?: number;
  output_depth?: number;

  // CNN parameters
  input_channels?: number;
  filters?: number;
  kernel_size?: number;
  stride?: number;
  padding?: number;

  // Attention parameters
  num_heads?: number;
  num_kv_heads?: number;
  d_model?: number;
  seq_length?: number;

  // Embedding parameters
  vocab_size?: number;
  embedding_dim?: number;

  // Grid position (for volumetric layout)
  z?: number;
  y?: number;
  x?: number;
  l?: number;

  // GPU tiling
  tile_size?: number;
}

export interface NetworkConfig {
  /** Grid depth (number of z-planes). Default: 1 */
  depth?: number;
  /** Grid rows (y-dimension). Default: 1 */
  rows?: number;
  /** Grid columns (x-dimension). Default: 1 */
  cols?: number;
  /** Layers per cell. Default: number of layers */
  layers_per_cell?: number;
  /** Layer definitions (flat list, laid out z→y→x→l) */
  layers: LayerSpec[];
}

// ──────────────────────────────────────────────────────────────────────────────
// Training
// ──────────────────────────────────────────────────────────────────────────────

export interface TrainingBatch {
  input: number[] | Float32Array;
  target: number[] | Float32Array;
}

export interface TrainingResult {
  final_loss: number;
  duration_ms: number;
  epochs_completed: number;
}

// ──────────────────────────────────────────────────────────────────────────────
// Systolic State (stepping API)
// ──────────────────────────────────────────────────────────────────────────────

export interface SystolicState {
  /**
   * Inject input into the first layer of the grid.
   * @param data Float32Array or number[] of input values
   */
  setInput(data: Float32Array | number[]): void;

  /**
   * Advance the systolic grid by one clock cycle.
   * @param captureHistory Whether to store history for backpropagation
   * @returns Duration of the step in milliseconds
   */
  step(captureHistory?: boolean): number;

  /**
   * Read the output of a layer (default: last active layer).
   * @param layerIdx Optional layer index (0-based)
   */
  getOutput(layerIdx?: number): Float32Array;

  /**
   * Backpropagate gradients through the stored history.
   * @param gradients Output gradient (Float32Array or number[])
   * @returns Input gradient as Float32Array
   */
  backward(gradients: Float32Array | number[]): Float32Array;

  /**
   * Apply target propagation (gradient-free alternative to backward).
   * @param target Global target tensor
   * @param lr Learning rate
   */
  applyTargetProp(target: Float32Array | number[], lr: number): void;

  /** Total number of systolic steps executed. */
  stepCount(): number;

  /** Release resources (no-op in WASM, included for API parity). */
  free(): void;
}

// ──────────────────────────────────────────────────────────────────────────────
// Target Propagation State
// ──────────────────────────────────────────────────────────────────────────────

export interface TargetPropState {
  /**
   * Forward pass through all layers, storing local targets.
   * @param input Input data
   * @returns Output as Float32Array
   */
  forward(input: Float32Array | number[]): Float32Array;

  /**
   * Backward pass using target propagation (gap-based, no chain rule).
   * @param target Desired output
   */
  backward(target: Float32Array | number[]): void;

  /**
   * Backward pass using the chain rule (standard backprop via TP state).
   * @param target Desired output
   */
  backwardChainRule(target: Float32Array | number[]): void;

  /**
   * Apply accumulated gap gradients to all layer weights.
   * @param lr Learning rate
   */
  applyGaps(lr?: number): void;

  /** Release resources (no-op in WASM). */
  free(): void;
}

// ──────────────────────────────────────────────────────────────────────────────
// Network
// ──────────────────────────────────────────────────────────────────────────────

export interface Network {
  /**
   * Full sequential forward pass through the network.
   * @param input Float32Array or number[] of inputs
   * @returns Output as Float32Array
   */
  sequentialForward(input: Float32Array | number[]): Float32Array;

  /**
   * Returns a JSON string with network shape info.
   * {depth, rows, cols, layers_per_cell, total_layers, use_gpu, default_dtype}
   */
  getInfo(): string;

  /**
   * Extract the network's DNA fingerprint as a JSON string.
   * Use compareLoomDNA() to compare two fingerprints.
   */
  extractDNA(): string;

  /**
   * Extract the network's full blueprint as a JSON string.
   * @param modelID Optional model identifier
   */
  extractBlueprint(modelID?: string): string;

  /** Total number of layers in the network. */
  getLayerCount(): number;

  /**
   * Get the specification of a single layer.
   * @param layerIdx 0-based layer index
   * @returns JSON string with layer spec
   */
  getLayerSpec(layerIdx: number): string;

  /**
   * Switch a layer's numerical type at runtime (zero-cost when cached).
   * @param layerIdx 0-based layer index
   * @param dtype DType constant
   * @returns JSON status/error
   */
  morphLayer(layerIdx: number, dtype: DTypeValue): string;

  /**
   * Initialize WebGPU for this network.
   * @returns Promise that resolves with status JSON
   */
  initGPU(): Promise<string>;

  /**
   * Upload all layer weights to GPU buffers.
   * @returns Promise that resolves with status JSON
   */
  syncToGPU(): Promise<string>;

  /** Download weights back to CPU and disable GPU mode. */
  syncToCPU(): void;

  /**
   * High-level supervised training loop.
   * @param batchesJSON JSON string of TrainingBatch[]
   * @param epochs Number of epochs
   * @param lr Learning rate
   * @returns JSON string with TrainingResult
   */
  train(batchesJSON: string, epochs: number, lr: number): string;

  /**
   * Create a SystolicState for the stepping API.
   */
  createSystolicState(): SystolicState;

  /**
   * Create a TargetPropState for gradient-free learning.
   * @param useChainRule If true, uses chain-rule backprop instead of gap-based TP
   */
  createTargetPropState(useChainRule?: boolean): TargetPropState;

  /** Release resources (no-op in WASM, included for API parity). */
  free(): void;
}

// ──────────────────────────────────────────────────────────────────────────────
// DNA Comparison Result
// ──────────────────────────────────────────────────────────────────────────────

export interface DNACompareResult {
  similarity: number;
  layer_count_match: boolean;
  depth_match: boolean;
  architecture_match: boolean;
  [key: string]: unknown;
}

// ──────────────────────────────────────────────────────────────────────────────
// Global WASM Exports (set on globalThis by the Go WASM module)
// ──────────────────────────────────────────────────────────────────────────────

declare global {
  /** Build a VolumetricNetwork from a JSON config string. */
  function createLoomNetwork(jsonConfig: string): Network;

  /** Load a network from a SafeTensors file path. */
  function loadLoomNetwork(path: string): Network;

  /** Initialize WebGPU (returns a Promise). */
  function setupWebGPU(): Promise<string>;

  /** Compare two DNA JSON strings for architectural similarity. */
  function compareLoomDNA(dnaA: string, dnaB: string): string;

  /** Get the default TargetPropConfig. */
  function getDefaultTargetPropConfig(): string;
}
