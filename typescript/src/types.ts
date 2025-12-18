/**
 * Type definitions for LOOM WASM API
 */

export interface LayerConfig {
  type: string;
  input_size?: number;
  output_size?: number;
  hidden_size?: number;
  seq_length?: number;
  activation?: string;
  combine_mode?: string;
  grid_output_rows?: number;
  grid_output_cols?: number;
  grid_output_layers?: number;
  grid_positions?: GridPosition[];
  branches?: LayerConfig[];
}

export interface NetworkConfig {
  batch_size: number;
  grid_rows?: number;
  grid_cols?: number;
  layers_per_cell?: number;
  layers: LayerConfig[];
}

export interface GridPosition {
  branch_index: number;
  target_row: number;
  target_col: number;
  target_layer: number;
}

export interface TrainingBatch {
  Input: number[];
  Target: number[];
}

export interface TrainingConfig {
  Epochs: number;
  LearningRate: number;
  LossType?: string;
  Verbose?: boolean;
  UseGPU?: boolean;
  PrintEveryBatch?: number;
  GradientClip?: number;
}

export interface TrainingResult {
  final_loss: number;
  duration_ms: number;
  epochs_completed: number;
}

/**
 * Network interface with all available methods
 * All methods take a JSON string of parameters (JSON array)
 * and return a JSON string of results (JSON array)
 */
export interface Network {
  // Forward/Backward propagation
  ForwardCPU(inputsJSON: string): string;
  ForwardGPU(inputsJSON: string): string;
  BackwardCPU(gradientsJSON: string): string;
  BackwardGPU(gradientsJSON: string): string;

  // Weight updates and training
  UpdateWeights(paramsJSON: string): string; // ["learningRate"]
  Train(paramsJSON: string): string; // [batches, config]
  ZeroGradients(paramsJSON: string): string; // []
  ResetGradients(paramsJSON: string): string; // []

  // Serialization - SaveModelToString requires modelID parameter
  SaveModelToString(paramsJSON: string): string; // ["modelID"]
  SerializeModel(paramsJSON: string): string; // ["modelID"]

  // Weight/Bias getters and setters
  GetWeights(paramsJSON: string): string; // [row, col, layer]
  SetWeights(paramsJSON: string): string; // [row, col, layer, weights]
  GetBiases(paramsJSON: string): string; // [row, col, layer]
  SetBiases(paramsJSON: string): string; // [row, col, layer, biases]

  // Layer configuration
  GetLayer(paramsJSON: string): string; // [row, col, layer]
  SetLayer(paramsJSON: string): string; // [row, col, layer, config]
  GetActivation(paramsJSON: string): string; // [row, col, layer]
  SetActivation(paramsJSON: string): string; // [row, col, layer, activation]
  GetLayerType(paramsJSON: string): string; // [row, col, layer]
  GetLayerSizes(paramsJSON: string): string; // [row, col, layer]

  // Network info
  GetBatchSize(paramsJSON: string): string; // []
  SetBatchSize(paramsJSON: string): string; // [batchSize]
  GetGridDimensions(paramsJSON: string): string; // []
  GetLayersPerCell(paramsJSON: string): string; // []
  TotalLayers(paramsJSON: string): string; // []
  GetNetworkInfo(paramsJSON: string): string; // []
  GetTotalParameters(paramsJSON: string): string; // []
  GetMemoryUsage(paramsJSON: string): string; // []
  ValidateArchitecture(paramsJSON: string): string; // []

  // State access
  GetLastOutput(paramsJSON: string): string; // []
  GetLastGradients(paramsJSON: string): string; // []
  Activations(paramsJSON: string): string; // []
  KernelGradients(paramsJSON: string): string; // []
  BiasGradients(paramsJSON: string): string; // []

  // Utilities
  Clone(paramsJSON: string): string; // []
  InitializeWeights(paramsJSON: string): string; // [] or [method]

  // GPU
  InitGPU(paramsJSON: string): string; // []
  ReleaseGPU(paramsJSON: string): string; // []

  // Introspection
  GetMethods(paramsJSON: string): string; // []
  GetMethodsJSON(paramsJSON: string): string; // []
  ListMethods(paramsJSON: string): string; // []
  HasMethod(paramsJSON: string): string; // [methodName]
  GetMethodSignature(paramsJSON: string): string; // [methodName]

  // Stepping API
  ApplyGradients(paramsJSON: string): string; // [learningRate]
  ApplyGradientsAdamW(paramsJSON: string): string; // [learningRate, beta1, beta2, weightDecay]
  ApplyGradientsRMSprop(paramsJSON: string): string; // [learningRate, alpha, epsilon, momentum]
  ApplyGradientsSGDMomentum(paramsJSON: string): string; // [learningRate, momentum, dampening, nesterov]
  
  createStepState(inputSize: number): StepState;
  createTweenState(useChainRule?: boolean): TweenState;
}

/**
 * StepState interface for stepping execution
 */
export interface StepState {
  setInput(data: Float32Array | number[]): void;
  stepForward(): number; // Returns duration in ms
  getOutput(): Float32Array;
  stepBackward(gradients: Float32Array | number[]): Float32Array;
}

/**
 * TweenState interface for neural tweening execution
 */
export interface TweenState {
  /**
   * Perform a tween training step
   * @param input - Input data
   * @param targetClass - Target class index
   * @param outputSize - Size of output layer
   * @param learningRate - Learning rate for this step
   * @returns Loss value
   */
  TweenStep(
    input: Float32Array | number[],
    targetClass: number,
    outputSize: number,
    learningRate: number
  ): number;

  /** Enable/disable chain rule mode */
  setChainRule(enabled: boolean): void;

  /** Get current chain rule setting */
  getChainRule(): boolean;

  /** Get number of tween steps performed */
  getTweenSteps(): number;
}

/**
 * Global WASM functions exposed by main.go
 * Only createLoomNetwork is exposed - use network.SaveModelToString() for saving
 */
declare global {
  function createLoomNetwork(jsonConfig: string): Network;
  function createAdaptationTracker(windowMs: number, totalMs: number): AdaptationTracker;
}

/**
 * AdaptationTracker interface for tracking accuracy during task changes
 */
export interface AdaptationTracker {
  setModelInfo(modelName: string, modeName: string): void;
  scheduleTaskChange(atOffsetMs: number, taskID: number, taskName: string): void;
  start(initialTask: string, initialTaskID: number): void;
  recordOutput(isCorrect: boolean): void;
  getCurrentTask(): number;
  finalize(): string; // Returns JSON result
}

