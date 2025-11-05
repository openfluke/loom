/**
 * LOOM Network instance returned by NewNetwork()
 * All Network methods are dynamically exposed via reflection
 */
export interface LoomNetwork {
  // Core training/inference methods
  ForwardCPU(inputJSON: string): string;
  ForwardGPU?(inputJSON: string): string;
  BackwardCPU(gradOutputJSON: string): string;
  BackwardGPU?(gradOutputJSON: string): string;
  UpdateWeights(learningRateJSON: string): string;
  Train(batchesAndConfigJSON: string): string;

  // Layer configuration
  SetLayer(configJSON: string): string;
  InitGPU?(argsJSON: string): string;
  ReleaseGPU?(): string;

  // Model serialization
  SaveModelToString(modelIDJSON: string): string;
  SaveModel?(pathAndIDJSON: string): string;

  // Introspection
  GetMethods(): string;
  ListMethods(): string;
  GetMethodSignature(methodNameJSON: string): string;
  HasMethod(methodName: string): boolean;

  // Additional methods dynamically exposed
  [key: string]: any;
}

/**
 * Global WASM functions exposed by LOOM
 */
export interface LoomAPI {
  /**
   * Create a new neural network
   * @param inputSize - Size of input layer
   * @param gridRows - Number of grid rows
   * @param gridCols - Number of grid columns
   * @param layersPerCell - Number of layers per grid cell
   * @returns LoomNetwork instance with all methods
   */
  NewNetwork: (
    inputSize: number,
    gridRows: number,
    gridCols: number,
    layersPerCell: number
  ) => LoomNetwork;

  /**
   * Load a model from JSON string
   * @param modelJSON - JSON string of the model
   * @param modelID - Model identifier
   * @returns LoomNetwork instance
   */
  LoadModelFromString: (modelJSON: string, modelID: string) => LoomNetwork;

  /**
   * Initialize a dense (fully-connected) layer configuration
   * @param inputSize - Input dimension
   * @param outputSize - Output dimension
   * @param activation - Activation type (0=ReLU, 1=Sigmoid, 2=Tanh, 3=Linear)
   * @returns JSON string of layer configuration
   */
  InitDenseLayer: (
    inputSize: number,
    outputSize: number,
    activation: number
  ) => string;

  /**
   * Initialize a multi-head attention layer configuration
   * @param dModel - Model dimension
   * @param numHeads - Number of attention heads
   * @param seqLength - Sequence length
   * @param activation - Activation type
   * @returns JSON string of layer configuration
   */
  InitMultiHeadAttentionLayer: (
    dModel: number,
    numHeads: number,
    seqLength: number,
    activation: number
  ) => string;

  /**
   * Initialize a 2D convolutional layer configuration
   * @param inputHeight - Input height
   * @param inputWidth - Input width
   * @param inputChannels - Number of input channels
   * @param kernelSize - Kernel size
   * @param stride - Stride
   * @param padding - Padding
   * @param filters - Number of filters
   * @param activation - Activation type
   * @returns JSON string of layer configuration
   */
  InitConv2DLayer: (
    inputHeight: number,
    inputWidth: number,
    inputChannels: number,
    kernelSize: number,
    stride: number,
    padding: number,
    filters: number,
    activation: number
  ) => string;

  /**
   * Initialize an RNN layer configuration
   * @param inputSize - Input size
   * @param hiddenSize - Hidden size
   * @param batchSize - Batch size
   * @param seqLength - Sequence length
   * @returns JSON string of layer configuration
   */
  InitRNNLayer: (
    inputSize: number,
    hiddenSize: number,
    batchSize: number,
    seqLength: number
  ) => string;

  /**
   * Initialize an LSTM layer configuration
   * @param inputSize - Input size
   * @param hiddenSize - Hidden size
   * @param batchSize - Batch size
   * @param seqLength - Sequence length
   * @returns JSON string of layer configuration
   */
  InitLSTMLayer: (
    inputSize: number,
    hiddenSize: number,
    batchSize: number,
    seqLength: number
  ) => string;
}

export interface InitOptions {
  /** Override where the WASM is read from. Useful in exotic deploys (Capacitor, CDN). */
  wasmUrl?: string | URL;
  /** Set false to skip injecting wasm_exec.js (e.g., if you already included it). Default: true */
  injectGoRuntime?: boolean;
}

/**
 * Activation function types
 */
export enum ActivationType {
  ReLU = 0,
  Sigmoid = 1,
  Tanh = 2,
  Softplus = 3,
  LeakyReLU = 4,
  Linear = 5,
}
