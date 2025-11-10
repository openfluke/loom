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
   * Call any layer initialization function from the registry
   * @param functionName - Name of the layer init function (e.g., "InitDenseLayer")
   * @param paramsJSON - JSON string of parameters array
   * @returns JSON string of layer configuration
   */
  CallLayerInit: (functionName: string, paramsJSON: string) => string;

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

// ============================================================================
// Transformer Inference Types
// ============================================================================

/**
 * Result from tokenizer loading
 */
export interface TokenizerLoadResult {
  success: boolean;
  vocab_size?: number;
  message?: string;
  error?: string;
}

/**
 * Result from transformer model loading
 */
export interface TransformerLoadResult {
  success: boolean;
  num_layers?: number;
  hidden_size?: number;
  vocab_size?: number;
  message?: string;
  error?: string;
}

/**
 * Result from text encoding
 */
export interface EncodeResult {
  success: boolean;
  ids?: number[];
  error?: string;
}

/**
 * Result from token decoding
 */
export interface DecodeResult {
  success: boolean;
  text?: string;
  error?: string;
}

/**
 * Result from text generation
 */
export interface GenerateResult {
  success: boolean;
  generated_text?: string;
  error?: string;
}

/**
 * Result from next token generation
 */
export interface NextTokenResult {
  success: boolean;
  token?: number;
  is_eos?: boolean;
  error?: string;
}

/**
 * Transformer API for LLM inference
 */
export interface TransformerAPI {
  /**
   * Load tokenizer from JSON bytes
   * @param tokenizerData - Uint8Array of tokenizer.json file
   */
  loadTokenizer(tokenizerData: Uint8Array): Promise<TokenizerLoadResult>;

  /**
   * Load transformer model from config and weights bytes
   * @param configData - Uint8Array of config.json file
   * @param weightsData - Uint8Array of model.safetensors file
   */
  loadModel(
    configData: Uint8Array,
    weightsData: Uint8Array
  ): Promise<TransformerLoadResult>;

  /**
   * Encode text to token IDs
   * @param text - Input text to encode
   * @param addSpecialTokens - Whether to add special tokens (default: true)
   */
  encode(text: string, addSpecialTokens?: boolean): Promise<EncodeResult>;

  /**
   * Decode token IDs to text
   * @param tokenIds - Array of token IDs
   * @param skipSpecialTokens - Whether to skip special tokens (default: true)
   */
  decode(
    tokenIds: number[],
    skipSpecialTokens?: boolean
  ): Promise<DecodeResult>;

  /**
   * Generate text from prompt (blocking, all tokens at once)
   * @param prompt - Input prompt
   * @param maxTokens - Maximum tokens to generate (default: 50)
   * @param temperature - Sampling temperature (default: 0.7)
   */
  generate(
    prompt: string,
    maxTokens?: number,
    temperature?: number
  ): Promise<GenerateResult>;

  /**
   * Generate text token-by-token (streaming)
   * @param prompt - Input prompt
   * @param maxTokens - Maximum tokens to generate (default: 50)
   * @param temperature - Sampling temperature (default: 0.7)
   * @yields Token text strings
   */
  generateStream(
    prompt: string,
    maxTokens?: number,
    temperature?: number
  ): AsyncGenerator<string, void, unknown>;
}
