# src/welvet/utils.py
"""
LOOM Python Bindings - Native library interface
Wraps the LOOM C ABI for Python access
"""

import sys
import json
import ctypes
import platform
from pathlib import Path
from typing import List, Optional, Any
from importlib.resources import files

PKG_DIR = files("welvet")
_RTLD_GLOBAL = getattr(ctypes, "RTLD_GLOBAL", 0)


def _lib_path() -> Path:
    """Determine the correct native library path for the current platform."""
    plat = sys.platform
    arch = platform.machine().lower()
    
    # Normalize architecture names
    arch_map = {
        "x86_64": "x86_64",
        "amd64": "x86_64",
        "aarch64": "arm64",
        "arm64": "arm64",
        "armv7l": "armv7",
        "i686": "x86",
        "i386": "x86",
    }
    arch_key = arch_map.get(arch, arch)
    
    if plat.startswith("linux"):
        lib_name = "libloom.so"
        platform_dir = f"linux_{arch_key}"
    elif plat == "darwin":
        lib_name = "libloom.dylib"
        # Try native architecture first
        platform_dir = f"darwin_{arch_key}"
        p = PKG_DIR / platform_dir / lib_name
        if not Path(p).is_file():
            # Fall back to universal binary
            platform_dir = "darwin_universal"
    elif plat.startswith("win"):
        lib_name = "libloom.dll"
        platform_dir = f"windows_{arch_key}"
    else:
        raise RuntimeError(f"Unsupported platform: {plat} ({arch})")
    
    lib_path = PKG_DIR / platform_dir / lib_name
    
    if not Path(lib_path).is_file():
        raise FileNotFoundError(
            f"LOOM native library not found at {lib_path}\n"
            f"Platform: {plat}, Architecture: {arch_key}\n"
            f"Expected directory: {platform_dir}"
        )
    
    return Path(lib_path)


# Load the native library
_LIB = ctypes.CDLL(str(_lib_path()), mode=_RTLD_GLOBAL)


def _sym(name: str):
    """Get a symbol from the loaded library."""
    try:
        return getattr(_LIB, name)
    except AttributeError:
        return None


def _steal(cptr) -> str:
    """Convert C string pointer to Python string and handle memory."""
    if not cptr:
        return ""
    return ctypes.cast(cptr, ctypes.c_char_p).value.decode("utf-8", errors="replace")


def _json(obj: Any) -> bytes:
    """Convert Python object to JSON bytes."""
    return json.dumps(obj).encode()


# ---- C Function Bindings ----

# Loom_NewNetwork: creates a network, returns JSON with handle
_NewNetwork = _sym("Loom_NewNetwork")
if not _NewNetwork:
    raise AttributeError("Loom_NewNetwork not found in LOOM library")
_NewNetwork.restype = ctypes.c_char_p
_NewNetwork.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_bool]

# Loom_Call: calls a method on a handle, returns JSON
_Call = _sym("Loom_Call")
if not _Call:
    raise AttributeError("Loom_Call not found in LOOM library")
_Call.restype = ctypes.c_char_p
_Call.argtypes = [ctypes.c_longlong, ctypes.c_char_p, ctypes.c_char_p]

# Loom_Free: frees a handle
_Free = _sym("Loom_Free")
if _Free:
    _Free.argtypes = [ctypes.c_longlong]

# Loom_FreeCString: frees C strings returned by the library
_FreeCString = _sym("Loom_FreeCString")
if _FreeCString:
    _FreeCString.argtypes = [ctypes.c_char_p]

# Loom_GetVersion: returns version string
_GetVersion = _sym("Loom_GetVersion")
if _GetVersion:
    _GetVersion.restype = ctypes.c_char_p

# Loom_InitDenseLayer: creates a dense layer configuration
_InitDenseLayer = _sym("Loom_InitDenseLayer")
if _InitDenseLayer:
    _InitDenseLayer.restype = ctypes.c_char_p
    _InitDenseLayer.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]

# Loom_CallLayerInit: generic layer init via registry
_CallLayerInit = _sym("Loom_CallLayerInit")
if _CallLayerInit:
    _CallLayerInit.restype = ctypes.c_char_p
    _CallLayerInit.argtypes = [ctypes.c_char_p, ctypes.c_char_p]

# Loom_ListLayerInitFunctions: list available layer init functions
_ListLayerInitFunctions = _sym("Loom_ListLayerInitFunctions")
if _ListLayerInitFunctions:
    _ListLayerInitFunctions.restype = ctypes.c_char_p
    _ListLayerInitFunctions.argtypes = []

# Loom_SetLayer: sets a layer in the network
_SetLayer = _sym("Loom_SetLayer")
if _SetLayer:
    _SetLayer.restype = ctypes.c_char_p
    _SetLayer.argtypes = [ctypes.c_longlong, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p]

# Loom_GetInfo: gets network information
_GetInfo = _sym("Loom_GetInfo")
if _GetInfo:
    _GetInfo.restype = ctypes.c_char_p
    _GetInfo.argtypes = [ctypes.c_longlong]

# Loom_LoadModel: loads a model from JSON string
_LoadModel = _sym("Loom_LoadModel")
if _LoadModel:
    _LoadModel.restype = ctypes.c_char_p
    _LoadModel.argtypes = [ctypes.c_char_p, ctypes.c_char_p]

# Loom_SaveModel: saves a model to JSON string
_SaveModel = _sym("Loom_SaveModel")
if _SaveModel:
    _SaveModel.restype = ctypes.c_char_p
    _SaveModel.argtypes = [ctypes.c_longlong, ctypes.c_char_p]

# ---- Transformer Functions ----
# Use c_void_p for return types to avoid Python's automatic string conversion
# which would corrupt the pointer before we can free it

# LoadTokenizerFromBytes: load tokenizer from bytes
_LoadTokenizerFromBytes = _sym("LoadTokenizerFromBytes")
if _LoadTokenizerFromBytes:
    _LoadTokenizerFromBytes.restype = ctypes.c_void_p
    _LoadTokenizerFromBytes.argtypes = [ctypes.c_char_p, ctypes.c_int]

# LoadTransformerFromBytes: load transformer model
_LoadTransformerFromBytes = _sym("LoadTransformerFromBytes")
if _LoadTransformerFromBytes:
    _LoadTransformerFromBytes.restype = ctypes.c_void_p
    _LoadTransformerFromBytes.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]

# EncodeText: encode text to token IDs
_EncodeText = _sym("EncodeText")
if _EncodeText:
    _EncodeText.restype = ctypes.c_void_p
    _EncodeText.argtypes = [ctypes.c_char_p, ctypes.c_bool]

# DecodeTokens: decode token IDs to text
_DecodeTokens = _sym("DecodeTokens")
if _DecodeTokens:
    _DecodeTokens.restype = ctypes.c_void_p
    _DecodeTokens.argtypes = [ctypes.c_char_p, ctypes.c_bool]

# GenerateText: generate text from prompt
_GenerateText = _sym("GenerateText")
if _GenerateText:
    _GenerateText.restype = ctypes.c_void_p
    _GenerateText.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_float]

# GenerateNextToken: generate next token
_GenerateNextToken = _sym("GenerateNextToken")
if _GenerateNextToken:
    _GenerateNextToken.restype = ctypes.c_void_p
    _GenerateNextToken.argtypes = [ctypes.c_char_p, ctypes.c_float]


# ---- Activation Types ----
class Activation:
    """Neural network activation function types."""
    SCALED_RELU = 0  # v * 1.1, then ReLU
    RELU = 0         # Alias for SCALED_RELU (default)
    SIGMOID = 1      # 1 / (1 + exp(-v))
    TANH = 2         # tanh(v)
    SOFTPLUS = 3     # log(1 + exp(v))
    LEAKY_RELU = 4   # v if v >= 0, else v * 0.1
    LINEAR = 3       # Alias for SOFTPLUS (deprecated)


# ---- Public Python API ----

def _json_call(handle, method, *args):
    """
    Call a method on a LOOM network handle using the reflection API.
    
    Args:
        handle: Network handle (int)
        method: Method name (str)
        *args: Method arguments (will be converted to JSON array)
    
    Returns:
        Parsed JSON response (will be an array of return values)
    """
    args_json = json.dumps(list(args)).encode('utf-8')
    method_bytes = method.encode('utf-8')
    
    response = _Call(int(handle), method_bytes, args_json)
    if not response:
        raise RuntimeError("No response from C library")
    
    result = json.loads(response.decode('utf-8'))
    
    # Check for error response
    if isinstance(result, dict) and "error" in result:
        error_msg = result["error"]
        raise RuntimeError(f"C library error: {error_msg}")
    
    return result


def create_network(input_size: int, hidden_size: int = None, output_size: int = None, 
                  use_gpu: bool = False, grid_rows: int = 2, grid_cols: int = 2, 
                  layers_per_cell: int = 3) -> int:
    """
    Create a new LOOM neural network.
    
    LOOM uses a grid-based architecture (gridRows × gridCols × layersPerCell).
    You can either:
    1. Provide grid_rows, grid_cols, layers_per_cell directly
    2. Provide hidden_size for simplified API (grid calculated automatically)
    
    Args:
        input_size: Size of input layer
        hidden_size: (Optional) Hidden layer size - if provided, calculates grid automatically
        output_size: (Optional) Output layer size (for compatibility, not used in grid API)
        use_gpu: Enable GPU acceleration (default: False)
        grid_rows: Number of rows in grid (default: 2)
        grid_cols: Number of columns in grid (default: 2)
        layers_per_cell: Number of layers per grid cell (default: 3)
    
    Returns:
        Network handle (integer)
    
    Raises:
        RuntimeError: If network creation fails
    """
    # If hidden_size provided, calculate grid parameters
    if hidden_size is not None:
        # Simple heuristic: grid_size ≈ sqrt(hidden_size / layers_per_cell)
        import math
        grid_size = max(1, int(math.sqrt(hidden_size / layers_per_cell)))
        grid_rows = grid_cols = grid_size
    
    response = _NewNetwork(
        int(input_size),
        int(grid_rows),
        int(grid_cols),
        int(layers_per_cell),
        bool(use_gpu)
    )
    
    if not response:
        raise RuntimeError("Failed to create network")
    
    result = json.loads(response.decode('utf-8'))
    
    # Check for error
    if "error" in result:
        error = result["error"]
        raise RuntimeError(f"Failed to create network: {error}")
    
    # Extract handle
    if "handle" not in result:
        raise RuntimeError("Network created but no handle returned")
    
    return result["handle"]


def load_model_from_string(model_json: str, model_id: str = "loaded_model") -> int:
    """
    Load a complete model from JSON string (with all weights and configuration).
    
    This is the easy way - just pass the JSON and get back a fully configured network!
    
    Args:
        model_json: JSON string containing the complete model (structure + weights)
        model_id: Optional model identifier
    
    Returns:
        Network handle (integer)
    
    Example:
        # Load model from file
        with open('model.json', 'r') as f:
            model_json = f.read()
        
        network = load_model_from_string(model_json, "my_model")
        
        # Use it immediately
        output = forward(network, input_data)
    """
    if not _LoadModel:
        raise RuntimeError("LoadModel function not available in library")
    
    response = _LoadModel(model_json.encode('utf-8'), model_id.encode('utf-8'))
    
    if not response:
        raise RuntimeError("Failed to load model")
    
    result = json.loads(response.decode('utf-8'))
    
    # Check for error
    if "error" in result:
        error = result["error"]
        raise RuntimeError(f"Failed to load model: {error}")
    
    # Extract handle
    if "handle" not in result:
        raise RuntimeError("Model loaded but no handle returned")
    
    return result["handle"]


def save_model_to_string(handle: int, model_id: str = "saved_model") -> str:
    """
    Save a model to JSON string (with all weights and configuration).
    
    Args:
        handle: Network handle
        model_id: Model identifier
    
    Returns:
        JSON string containing the complete model
    
    Example:
        model_json = save_model_to_string(network, "my_model")
        
        # Save to file
        with open('model.json', 'w') as f:
            f.write(model_json)
    """
    if not _SaveModel:
        raise RuntimeError("SaveModel function not available in library")
    
    response = _SaveModel(int(handle), model_id.encode('utf-8'))
    
    if not response:
        raise RuntimeError("Failed to save model")
    
    # The response should be the JSON string directly
    result_str = response.decode('utf-8')
    
    # Check if it's an error wrapped in JSON
    try:
        result = json.loads(result_str)
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(f"Failed to save model: {result['error']}")
    except json.JSONDecodeError:
        # Not JSON, probably the raw model string - this is fine
        pass
    
    return result_str


def free_network(handle: int) -> None:
    """
    Free network resources.
    
    Args:
        handle: Network handle from create_network()
    """
    if _Free:
        _Free(int(handle))


def forward(handle: int, input_data: List[float]) -> List[float]:
    """
    Perform forward pass through the network.
    
    Args:
        handle: Network handle
        input_data: Input vector as list of floats
    
    Returns:
        Output vector as list of floats
    """
    result = _json_call(handle, "ForwardCPU", input_data)
    
    # Result is an array of return values - first element is the output
    if isinstance(result, list) and len(result) > 0:
        return result[0]
    
    raise RuntimeError(f"Unexpected forward response format: {result}")


def get_output(handle: int, output_size: int) -> List[float]:
    """
    Get network output from last forward pass.
    
    Note: In the new API, output is returned directly from forward().
    This function is kept for compatibility.
    
    Args:
        handle: Network handle
        output_size: Expected size of output vector
    
    Returns:
        Output vector as list of floats
    """
    # GetOutput might not exist - just return empty
    return []


def backward(handle: int, target_data: List[float]) -> None:
    """
    Perform backward pass for training.
    
    Args:
        handle: Network handle
        target_data: Target/label vector as list of floats
    """
    _json_call(handle, "BackwardCPU", target_data)


def update_weights(handle: int, learning_rate: float) -> None:
    """
    Update network weights using computed gradients.
    
    Args:
        handle: Network handle
        learning_rate: Learning rate for gradient descent
    """
    _json_call(handle, "UpdateWeights", float(learning_rate))


def initialize_gpu(handle: int) -> bool:
    """
    Explicitly initialize GPU resources.
    
    Args:
        handle: Network handle
    
    Returns:
        True if GPU initialized successfully, False otherwise
    """
    try:
        _json_call(handle, "InitGPU")
        return True
    except RuntimeError:
        return False


def cleanup_gpu(handle: int) -> None:
    """
    Clean up GPU resources.
    
    Args:
        handle: Network handle
    """
    try:
        _json_call(handle, "ReleaseGPU")
    except Exception:
        pass  # Best effort cleanup



def get_output(handle: int, output_size: int) -> List[float]:
    """
    Get network output from last forward pass.
    
    Note: In the new API, output is returned directly from forward().
    This function is kept for compatibility.
    
    Args:
        handle: Network handle
        output_size: Expected size of output vector
    
    Returns:
        Output vector as list of floats
    """
    # GetOutput might not exist - try getting info or just use forward
    try:
        result = _json_call(handle, "GetOutput", {})
        
        if "output" in result:
            return result["output"]
        
        if isinstance(result, list):
            return result
        
        if "return" in result:
            return result["return"]
    except RuntimeError:
        # Method might not exist
        pass
    
    return []


def backward(handle: int, target_data: List[float]) -> None:
    """
    Perform backward pass for training.
    
    Args:
        handle: Network handle
        target_data: Target/label vector as list of floats
    """
    _json_call(handle, "BackwardCPU", target_data)


def update_weights(handle: int, learning_rate: float) -> None:
    """
    Update network weights using computed gradients.
    
    Args:
        handle: Network handle
        learning_rate: Learning rate for gradient descent
    """
    _json_call(handle, "UpdateWeights", float(learning_rate))


def initialize_gpu(handle: int) -> bool:
    """
    Explicitly initialize GPU resources.
    
    Args:
        handle: Network handle
    
    Returns:
        True if GPU initialized successfully, False otherwise
    """
    try:
        _json_call(handle, "InitGPU")
        return True
    except RuntimeError:
        return False


def cleanup_gpu(handle: int) -> None:
    """
    Clean up GPU resources.
    
    Args:
        handle: Network handle
    """
    try:
        _json_call(handle, "ReleaseGPU")
    except Exception:
        pass  # Best effort cleanup


def get_version() -> str:
    """
    Get LOOM library version string.
    
    Returns:
        Version string (e.g., "0.0.1")
    """
    if not _GetVersion:
        return "unknown"
    
    version_ptr = _GetVersion()
    if version_ptr:
        return version_ptr.decode("utf-8")
    return "unknown"


# ---- Layer Configuration API ----

def init_dense_layer(input_size: int, output_size: int, activation: int = 0) -> dict:
    """
    Initialize a dense (fully connected) layer configuration.
    
    Args:
        input_size: Number of input neurons
        output_size: Number of output neurons
        activation: Activation function (use Activation constants: RELU=0, SIGMOID=1, TANH=2, LINEAR=3)
    
    Returns:
        Layer configuration dict
    
    Example:
        layer_config = init_dense_layer(784, 128, Activation.RELU)
    """
    if not _InitDenseLayer:
        raise RuntimeError("InitDenseLayer function not available")
    
    response = _InitDenseLayer(int(input_size), int(output_size), int(activation))
    if not response:
        raise RuntimeError("Failed to initialize dense layer")
    
    config = json.loads(response.decode('utf-8'))
    
    if isinstance(config, dict) and "error" in config:
        raise RuntimeError(f"Failed to initialize layer: {config['error']}")
    
    return config


def call_layer_init(function_name: str, *args) -> dict:
    """
    Call any layer initialization function via the registry.
    
    This is a generic interface to call any of the layer init functions:
    - InitDenseLayer
    - InitConv2DLayer
    - InitMultiHeadAttentionLayer
    - InitRNNLayer
    - InitLSTMLayer
    
    Args:
        function_name: Name of the function to call (e.g., "InitDenseLayer")
        *args: Arguments to pass to the function
    
    Returns:
        Layer configuration dict
    
    Examples:
        # Dense layer: init_dense_layer(32, 64, 0)
        config = call_layer_init("InitDenseLayer", 32, 64, 0)
        
        # Conv2D layer: init_conv2d_layer(28, 28, 1, 3, 1, 1, 32, 0)
        config = call_layer_init("InitConv2DLayer", 28, 28, 1, 3, 1, 1, 32, 0)
        
        # LSTM layer: init_lstm_layer(128, 256, 1, 10)
        config = call_layer_init("InitLSTMLayer", 128, 256, 1, 10)
    """
    if not _CallLayerInit:
        raise RuntimeError("CallLayerInit function not available")
    
    args_json = json.dumps(list(args))
    response = _CallLayerInit(function_name.encode('utf-8'), args_json.encode('utf-8'))
    
    if not response:
        raise RuntimeError(f"Failed to call {function_name}")
    
    config = json.loads(response.decode('utf-8'))
    
    if isinstance(config, dict) and "error" in config:
        raise RuntimeError(f"Failed to call {function_name}: {config['error']}")
    
    return config


def list_layer_init_functions() -> list:
    """
    List all available layer initialization functions.
    
    Returns:
        List of dicts with function metadata including:
        - Name: Function name
        - NumArgs: Number of arguments
        - ArgTypes: List of argument type names
    
    Example:
        functions = list_layer_init_functions()
        for fn in functions:
            print(f"{fn['Name']}: {fn['NumArgs']} args - {fn['ArgTypes']}")
    """
    if not _ListLayerInitFunctions:
        raise RuntimeError("ListLayerInitFunctions function not available")
    
    response = _ListLayerInitFunctions()
    
    if not response:
        raise RuntimeError("Failed to list layer init functions")
    
    result = json.loads(response.decode('utf-8'))
    
    if isinstance(result, dict) and "error" in result:
        raise RuntimeError(f"Failed to list functions: {result['error']}")
    
    return result


def set_layer(handle: int, row: int, col: int, layer_index: int, layer_config: dict) -> None:
    """
    Set a layer in the network grid.
    
    Args:
        handle: Network handle
        row: Grid row (0-indexed)
        col: Grid column (0-indexed)
        layer_index: Layer index within the grid cell (0-indexed)
        layer_config: Layer configuration from init_dense_layer()
    
    Example:
        layer = init_dense_layer(784, 128, Activation.RELU)
        set_layer(net, row=0, col=0, layer_index=0, layer_config=layer)
    """
    if not _SetLayer:
        raise RuntimeError("SetLayer function not available")
    
    config_json = json.dumps(layer_config).encode('utf-8')
    response = _SetLayer(int(handle), int(row), int(col), int(layer_index), config_json)
    
    if not response:
        raise RuntimeError("Failed to set layer")
    
    result = json.loads(response.decode('utf-8'))
    
    if isinstance(result, dict) and "error" in result:
        raise RuntimeError(f"Failed to set layer: {result['error']}")


def get_network_info(handle: int) -> dict:
    """
    Get detailed information about a network.
    
    Args:
        handle: Network handle
    
    Returns:
        Dict with network information including:
        - type: Network type
        - gpu_enabled: Whether GPU is enabled
        - grid_rows, grid_cols: Grid dimensions
        - layers_per_cell: Layers per grid cell
        - total_layers: Total number of layers
    """
    if not _GetInfo:
        raise RuntimeError("GetInfo function not available")
    
    response = _GetInfo(int(handle))
    if not response:
        raise RuntimeError("Failed to get network info")
    
    info = json.loads(response.decode('utf-8'))
    
    if isinstance(info, dict) and "error" in info:
        raise RuntimeError(f"Failed to get network info: {info['error']}")
    
    return info


# ---- High-Level Training Helpers ----

def configure_sequential_network(handle: int, layer_sizes: List[int], 
                                activations: List[int] = None) -> None:
    """
    Configure a simple sequential (feedforward) network in a 1x1 grid.
    
    This is a convenience function for the most common network architecture.
    The network must be created with grid_rows=1, grid_cols=1, and 
    layers_per_cell equal to len(layer_sizes)-1.
    
    Args:
        handle: Network handle
        layer_sizes: List of layer sizes [input, hidden1, hidden2, ..., output]
        activations: List of activation functions for each layer (excluding input).
                    If None, uses ReLU for hidden layers and Sigmoid for output.
    
    Example:
        # Create network with 1x1 grid and 2 layers
        net = create_network(input_size=784, grid_rows=1, grid_cols=1, layers_per_cell=2)
        
        # Configure as: 784 -> 128 (ReLU) -> 10 (Sigmoid)
        configure_sequential_network(net, [784, 128, 10])
    """
    if len(layer_sizes) < 2:
        raise ValueError("Need at least input and output layer sizes")
    
    num_layers = len(layer_sizes) - 1
    
    # Default activations: ReLU for hidden, Sigmoid for output
    if activations is None:
        activations = [Activation.RELU] * (num_layers - 1) + [Activation.SIGMOID]
    
    if len(activations) != num_layers:
        raise ValueError(f"Need {num_layers} activations for {len(layer_sizes)} layer sizes")
    
    # Configure each layer
    for i in range(num_layers):
        layer_config = init_dense_layer(
            input_size=layer_sizes[i],
            output_size=layer_sizes[i + 1],
            activation=activations[i]
        )
        set_layer(handle, row=0, col=0, layer_index=i, layer_config=layer_config)


def train_epoch(handle: int, inputs: List[List[float]], targets: List[List[float]], 
               learning_rate: float = 0.01) -> float:
    """
    Train the network for one epoch.
    
    Args:
        handle: Network handle
        inputs: List of input vectors
        targets: List of target vectors
        learning_rate: Learning rate for weight updates
    
    Returns:
        Average loss for the epoch (MSE)
    
    Example:
        loss = train_epoch(net, train_inputs, train_targets, learning_rate=0.01)
        print(f"Epoch loss: {loss:.4f}")
    """
    if len(inputs) != len(targets):
        raise ValueError("Number of inputs and targets must match")
    
    total_loss = 0.0
    
    for input_vec, target_vec in zip(inputs, targets):
        # Forward pass
        output = forward(handle, input_vec)
        
        if len(output) == 0:
            raise RuntimeError("Network produced no output. Are layers configured?")
        
        # Calculate MSE loss
        loss = sum((o - t) ** 2 for o, t in zip(output, target_vec)) / len(output)
        total_loss += loss
        
        # Backward pass
        backward(handle, target_vec)
        
        # Update weights
        update_weights(handle, learning_rate)
    
    return total_loss / len(inputs)

def train(handle: int, batches: List[tuple], config: dict = None) -> dict:
    """
    Train the network using the high-level Train API.
    
    Args:
        handle: Network handle
        batches: List of (input, target) tuples where:
                 - input: List of floats (input data)
                 - target: List of floats (target/label data)
        config: Optional training configuration dict with keys:
                - epochs: int (default: 5)
                - learning_rate: float (default: 0.001)
                - use_gpu: bool (default: False)
                - print_every_batch: int (default: 0)
                - gradient_clip: float (default: 0.0)
                - loss_type: str (default: "mse")
                - verbose: bool (default: True)
    
    Returns:
        Training result dict with keys:
        - FinalLoss: float
        - BestLoss: float
        - TotalTime: int (nanoseconds)
        - AvgThroughput: float (samples/sec)
        - LossHistory: List[float]
    
    Example:
        batches = [
            ([0.1, 0.2], [1.0, 0.0]),
            ([0.8, 0.9], [0.0, 1.0]),
        ]
        result = train(net, batches, {
            'epochs': 10,
            'learning_rate': 0.01,
            'verbose': True
        })
        print(f"Final loss: {result['FinalLoss']}")
    """
    # Convert batches to the format expected by Go
    training_batches = []
    for input_data, target_data in batches:
        training_batches.append({
            "Input": input_data,
            "Target": target_data
        })
    
    # Default configuration
    if config is None:
        config = {}
    
    training_config = {
        "Epochs": config.get("epochs", 5),
        "LearningRate": config.get("learning_rate", 0.001),
        "UseGPU": config.get("use_gpu", False),
        "PrintEveryBatch": config.get("print_every_batch", 0),
        "GradientClip": config.get("gradient_clip", 0.0),
        "LossType": config.get("loss_type", "mse"),
        "Verbose": config.get("verbose", True),
        "EvaluateEveryN": config.get("evaluate_every_n", 0),
        "ValidationInputs": config.get("validation_inputs"),
        "ValidationTargets": config.get("validation_targets"),
    }
    
    # Call Train method via reflection
    result = _json_call(handle, "Train", training_batches, training_config)
    
    # Result is [TrainingResult, error] - we want the first element
    if isinstance(result, list) and len(result) > 0:
        training_result = result[0]
        if training_result is None:
            # Check if there was an error (second element)
            if len(result) > 1 and result[1]:
                raise RuntimeError(f"Training failed: {result[1]}")
            raise RuntimeError("Training returned no result")
        return training_result
    
    raise RuntimeError(f"Unexpected Train response format: {result}")


# ---- Transformer Inference API ----

def _free_cstring_ptr(ptr):
    """Helper to free a c_void_p by casting it to c_char_p."""
    if _FreeCString and ptr:
        _FreeCString(ctypes.cast(ptr, ctypes.c_char_p))


def load_tokenizer_from_bytes(data: bytes) -> dict:
    """
    Load tokenizer from bytes.
    
    Args:
        data: Tokenizer JSON bytes
    
    Returns:
        Dict with 'success', 'vocab_size', etc.
    """
    if not _LoadTokenizerFromBytes:
        raise RuntimeError("LoadTokenizerFromBytes not available in library")
    
    result_ptr = _LoadTokenizerFromBytes(data, len(data))
    if not result_ptr:
        raise RuntimeError("Failed to load tokenizer")
    
    result_json = ctypes.string_at(result_ptr).decode('utf-8')
    _free_cstring_ptr(result_ptr)
    
    result = json.loads(result_json)
    if not result.get('success'):
        raise RuntimeError(f"Failed to load tokenizer: {result.get('error', 'Unknown error')}")
    
    return result


def load_transformer_from_bytes(config_data: bytes, weights_data: bytes) -> dict:
    """
    Load transformer model from config and weights bytes.
    
    Args:
        config_data: Config JSON bytes
        weights_data: Model weights (safetensors format)
    
    Returns:
        Dict with 'success', 'num_layers', 'hidden_size', etc.
    """
    if not _LoadTransformerFromBytes:
        raise RuntimeError("LoadTransformerFromBytes not available in library")
    
    result_ptr = _LoadTransformerFromBytes(
        config_data, len(config_data),
        weights_data, len(weights_data)
    )
    if not result_ptr:
        raise RuntimeError("Failed to load transformer")
    
    result_json = ctypes.string_at(result_ptr).decode('utf-8')
    _free_cstring_ptr(result_ptr)
    
    result = json.loads(result_json)
    if not result.get('success'):
        raise RuntimeError(f"Failed to load transformer: {result.get('error', 'Unknown error')}")
    
    return result


def encode_text(text: str, add_special_tokens: bool = True) -> list:
    """
    Encode text to token IDs.
    
    Args:
        text: Text to encode
        add_special_tokens: Whether to add special tokens
    
    Returns:
        List of token IDs
    """
    if not _EncodeText:
        raise RuntimeError("EncodeText not available in library")
    
    result_ptr = _EncodeText(text.encode('utf-8'), add_special_tokens)
    if not result_ptr:
        raise RuntimeError("Failed to encode text")
    
    result_json = ctypes.string_at(result_ptr).decode('utf-8')
    _free_cstring_ptr(result_ptr)
    
    result = json.loads(result_json)
    if not result.get('success'):
        raise RuntimeError(f"Failed to encode text: {result.get('error', 'Unknown error')}")
    
    return result['ids']


def decode_tokens(ids: list, skip_special_tokens: bool = True) -> str:
    """
    Decode token IDs to text.
    
    Args:
        ids: List of token IDs
        skip_special_tokens: Whether to skip special tokens
    
    Returns:
        Decoded text string
    """
    if not _DecodeTokens:
        raise RuntimeError("DecodeTokens not available in library")
    
    ids_json = json.dumps(ids).encode('utf-8')
    result_ptr = _DecodeTokens(ids_json, skip_special_tokens)
    if not result_ptr:
        raise RuntimeError("Failed to decode tokens")
    
    result_json = ctypes.string_at(result_ptr).decode('utf-8')
    _free_cstring_ptr(result_ptr)
    
    result = json.loads(result_json)
    if not result.get('success'):
        raise RuntimeError(f"Failed to decode tokens: {result.get('error', 'Unknown error')}")
    
    return result['text']


def generate_text(prompt: str, max_tokens: int = 50, temperature: float = 0.7) -> str:
    """
    Generate text from prompt (all at once).
    
    Args:
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
    
    Returns:
        Generated text
    """
    if not _GenerateText:
        raise RuntimeError("GenerateText not available in library")
    
    result_ptr = _GenerateText(prompt.encode('utf-8'), max_tokens, temperature)
    if not result_ptr:
        raise RuntimeError("Failed to generate text")
    
    result_json = ctypes.string_at(result_ptr).decode('utf-8')
    _free_cstring_ptr(result_ptr)
    
    result = json.loads(result_json)
    if not result.get('success'):
        raise RuntimeError(f"Failed to generate text: {result.get('error', 'Unknown error')}")
    
    return result['generated_text']


def generate_stream(prompt: str, max_tokens: int = 50, temperature: float = 0.7):
    """
    Generate text token-by-token (streaming).
    
    Args:
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
    
    Yields:
        Token text strings
    
    Example:
        for token in generate_stream("Once upon a time", max_tokens=50):
            print(token, end='', flush=True)
    """
    if not _EncodeText or not _GenerateNextToken or not _DecodeTokens:
        raise RuntimeError("Streaming functions not available in library")
    
    # Encode prompt
    tokens = encode_text(prompt, add_special_tokens=True)
    
    # Generate tokens one at a time
    for _ in range(max_tokens):
        # Generate next token
        tokens_json = json.dumps(tokens).encode('utf-8')
        result_ptr = _GenerateNextToken(tokens_json, temperature)
        if not result_ptr:
            break
        
        result_json = ctypes.string_at(result_ptr).decode('utf-8')
        _free_cstring_ptr(result_ptr)
        
        result = json.loads(result_json)
        if not result.get('success'):
            break
        
        next_token = result['token']
        tokens.append(next_token)
        
        # Decode just this token
        token_text = decode_tokens([next_token], skip_special_tokens=True)
        yield token_text
        
        # Check for EOS
        if result.get('is_eos', False):
            break
