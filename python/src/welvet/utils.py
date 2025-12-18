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

# Loom_NewNetwork: creates a network, returns JSON with handle (legacy API - optional)
_NewNetwork = _sym("Loom_NewNetwork")
if _NewNetwork:
    _NewNetwork.restype = ctypes.c_char_p
    _NewNetwork.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_bool]

# Loom_Call: calls a method on a handle, returns JSON (legacy API - optional)
_Call = _sym("Loom_Call")
if _Call:
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

# ---- New Simple API (global network instance) ----

# CreateLoomNetwork: creates a network from JSON config
_CreateLoomNetwork = _sym("CreateLoomNetwork")
if _CreateLoomNetwork:
    _CreateLoomNetwork.restype = ctypes.c_char_p
    _CreateLoomNetwork.argtypes = [ctypes.c_char_p]

# LoomForward: forward pass with float array
_LoomForward = _sym("LoomForward")
if _LoomForward:
    _LoomForward.restype = ctypes.c_char_p
    _LoomForward.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]

# LoomBackward: backward pass with gradients
_LoomBackward = _sym("LoomBackward")
if _LoomBackward:
    _LoomBackward.restype = ctypes.c_char_p
    _LoomBackward.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]

# LoomUpdateWeights: update weights with learning rate
_LoomUpdateWeights = _sym("LoomUpdateWeights")
if _LoomUpdateWeights:
    _LoomUpdateWeights.restype = None
    _LoomUpdateWeights.argtypes = [ctypes.c_float]

# LoomTrain: train network with batches
_LoomTrain = _sym("LoomTrain")
if _LoomTrain:
    _LoomTrain.restype = ctypes.c_char_p
    _LoomTrain.argtypes = [ctypes.c_char_p, ctypes.c_char_p]

# LoomSaveModel: save model to JSON string
_LoomSaveModel = _sym("LoomSaveModel")
if _LoomSaveModel:
    _LoomSaveModel.restype = ctypes.c_char_p
    _LoomSaveModel.argtypes = [ctypes.c_char_p]

# LoomLoadModel: load model from JSON string
_LoomLoadModel = _sym("LoomLoadModel")
if _LoomLoadModel:
    _LoomLoadModel.restype = ctypes.c_char_p
    _LoomLoadModel.argtypes = [ctypes.c_char_p, ctypes.c_char_p]

# LoomGetNetworkInfo: get network information
_LoomGetNetworkInfo = _sym("LoomGetNetworkInfo")
if _LoomGetNetworkInfo:
    _LoomGetNetworkInfo.restype = ctypes.c_char_p
    _LoomGetNetworkInfo.argtypes = []

# LoomEvaluateNetwork: evaluate network
_LoomEvaluateNetwork = _sym("LoomEvaluateNetwork")
if _LoomEvaluateNetwork:
    _LoomEvaluateNetwork.restype = ctypes.c_char_p
    _LoomEvaluateNetwork.argtypes = [ctypes.c_char_p, ctypes.c_char_p]

# FreeLoomString: free C strings returned by LOOM
_FreeLoomString = _sym("FreeLoomString")
if _FreeLoomString:
    _FreeLoomString.argtypes = [ctypes.c_char_p]

# ---- TweenState API ----

# LoomCreateTweenState: create a tween state
_LoomCreateTweenState = _sym("LoomCreateTweenState")
if _LoomCreateTweenState:
    _LoomCreateTweenState.restype = ctypes.c_longlong
    _LoomCreateTweenState.argtypes = [ctypes.c_int]

# LoomTweenStep: apply a tween step
_LoomTweenStep = _sym("LoomTweenStep")
if _LoomTweenStep:
    _LoomTweenStep.restype = ctypes.c_float
    _LoomTweenStep.argtypes = [
        ctypes.c_longlong,  # handle
        ctypes.POINTER(ctypes.c_float),  # input
        ctypes.c_int,  # inputLen
        ctypes.c_int,  # targetClass
        ctypes.c_int,  # outputSize
        ctypes.c_float  # learningRate
    ]

# LoomFreeTweenState: free tween state
_LoomFreeTweenState = _sym("LoomFreeTweenState")
if _LoomFreeTweenState:
    _LoomFreeTweenState.argtypes = [ctypes.c_longlong]

# ---- AdaptationTracker API ----

# LoomCreateAdaptationTracker: create tracker
_LoomCreateAdaptationTracker = _sym("LoomCreateAdaptationTracker")
if _LoomCreateAdaptationTracker:
    _LoomCreateAdaptationTracker.restype = ctypes.c_longlong
    _LoomCreateAdaptationTracker.argtypes = [ctypes.c_int, ctypes.c_int]

# LoomTrackerSetModelInfo: set model info
_LoomTrackerSetModelInfo = _sym("LoomTrackerSetModelInfo")
if _LoomTrackerSetModelInfo:
    _LoomTrackerSetModelInfo.argtypes = [ctypes.c_longlong, ctypes.c_char_p, ctypes.c_char_p]

# LoomTrackerScheduleTaskChange: schedule task change
_LoomTrackerScheduleTaskChange = _sym("LoomTrackerScheduleTaskChange")
if _LoomTrackerScheduleTaskChange:
    _LoomTrackerScheduleTaskChange.argtypes = [ctypes.c_longlong, ctypes.c_int, ctypes.c_int, ctypes.c_char_p]

# LoomTrackerStart: start tracking
_LoomTrackerStart = _sym("LoomTrackerStart")
if _LoomTrackerStart:
    _LoomTrackerStart.argtypes = [ctypes.c_longlong, ctypes.c_char_p, ctypes.c_int]

# LoomTrackerRecordOutput: record an output
_LoomTrackerRecordOutput = _sym("LoomTrackerRecordOutput")
if _LoomTrackerRecordOutput:
    _LoomTrackerRecordOutput.restype = ctypes.c_int
    _LoomTrackerRecordOutput.argtypes = [ctypes.c_longlong, ctypes.c_int]

# LoomTrackerGetCurrentTask: get current task
_LoomTrackerGetCurrentTask = _sym("LoomTrackerGetCurrentTask")
if _LoomTrackerGetCurrentTask:
    _LoomTrackerGetCurrentTask.restype = ctypes.c_int
    _LoomTrackerGetCurrentTask.argtypes = [ctypes.c_longlong]

# LoomTrackerFinalize: finalize and get results
_LoomTrackerFinalize = _sym("LoomTrackerFinalize")
if _LoomTrackerFinalize:
    _LoomTrackerFinalize.restype = ctypes.c_char_p
    _LoomTrackerFinalize.argtypes = [ctypes.c_longlong]

# LoomFreeTracker: free tracker
_LoomFreeTracker = _sym("LoomFreeTracker")
if _LoomFreeTracker:
    _LoomFreeTracker.argtypes = [ctypes.c_longlong]


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


# ---- New Simple API (global network instance) ----

def create_network_from_json(json_config: str) -> None:
    """
    Create a network from JSON configuration (new simple API).
    Uses a global network instance - no handle management needed!
    
    Args:
        json_config: JSON string or dict with network configuration
    
    Raises:
        RuntimeError: If network creation fails
    
    Example:
        config = {
            "batch_size": 1,
            "grid_rows": 1,
            "grid_cols": 3,
            "layers_per_cell": 1,
            "layers": [
                {"type": "dense", "input_size": 8, "output_size": 16, "activation": "relu"},
                # ... more layers
            ]
        }
        create_network_from_json(json.dumps(config))
    """
    if not _CreateLoomNetwork:
        raise RuntimeError("CreateLoomNetwork not available in library")
    
    # Convert dict to JSON string if needed
    if isinstance(json_config, dict):
        json_config = json.dumps(json_config)
    
    response = _CreateLoomNetwork(json_config.encode('utf-8'))
    if not response:
        raise RuntimeError("Failed to create network")
    
    result = json.loads(response.decode('utf-8'))
    
    if "error" in result:
        raise RuntimeError(f"Failed to create network: {result['error']}")


def forward_simple(inputs: List[float]) -> List[float]:
    """
    Forward pass with the global network (new simple API).
    
    Args:
        inputs: Input vector as list of floats
    
    Returns:
        Output vector as list of floats
    """
    if not _LoomForward:
        raise RuntimeError("LoomForward not available in library")
    
    # Convert to ctypes array
    input_array = (ctypes.c_float * len(inputs))(*inputs)
    
    response = _LoomForward(input_array, len(inputs))
    if not response:
        raise RuntimeError("Forward pass failed")
    
    result = json.loads(response.decode('utf-8'))
    
    if isinstance(result, dict) and "error" in result:
        raise RuntimeError(f"Forward pass failed: {result['error']}")
    
    return result


def backward_simple(gradients: List[float]) -> None:
    """
    Backward pass with the global network (new simple API).
    
    Args:
        gradients: Gradient vector as list of floats
    """
    if not _LoomBackward:
        raise RuntimeError("LoomBackward not available in library")
    
    # Convert to ctypes array
    grad_array = (ctypes.c_float * len(gradients))(*gradients)
    
    response = _LoomBackward(grad_array, len(gradients))
    if not response:
        raise RuntimeError("Backward pass failed")
    
    result = json.loads(response.decode('utf-8'))
    
    if isinstance(result, dict) and "error" in result:
        raise RuntimeError(f"Backward pass failed: {result['error']}")


def update_weights_simple(learning_rate: float) -> None:
    """
    Update weights with the global network (new simple API).
    
    Args:
        learning_rate: Learning rate for gradient descent
    """
    if not _LoomUpdateWeights:
        raise RuntimeError("LoomUpdateWeights not available in library")
    
    _LoomUpdateWeights(float(learning_rate))


def train_simple(batches: List[dict], config: dict) -> dict:
    """
    Train the global network with batches (new simple API).
    
    Args:
        batches: List of training batches, each with "inputs" and "targets"
        config: Training configuration with "epochs", "learning_rate", etc.
    
    Returns:
        Training result dictionary
    
    Example:
        batches = [
            {"inputs": [[1, 2], [3, 4]], "targets": [[0, 1], [1, 0]]}
        ]
        config = {"epochs": 100, "learning_rate": 0.1}
        result = train_simple(batches, config)
    """
    if not _LoomTrain:
        raise RuntimeError("LoomTrain not available in library")
    
    batches_json = json.dumps(batches).encode('utf-8')
    config_json = json.dumps(config).encode('utf-8')
    
    response = _LoomTrain(batches_json, config_json)
    if not response:
        raise RuntimeError("Training failed")
    
    result = json.loads(response.decode('utf-8'))
    
    if isinstance(result, dict) and "error" in result:
        raise RuntimeError(f"Training failed: {result['error']}")
    
    return result


def save_model_simple(model_id: str = "my_model") -> str:
    """
    Save the global network to JSON string (new simple API).
    
    Args:
        model_id: Model identifier
    
    Returns:
        JSON string with complete model (structure + weights)
    """
    if not _LoomSaveModel:
        raise RuntimeError("LoomSaveModel not available in library")
    
    response = _LoomSaveModel(model_id.encode('utf-8'))
    if not response:
        raise RuntimeError("Failed to save model")
    
    result_str = response.decode('utf-8')
    
    # Check if it's an error
    try:
        result = json.loads(result_str)
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(f"Failed to save model: {result['error']}")
    except json.JSONDecodeError:
        pass
    
    return result_str


def load_model_simple(json_string: str, model_id: str = "my_model") -> None:
    """
    Load a model into the global network (new simple API).
    
    Args:
        json_string: JSON string with complete model
        model_id: Model identifier
    """
    if not _LoomLoadModel:
        raise RuntimeError("LoomLoadModel not available in library")
    
    response = _LoomLoadModel(json_string.encode('utf-8'), model_id.encode('utf-8'))
    if not response:
        raise RuntimeError("Failed to load model")
    
    result = json.loads(response.decode('utf-8'))
    
    if isinstance(result, dict) and "error" in result:
        raise RuntimeError(f"Failed to load model: {result['error']}")


def get_network_info_simple() -> dict:
    """
    Get information about the global network (new simple API).
    
    Returns:
        Dictionary with network information
    """
    if not _LoomGetNetworkInfo:
        raise RuntimeError("LoomGetNetworkInfo not available in library")
    
    response = _LoomGetNetworkInfo()
    if not response:
        raise RuntimeError("Failed to get network info")
    
    result = json.loads(response.decode('utf-8'))
    
    if isinstance(result, dict) and "error" in result:
        raise RuntimeError(f"Failed to get network info: {result['error']}")
    
    return result


def evaluate_network_simple(inputs: List[List[float]], expected_outputs: List[float]) -> dict:
    """
    Evaluate the global network with deviation metrics (new simple API).
    
    Args:
        inputs: 2D array of input samples
        expected_outputs: 1D array of expected class labels
    
    Returns:
        Dictionary with evaluation metrics including deviation buckets
    
    Example:
        inputs = [[1, 2, 3], [4, 5, 6]]
        expected = [0, 1]
        metrics = evaluate_network_simple(inputs, expected)
        print(f"Quality Score: {metrics['score']}/100")
    """
    if not _LoomEvaluateNetwork:
        raise RuntimeError("LoomEvaluateNetwork not available in library")
    
    inputs_json = json.dumps(inputs).encode('utf-8')
    expected_json = json.dumps(expected_outputs).encode('utf-8')
    
    response = _LoomEvaluateNetwork(inputs_json, expected_json)
    if not response:
        raise RuntimeError("Evaluation failed")
    
    result = json.loads(response.decode('utf-8'))
    
    if isinstance(result, dict) and "error" in result:
        raise RuntimeError(f"Evaluation failed: {result['error']}")
    
    return result


# ---- Stepping API (Fine-grained control) ----

# LoomInitStepState: initialize step state
_LoomInitStepState = _sym("LoomInitStepState")
if _LoomInitStepState:
    _LoomInitStepState.restype = ctypes.c_longlong
    _LoomInitStepState.argtypes = [ctypes.c_int]

# LoomSetInput: set input for step
_LoomSetInput = _sym("LoomSetInput")
if _LoomSetInput:
    _LoomSetInput.restype = None
    _LoomSetInput.argtypes = [ctypes.c_longlong, ctypes.POINTER(ctypes.c_float), ctypes.c_int]

# LoomStepForward: step forward
_LoomStepForward = _sym("LoomStepForward")
if _LoomStepForward:
    _LoomStepForward.restype = ctypes.c_longlong
    _LoomStepForward.argtypes = [ctypes.c_longlong]

# LoomGetOutput: get output for step
_LoomGetOutput = _sym("LoomGetOutput")
if _LoomGetOutput:
    _LoomGetOutput.restype = ctypes.c_char_p
    _LoomGetOutput.argtypes = [ctypes.c_longlong]

# LoomStepBackward: step backward
_LoomStepBackward = _sym("LoomStepBackward")
if _LoomStepBackward:
    _LoomStepBackward.restype = ctypes.c_char_p
    _LoomStepBackward.argtypes = [ctypes.c_longlong, ctypes.POINTER(ctypes.c_float), ctypes.c_int]

# LoomApplyGradients: apply gradients
_LoomApplyGradients = _sym("LoomApplyGradients")
_LoomApplyGradientsAdamW = _sym("LoomApplyGradientsAdamW")
_LoomApplyGradientsRMSprop = _sym("LoomApplyGradientsRMSprop")
_LoomApplyGradientsSGDMomentum = _sym("LoomApplyGradientsSGDMomentum")
if _LoomApplyGradients:
    _LoomApplyGradients.restype = None
    _LoomApplyGradients.argtypes = [ctypes.c_float]
if _LoomApplyGradientsAdamW:
    _LoomApplyGradientsAdamW.restype = None
    _LoomApplyGradientsAdamW.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
if _LoomApplyGradientsRMSprop:
    _LoomApplyGradientsRMSprop.restype = None
    _LoomApplyGradientsRMSprop.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
if _LoomApplyGradientsSGDMomentum:
    _LoomApplyGradientsSGDMomentum.restype = None
    _LoomApplyGradientsSGDMomentum.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int]

# LoomFreeStepState: free step state
_LoomFreeStepState = _sym("LoomFreeStepState")
if _LoomFreeStepState:
    _LoomFreeStepState.restype = None
    _LoomFreeStepState.argtypes = [ctypes.c_longlong]


class StepState:
    """
    Manages the state for fine-grained stepping execution of the network.
    Useful for RNNs, LSTMs, and other stateful architectures where you need
    control over the execution loop.
    """
    
    def __init__(self, input_size: int):
        """
        Initialize a new stepping state.
        
        Args:
            input_size: Size of the input vector
        """
        if not _LoomInitStepState:
            raise RuntimeError("LoomInitStepState not available in library")
            
        self.handle = _LoomInitStepState(int(input_size))
        if self.handle < 0:
            raise RuntimeError("Failed to initialize step state")
            
    def set_input(self, input_data: List[float]) -> None:
        """
        Set the input data for the current step.
        
        Args:
            input_data: Input vector as list of floats
        """
        if not _LoomSetInput:
            raise RuntimeError("LoomSetInput not available")
            
        # Convert to ctypes array
        input_array = (ctypes.c_float * len(input_data))(*input_data)
        _LoomSetInput(self.handle, input_array, len(input_data))
        
    def step_forward(self) -> int:
        """
        Execute forward pass for one step.
        
        Returns:
            Duration in nanoseconds
        """
        if not _LoomStepForward:
            raise RuntimeError("LoomStepForward not available")
            
        return _LoomStepForward(self.handle)
        
    def get_output(self) -> List[float]:
        """
        Get the output of the network for the current step.
        
        Returns:
            Output vector as list of floats
        """
        if not _LoomGetOutput:
            raise RuntimeError("LoomGetOutput not available")
            
        response = _LoomGetOutput(self.handle)
        if not response:
            raise RuntimeError("Failed to get output")
            
        result = json.loads(response.decode('utf-8'))
        
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(f"Failed to get output: {result['error']}")
            
        # If result is a list, it's the output
        if isinstance(result, list):
            return result
            
        return []
        
    def step_backward(self, gradients: List[float]) -> dict:
        """
        Execute backward pass for one step.
        
        Args:
            gradients: Gradient vector from the next layer/step
            
        Returns:
            Dict containing 'grad_input' and 'duration'
        """
        if not _LoomStepBackward:
            raise RuntimeError("LoomStepBackward not available")
            
        # Convert to ctypes array
        grad_array = (ctypes.c_float * len(gradients))(*gradients)
        
        response = _LoomStepBackward(self.handle, grad_array, len(gradients))
        if not response:
            raise RuntimeError("Failed to step backward")
            
        result = json.loads(response.decode('utf-8'))
        
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(f"Failed to step backward: {result['error']}")
            
        return result
        
    def __del__(self):
        """Clean up the native step state."""
        if _LoomFreeStepState and hasattr(self, 'handle') and self.handle >= 0:
            _LoomFreeStepState(self.handle)


def apply_gradients(learning_rate: float) -> None:
    """
    Apply accumulated gradients to update network weights.
    
    Args:
        learning_rate: Learning rate
    """
    if not _LoomApplyGradients:
        raise RuntimeError("LoomApplyGradients not available")
        
    _LoomApplyGradients(float(learning_rate))


def apply_gradients_adamw(
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    weight_decay: float = 0.01
) -> None:
    """
    Apply accumulated gradients using AdamW optimizer.
    
    AdamW is Adam with decoupled weight decay - state-of-the-art optimizer
    for many deep learning tasks.
    
    Args:
        learning_rate: Learning rate
        beta1: Exponential decay rate for first moment estimates (default: 0.9)
        beta2: Exponential decay rate for second moment estimates (default: 0.999)
        weight_decay: Weight decay coefficient (default: 0.01)
    """
    if not _LoomApplyGradientsAdamW:
        raise RuntimeError("LoomApplyGradientsAdamW not available")
        
    _LoomApplyGradientsAdamW(
        float(learning_rate),
        float(beta1),
        float(beta2),
        float(weight_decay)
    )


def apply_gradients_rmsprop(
    learning_rate: float,
    alpha: float = 0.99,
    epsilon: float = 1e-8,
    momentum: float = 0.0
) -> None:
    """
    Apply accumulated gradients using RMSprop optimizer.
    
    RMSprop adapts the learning rate for each parameter based on recent gradients.
    
    Args:
        learning_rate: Learning rate
        alpha: Smoothing constant (default: 0.99)
        epsilon: Small constant for numerical stability (default: 1e-8)
        momentum: Momentum factor (default: 0.0, no momentum)
    """
    if not _LoomApplyGradientsRMSprop:
        raise RuntimeError("LoomApplyGradientsRMSprop not available")
        
    _LoomApplyGradientsRMSprop(
        float(learning_rate),
        float(alpha),
        float(epsilon),
        float(momentum)
    )


def apply_gradients_sgd_momentum(
    learning_rate: float,
    momentum: float = 0.9,
    dampening: float = 0.0,
    nesterov: bool = False
) -> None:
    """
    Apply accumulated gradients using SGD with momentum.
    
    Momentum helps accelerate SGD in the relevant direction and dampens oscillations.
    
    Args:
        learning_rate: Learning rate
        momentum: Momentum factor (default: 0.9)
        dampening: Dampening for momentum (default: 0.0)
        nesterov: Whether to use Nesterov momentum (default: False)
    """
    if not _LoomApplyGradientsSGDMomentum:
        raise RuntimeError("LoomApplyGradientsSGDMomentum not available")
        
    _LoomApplyGradientsSGDMomentum(
        float(learning_rate),
        float(momentum),
        float(dampening),
        int(nesterov)
    )


# ---- TweenState Wrapper Class ----

class TweenState:
    """
    Wrapper for LOOM TweenState - enables neural tweening for real-time learning.
    
    Neural tweening allows networks to adapt to new targets without full backpropagation,
    making it suitable for real-time applications.
    
    Example:
        # Create network first
        create_network_from_json(config)
        
        # Create tween state with chain rule
        tween = TweenState(use_chain_rule=True)
        
        # Apply tween steps
        for obs, target in observations:
            gap = tween.step(obs, target_class=target, output_size=4, learning_rate=0.02)
            print(f"Gap: {gap}")
        
        # Cleanup
        tween.close()
        
        # Or use context manager:
        with TweenState(use_chain_rule=True) as tween:
            tween.step(obs, target_class=0, output_size=4)
    """
    
    def __init__(self, use_chain_rule: bool = False):
        """
        Create a new TweenState.
        
        Args:
            use_chain_rule: If True, use chain rule for tween (TweenChain mode)
        """
        if not _LoomCreateTweenState:
            raise RuntimeError("LoomCreateTweenState not available in library")
        
        self._handle = _LoomCreateTweenState(1 if use_chain_rule else 0)
        if self._handle < 0:
            raise RuntimeError("Failed to create TweenState - ensure network is created first")
    
    def step(self, input_data: List[float], target_class: int, output_size: int, 
             learning_rate: float = 0.02) -> float:
        """
        Apply one tween step.
        
        Args:
            input_data: Input vector
            target_class: Target class index (0-based)
            output_size: Number of output classes
            learning_rate: Learning rate for weight updates
        
        Returns:
            Gap value (distance to target)
        """
        if not _LoomTweenStep:
            raise RuntimeError("LoomTweenStep not available")
        
        # Convert to C array
        c_input = (ctypes.c_float * len(input_data))(*input_data)
        
        return _LoomTweenStep(
            self._handle,
            c_input,
            len(input_data),
            target_class,
            output_size,
            learning_rate
        )
    
    def close(self):
        """Free the TweenState resources."""
        if self._handle >= 0 and _LoomFreeTweenState:
            _LoomFreeTweenState(self._handle)
            self._handle = -1
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# ---- AdaptationTracker Wrapper Class ----

class AdaptationTracker:
    """
    Wrapper for LOOM AdaptationTracker - measures accuracy across task changes.
    
    The tracker divides the test duration into windows and records accuracy
    in each window. It also supports scheduling task changes at specific times.
    
    Example:
        tracker = AdaptationTracker(window_duration_ms=1000, total_duration_ms=10000)
        tracker.set_model_info("Dense-5L", "StepTweenChain")
        
        # Schedule task changes at 1/3 and 2/3 of test
        tracker.schedule_task_change(3333, task_id=1, task_name="AVOID")
        tracker.schedule_task_change(6666, task_id=0, task_name="CHASE")
        
        tracker.start("CHASE", task_id=0)
        
        while running:
            current_task = tracker.get_current_task()
            # ... run network ...
            tracker.record_output(is_correct=True)
        
        results = tracker.finalize()
        print(f"Avg accuracy: {results['avg_accuracy']}")
    """
    
    def __init__(self, window_duration_ms: int = 1000, total_duration_ms: int = 10000):
        """
        Create a new AdaptationTracker.
        
        Args:
            window_duration_ms: Duration of each accuracy window in milliseconds
            total_duration_ms: Total test duration in milliseconds
        """
        if not _LoomCreateAdaptationTracker:
            raise RuntimeError("LoomCreateAdaptationTracker not available in library")
        
        self._handle = _LoomCreateAdaptationTracker(window_duration_ms, total_duration_ms)
        if self._handle < 0:
            raise RuntimeError("Failed to create AdaptationTracker")
    
    def set_model_info(self, model_name: str, mode_name: str):
        """
        Set model information for the tracker.
        
        Args:
            model_name: Name of the model (e.g., "Dense-5L")
            mode_name: Training mode name (e.g., "StepTweenChain")
        """
        if _LoomTrackerSetModelInfo:
            _LoomTrackerSetModelInfo(
                self._handle,
                model_name.encode('utf-8'),
                mode_name.encode('utf-8')
            )
    
    def schedule_task_change(self, at_offset_ms: int, task_id: int, task_name: str):
        """
        Schedule a task change at a specific offset from start.
        
        Args:
            at_offset_ms: Milliseconds from start when task should change
            task_id: New task ID
            task_name: New task name
        """
        if _LoomTrackerScheduleTaskChange:
            _LoomTrackerScheduleTaskChange(
                self._handle,
                at_offset_ms,
                task_id,
                task_name.encode('utf-8')
            )
    
    def start(self, task_name: str, task_id: int = 0):
        """
        Start the tracker with an initial task.
        
        Args:
            task_name: Initial task name
            task_id: Initial task ID
        """
        if _LoomTrackerStart:
            _LoomTrackerStart(self._handle, task_name.encode('utf-8'), task_id)
    
    def record_output(self, is_correct: bool) -> int:
        """
        Record an output and whether it was correct.
        
        Args:
            is_correct: Whether the network output was correct
        
        Returns:
            Previous task ID (for detecting task changes)
        """
        if not _LoomTrackerRecordOutput:
            return -1
        return _LoomTrackerRecordOutput(self._handle, 1 if is_correct else 0)
    
    def get_current_task(self) -> int:
        """
        Get the current task ID.
        
        Returns:
            Current task ID
        """
        if not _LoomTrackerGetCurrentTask:
            return 0
        return _LoomTrackerGetCurrentTask(self._handle)
    
    def finalize(self) -> dict:
        """
        Finalize tracking and get results.
        
        Returns:
            Dict with results including:
            - model_name: str
            - mode_name: str
            - avg_accuracy: float
            - total_outputs: int
            - window_accuracies: List[float]
        """
        if not _LoomTrackerFinalize:
            return {"error": "LoomTrackerFinalize not available"}
        
        result_json = _LoomTrackerFinalize(self._handle)
        if not result_json:
            return {"error": "No result from tracker"}
        
        return json.loads(result_json.decode('utf-8'))
    
    def close(self):
        """Free the tracker resources."""
        if self._handle >= 0 and _LoomFreeTracker:
            _LoomFreeTracker(self._handle)
            self._handle = -1
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

