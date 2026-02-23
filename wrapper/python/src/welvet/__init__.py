# LOOM Python Bindings
# High-performance neural networks with WebGPU acceleration

from .utils import (
    # Core network functions
    create_network,
    load_model_from_string,
    save_model_to_string,
    free_network,
    forward,
    get_output,
    backward,
    update_weights,
    initialize_gpu,
    cleanup_gpu,
    get_version,
    # Layer configuration
    init_dense_layer,
    set_layer,
    get_network_info,
    Activation,
    # Registry-based layer initialization
    call_layer_init,
    list_layer_init_functions,
    # High-level helpers
    configure_sequential_network,
    train_epoch,
    train,
    # Transformer inference
    load_tokenizer_from_bytes,
    load_transformer_from_bytes,
    encode_text,
    decode_tokens,
    generate_text,
    generate_stream,
    # New Simple API (global network instance)
    create_network_from_json,
    forward_simple,
    backward_simple,
    update_weights_simple,
    train_simple,
    save_model_simple,
    load_model_simple,
    get_network_info_simple,
    evaluate_network_simple,
    # Stepping API
    StepState,
    apply_gradients,
    apply_gradients_adamw,
    apply_gradients_rmsprop,
    apply_gradients_sgd_momentum,
    # Tween and Adaptation API
    TweenState,
    AdaptationTracker,
)

__version__ = "0.0.1"
__all__ = [
    # Core API
    "create_network",
    "load_model_from_string",
    "save_model_to_string",
    "free_network",
    "forward",
    "get_output",
    "backward",
    "update_weights",
    "initialize_gpu",
    "cleanup_gpu",
    "get_version",
    # Layer configuration
    "init_dense_layer",
    "set_layer",
    "get_network_info",
    "Activation",
    # Registry-based layer initialization
    "call_layer_init",
    "list_layer_init_functions",
    # High-level helpers
    "configure_sequential_network",
    "train_epoch",
    "train",
    # Transformer inference
    "load_tokenizer_from_bytes",
    "load_transformer_from_bytes",
    "encode_text",
    "decode_tokens",
    "generate_text",
    "generate_stream",
    # New Simple API
    "create_network_from_json",
    "forward_simple",
    "backward_simple",
    "update_weights_simple",
    "train_simple",
    "save_model_simple",
    "load_model_simple",
    "get_network_info_simple",
    "evaluate_network_simple",
    # Stepping API
    "StepState",
    "apply_gradients",
    "apply_gradients_adamw",
    "apply_gradients_rmsprop",
    "apply_gradients_sgd_momentum",
    # Tween and Adaptation API
    "TweenState",
    "AdaptationTracker",
]

