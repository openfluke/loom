# LOOM Python Bindings
# High-performance neural networks with WebGPU acceleration

from .utils import (
    # Core network functions
    create_network,
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
    # High-level helpers
    configure_sequential_network,
    train_epoch,
)

__version__ = "0.0.1"
__all__ = [
    # Core API
    "create_network",
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
    # High-level helpers
    "configure_sequential_network",
    "train_epoch",
]
