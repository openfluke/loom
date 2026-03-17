# welvet — LOOM Python Bindings (M-POLY-VTD)
# High-performance polymorphic AI engine with WebGPU acceleration

from .utils import (
    # Type constants
    DType,
    LayerType,
    Activation,
    # High-level OOP API (recommended)
    Network,
    SystolicState,
    TargetPropState,
    Tokenizer,
    # Network lifecycle
    build_network,
    load_network,
    load_network_detailed,
    free_network,
    get_network_info,
    get_layer_telemetry,
    get_available_methods,
    # Inference
    sequential_forward,
    create_systolic_state,
    free_systolic_state,
    set_input,
    systolic_step,
    get_output,
    # GPU inference
    init_wgpu,
    sync_to_gpu,
    sync_to_cpu,
    forward_wgpu,
    forward_token_ids_wgpu,
    # GPU buffer management
    create_gpu_buffer,
    free_gpu_buffer,
    # GPU shader sources
    shader_dense_backward_dx,
    shader_dense_backward_dw,
    # GPU backward dispatch
    dispatch_dense_backward_dx,
    dispatch_dense_backward_dw,
    dispatch_swiglu_backward,
    dispatch_rmsnorm_backward,
    dispatch_embedding_backward,
    dispatch_residual_backward,
    dispatch_cnn1_backward_dx,
    dispatch_cnn1_backward_dw,
    dispatch_cnn2_backward_dx,
    dispatch_cnn2_backward_dw,
    dispatch_cnn3_backward_dx,
    dispatch_cnn3_backward_dw,
    dispatch_mha_backward,
    dispatch_apply_gradients,
    dispatch_mse_grad_partial_loss,
    dispatch_forward_layer,
    dispatch_backward_layer,
    dispatch_activation,
    dispatch_activation_backward,
    # Training
    systolic_backward,
    compute_loss_gradient,
    apply_gradients,
    apply_recursive_gradients,
    apply_target_prop,
    # Target propagation
    create_target_prop_state,
    target_prop_forward,
    target_prop_backward,
    target_prop_backward_chain_rule,
    get_default_target_prop_config,
    # Weight morphing (M-POLY-VTD)
    morph_layer,
    # SafeTensors / model I/O
    load_safetensors,
    load_safetensors_from_bytes,
    load_safetensors_with_shapes,
    load_weights_with_prefixes,
    # DNA / introspection
    extract_dna,
    compare_dna,
    extract_network_blueprint,
    # Tokenizer
    load_tokenizer,
    tokenize,
    detokenize,
    free_tokenizer,
    # Per-layer dispatch
    layer_forward,
    layer_backward,
    # Training helper
    train_network,
    # DNA Splice / Crossover
    default_splice_config,
    splice_dna,
    splice_dna_with_report,
    # NEAT Mutation / Evolution
    default_neat_config,
    neat_mutate,
    new_neat_population,
    neat_population_size,
    neat_population_get_network,
    neat_population_evolve,
    neat_population_best,
    neat_population_best_fitness,
    neat_population_summary,
    free_neat_population,
)

__version__ = "0.7.0"

__all__ = [
    # Version
    "__version__",
    # Type constants
    "DType",
    "LayerType",
    "Activation",
    # High-level OOP
    "Network",
    "SystolicState",
    "TargetPropState",
    "Tokenizer",
    # Network lifecycle
    "build_network",
    "load_network",
    "load_network_detailed",
    "free_network",
    "get_network_info",
    "get_layer_telemetry",
    "get_available_methods",
    # Inference
    "sequential_forward",
    "create_systolic_state",
    "free_systolic_state",
    "set_input",
    "systolic_step",
    "get_output",
    # GPU inference
    "init_wgpu",
    "sync_to_gpu",
    "sync_to_cpu",
    "forward_wgpu",
    "forward_token_ids_wgpu",
    # GPU buffer management
    "create_gpu_buffer",
    "free_gpu_buffer",
    # GPU shader sources
    "shader_dense_backward_dx",
    "shader_dense_backward_dw",
    # GPU backward dispatch
    "dispatch_dense_backward_dx",
    "dispatch_dense_backward_dw",
    "dispatch_swiglu_backward",
    "dispatch_rmsnorm_backward",
    "dispatch_embedding_backward",
    "dispatch_residual_backward",
    "dispatch_cnn1_backward_dx",
    "dispatch_cnn1_backward_dw",
    "dispatch_cnn2_backward_dx",
    "dispatch_cnn2_backward_dw",
    "dispatch_cnn3_backward_dx",
    "dispatch_cnn3_backward_dw",
    "dispatch_mha_backward",
    "dispatch_apply_gradients",
    "dispatch_mse_grad_partial_loss",
    "dispatch_forward_layer",
    "dispatch_backward_layer",
    "dispatch_activation",
    "dispatch_activation_backward",
    # Training
    "systolic_backward",
    "compute_loss_gradient",
    "apply_gradients",
    "apply_recursive_gradients",
    "apply_target_prop",
    # Target propagation
    "create_target_prop_state",
    "target_prop_forward",
    "target_prop_backward",
    "target_prop_backward_chain_rule",
    "get_default_target_prop_config",
    # Weight morphing
    "morph_layer",
    # SafeTensors
    "load_safetensors",
    "load_safetensors_from_bytes",
    "load_safetensors_with_shapes",
    "load_weights_with_prefixes",
    # DNA
    "extract_dna",
    "compare_dna",
    "extract_network_blueprint",
    # Tokenizer
    "load_tokenizer",
    "tokenize",
    "detokenize",
    "free_tokenizer",
    # Per-layer
    "layer_forward",
    "layer_backward",
    # Training helper
    "train_network",
    # DNA Splice / Crossover
    "default_splice_config",
    "splice_dna",
    "splice_dna_with_report",
    # NEAT Mutation / Evolution
    "default_neat_config",
    "neat_mutate",
    "new_neat_population",
    "neat_population_size",
    "neat_population_get_network",
    "neat_population_evolve",
    "neat_population_best",
    "neat_population_best_fitness",
    "neat_population_summary",
    "free_neat_population",
]
