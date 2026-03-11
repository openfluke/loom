namespace Welvet;

/// <summary>
/// Neural network activation function types.
/// </summary>
public enum Activation
{
    /// <summary>
    /// Scaled ReLU: v * 1.1, then ReLU (default)
    /// </summary>
    ScaledReLU = 0,

    /// <summary>
    /// ReLU: max(0, v) - Alias for ScaledReLU
    /// </summary>
    ReLU = 0,

    /// <summary>
    /// Sigmoid: 1 / (1 + exp(-v))
    /// </summary>
    Sigmoid = 1,

    /// <summary>
    /// Tanh: hyperbolic tangent
    /// </summary>
    Tanh = 2,

    /// <summary>
    /// Softplus: log(1 + exp(v))
    /// </summary>
    Softplus = 3,

    /// <summary>
    /// Leaky ReLU: v if v >= 0, else v * 0.1
    /// </summary>
    LeakyReLU = 4,

    /// <summary>
    /// Linear: no activation (identity function) - Alias for Softplus (deprecated)
    /// </summary>
    Linear = 3
}
