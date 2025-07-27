#pragma once

// Include all activation functions
#include "ActivationFunction.hpp"
#include "ReLU.hpp"
#include "Sigmoid.hpp"
#include "Tanh.hpp"
#include "LeakyReLU.hpp"

/**
 * Activation Functions Library
 * 
 * This header provides access to all implemented activation functions:
 * 
 * 1. ReLU - Rectified Linear Unit
 *    - Good for hidden layers
 *    - Fast computation
 *    - Can suffer from "dying ReLU" problem
 * 
 * 2. Sigmoid - Logistic function
 *    - Good for binary classification output
 *    - Output range [0, 1]
 *    - Can suffer from vanishing gradients
 * 
 * 3. Tanh - Hyperbolic tangent
 *    - Output range [-1, 1]
 *    - Zero-centered output
 *    - Better than sigmoid for hidden layers
 * 
 * 4. LeakyReLU - Leaky Rectified Linear Unit
 *    - Solves "dying ReLU" problem
 *    - Small slope for negative values
 *    - Configurable alpha parameter
 * 
 * Usage example:
 * ```cpp
 * #include "activation/Activations.hpp"
 * 
 * ReLU relu;
 * Sigmoid sigmoid;
 * LeakyReLU leaky_relu(0.01);
 * 
 * Matrix input(3, 3);
 * Matrix activated = relu.activate(input);
 * Matrix derivative = relu.derivative(input);
 * ```
 */
