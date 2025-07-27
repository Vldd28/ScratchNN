#pragma once

// Include all layer types
#include "Layer.hpp"
#include "FullyConnected.hpp"
#include "Activation.hpp"

// Include activation functions for convenience
#include "../activation/Activations.hpp"

/**
 * Neural Network Layers Library
 * 
 * This header provides access to all implemented layer types:
 * 
 * 1. Layer (Abstract Base Class)
 *    - Defines the interface for all layers
 *    - Virtual functions: forward(), backward(), updateParameters()
 * 
 * 2. FullyConnected (Dense Layer)
 *    - Linear transformation: output = input * weights + bias
 *    - Trainable parameters: weights and biases
 *    - Uses Xavier/Glorot initialization
 * 
 * 3. Activation Layer
 *    - Applies activation functions element-wise
 *    - No trainable parameters
 *    - Works with any ActivationFunction
 * 
 * Usage example:
 * ```cpp
 * #include "layers/Layers.hpp"
 * 
 * // Create layers
 * FullyConnected fc1(784, 128);  // Input: 784, Output: 128
 * Activation relu1(std::make_unique<ReLU>());
 * FullyConnected fc2(128, 10);   // Input: 128, Output: 10
 * Activation softmax(std::make_unique<Sigmoid>());
 * 
 * // Initialize layers
 * fc1.initialize();
 * fc2.initialize();
 * 
 * // Forward pass
 * Matrix input(32, 784);  // Batch of 32 samples
 * Matrix h1 = fc1.forward(input);
 * Matrix a1 = relu1.forward(h1);
 * Matrix h2 = fc2.forward(a1);
 * Matrix output = softmax.forward(h2);
 * ```
 */
