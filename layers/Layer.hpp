#pragma once
#include "../math/Matrix.hpp"
#include <memory>

/**
 * Abstract base class for all neural network layers
 */
class Layer {
protected:
    bool is_training = true;  // Training mode flag
    Matrix last_input;        // Store input for backward pass
    
public:
    /**
     * Constructor - initializes last_input with default size
     */
    Layer() : last_input(1, 1) {}
    
    virtual ~Layer() = default;
    
    /**
     * Forward pass through the layer
     * @param input Input matrix
     * @return Output matrix after processing
     */
    virtual Matrix forward(const Matrix& input) = 0;
    
    /**
     * Backward pass through the layer (for training)
     * @param gradient_output Gradient from the next layer
     * @return Gradient to pass to the previous layer
     */
    virtual Matrix backward(const Matrix& gradient_output) = 0;
    
    /**
     * Update layer parameters (weights, biases) using gradients
     * @param learning_rate Learning rate for parameter updates
     */
    virtual void updateParameters(double learning_rate) = 0;
    
    /**
     * Get the name/type of this layer
     * @return String identifier for the layer type
     */
    virtual std::string getLayerType() const = 0;
    
    /**
     * Get the output shape of this layer
     * @return Pair of (rows, cols) representing output dimensions
     */
    virtual std::pair<int, int> getOutputShape() const = 0;
    
    /**
     * Initialize layer parameters (if applicable)
     * Called when the layer is added to the network
     */
    virtual void initialize() {}
    
    /**
     * Set the layer to training or inference mode
     * @param training True for training, false for inference
     */
    virtual void setTrainingMode(bool training) {
        is_training = training;
    }
};
