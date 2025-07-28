#pragma once
#include "Layer.hpp"
#include <random>

/**
 * Fully Connected (Dense) Layer
 * Performs: output = input * weights + bias
 */
class FullyConnected : public Layer {
private:
    int input_size;
    int output_size;
    
    Matrix weights;        // Weight matrix (input_size x output_size)
    Matrix bias;          // Bias vector (1 x output_size)
    
    // Gradients for backpropagation
    Matrix weight_gradients;
    Matrix bias_gradients;
    
    // Random number generator for initialization
    mutable std::random_device rd;
    mutable std::mt19937 gen;
    
public:
    /**
     * Constructor
     * @param input_size Number of input neurons
     * @param output_size Number of output neurons
     */
    FullyConnected(int input_size, int output_size);
    
    /**
     * Forward pass: output = input * weights + bias
     * @param input Input matrix (batch_size x input_size)
     * @return Output matrix (batch_size x output_size)
     */
    Matrix forward(const Matrix& input) override;
    
    /**
     * Backward pass: compute gradients and return input gradient
     * @param gradient_output Gradient from next layer (batch_size x output_size)
     * @return Input gradient (batch_size x input_size)
     */
    Matrix backward(const Matrix& gradient_output) override;
    
    /**
     * Update weights and biases using computed gradients
     * @param learning_rate Learning rate for parameter updates
     */
    void updateParameters(double learning_rate) override;
    
    /**
     * Initialize weights and biases
     * Uses Xavier/Glorot initialization
     */
    void initialize() override;
    
    /**
     * Get layer type identifier
     * @return "FullyConnected"
     */
    std::string getLayerType() const override;
    
    /**
     * Get output shape of this layer
     * @return (batch_size, output_size) - batch_size is dynamic
     */
    std::pair<int, int> getOutputShape() const override;
    
    /**
     * Get the weight matrix (for inspection/debugging)
     * @return Reference to weights matrix
     */
    const Matrix& getWeights() const;
    
    /**
     * Get the bias vector (for inspection/debugging)
     * @return Reference to bias matrix
     */
    const Matrix& getBias() const;
    
    /**
     * Get input and output sizes
     */
    int getInputSize() const { return input_size; }
    int getOutputSize() const { return output_size; }
};