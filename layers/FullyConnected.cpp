#include "FullyConnected.hpp"
#include <cmath>
#include <stdexcept>

FullyConnected::FullyConnected(int input_size, int output_size) 
    : Layer(), input_size(input_size), output_size(output_size),
      weights(input_size, output_size), bias(1, output_size),
      weight_gradients(input_size, output_size), bias_gradients(1, output_size),
      gen(rd()) {
    
    if (input_size <= 0 || output_size <= 0) {
        throw std::invalid_argument("Input and output sizes must be positive");
    }
}

Matrix FullyConnected::forward(const Matrix& input) {
    // Store input for backward pass
    last_input = input;
    
    // Check input dimensions
    if (input.cols != input_size) {
        throw std::invalid_argument("Input size doesn't match layer input size");
    }
    
    // Forward pass: output = input * weights + bias
    Matrix output = input.multiplyCPU(weights);
    
    // Add bias to each row (broadcast bias across batch)
    for (int i = 0; i < output.rows; ++i) {
        for (int j = 0; j < output.cols; ++j) {
            output(i, j) += bias(0, j);
        }
    }
    
    return output;
}

Matrix FullyConnected::backward(const Matrix& gradient_output) {
    int batch_size = gradient_output.rows;
    
    // Compute weight gradients: weight_grad = input^T * gradient_output
    Matrix input_transposed(last_input.cols, last_input.rows);
    
    // Transpose last_input
    for (int i = 0; i < last_input.rows; ++i) {
        for (int j = 0; j < last_input.cols; ++j) {
            input_transposed(j, i) = last_input(i, j);
        }
    }
    
    weight_gradients = input_transposed.multiplyCPU(gradient_output);
    
    // Compute bias gradients: sum gradient_output along batch dimension
    for (int j = 0; j < output_size; ++j) {
        double bias_grad_sum = 0.0;
        for (int i = 0; i < batch_size; ++i) {
            bias_grad_sum += gradient_output(i, j);
        }
        bias_gradients(0, j) = bias_grad_sum;
    }
    
    // Compute input gradients: input_grad = gradient_output * weights^T
    Matrix weights_transposed(weights.cols, weights.rows);
    
    // Transpose weights
    for (int i = 0; i < weights.rows; ++i) {
        for (int j = 0; j < weights.cols; ++j) {
            weights_transposed(j, i) = weights(i, j);
        }
    }
    
    Matrix input_gradients = gradient_output.multiplyCPU(weights_transposed);
    
    return input_gradients;
}

void FullyConnected::updateParameters(double learning_rate) {
    // Update weights: weights = weights - learning_rate * weight_gradients
    for (int i = 0; i < weights.rows; ++i) {
        for (int j = 0; j < weights.cols; ++j) {
            weights(i, j) -= learning_rate * weight_gradients(i, j);
        }
    }
    
    // Update bias: bias = bias - learning_rate * bias_gradients
    for (int j = 0; j < bias.cols; ++j) {
        bias(0, j) -= learning_rate * bias_gradients(0, j);
    }
}

void FullyConnected::initialize() {
    // Xavier/Glorot initialization
    double limit = std::sqrt(6.0 / (input_size + output_size));
    std::uniform_real_distribution<double> dist(-limit, limit);
    
    // Initialize weights
    for (int i = 0; i < weights.rows; ++i) {
        for (int j = 0; j < weights.cols; ++j) {
            weights(i, j) = dist(gen);
        }
    }
    
    // Initialize bias to zero
    for (int j = 0; j < bias.cols; ++j) {
        bias(0, j) = 0.0;
    }
}

std::string FullyConnected::getLayerType() const {
    return "FullyConnected";
}

std::pair<int, int> FullyConnected::getOutputShape() const {
    return {-1, output_size}; // -1 indicates dynamic batch size
}

const Matrix& FullyConnected::getWeights() const {
    return weights;
}

const Matrix& FullyConnected::getBias() const {
    return bias;
}
