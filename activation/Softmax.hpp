#pragma once
#include "ActivationFunction.hpp"

/**
 * Softmax activation function
 * f(x_i) = e^(x_i) / sum(e^(x_j)) for all j
 * 
 * Softmax converts a vector of real numbers into a probability distribution.
 * It's commonly used in the output layer for multi-class classification.
 * 
 * Note: The derivative is more complex than other activation functions
 * and depends on the output of the softmax function itself.
 */
class Softmax : public ActivationFunction {
public:
    /**
     * Apply Softmax activation function
     * @param input Input matrix (each row represents a sample, each column a class)
     * @return Output matrix with Softmax applied (probability distribution)
     */
    Matrix activate(const Matrix& input) const override;
    
    /**
     * Compute derivative of Softmax
     * @param input Original input to the activation function
     * @return Matrix of derivatives (Jacobian matrix for each sample)
     */
    Matrix derivative(const Matrix& input) const override;
    
    /**
     * Get the name of this activation function
     * @return "Softmax"
     */
    std::string getName() const override;
}; 