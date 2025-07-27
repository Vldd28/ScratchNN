#pragma once
#include "ActivationFunction.hpp"

/**
 * Sigmoid activation function
 * f(x) = 1 / (1 + e^(-x))
 * f'(x) = f(x) * (1 - f(x))
 */
class Sigmoid : public ActivationFunction {
public:
    /**
     * Apply Sigmoid activation function
     * @param input Input matrix
     * @return Output matrix with Sigmoid applied element-wise
     */
    Matrix activate(const Matrix& input) const override;
    
    /**
     * Compute derivative of Sigmoid
     * @param input Original input to the activation function
     * @return Matrix of derivatives: sigmoid(x) * (1 - sigmoid(x))
     */
    Matrix derivative(const Matrix& input) const override;
    
    /**
     * Get the name of this activation function
     * @return "Sigmoid"
     */
    std::string getName() const override;
};
