#pragma once
#include "ActivationFunction.hpp"

/**
 * Hyperbolic Tangent (tanh) activation function
 * f(x) = tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
 * f'(x) = 1 - tanh^2(x)
 */
class Tanh : public ActivationFunction {
public:
    /**
     * Apply Tanh activation function
     * @param input Input matrix
     * @return Output matrix with Tanh applied element-wise
     */
    Matrix activate(const Matrix& input) const override;
    
    /**
     * Compute derivative of Tanh
     * @param input Original input to the activation function
     * @return Matrix of derivatives: 1 - tanh^2(x)
     */
    Matrix derivative(const Matrix& input) const override;
    
    /**
     * Get the name of this activation function
     * @return "Tanh"
     */
    std::string getName() const override;
};
