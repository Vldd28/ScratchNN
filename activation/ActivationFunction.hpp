#pragma once
#include "../math/Matrix.hpp"

/**
 * Abstract base class for activation functions
 */
class ActivationFunction {
public:
    virtual ~ActivationFunction() = default;
    
    /**
     * Apply the activation function to a matrix
     * @param input Input matrix
     * @return Output matrix after applying activation
     */
    virtual Matrix activate(const Matrix& input) const = 0;
    
    /**
     * Compute the derivative of the activation function
     * @param input Input matrix (the original input to the activation function)
     * @return Matrix of derivatives
     */
    virtual Matrix derivative(const Matrix& input) const = 0;
    
    /**
     * Get the name of the activation function
     * @return String name of the activation function
     */
    virtual std::string getName() const = 0;
};
