#pragma once
#include "ActivationFunction.hpp"

/**
 * Leaky ReLU activation function
 * f(x) = x if x > 0, alpha * x otherwise
 * f'(x) = 1 if x > 0, alpha otherwise
 */
class LeakyReLU : public ActivationFunction {
private:
    double alpha;  // Slope for negative values
    
public:
    /**
     * Constructor
     * @param alpha Slope for negative values (default 0.01)
     */
    explicit LeakyReLU(double alpha = 0.01);
    
    /**
     * Apply Leaky ReLU activation function
     * @param input Input matrix
     * @return Output matrix with Leaky ReLU applied element-wise
     */
    Matrix activate(const Matrix& input) const override;
    
    /**
     * Compute derivative of Leaky ReLU
     * @param input Original input to the activation function
     * @return Matrix of derivatives (1 for positive, alpha for negative)
     */
    Matrix derivative(const Matrix& input) const override;
    
    /**
     * Get the name of this activation function
     * @return "LeakyReLU"
     */
    std::string getName() const override;
    
    /**
     * Get the alpha parameter
     * @return Alpha value
     */
    double getAlpha() const;
};
