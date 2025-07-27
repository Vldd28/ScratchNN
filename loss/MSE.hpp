#pragma once
#include "LossFunction.hpp"

/**
 * Mean Squared Error loss function
 * L = (1/2N) * Σ(y_pred - y_true)²
 * dL/dy_pred = (1/N) * (y_pred - y_true)
 */
class MSE : public LossFunction {
public:
    /**
     * Compute MSE loss
     * @param predictions Model predictions
     * @param targets True targets
     * @return Mean squared error
     */
    double computeLoss(const Matrix& predictions, const Matrix& targets) const override;
    
    /**
     * Compute gradient of MSE loss
     * @param predictions Model predictions
     * @param targets True targets
     * @return Gradient matrix
     */
    Matrix computeGradient(const Matrix& predictions, const Matrix& targets) const override;
    
    /**
     * Get loss function name
     * @return "MSE"
     */
    std::string getName() const override;
};