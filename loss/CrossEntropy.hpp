#pragma once
#include "LossFunction.hpp"

/**
 * Cross-Entropy loss function for classification
 * L = -(1/N) * Σ(y_true * log(y_pred))
 * dL/dy_pred = -(1/N) * (y_true / y_pred)
 */
class CrossEntropy : public LossFunction {
private:
    double epsilon = 1e-15;  // Small value to prevent log(0)
    
public:
    /**
     * Constructor
     * @param eps Small epsilon value to prevent numerical instability
     */
    explicit CrossEntropy(double eps = 1e-15) : epsilon(eps) {}
    
    /**
     * Compute cross-entropy loss
     * @param predictions Model predictions (should be probabilities)
     * @param targets True targets (one-hot encoded)
     * @return Cross-entropy loss
     */
    double computeLoss(const Matrix& predictions, const Matrix& targets) const override;
    
    /**
     * Compute gradient of cross-entropy loss
     * @param predictions Model predictions
     * @param targets True targets
     * @return Gradient matrix
     */
    Matrix computeGradient(const Matrix& predictions, const Matrix& targets) const override;
    
    /**
     * Get loss function name
     * @return "CrossEntropy"
     */
    std::string getName() const override;
    
    /**
     * Get epsilon value
     * @return Epsilon for numerical stability
     */
    double getEpsilon() const { return epsilon; }
};