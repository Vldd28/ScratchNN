#pragma once
#include "../math/Matrix.hpp"

/**
 * Abstract base class for loss functions
 */
class LossFunction {
public:
    virtual ~LossFunction() = default;
    
    /**
     * Compute the loss between predictions and targets
     * @param predictions Model predictions (batch_size x num_classes)
     * @param targets True labels (batch_size x num_classes)
     * @return Scalar loss value (averaged over batch)
     */
    virtual double computeLoss(const Matrix& predictions, const Matrix& targets) const = 0;
    
    /**
     * Compute the gradient of the loss with respect to predictions
     * @param predictions Model predictions (batch_size x num_classes)
     * @param targets True labels (batch_size x num_classes)
     * @return Gradient matrix (same shape as predictions)
     */
    virtual Matrix computeGradient(const Matrix& predictions, const Matrix& targets) const = 0;
    
    /**
     * Get the name of the loss function
     * @return String name of the loss function
     */
    virtual std::string getName() const = 0;
};