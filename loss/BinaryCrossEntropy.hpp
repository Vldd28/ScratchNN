#pragma once
#include "LossFunction.hpp"

/**
 * Binary Cross-Entropy loss function
 * L = -(1/N) * Σ[y*log(p) + (1-y)*log(1-p)]
 * dL/dp = -(1/N) * [y/p - (1-y)/(1-p)]
 */
class BinaryCrossEntropy : public LossFunction {
private:
    double epsilon = 1e-15;
    
public:
    explicit BinaryCrossEntropy(double eps = 1e-15) : epsilon(eps) {}
    
    double computeLoss(const Matrix& predictions, const Matrix& targets) const override;
    Matrix computeGradient(const Matrix& predictions, const Matrix& targets) const override;
    std::string getName() const override;
    double getEpsilon() const { return epsilon; }
};