#include "CrossEntropy.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

double CrossEntropy::computeLoss(const Matrix& predictions, const Matrix& targets) const {
    if (predictions.rows != targets.rows || predictions.cols != targets.cols) {
        throw std::invalid_argument("Predictions and targets must have the same shape");
    }
    
    double total_loss = 0.0;
    
    for (int i = 0; i < predictions.rows; ++i) {
        for (int j = 0; j < predictions.cols; ++j) {
            double pred = std::max(epsilon, std::min(1.0 - epsilon, predictions(i, j)));
            double target = targets(i, j);
            
            if (target > 0) {  // Only compute loss for non-zero targets
                total_loss -= target * std::log(pred);
            }
        }
    }
    
    return total_loss / predictions.rows;  // Average over batch
}

Matrix CrossEntropy::computeGradient(const Matrix& predictions, const Matrix& targets) const {
    if (predictions.rows != targets.rows || predictions.cols != targets.cols) {
        throw std::invalid_argument("Predictions and targets must have the same shape");
    }
    
    Matrix gradient(predictions.rows, predictions.cols);
    
    for (int i = 0; i < predictions.rows; ++i) {
        for (int j = 0; j < predictions.cols; ++j) {
            double pred = std::max(epsilon, std::min(1.0 - epsilon, predictions(i, j)));
            double target = targets(i, j);
            
            // Gradient: -(1/N) * (target / prediction)
            gradient(i, j) = -(target / pred) / predictions.rows;
        }
    }
    
    return gradient;
}

std::string CrossEntropy::getName() const {
    return "CrossEntropy";
}