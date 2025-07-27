#include "BinaryCrossEntropy.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

double BinaryCrossEntropy::computeLoss(const Matrix& predictions, const Matrix& targets) const {
    if (predictions.rows != targets.rows || predictions.cols != targets.cols) {
        throw std::invalid_argument("Predictions and targets must have the same shape");
    }
    
    double total_loss = 0.0;
    int total_elements = predictions.rows * predictions.cols;
    
    for (int i = 0; i < total_elements; ++i) {
        double pred = std::max(epsilon, std::min(1.0 - epsilon, predictions.data[i]));
        double target = targets.data[i];
        
        total_loss -= target * std::log(pred) + (1.0 - target) * std::log(1.0 - pred);
    }
    
    return total_loss / predictions.rows;
}

Matrix BinaryCrossEntropy::computeGradient(const Matrix& predictions, const Matrix& targets) const {
    if (predictions.rows != targets.rows || predictions.cols != targets.cols) {
        throw std::invalid_argument("Predictions and targets must have the same shape");
    }
    
    Matrix gradient(predictions.rows, predictions.cols);
    
    for (int i = 0; i < predictions.rows * predictions.cols; ++i) {
        double pred = std::max(epsilon, std::min(1.0 - epsilon, predictions.data[i]));
        double target = targets.data[i];
        
        // Gradient: -(1/N) * [y/p - (1-y)/(1-p)]
        gradient.data[i] = -(target / pred - (1.0 - target) / (1.0 - pred)) / predictions.rows;
    }
    
    return gradient;
}

std::string BinaryCrossEntropy::getName() const {
    return "BinaryCrossEntropy";
}