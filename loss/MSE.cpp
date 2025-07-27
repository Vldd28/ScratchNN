#include "MSE.hpp"
#include <stdexcept>

double MSE::computeLoss(const Matrix& predictions, const Matrix& targets) const {
    if (predictions.rows != targets.rows || predictions.cols != targets.cols) {
        throw std::invalid_argument("Predictions and targets must have the same shape");
    }
    
    double total_loss = 0.0;
    int total_elements = predictions.rows * predictions.cols;
    
    for (int i = 0; i < total_elements; ++i) {
        double diff = predictions.data[i] - targets.data[i];
        total_loss += diff * diff;
    }
    
    // Return mean squared error (divide by 2N for easier gradient computation)
    return total_loss / (2.0 * predictions.rows);
}

Matrix MSE::computeGradient(const Matrix& predictions, const Matrix& targets) const {
    if (predictions.rows != targets.rows || predictions.cols != targets.cols) {
        throw std::invalid_argument("Predictions and targets must have the same shape");
    }
    
    Matrix gradient(predictions.rows, predictions.cols);
    
    // Gradient: (1/N) * (predictions - targets)
    for (int i = 0; i < predictions.rows * predictions.cols; ++i) {
        gradient.data[i] = (predictions.data[i] - targets.data[i]) / predictions.rows;
    }
    
    return gradient;
}

std::string MSE::getName() const {
    return "MSE";
}