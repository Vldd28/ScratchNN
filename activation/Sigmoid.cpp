#include "Sigmoid.hpp"
#include <cmath>

Matrix Sigmoid::activate(const Matrix& input) const {
    Matrix result(input.rows, input.cols);
    
    for (int i = 0; i < input.rows * input.cols; ++i) {
        // Sigmoid: 1 / (1 + e^(-x))
        // Use stable computation to avoid overflow
        double x = input.data[i];
        if (x >= 0) {
            double exp_neg_x = std::exp(-x);
            result.data[i] = 1.0 / (1.0 + exp_neg_x);
        } else {
            double exp_x = std::exp(x);
            result.data[i] = exp_x / (1.0 + exp_x);
        }
    }
    
    return result;
}

Matrix Sigmoid::derivative(const Matrix& input) const {
    Matrix sigmoid_output = activate(input);
    Matrix result(input.rows, input.cols);
    
    for (int i = 0; i < input.rows * input.cols; ++i) {
        double s = sigmoid_output.data[i];
        result.data[i] = s * (1.0 - s);
    }
    
    return result;
}

std::string Sigmoid::getName() const {
    return "Sigmoid";
}
