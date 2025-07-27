#include "Tanh.hpp"
#include <cmath>

Matrix Tanh::activate(const Matrix& input) const {
    Matrix result(input.rows, input.cols);
    
    for (int i = 0; i < input.rows * input.cols; ++i) {
        result.data[i] = std::tanh(input.data[i]);
    }
    
    return result;
}

Matrix Tanh::derivative(const Matrix& input) const {
    Matrix tanh_output = activate(input);
    Matrix result(input.rows, input.cols);
    
    for (int i = 0; i < input.rows * input.cols; ++i) {
        double t = tanh_output.data[i];
        result.data[i] = 1.0 - t * t;
    }
    
    return result;
}

std::string Tanh::getName() const {
    return "Tanh";
}
