#include "ReLU.hpp"
#include <algorithm>

Matrix ReLU::activate(const Matrix& input) const {
    Matrix result(input.rows, input.cols);
    
    for (int i = 0; i < input.rows * input.cols; ++i) {
        result.data[i] = std::max(0.0, input.data[i]);
    }
    
    return result;
}

Matrix ReLU::derivative(const Matrix& input) const {
    Matrix result(input.rows, input.cols);
    
    for (int i = 0; i < input.rows * input.cols; ++i) {
        result.data[i] = (input.data[i] > 0.0) ? 1.0 : 0.0;
    }
    
    return result;
}

std::string ReLU::getName() const {
    return "ReLU";
}
