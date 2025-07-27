#include "LeakyReLU.hpp"

LeakyReLU::LeakyReLU(double alpha) : alpha(alpha) {}

Matrix LeakyReLU::activate(const Matrix& input) const {
    Matrix result(input.rows, input.cols);
    
    for (int i = 0; i < input.rows * input.cols; ++i) {
        double x = input.data[i];
        result.data[i] = (x > 0.0) ? x : alpha * x;
    }
    
    return result;
}

Matrix LeakyReLU::derivative(const Matrix& input) const {
    Matrix result(input.rows, input.cols);
    
    for (int i = 0; i < input.rows * input.cols; ++i) {
        result.data[i] = (input.data[i] > 0.0) ? 1.0 : alpha;
    }
    
    return result;
}

std::string LeakyReLU::getName() const {
    return "LeakyReLU";
}

double LeakyReLU::getAlpha() const {
    return alpha;
}
