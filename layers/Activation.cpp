#include "Activation.hpp"

Activation::Activation(std::unique_ptr<ActivationFunction> activation_func)
    : activation_function(std::move(activation_func)) {

    if (!activation_function) {
        throw std::invalid_argument("Activation function cannot be null");
    }
}

Matrix Activation::forward(const Matrix& input) {
    // Store input for backward pass
    last_input = input;

    // Apply activation function
    return activation_function->activate(input);
}

Matrix Activation::backward(const Matrix& gradient_output) {
    // Compute activation derivative at the stored input
    Matrix activation_derivative = activation_function->derivative(last_input);

    // Element-wise multiplication: gradient_input = gradient_output * activation_derivative
    Matrix gradient_input(gradient_output.rows, gradient_output.cols);

    for (int i = 0; i < gradient_output.rows * gradient_output.cols; ++i) {
        gradient_input.data[i] = gradient_output.data[i] * activation_derivative.data[i];
    }

    return gradient_input;
}

void Activation::updateParameters(double learning_rate) {
    // Activation layers have no parameters to update
    // This function is intentionally empty
}

void Activation::initialize() {
    // Activation layers have no parameters to initialize
    // This function is intentionally empty
}

std::string Activation::getLayerType() const {
    return "Activation_" + activation_function->getName();
}

std::pair<int, int> Activation::getOutputShape() const {
    return {-1, -1}; // Shape is preserved (dynamic)
}

std::string Activation::getActivationName() const {
    return activation_function->getName();
}
