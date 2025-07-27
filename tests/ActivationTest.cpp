#include "../activation/Activations.hpp"
#include "../math/Matrix.hpp"
#include <iostream>
#include <vector>

/**
 * Simple test program for activation functions
 */
int main() {
    // Create test input matrix
    std::vector<std::vector<double>> testData = {
        {-2.0, -1.0, 0.0},
        { 1.0,  2.0, 3.0},
        {-0.5,  0.5, 1.5}
    };
    
    Matrix input(testData);
    
    std::cout << "Input Matrix:\n" << input.toString() << std::endl;
    
    // Test ReLU
    ReLU relu;
    Matrix relu_output = relu.activate(input);
    Matrix relu_derivative = relu.derivative(input);
    
    std::cout << "ReLU Output:\n" << relu_output.toString() << std::endl;
    std::cout << "ReLU Derivative:\n" << relu_derivative.toString() << std::endl;
    
    // Test Sigmoid
    Sigmoid sigmoid;
    Matrix sigmoid_output = sigmoid.activate(input);
    Matrix sigmoid_derivative = sigmoid.derivative(input);
    
    std::cout << "Sigmoid Output:\n" << sigmoid_output.toString() << std::endl;
    std::cout << "Sigmoid Derivative:\n" << sigmoid_derivative.toString() << std::endl;
    
    // Test Tanh
    Tanh tanh;
    Matrix tanh_output = tanh.activate(input);
    Matrix tanh_derivative = tanh.derivative(input);
    
    std::cout << "Tanh Output:\n" << tanh_output.toString() << std::endl;
    std::cout << "Tanh Derivative:\n" << tanh_derivative.toString() << std::endl;
    
    // Test LeakyReLU
    LeakyReLU leaky_relu(0.1);  // alpha = 0.1
    Matrix leaky_output = leaky_relu.activate(input);
    Matrix leaky_derivative = leaky_relu.derivative(input);
    
    std::cout << "LeakyReLU Output (alpha=0.1):\n" << leaky_output.toString() << std::endl;
    std::cout << "LeakyReLU Derivative:\n" << leaky_derivative.toString() << std::endl;
    
    return 0;
}
