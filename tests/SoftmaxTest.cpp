#include "../activation/Activations.hpp"
#include <iostream>
#include <iomanip>

void testSoftmax() {
    std::cout << "Testing Softmax Activation Function" << std::endl;
    std::cout << "==================================" << std::endl;
    
    Softmax softmax;
    
    // Test 1: Simple 1D case
    std::cout << "\nTest 1: Simple 1D case" << std::endl;
    Matrix input1(1, 3);
    input1(0, 0) = 1.0;
    input1(0, 1) = 2.0;
    input1(0, 2) = 3.0;
    
    Matrix output1 = softmax.activate(input1);
    std::cout << "Input: " << input1.toString();
    std::cout << "Output: " << output1.toString();
    
    // Verify probabilities sum to 1
    double sum = 0.0;
    for (int j = 0; j < output1.cols; ++j) {
        sum += output1(0, j);
    }
    std::cout << "Sum of probabilities: " << sum << std::endl;
    
    // Test 2: Multiple samples
    std::cout << "\nTest 2: Multiple samples" << std::endl;
    Matrix input2(2, 3);
    input2(0, 0) = 1.0; input2(0, 1) = 2.0; input2(0, 2) = 3.0;
    input2(1, 0) = 0.5; input2(1, 1) = 1.5; input2(1, 2) = 2.5;
    
    Matrix output2 = softmax.activate(input2);
    std::cout << "Input: " << input2.toString();
    std::cout << "Output: " << output2.toString();
    
    // Test 3: Large numbers (test numerical stability)
    std::cout << "\nTest 3: Large numbers (numerical stability)" << std::endl;
    Matrix input3(1, 3);
    input3(0, 0) = 1000.0;
    input3(0, 1) = 1001.0;
    input3(0, 2) = 1002.0;
    
    Matrix output3 = softmax.activate(input3);
    std::cout << "Input: " << input3.toString();
    std::cout << "Output: " << output3.toString();
    
    // Test 4: Negative numbers
    std::cout << "\nTest 4: Negative numbers" << std::endl;
    Matrix input4(1, 3);
    input4(0, 0) = -1.0;
    input4(0, 1) = 0.0;
    input4(0, 2) = 1.0;
    
    Matrix output4 = softmax.activate(input4);
    std::cout << "Input: " << input4.toString();
    std::cout << "Output: " << output4.toString();
    
    // Test derivative
    std::cout << "\nTest 5: Derivative" << std::endl;
    Matrix derivative = softmax.derivative(input1);
    std::cout << "Derivative shape: " << derivative.rows << "x" << derivative.cols << std::endl;
    std::cout << "First few elements of derivative: " << std::endl;
    for (int i = 0; i < std::min(3, derivative.rows); ++i) {
        for (int j = 0; j < std::min(3, derivative.cols); ++j) {
            std::cout << std::setw(10) << derivative(i, j) << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\nSoftmax test completed!" << std::endl;
}

int main() {
    testSoftmax();
    return 0;
} 