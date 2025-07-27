#include "../layers/Layers.hpp"
#include "../math/Matrix.hpp"
#include <iostream>
#include <vector>

/**
 * Test program for neural network layers
 */
int main() {
    try {
        std::cout << "=== Neural Network Layers Test ===" << std::endl;
        
        // Create a simple 2-layer network: 4 -> 3 -> 2
        FullyConnected fc1(4, 3);
        Activation relu1(std::make_unique<ReLU>());
        FullyConnected fc2(3, 2);
        Activation sigmoid1(std::make_unique<Sigmoid>());
        
        // Initialize layers
        fc1.initialize();
        fc2.initialize();
        
        std::cout << "Layers created and initialized successfully!" << std::endl;
        std::cout << "Network architecture: 4 -> FC(3) -> ReLU -> FC(2) -> Sigmoid" << std::endl;
        
        // Create test input (batch size 2, features 4)
        std::vector<std::vector<double>> test_data = {
            {1.0, 0.5, -0.2, 0.8},  // Sample 1
            {-0.5, 1.2, 0.3, -0.1}  // Sample 2
        };
        Matrix input(test_data);
        
        std::cout << "\nInput Matrix (2x4):\n" << input.toString() << std::endl;
        
        // Forward pass
        std::cout << "=== Forward Pass ===" << std::endl;
        
        Matrix h1 = fc1.forward(input);
        std::cout << "After FC1 (2x3):\n" << h1.toString() << std::endl;
        
        Matrix a1 = relu1.forward(h1);
        std::cout << "After ReLU (2x3):\n" << a1.toString() << std::endl;
        
        Matrix h2 = fc2.forward(a1);
        std::cout << "After FC2 (2x2):\n" << h2.toString() << std::endl;
        
        Matrix output = sigmoid1.forward(h2);
        std::cout << "Final Output (2x2):\n" << output.toString() << std::endl;
        
        // Test backward pass with dummy gradients
        std::cout << "=== Backward Pass Test ===" << std::endl;
        
        // Create dummy output gradients
        std::vector<std::vector<double>> grad_data = {
            {0.1, -0.2},  // Gradient for sample 1
            {-0.1, 0.3}   // Gradient for sample 2
        };
        Matrix grad_output(grad_data);
        
        std::cout << "Output Gradient (2x2):\n" << grad_output.toString() << std::endl;
        
        // Backward pass
        Matrix grad4 = sigmoid1.backward(grad_output);
        std::cout << "Gradient after Sigmoid (2x2):\n" << grad4.toString() << std::endl;
        
        Matrix grad3 = fc2.backward(grad4);
        std::cout << "Gradient after FC2 (2x3):\n" << grad3.toString() << std::endl;
        
        Matrix grad2 = relu1.backward(grad3);
        std::cout << "Gradient after ReLU (2x3):\n" << grad2.toString() << std::endl;
        
        Matrix grad1 = fc1.backward(grad2);
        std::cout << "Gradient after FC1 (2x4):\n" << grad1.toString() << std::endl;
        
        // Test parameter updates
        std::cout << "=== Parameter Update Test ===" << std::endl;
        double learning_rate = 0.01;
        
        std::cout << "FC1 weights before update:\n" << fc1.getWeights().toString() << std::endl;
        
        fc1.updateParameters(learning_rate);
        fc2.updateParameters(learning_rate);
        
        std::cout << "FC1 weights after update:\n" << fc1.getWeights().toString() << std::endl;
        
        std::cout << "\n=== Test Completed Successfully! ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
