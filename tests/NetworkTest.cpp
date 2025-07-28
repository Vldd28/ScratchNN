#include "../nn/NeuralNetwork.hpp"
#include "../layers/Layers.hpp"
#include "../loss/Loses.hpp"
#include <iostream>
#include <vector>

/**
 * Test program that actually uses the NeuralNetwork class
 */
int main() {
    try {
        std::cout << "=== NeuralNetwork Class Test ===" << std::endl;
        
        // Create a neural network with learning rate 0.01
        NeuralNetwork network(0.01);
        
        // Build the same architecture: 4 -> 3 -> ReLU -> 2 -> Sigmoid
        network.addLayer(std::make_unique<FullyConnected>(4, 3));
        network.addLayer(std::make_unique<Activation>(std::make_unique<ReLU>()));
        network.addLayer(std::make_unique<FullyConnected>(3, 2));
        network.addLayer(std::make_unique<Activation>(std::make_unique<Sigmoid>()));
        
        // Set loss function
        network.setLossFunction(std::make_unique<MSE>());
        
        // Initialize the network
        network.initialize();
        
        std::cout << "Neural network created and initialized!" << std::endl;
        
        // Create test data (same as LayersTest)
        std::vector<std::vector<double>> input_data = {
            {1.0, 0.5, -0.2, 0.8},  // Sample 1
            {-0.5, 1.2, 0.3, -0.1}  // Sample 2
        };
        Matrix input(input_data);
        
        // Create dummy target data
        std::vector<std::vector<double>> target_data = {
            {0.8, 0.2},  // Target for sample 1
            {0.3, 0.7}   // Target for sample 2
        };
        Matrix targets(target_data);
        
        std::cout << "\nInput Matrix (2x4):\n" << input.toString() << std::endl;
        std::cout << "Target Matrix (2x2):\n" << targets.toString() << std::endl;
        
        // Test prediction (forward pass through entire network)
        std::cout << "\n=== Forward Pass Through Network ===" << std::endl;
        Matrix predictions = network.predict(input);
        std::cout << "Network Predictions (2x2):\n" << predictions.toString() << std::endl;
        
        // Test training (this uses forward + backward + parameter updates)
        std::cout << "\n=== Training Test ===" << std::endl;
        std::cout << "Training for 10 epochs..." << std::endl;
        
        for (int epoch = 1; epoch <= 10; ++epoch) {
            network.train(input, targets, 1, false);  // 1 epoch, no verbose
            
            // Show loss every few epochs
            if (epoch % 3 == 0 || epoch == 1) {
                Matrix current_pred = network.predict(input);
                double loss = network.evaluate(input, targets);
                std::cout << "Epoch " << epoch << " - Loss: " << loss << std::endl;
            }
        }
        
        // Final predictions after training
        std::cout << "\n=== After Training ===" << std::endl;
        Matrix final_predictions = network.predict(input);
        std::cout << "Final Predictions:\n" << final_predictions.toString() << std::endl;
        std::cout << "Targets:\n" << targets.toString() << std::endl;
        
        double final_loss = network.evaluate(input, targets);
        std::cout << "Final Loss: " << final_loss << std::endl;
        
        std::cout << "\n=== NeuralNetwork Test Completed Successfully! ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
