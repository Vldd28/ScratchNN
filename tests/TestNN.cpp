// #include "../nn/NeuralNetwork.hpp"
// #include "../layers/Layers.hpp"
// #include "../loss/Loses.hpp"
// #include "../activation/Activations.hpp"
// #include <iostream>
// #include <random>
// #include <iomanip>
// #include "../mnist/MNISTLoader.hpp"

// /**
//  * Example: Creating and training a neural network for binary classification
//  */
// void binaryClassificationExample() {
//     std::cout << "=== Binary Classification Example ===\n\n";

//     // Create network
//     NeuralNetwork network(0.01);  // Learning rate = 0.01

//     // Add layers
//     network.addLayer(std::make_unique<FullyConnected>(2, 8));    // Input: 2 features → 8 hidden
//     network.addLayer(std::make_unique<Activation>(std::make_unique<ReLU>()));

//     network.addLayer(std::make_unique<FullyConnected>(8, 4));    // 8 hidden → 4 hidden
//     network.addLayer(std::make_unique<Activation>(std::make_unique<ReLU>()));

//     network.addLayer(std::make_unique<FullyConnected>(4, 1));    // 4 hidden → 1 output
//     network.addLayer(std::make_unique<Activation>(std::make_unique<Sigmoid>()));

//     // Set loss function
//     network.setLossFunction(std::make_unique<BinaryCrossEntropy>());

//     // Initialize network
//     network.initialize();

//     // Create simple XOR-like training data
//     Matrix inputs(4, 2);
//     Matrix targets(4, 1);

//     // XOR problem: [0,0]→0, [0,1]→1, [1,0]→1, [1,1]→0
//     inputs(0, 0) = 0.0; inputs(0, 1) = 0.0; targets(0, 0) = 0.0;
//     inputs(1, 0) = 0.0; inputs(1, 1) = 1.0; targets(1, 0) = 1.0;
//     inputs(2, 0) = 1.0; inputs(2, 1) = 0.0; targets(2, 0) = 1.0;
//     inputs(3, 0) = 1.0; inputs(3, 1) = 1.0; targets(3, 0) = 0.0;

//     // Train the network
//     network.train(inputs, targets, 1000, true);

//     // Test predictions
//     std::cout << "\nTesting predictions:\n";
//     Matrix predictions = network.predict(inputs);

//     for (int i = 0; i < inputs.rows; ++i) {
//         std::cout << "Input: [" << inputs(i, 0) << ", " << inputs(i, 1)
//                   << "] → Predicted: " << std::fixed << std::setprecision(4)
//                   << predictions(i, 0) << ", Target: " << targets(i, 0) << std::endl;
//     }

//     // Evaluate final performance
//     double final_loss = network.evaluate(inputs, targets);
//     std::cout << "\nFinal test loss: " << final_loss << std::endl;
// }

// /**
//  * Example: Multi-class classification (like MNIST digits)
//  */
//  void multiClassExample() {
//      std::cout << "\n=== Multi-Class Classification Example ===\n\n";

//      NeuralNetwork network(0.001);

//      network.addLayer(std::make_unique<FullyConnected>(784, 128));
//      network.addLayer(std::make_unique<Activation>(std::make_unique<ReLU>()));
//      network.addLayer(std::make_unique<FullyConnected>(128, 64));
//      network.addLayer(std::make_unique<Activation>(std::make_unique<ReLU>()));
//      network.addLayer(std::make_unique<FullyConnected>(64, 10));
//      network.addLayer(std::make_unique<Activation>(std::make_unique<Sigmoid>()));
//      network.setLossFunction(std::make_unique<CrossEntropy>());

//      network.initialize();

//      int num_images;
//      Matrix inputs = loadMNISTImages("mnist/train-images.idx3-ubyte", num_images);
//      Matrix targets = loadMNISTLabels("mnist/train-labels.idx1-ubyte");

//      std::cout << "Training on MNIST (" << inputs.rows << " samples)...\n";

//      // Train only on first 1000 samples for speed (adjust as needed)
//      Matrix train_inputs(1000, 784);
//      Matrix train_targets(1000, 10);

//      for (int i = 0; i < 1000; ++i) {
//          for (int j = 0; j < 784; ++j) {
//              train_inputs(i, j) = inputs(i, j);
//          }
//          for (int j = 0; j < 10; ++j) {
//              train_targets(i, j) = targets(i, j);
//          }
//      }

//      network.train(train_inputs, train_targets, 50, true);

//      Matrix predictions = network.predict(train_inputs);

//      std::cout << "\nSample predictions:\n";
//      for (int i = 0; i < 5; ++i) {
//          std::cout << "Sample " << i + 1 << ": [";
//          for (int j = 0; j < 10; ++j) {
//              std::cout << std::fixed << std::setprecision(2) << predictions(i, j);
//              if (j < 9) std::cout << ", ";
//          }
//          std::cout << "]\n";
//      }
//  }

// /**
//  * Example: Regression task
//  */
// void regressionExample() {
//     std::cout << "\n=== Regression Example ===\n\n";

//     // Create network for regression
//     NeuralNetwork network(0.01);

//     // Simple network: 1 input → 8 hidden → 1 output
//     network.addLayer(std::make_unique<FullyConnected>(1, 8));
//     network.addLayer(std::make_unique<Activation>(std::make_unique<Tanh>()));

//     network.addLayer(std::make_unique<FullyConnected>(8, 8));
//     network.addLayer(std::make_unique<Activation>(std::make_unique<Tanh>()));

//     network.addLayer(std::make_unique<FullyConnected>(8, 1));
//     // No activation on output for regression

//     // Set MSE loss for regression
//     network.setLossFunction(std::make_unique<MSE>());

//     network.initialize();

//     // Create training data: y = x^2 function
//     Matrix inputs(20, 1);
//     Matrix targets(20, 1);

//     for (int i = 0; i < 20; ++i) {
//         double x = (i - 10) / 5.0;  // x from -2 to 2
//         inputs(i, 0) = x;
//         targets(i, 0) = x * x;      // y = x^2
//     }

//     std::cout << "Learning y = x^2 function...\n";
//     network.train(inputs, targets, 100, true);

//     // Test predictions
//     std::cout << "\nTesting on training data:\n";
//     Matrix predictions = network.predict(inputs);

//     for (int i = 0; i < inputs.rows; i += 4) {  // Print every 4th sample
//         std::cout << "x = " << std::fixed << std::setprecision(2) << inputs(i, 0)
//                   << " → Predicted: " << std::setprecision(4) << predictions(i, 0)
//                   << ", Target: " << targets(i, 0) << std::endl;
//     }
// }

// int main() {
//     try {
//         binaryClassificationExample();
//         multiClassExample();
//         regressionExample();

//         std::cout << "\n=== All examples completed successfully! ===\n";

//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//         return 1;
//     }

//     return 0;
// }
