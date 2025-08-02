// #include "../nn/NeuralNetwork.hpp"
// #include "../layers/Layers.hpp"
// #include "../loss/Loses.hpp"
// #include "../activation/Activations.hpp"
// #include <iostream>
// #include <memory>
// #include <random>
// #include <iomanip>
// #include "../mnist/MNISTLoader.hpp"





// void multiClassExample() {
//     std::cout << "\n=== Multi-Class Classification Example ===\n\n";

//     NeuralNetwork network(0.001);

//     network.addLayer(std::make_unique<FullyConnected>(784, 128));
//     network.addLayer(std::make_unique<Activation>(std::make_unique<ReLU>()));
//     network.addLayer(std::make_unique<FullyConnected>(128, 64));
//     network.addLayer(std::make_unique<Activation>(std::make_unique<ReLU>()));
//     network.addLayer(std::make_unique<FullyConnected>(64, 10));
//     //network.addLayer(std::make_unique<Activation>(std::make_unique<Sigmoid>()));
//     network.addLayer(std::make_unique<Activation>(std::make_unique<Softmax>()));
//     network.setLossFunction(std::make_unique<CrossEntropy>());

//     network.initialize();

//     int num_images;
//     Matrix inputs = loadMNISTImages("mnist/train-images.idx3-ubyte", num_images);
//     Matrix targets = loadMNISTLabels("mnist/train-labels.idx1-ubyte");

//     std::cout << "Training on MNIST (" << inputs.rows << " samples)...\n";

//     // Train only on first 1000 samples for speed (adjust as needed)
//     Matrix train_inputs(1000, 784);
//     Matrix train_targets(1000, 10);

//     for (int i = 0; i < 1000; ++i) {
//         for (int j = 0; j < 784; ++j) {
//             train_inputs(i, j) = inputs(i, j);
//         }
//         for (int j = 0; j < 10; ++j) {
//             train_targets(i, j) = targets(i, j);
//         }
//     }

//     network.train(train_inputs, train_targets, 50, true);

//     Matrix predictions = network.predict(train_inputs);

//     std::cout << "\nSample predictions:\n";
//     for (int i = 0; i < 5; ++i) {
//         std::cout << "Sample " << i + 1 << ": [";
//         for (int j = 0; j < 10; ++j) {
//             std::cout << std::fixed << std::setprecision(2) << predictions(i, j);
//             if (j < 9) std::cout << ", ";
//         }
//         std::cout << "]\n";
//     }
// }

// int main()
// {
//     multiClassExample();
// }
