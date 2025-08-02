#include "../nn/NeuralNetwork.hpp"
#include "../layers/Layers.hpp"
#include "../loss/Loses.hpp"
#include "../activation/Activations.hpp"
#include <iostream>
#include <memory>
#include <random>
#include <iomanip>
#include "../mnist/MNISTLoader.hpp"


int main()
{
    std::vector<float> xs, ys;
    for (int i = 0; i < 100; ++i) {
        float x = i / 100.0f;
        xs.push_back(x);
        ys.push_back(2 * x + 1);
    }

    NeuralNetwork regression_net(0.05);

    regression_net.addLayer(std::make_unique<FullyConnected>(1, 1));
    regression_net.setLossFunction(std::make_unique<MSE>());
    regression_net.initialize();

    int train_size = 80;
    int test_size = 20;

    Matrix train_inputs(train_size, 1);
    Matrix train_targets(train_size, 1);
    Matrix test_inputs(test_size, 1);
    Matrix test_targets(test_size, 1);

    for (int i = 0; i < 100; ++i) {
        float x = i / 100.0f;
        float y = 2 * x + 1;

        if (i < train_size) {
            train_inputs(i, 0) = x;
            train_targets(i, 0) = y;
        } else {
            test_inputs(i - train_size, 0) = x;
            test_targets(i - train_size, 0) = y;
        }
    }

    regression_net.train(train_inputs, train_targets, 1000, true);

    Matrix test_preds = regression_net.predict(test_inputs);


    std::cout << "\nGeneralization Test Predictions:\n";
    for (int i = 0; i < 5; ++i) {
        std::cout << "x = " << test_inputs(i, 0)
                  << " → predicted y = " << test_preds(i, 0)
                  << ", expected y = " << test_targets(i, 0)
                  << std::endl;
    }
}
