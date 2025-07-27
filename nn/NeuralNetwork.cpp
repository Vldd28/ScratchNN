#include "NeuralNetwork.hpp"
#include "../layers/Layers.hpp"
#include "../loss/LossFunction.hpp"
#include <stdexcept>
#include <iomanip>
#include <sstream>

NeuralNetwork::NeuralNetwork(double learning_rate)
    : learning_rate(learning_rate), is_training(true), current_epoch(0) {
}

void NeuralNetwork::addLayer(std::unique_ptr<Layer> layer) {
    if (!layer) {
        throw std::invalid_argument("Layer cannot be null");
    }
    layers.push_back(std::move(layer));
}

void NeuralNetwork::setLossFunction(std::unique_ptr<LossFunction> loss_func) {
    if (!loss_func) {
        throw std::invalid_argument("Loss function cannot be null");
    }
    loss_function = std::move(loss_func);
}

void NeuralNetwork::initialize() {
    if (layers.empty()) {
        throw std::runtime_error("Cannot initialize empty network. Add layers first.");
    }

    // Initialize all layers
    for (auto& layer : layers) {
        layer->initialize();
    }

    std::cout << "Network initialized with " << layers.size() << " layers.\n";
    std::cout << getSummary() << std::endl;
}

Matrix NeuralNetwork::forward(const Matrix& input) {
    if (layers.empty()) {
        throw std::runtime_error("Cannot perform forward pass on empty network");
    }

    Matrix output = input;

    // Pass through each layer
    for (auto& layer : layers) {
        output = layer->forward(output);
    }

    return output;
}

double NeuralNetwork::backward(const Matrix& predictions, const Matrix& targets) {
    if (!loss_function) {
        throw std::runtime_error("Loss function not set. Call setLossFunction() first.");
    }

    if (layers.empty()) {
        throw std::runtime_error("Cannot perform backward pass on empty network");
    }

    // Compute loss
    double loss = loss_function->computeLoss(predictions, targets);

    // Compute loss gradient
    Matrix gradient = loss_function->computeGradient(predictions, targets);

    // Backpropagate through layers (reverse order)
    for (int i = layers.size() - 1; i >= 0; --i) {
        gradient = layers[i]->backward(gradient);
    }

    return loss;
}

void NeuralNetwork::updateParameters() {
    // Update parameters in all layers
    for (auto& layer : layers) {
        layer->updateParameters(learning_rate);
    }
}

double NeuralNetwork::trainEpoch(const Matrix& inputs, const Matrix& targets) {
    if (!is_training) {
        throw std::runtime_error("Network is in inference mode. Call setTrainingMode(true) first.");
    }

    // Forward pass
    Matrix predictions = forward(inputs);

    // Backward pass and compute loss
    double loss = backward(predictions, targets);

    // Update parameters
    updateParameters();

    // Store loss history
    loss_history.push_back(loss);

    return loss;
}

void NeuralNetwork::train(const Matrix& inputs, const Matrix& targets, int epochs, bool verbose) {
    if (epochs <= 0) {
        throw std::invalid_argument("Number of epochs must be positive");
    }

    if (inputs.rows != targets.rows) {
        throw std::invalid_argument("Input and target batch sizes must match");
    }

    setTrainingMode(true);

    if (verbose) {
        std::cout << "Starting training for " << epochs << " epochs...\n";
        std::cout << "Learning rate: " << learning_rate << "\n";
        std::cout << "Batch size: " << inputs.rows << "\n\n";
    }

    for (int epoch = 0; epoch < epochs; ++epoch) {
        current_epoch = epoch + 1;

        // Train one epoch
        double epoch_loss = trainEpoch(inputs, targets);

        // Print progress
        if (verbose) {
            if (epoch % 10 == 0 || epoch == epochs - 1) {
                std::cout << "Epoch " << std::setw(4) << (epoch + 1)
                         << "/" << epochs
                         << " - Loss: " << std::fixed << std::setprecision(6)
                         << epoch_loss << std::endl;
            }
        }
    }

    if (verbose) {
        std::cout << "\nTraining completed!\n";
    }
}

Matrix NeuralNetwork::predict(const Matrix& input) {
    setTrainingMode(false);
    return forward(input);
}

double NeuralNetwork::evaluate(const Matrix& inputs, const Matrix& targets) {
    if (!loss_function) {
        throw std::runtime_error("Loss function not set");
    }

    // Set to inference mode
    bool was_training = is_training;
    setTrainingMode(false);

    // Make predictions
    Matrix predictions = forward(inputs);

    // Compute loss
    double loss = loss_function->computeLoss(predictions, targets);

    // Restore training mode
    setTrainingMode(was_training);

    return loss;
}

void NeuralNetwork::setTrainingMode(bool training) {
    is_training = training;

    // Set training mode for all layers
    for (auto& layer : layers) {
        layer->setTrainingMode(training);
    }
}

std::string NeuralNetwork::getSummary() const {
    std::ostringstream oss;
    oss << "Neural Network Architecture:\n";
    oss << "==========================\n";

    for (size_t i = 0; i < layers.size(); ++i) {
        auto shape = layers[i]->getOutputShape();
        oss << "Layer " << (i + 1) << ": " << layers[i]->getLayerType();

        if (shape.first != -1 && shape.second != -1) {
            oss << " - Output: (" << shape.first << ", " << shape.second << ")";
        } else if (shape.second != -1) {
            oss << " - Output: (batch_size, " << shape.second << ")";
        } else {
            oss << " - Output: (preserves input shape)";
        }
        oss << "\n";
    }

    oss << "\nLoss Function: " << (loss_function ? loss_function->getName() : "Not Set") << "\n";
    oss << "Learning Rate: " << learning_rate << "\n";
    oss << "Total Layers: " << layers.size() << "\n";

    return oss.str();
}

const std::vector<double>& NeuralNetwork::getLossHistory() const {
    return loss_history;
}

void NeuralNetwork::setLearningRate(double lr) {
    if (lr <= 0) {
        throw std::invalid_argument("Learning rate must be positive");
    }
    learning_rate = lr;
}

double NeuralNetwork::getLearningRate() const {
    return learning_rate;
}

size_t NeuralNetwork::getLayerCount() const {
    return layers.size();
}

void NeuralNetwork::clearHistory() {
    loss_history.clear();
    current_epoch = 0;
}
