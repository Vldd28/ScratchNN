#pragma once
#include "../layers/Layers.hpp"
#include "../loss/LossFunction.hpp"
#include <vector>
#include <memory>
#include <iostream>

/**
 * Neural Network class that manages layers, training, and inference
 */
class NeuralNetwork {
private:
    std::vector<std::unique_ptr<Layer>> layers;
    std::unique_ptr<LossFunction> loss_function;
    double learning_rate;
    bool is_training;
    
    // Training statistics
    std::vector<double> loss_history;
    int current_epoch;
    
public:
    /**
     * Constructor
     * @param learning_rate Learning rate for training (default: 0.001)
     */
    explicit NeuralNetwork(double learning_rate = 0.001);
    
    /**
     * Destructor
     */
    ~NeuralNetwork() = default;
    
    /**
     * Add a layer to the network
     * @param layer Unique pointer to the layer
     */
    void addLayer(std::unique_ptr<Layer> layer);
    
    /**
     * Set the loss function
     * @param loss_func Unique pointer to the loss function
     */
    void setLossFunction(std::unique_ptr<LossFunction> loss_func);
    
    /**
     * Initialize all layers in the network
     * Call this after adding all layers
     */
    void initialize();
    
    /**
     * Forward pass through the entire network
     * @param input Input matrix (batch_size x input_features)
     * @return Output matrix (batch_size x output_features)
     */
    Matrix forward(const Matrix& input);
    
    /**
     * Backward pass through the entire network
     * @param predictions Network predictions
     * @param targets True targets
     * @return Loss value
     */
    double backward(const Matrix& predictions, const Matrix& targets);
    
    /**
     * Update all parameters in the network
     */
    void updateParameters();
    
    /**
     * Train the network for one epoch
     * @param inputs Training inputs (batch_size x input_features)
     * @param targets Training targets (batch_size x output_features)
     * @return Average loss for the epoch
     */
    double trainEpoch(const Matrix& inputs, const Matrix& targets);
    
    /**
     * Train the network for multiple epochs
     * @param inputs Training inputs
     * @param targets Training targets
     * @param epochs Number of epochs to train
     * @param verbose Print training progress
     */
    void train(const Matrix& inputs, const Matrix& targets, int epochs, bool verbose = true);
    
    /**
     * Make predictions (inference mode)
     * @param input Input matrix
     * @return Predictions
     */
    Matrix predict(const Matrix& input);
    
    /**
     * Evaluate the network on test data
     * @param inputs Test inputs
     * @param targets Test targets
     * @return Test loss
     */
    double evaluate(const Matrix& inputs, const Matrix& targets);
    
    /**
     * Set training mode
     * @param training True for training, false for inference
     */
    void setTrainingMode(bool training);
    
    /**
     * Get network architecture summary
     * @return String description of the network
     */
    std::string getSummary() const;
    
    /**
     * Get loss history
     * @return Vector of loss values from training
     */
    const std::vector<double>& getLossHistory() const;
    
    /**
     * Set learning rate
     * @param lr New learning rate
     */
    void setLearningRate(double lr);
    
    /**
     * Get current learning rate
     * @return Current learning rate
     */
    double getLearningRate() const;
    
    /**
     * Get number of layers
     * @return Number of layers in the network
     */
    size_t getLayerCount() const;
    
    /**
     * Clear loss history
     */
    void clearHistory();
};