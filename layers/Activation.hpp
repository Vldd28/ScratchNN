#pragma once
#include "Layer.hpp"
#include "../activation/ActivationFunction.hpp"
#include <memory>

/**
 * Activation Layer
 * Applies an activation function element-wise to the input
 */
class Activation : public Layer {
private:
    std::unique_ptr<ActivationFunction> activation_function;
    
public:
    /**
     * Constructor
     * @param activation_func Unique pointer to activation function
     */
    explicit Activation(std::unique_ptr<ActivationFunction> activation_func);
    
    /**
     * Forward pass: apply activation function to input
     * @param input Input matrix
     * @return Output matrix after activation
     */
    Matrix forward(const Matrix& input) override;
    
    /**
     * Backward pass: compute gradient using activation derivative
     * @param gradient_output Gradient from next layer
     * @return Input gradient (gradient_output * activation_derivative)
     */
    Matrix backward(const Matrix& gradient_output) override;
    
    /**
     * No parameters to update in activation layer
     * @param learning_rate Unused for activation layers
     */
    void updateParameters(double learning_rate) override;
    
    /**
     * No parameters to initialize
     */
    void initialize() override;
    
    /**
     * Get layer type identifier
     * @return "Activation_<function_name>"
     */
    std::string getLayerType() const override;
    
    /**
     * Output shape is same as input shape
     * @return (-1, -1) indicating shape is preserved
     */
    std::pair<int, int> getOutputShape() const override;
    
    /**
     * Get the activation function name
     * @return Name of the activation function
     */
    std::string getActivationName() const;
};