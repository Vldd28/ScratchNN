#include "Softmax.hpp"
#include <cmath>
#include <algorithm>

Matrix Softmax::activate(const Matrix& input) const {
    Matrix result(input.rows, input.cols);
    
    // Apply softmax row by row (each row is a sample)
    for (int i = 0; i < input.rows; ++i) {
        // Find the maximum value in this row for numerical stability
        double max_val = input(i, 0);
        for (int j = 1; j < input.cols; ++j) {
            if (input(i, j) > max_val) {
                max_val = input(i, j);
            }
        }
        
        // Compute exp(x - max_val) for numerical stability
        double sum_exp = 0.0;
        for (int j = 0; j < input.cols; ++j) {
            double exp_val = std::exp(input(i, j) - max_val);
            result(i, j) = exp_val;
            sum_exp += exp_val;
        }
        
        // Normalize to get probabilities
        for (int j = 0; j < input.cols; ++j) {
            result(i, j) /= sum_exp;
        }
    }
    
    return result;
}

Matrix Softmax::derivative(const Matrix& input) const {
    // For softmax, the derivative is more complex
    // We need to compute the Jacobian matrix for each sample
    // The derivative of softmax is: softmax(x_i) * (1 - softmax(x_i)) for diagonal
    // and -softmax(x_i) * softmax(x_j) for off-diagonal elements
    
    Matrix softmax_output = activate(input);
    Matrix result(input.rows * input.cols, input.cols);
    
    // For each sample (row in input)
    for (int sample = 0; sample < input.rows; ++sample) {
        // For each output class i
        for (int i = 0; i < input.cols; ++i) {
            // For each input class j
            for (int j = 0; j < input.cols; ++j) {
                int result_row = sample * input.cols + i;
                int result_col = j;
                
                if (i == j) {
                    // Diagonal element: s_i * (1 - s_i)
                    result(result_row, result_col) = softmax_output(sample, i) * 
                                                   (1.0 - softmax_output(sample, i));
                } else {
                    // Off-diagonal element: -s_i * s_j
                    result(result_row, result_col) = -softmax_output(sample, i) * 
                                                   softmax_output(sample, j);
                }
            }
        }
    }
    
    return result;
}

std::string Softmax::getName() const {
    return "Softmax";
} 