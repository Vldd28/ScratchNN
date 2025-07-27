#pragma once

// Include all loss functions
#include "LossFunction.hpp"
#include "MSE.hpp"
#include "CrossEntropy.hpp"
#include "BinaryCrossEntropy.hpp"

/**
 * Convenience header that includes all loss functions
 * 
 * Available loss functions:
 * - MSE: Mean Squared Error for regression
 * - CrossEntropy: For multi-class classification  
 * - BinaryCrossEntropy: For binary classification
 * 
 * Usage:
 *   #include "loss/Losses.hpp"
 *   
 *   MSE mse_loss;
 *   CrossEntropy ce_loss;
 *   BinaryCrossEntropy bce_loss;
 */