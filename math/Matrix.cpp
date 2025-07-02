#pragma once

#include <vector>
#include <stdexcept>

class Matrix {
public:
    int rows;
    int cols;
    std::vector<std::vector<double>> data;

    // Constructors
    Matrix(int rows, int cols);
    Matrix(const std::vector<std::vector<double>>& inputData);

    // Get a column as a vector
    std::vector<double> getCol(int col) const;

    // Matrix multiplication
    Matrix multiply(const Matrix& other) const;

    // Element-wise addition of vectors (assumes column vectors)
    void addVectors(const Matrix& other);

    // Static dot product of two vectors
    static double dotProduct(const std::vector<double>& v1, const std::vector<double>& v2);
};
