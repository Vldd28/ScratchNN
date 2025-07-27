#pragma once
#include <vector>
#include <stdexcept>
#include <string>

class Matrix {
public:
    int rows;
    int cols;
    std::vector<double> data;  // Flattened data (row-major)


    // Constructors
    Matrix(int rows, int cols);
    Matrix(const std::vector<std::vector<double>>& inputData);


    // Access elements
    double& operator()(int r, int c);
    const double& operator()(int r, int c) const;

    // Matrix operations (CPU)
    Matrix multiplyCPU(const Matrix& other) const;

    // Matrix operations (GPU)
    Matrix multiplyCUDA(const Matrix& other) const;

    // Utility
    static double dotProduct(const std::vector<double>& v1, const std::vector<double>& v2);

    // Print matrix (for debugging)
    std::string toString() const;
};
