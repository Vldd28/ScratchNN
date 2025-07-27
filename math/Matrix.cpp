#pragma once
#include "matrix.hpp"
#include <sstream>
#include <iomanip>
#include <stdexcept>
using namespace std;

// Constructor
Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols), data(rows * cols, 0.0) {
    if (rows <= 0 || cols <= 0)
        throw invalid_argument("Rows and columns must be positive.");
}

// Constructor from 2D vector
Matrix::Matrix(const vector<vector<double>> &inputData){
    rows = static_cast<int>(inputData.size());
    if(rows == 0)
    {
        cols = 0;
        return;
    }
    cols = static_cast<int>(inputData[0].size());
    for (int i = 0; i < rows; ++i) {
            if (inputData[i].size() != static_cast<size_t>(cols)) {
                throw std::invalid_argument("All rows must have the same number of columns.");
            }
            for (int j = 0; j < cols; ++j) {
                data.push_back(inputData[i][j]);
            }
        }
}
// Access operators
double& Matrix::operator()(int r, int c) { return data[r  * cols + c]; }
const double& Matrix::operator()(int r, int c) const { return data[r * cols + c]; }

// CPU matrix multiplication
Matrix Matrix::multiplyCPU(const Matrix& other) const{
    if(cols != other.rows){
        throw invalid_argument("Matrix multiplication error: columns of A must match rows of B.");
    }
    Matrix result(rows,other.cols);

    for(int row = 0; row < rows; ++row)
    {
        for(int col = 0; col < other.cols; ++col)
        {
            double dotProduct = 0.0;
            for(int k = 0; k < cols; ++k)
            {
                dotProduct = dotProduct + (*this)(row,k) * other(k, col);
            }
            result(row, col) = dotProduct;
        }
    }
    return result;
}

// Dot product
double Matrix::dotProduct(const vector<double>& v1, const vector<double>& v2) {
    if (v1.size() != v2.size())
        throw invalid_argument("Vectors must have the same size.");
    double sum = 0;
    for (size_t i = 0; i < v1.size(); ++i)
        sum += v1[i] * v2[i];
    return sum;
}

// For debugging
// This is witchcraft, good luck.
string Matrix::toString() const {
    ostringstream oss;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            oss << setw(8) << (*this)(i, j) << " ";
        }
        oss << "\n";
    }
    return oss.str();
}
