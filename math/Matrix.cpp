#include "matrix.hpp"
#include <sstream>
#include <iomanip>
using namespace std;

// Constructor
Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols), data(rows * cols, 0.0) {
    if (rows <= 0 || cols <= 0)
        throw invalid_argument("Rows and columns must be positive.");
}

// Constructor from 2D vector
Matrix::Matrix(const vector<vector<double>>& inputData)
    : rows(inputData.size()), cols(inputData.empty() ? 0 : inputData[0].size()) {
    for (const auto& row : inputData) {
        if (row.size() != cols) throw invalid_argument("All rows must have same columns.");
        data.insert(data.end(), row.begin(), row.end());
    }
}

// Access operators
double& Matrix::operator()(int r, int c) { return data[r  * cols + c]; }
const double& Matrix::operator()(int r, int c) const { return data[r * cols + c]; }

// CPU matrix multiplication
Matrix Matrix::multiplyCPU(const Matrix& other) const {
    if (cols != other.rows)
        throw invalid_argument("Matrix dimensions don't match for multiplication.");
    Matrix result(rows, other.cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < other.cols; ++j) {
            double sum = 0;
            for (int k = 0; k < cols; ++k) {
                sum += (*this)(i, k) * other(k, j);
            }
            result(i, j) = sum;
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
