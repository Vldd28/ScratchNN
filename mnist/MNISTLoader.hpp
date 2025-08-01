#pragma once
#include <fstream>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include "../math/Matrix.hpp"

uint32_t readBigEndianUint32(std::ifstream& f) {
    uint8_t bytes[4];
    f.read(reinterpret_cast<char*>(bytes), 4);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

Matrix loadMNISTImages(const std::string& filename, int& out_num_images) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open image file.");

    uint32_t magic = readBigEndianUint32(file);
    if (magic != 2051) throw std::runtime_error("Invalid MNIST image file.");

    uint32_t num_images = readBigEndianUint32(file);
    uint32_t num_rows   = readBigEndianUint32(file);
    uint32_t num_cols   = readBigEndianUint32(file);
    out_num_images = num_images;

    int image_size = num_rows * num_cols;
    Matrix data(num_images, image_size);

    for (int i = 0; i < num_images; ++i) {
        for (int j = 0; j < image_size; ++j) {
            uint8_t pixel;
            file.read(reinterpret_cast<char*>(&pixel), 1);
            data(i, j) = pixel / 255.0;  // normalize to [0,1]
        }
    }

    return data;
}

Matrix loadMNISTLabels(const std::string& filename, int num_classes = 10) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open label file.");

    uint32_t magic = readBigEndianUint32(file);
    if (magic != 2049) throw std::runtime_error("Invalid MNIST label file.");

    uint32_t num_labels = readBigEndianUint32(file);
    Matrix labels(num_labels, num_classes);

    for (int i = 0; i < num_labels; ++i) {
        uint8_t label;
        file.read(reinterpret_cast<char*>(&label), 1);
        for (int j = 0; j < num_classes; ++j) {
            labels(i, j) = (j == label) ? 1.0 : 0.0;  // one-hot
        }
    }

    return labels;
}
