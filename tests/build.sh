#!/bin/bash

# Build script for ScratchNN
echo "Building ScratchNN..."

# Source files
SOURCES="../math/Matrix.cpp ../activation/ReLU.cpp ../activation/Sigmoid.cpp ../activation/Tanh.cpp ../activation/LeakyReLU.cpp ../layers/FullyConnected.cpp ../layers/Activation.cpp ../loss/MSE.cpp ../loss/CrossEntropy.cpp ../loss/BinaryCrossEntropy.cpp ../nn/NeuralNetwork.cpp"

# Compile
g++ -I.. -std=c++14 -O2 TestNN.cpp $SOURCES -o TestNN

if [ $? -eq 0 ]; then
    echo "✅ Build successful! Run with: ./TestNN"
else
    echo "❌ Build failed!"
    exit 1
fi
