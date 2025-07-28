# ScratchNN - Neural Network from Scratch

A C++ implementation of a neural network built from scratch with support for various activation functions, layers, and loss functions.

## Quick Start

### Using Docker (Recommended)

#### For All Platforms (Intel, AMD, Apple M1/M2):
```bash
# Clone the repository
git clone https://github.com/Vldd28/ScratchNN.git
cd ScratchNN

# Build and run with Docker
docker build -t scratchnn .
docker run --rm scratchnn

# Or use Docker Compose
docker compose up --build
```

#### For Apple M1/M2 Macs:
The Docker setup works natively on Apple Silicon. Just run the commands above - Docker will automatically use the ARM64 architecture.

### Manual Compilation

If you prefer to compile directly:

```bash
# Install dependencies (varies by OS)
# On macOS: brew install gcc
# On Ubuntu: sudo apt-get install build-essential

# Compile and run
cd tests
chmod +x build.sh
./build.sh
./TestNN
```

## Features

- **Activation Functions**: ReLU, Sigmoid, Tanh, LeakyReLU
- **Layer Types**: Fully Connected, Activation layers
- **Loss Functions**: MSE, Cross Entropy, Binary Cross Entropy
- **Training**: Backpropagation with configurable learning rates
- **Examples**: Binary classification (XOR problem), regression

## Project Structure

```
ScratchNN/
├── activation/          # Activation function implementations
├── layers/             # Neural network layer implementations  
├── loss/               # Loss function implementations
├── math/               # Matrix operations and utilities
├── nn/                 # Main neural network class
├── tests/              # Example implementations and tests
├── Dockerfile          # Docker configuration
└── compose.yaml        # Docker Compose configuration
```

## Docker Multi-Architecture Support

This project supports both AMD64 (Intel/AMD) and ARM64 (Apple M1/M2) architectures. The Docker images are built to work optimally on your specific hardware.

## Examples

The main example demonstrates solving the XOR problem using a neural network with:
- 2 input features
- 2 hidden layers (8 and 4 neurons)
- ReLU activation functions
- 1 output neuron with Sigmoid activation
- Binary cross-entropy loss

## Contributing

Feel free to submit issues and pull requests to improve the implementation!
