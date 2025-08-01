# syntax=docker/dockerfile:1

# Dockerfile for ScratchNN - A C++ Neural Network Implementation
# This builds and runs the neural network test application
# Supports both AMD64 (Intel/AMD) and ARM64 (Apple M1/M2) architectures

################################################################################
# Build stage - Install dependencies and compile the application
FROM --platform=$BUILDPLATFORM gcc:latest as build

# Install any additional dependencies if needed
RUN apt-get update && apt-get install -y \
    make \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy source code
COPY . .

# Build the application
# Using the same compilation flags as in tests/build.sh
RUN g++ -I. -std=c++14 -O2 \
    pipeline/mainNN.cpp \
    math/Matrix.cpp \
    activation/ReLU.cpp \
    activation/Sigmoid.cpp \
    activation/Tanh.cpp \
    activation/LeakyReLU.cpp \
    activation/Softmax.cpp \
    layers/FullyConnected.cpp \
    layers/Activation.cpp \
    loss/MSE.cpp \
    loss/CrossEntropy.cpp \
    loss/BinaryCrossEntropy.cpp \
    nn/NeuralNetwork.cpp \
    -o ScratchNN

################################################################################
# Runtime stage - Use the same base as build to ensure library compatibility
FROM --platform=$BUILDPLATFORM gcc:latest AS final

# Install minimal runtime dependencies (already included in gcc image)
# Remove unnecessary build tools to reduce image size
RUN apt-get update && apt-get remove -y \
    make \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Create a non-privileged user that the app will run under
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Create app directory and set permissions
RUN mkdir /app && chown appuser:appuser /app
WORKDIR /app

# Copy the executable from the build stage
COPY --from=build --chown=appuser:appuser /app/ScratchNN /app/

# Copy MNIST data files
COPY --from=build --chown=appuser:appuser /app/mnist/ /app/mnist/

# Switch to non-privileged user
USER appuser

# What the container should run when it is started
ENTRYPOINT [ "./ScratchNN" ]
