FROM nvidia/cuda:11.4.2-devel-ubuntu20.04

# Install additional dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set the working directory
WORKDIR /app

# Copy your CUDA kernel code to the container
COPY . .

# Compile the CUDA kernel code to PTX assembly
RUN nvcc -ptx src/cosine_similarity.cu -o src/cosine_similarity.ptx


