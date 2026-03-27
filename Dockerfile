# Prelude dev container — works with or without GPU.
#
# Base: CUDA 12.8 + Ubuntu 24.04 (CUDA toolkit installed, GPU optional at runtime).
# Without GPU: compiles CPU-only (default features), runs CPU tests/benchmarks.
# With GPU: `--features cuda` for full CUTLASS/DeepGEMM/FlashAttn support.
#
# All tools baked into image → `docker pull` on any machine = identical environment.
# To add a new tool: add a RUN line below, rebuild, push.

FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# ── System dependencies ───────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    git \
    curl \
    wget \
    ca-certificates \
    libssl-dev \
    libgomp1 \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# ── Rust ───────────────────────────────────────────────────────────────────
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable \
    && rustup component add rust-analyzer clippy rustfmt

# ── CUDA environment ──────────────────────────────────────────────────────
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# ── Comparison tools (for benchmarking) ────────────────────────────────────

# llama.cpp — quantized kernel benchmark baseline
RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp /opt/llama.cpp \
    && cd /opt/llama.cpp \
    && cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_AVX512=ON -DGGML_CUDA=OFF \
    && cmake --build build -j$(nproc) --target llama-bench \
    && ln -s /opt/llama.cpp/build/bin/llama-bench /usr/local/bin/llama-bench

# ── Claude Code ────────────────────────────────────────────────────────────
RUN curl -fsSL https://claude.ai/install.sh | bash

# Add more comparison tools here as needed:
# RUN git clone ... /opt/some-tool && cd /opt/some-tool && ...

WORKDIR /workspace
