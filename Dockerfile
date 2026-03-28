# Prelude inference server — multi-stage build.
#
# FA4 kernel compilation requires GPU access at build time (CuTeDSL tracing).
# Use BuildKit with GPU support:
#
#   docker buildx create --use --driver docker-container \
#     --driver-opt default-load=true \
#     --buildkitd-config /dev/stdin <<< $'[worker.oci]\n  gpus = ["all"]'
#   docker buildx build -t prelude .
#
# Or simpler (requires Docker >= 25.0 + nvidia-container-toolkit):
#   docker build --gpus all -t prelude .
#
# Run:
#   docker run --gpus all -p 8000:8000 -v ~/.cache/huggingface:/root/.cache/huggingface \
#     prelude --model Qwen/Qwen3-4B

# ============================================================================
# Stage 1: Builder
# ============================================================================
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# ── System deps ──────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build pkg-config \
    git curl wget ca-certificates \
    libssl-dev libgomp1 \
    python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --break-system-packages jinja2

# ── Rust ─────────────────────────────────────────────────────────────────
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable

# ── CUDA env ─────────────────────────────────────────────────────────────
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# CUDA driver stub for build-time linking (real driver injected by nvidia-docker at runtime)
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1

# ── Clone and build ──────────────────────────────────────────────────────
ARG REPO=https://github.com/Virtue-Research/prelude.git
ARG BRANCH=refactor

RUN git clone --depth 1 --branch ${BRANCH} ${REPO} /build/prelude
WORKDIR /build/prelude

# Build with full features using dist profile (fat LTO + codegen-units=1).
# FA4 CuTeDSL kernel compilation requires GPU access — see build instructions above.
RUN cargo build -p prelude-server --profile dist --features full

# Remove the build-time driver stub
RUN rm -f /usr/lib/x86_64-linux-gnu/libcuda.so.1

# ============================================================================
# Stage 2: Runtime
# ============================================================================
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libssl3 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

COPY --from=builder /build/prelude/target/dist/prelude-server /usr/local/bin/prelude-server

EXPOSE 8000

ENTRYPOINT ["prelude-server", "--host", "0.0.0.0", "--port", "8000"]
CMD ["--model", "Qwen/Qwen3-0.6B"]
