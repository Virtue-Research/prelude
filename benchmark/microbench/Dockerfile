# Dev image — toolchain + llama.cpp, mount local source at runtime.
#
#   docker build -f benchmark/microbench/Dockerfile -t prelude-microbench-dev .
#   docker run --rm -v $(pwd):/workspace prelude-microbench-dev quant

FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build pkg-config \
    git curl ca-certificates libssl-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Rust
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable

# CUDA env
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# llama.cpp — build ggml static library + llama-bench for forward comparison
RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp /opt/llama.cpp \
    && cd /opt/llama.cpp \
    && cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=OFF -DGGML_AVX512=ON -DBUILD_SHARED_LIBS=OFF \
    && cmake --build build -j$(nproc) --target ggml-cpu \
    && cmake --build build -j$(nproc) --target llama-bench \
    && cp build/bin/llama-bench /usr/local/bin/

ENV GGML_LIB=/opt/llama.cpp/build
ENV GGML_INCLUDE=/opt/llama.cpp/ggml/include:/opt/llama.cpp/ggml/src:/opt/llama.cpp/ggml/src/ggml-cpu
ENV LLAMA_BENCH=/usr/local/bin/llama-bench

WORKDIR /workspace

ENTRYPOINT ["/bin/bash", "-c", "cd /workspace && cargo run -p prelude-microbench --release -- $@", "--"]
CMD ["--help"]
