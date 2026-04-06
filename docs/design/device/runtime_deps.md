## Runtime Dependencies

Binary distribution: the compiled binary should run with **only the GPU driver installed**.
No SDK, no toolkit, no pip packages. This constrains which kernel libraries we can use —
everything must be vendored and AOT-compiled into the binary at build time.

**Two binaries cover all platforms.** Device features are additive — multiple GPU backends
compile into a single binary. Runtime auto-detection selects the best available backend.

```
# Linux: full-featured binary with all GPU backends
cargo build --features cuda,rocm,vulkan    →  prelude-linux-x86_64

# macOS: Metal + CPU (CUDA/ROCm unavailable on macOS)
cargo build --features metal               →  prelude-darwin-aarch64
```

Metal and CUDA/ROCm cannot coexist in one binary — not a design limitation but an OS
limitation (Metal framework only on macOS, nvcc/hipcc only on Linux).

NCCL and RCCL are **dlopen'd at runtime** (not statically linked), avoiding symbol
conflicts when both CUDA and ROCm backends are compiled into the same binary.

Model code compiles identically in all targets — only the `OpsBundle` implementation differs.

| Device | Runtime dependency | Build dependency (not needed at runtime) |
|--------|-------------------|----------------------------------------|
| **CUDA** | NVIDIA driver (libcuda.so) | CUDA toolkit (nvcc) for AOT compilation |
| **ROCm** | AMD driver + HIP runtime (libamdhip64.so) | ROCm SDK (hipcc) for AOT compilation |
| **Metal** | macOS (Metal.framework built-in) | Xcode command line tools |
| **Vulkan** | GPU driver (Vulkan ICD) | Vulkan SDK (glslc) for SPIR-V compilation |
| **TPU** | PJRT runtime (libpjrt_tpu.so), dlopen at runtime | JAX/XLA build tools |
| **CPU** | None | C compiler |

**What this means for kernel selection:**

- **No cuBLAS, no cuDNN, no hipBLAS, no hipDNN.** These are CUDA/ROCm toolkit components,
  not part of the driver. We vendor and AOT-compile everything:
  - GEMM: DeepGEMM (vendored) → CUTLASS (vendored, header-only) → no fallback needed
  - Attention: FA4 (vendored, TVM AOT) → FlashInfer (vendored, AOT)
  - Conv: CUTLASS conv (vendored) or custom CUDA/HIP kernels
  - Norm/activation/element-wise: ComposedOps via CubeCL TensorOps (JIT-compiled at first run, cached)
- **ROCm follows the same pattern**: CK (header-only, vendored) for GEMM + attention,
  aiter kernels (vendored, AOT) for flash attention. All compiled with hipcc at build time,
  runtime only needs libamdhip64.so from the driver package.
- **NCCL/RCCL** for communication: dlopen'd at runtime from third_party/ or system-installed version.
- **TPU exception**: TPU has no public low-level API — the only programming interface is
  XLA/PJRT runtime (libpjrt_tpu.so). There is no way to AOT-compile TPU kernels
  independently; all kernel generation happens inside XLA JIT. The binary dlopen's
  libpjrt_tpu.so at runtime — same pattern as CUDA dlopen's libcuda.so from the driver.
  GCP TPU VMs already have PJRT pre-installed. Users do NOT need Python, JAX, or any SDK.
