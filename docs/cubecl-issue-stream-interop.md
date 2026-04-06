# Feature Request: Expose raw CUDA/HIP stream for external kernel interop

## Summary

Add a public API to retrieve the raw `CUstream` (CUDA) / `hipStream_t` (HIP) from `ComputeClient`, enabling external CUDA/HIP kernels to launch on the same stream as CubeCL operations without synchronization overhead.

## Motivation

We are building an LLM inference engine in Rust that uses CubeCL as the tensor storage and default ops backend. For hot-path operations (GEMM, attention), we use specialized pre-compiled CUDA kernel libraries (FlashInfer, CUTLASS, DeepGEMM) that significantly outperform general-purpose generated kernels.

The integration pattern is analogous to how PyTorch extensions work:
- CubeCL owns GPU memory allocation (like PyTorch's caching allocator)
- We extract raw device pointers via `client.get_resource(handle)` (like `tensor.data_ptr()`)
- We launch external CUDA kernels with those pointers

The missing piece: we need the raw `CUstream` to launch external kernels on the **same stream** as CubeCL ops. Without this, every transition between a CubeCL op and an external kernel requires `cudaStreamSynchronize`, adding ~5-10us per sync point. In a typical LLM forward pass with 32 transformer layers and ~4 transitions per layer, this totals ~0.6-1.3ms overhead (~4-7% of a 15ms forward pass).

PyTorch solves this with `at::cuda::getCurrentCUDAStream()` which returns the raw `cudaStream_t`. All PyTorch ops and all custom extensions launch on this shared stream, achieving zero synchronization overhead.

## Proposed API

```rust
// On ComputeClient or CudaServer:
impl CudaServer {
    /// Returns the raw CUDA stream for the given logical stream ID.
    /// Enables external CUDA kernels to launch on the same stream as CubeCL operations.
    pub fn cuda_stream(&mut self, stream_id: StreamId) -> cudarc::driver::sys::CUstream {
        self.streams.resolve(stream_id, [].into_iter(), false)
            .unwrap().current().sys
    }
}
```

Usage from application code:

```rust
let cu_stream = client.device.submit_blocking(|server| {
    server.cuda_stream(StreamId::current())
});

// Launch external kernel on the same stream — zero synchronization needed
unsafe {
    cuLaunchKernel(my_kernel, grid, block, args, shared_mem, cu_stream, ...);
}
```

## Current State

The raw `CUstream` is already accessible internally:
- `Stream.sys` is `pub` (`cubecl-cuda/src/compute/stream.rs`)
- `MultiStream::resolve()` is `pub` (`cubecl-runtime/src/stream/event.rs`)
- `ResolvedStreams::current()` is `pub`

The only barrier is `CudaServer.streams` being `pub(crate)`. The proposed change is minimal.

## Alternative Considered

Without this API, the workaround is inserting `cudaStreamSynchronize` at every CubeCL-to-external-kernel boundary. This works correctly but has measurable performance impact for latency-sensitive inference workloads.

## Use Case: LLM Inference Engine Architecture

```
Forward pass (per transformer layer):
  CubeCL GenericOps: rms_norm(x)           ← CubeCL stream
  External kernel:   flash_attention(q,k,v) ← needs same stream
  CubeCL GenericOps: silu(x) * up(x)       ← CubeCL stream  
  External kernel:   cutlass_gemm(x, w)     ← needs same stream
```

With stream sharing: all 4 ops submit to the same `CUstream`, CUDA guarantees sequential execution. Zero sync overhead.

Without stream sharing: need `cudaStreamSynchronize` between every pair. 4 syncs per layer x 32 layers = 128 syncs = ~1ms overhead.

## Context

This is the same pattern used by the entire PyTorch CUDA ecosystem. Every PyTorch C++ extension (FlashAttention, xformers, Triton, vLLM custom ops, etc.) calls `at::cuda::getCurrentCUDAStream()` to get the shared stream. Exposing the raw stream is a prerequisite for CubeCL to interoperate with the existing CUDA kernel ecosystem.
