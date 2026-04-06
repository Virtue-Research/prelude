# Question: Plans to separate compiler from runtime?

Is there a plan to split cutile into a build-time compiler (depends on MLIR) and a lightweight runtime (only needs CUDA driver to load pre-compiled PTX/cubin)? This would allow distributing binaries to end users without requiring LLVM/MLIR at runtime — same model as nvcc/cudart or Triton's ahead-of-time workflow.
