cmake_minimum_required(VERSION 3.18)
project(onednn_ffi C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── oneDNN minimal build ──────────────────────────────────────────────
# Only matmul + reorder, CPU inference only
set(DNNL_LIBRARY_TYPE STATIC CACHE STRING "" FORCE)
set(DNNL_GPU_RUNTIME NONE CACHE STRING "" FORCE)
set(DNNL_ENABLE_WORKLOAD INFERENCE CACHE STRING "" FORCE)
set(DNNL_ENABLE_PRIMITIVE "MATMUL;REORDER" CACHE STRING "" FORCE)
set(DNNL_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(DNNL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(DNNL_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(ONEDNN_BUILD_GRAPH OFF CACHE BOOL "" FORCE)
set(DNNL_ENABLE_JIT_PROFILING OFF CACHE BOOL "" FORCE)
set(DNNL_ENABLE_ITT_TASKS OFF CACHE BOOL "" FORCE)

# Enable brgemm micro-kernel API (experimental)
set(DNNL_EXPERIMENTAL_UKERNEL ON CACHE BOOL "" FORCE)

# Threading: THREADPOOL — oneDNN calls our RayonThreadPool adapter.
# No OpenMP runtime needed. Eliminates rayon↔OpenMP contention.
set(DNNL_CPU_RUNTIME THREADPOOL CACHE STRING "" FORCE)
message(STATUS "oneDNN FFI: using THREADPOOL runtime (rayon-backed)")

# Use local submodule (third_party/oneDNN) instead of FetchContent download.
set(ONEDNN_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/oneDNN")
if(NOT EXISTS "${ONEDNN_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "third_party/oneDNN not found. Run: git submodule update --init third_party/oneDNN")
endif()
add_subdirectory(${ONEDNN_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/onednn)

# ── Our FFI wrapper (static) ─────────────────────────────────────────
# Both libonednn_ffi.a and libdnnl.a are linked statically into the Rust binary.
# No .so needed at runtime — only system libs (libc, libstdc++).
add_library(onednn_ffi STATIC src/onednn_ffi.cpp src/amx_gemm.c)
target_include_directories(onednn_ffi PUBLIC include)
target_include_directories(onednn_ffi PRIVATE ${ONEDNN_SOURCE_DIR}/include)
target_link_libraries(onednn_ffi PRIVATE dnnl)

# ── AMX GEMM compile flags ──────────────────────────────────────────
# amx_gemm.c needs AMX + AVX-512 BF16 intrinsics. If the compiler doesn't
# support these flags, the file compiles with stub implementations.
include(CheckCCompilerFlag)
check_c_compiler_flag("-mamx-tile" HAS_AMX_TILE)
if(HAS_AMX_TILE)
    set_source_files_properties(src/amx_gemm.c PROPERTIES
        COMPILE_FLAGS "-mamx-tile -mamx-bf16 -mavx512f -mavx512bw -mavx512bf16")
    message(STATUS "AMX GEMM: enabled (compiler supports -mamx-tile)")
else()
    message(STATUS "AMX GEMM: disabled (compiler does not support -mamx-tile, stubs used)")
endif()
