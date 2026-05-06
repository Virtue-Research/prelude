/// C bridge for calling methods on TVM Module objects from Rust.
///
/// When `__tvm_ffi_init` returns a Module, the Rust side holds the Module
/// as a TVMFFIAny (type_index + opaque pointer). This helper lets Rust
/// invoke methods on that Module without needing the full TVM object system.
///
/// Also provides a CUDA DLPack tensor allocator for TVM-FFI's internal
/// workspace allocation (used by CUTLASS MoE runner).

#include <tvm/ffi/c_api.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <cuda_runtime.h>
#include <cstring>

// TVM allocator API
#include <tvm/ffi/extra/c_env_api.h>

using namespace tvm::ffi;

struct CudaAllocCtx {
    cudaStream_t stream;
    bool async_alloc;
};

/// Call a method on a TVM Module object.
///
/// @param module_any  Pointer to a TVMFFIAny holding the Module object
///                    (returned from a previous call_tvm_ffi of __tvm_ffi_init).
/// @param method_name Null-terminated method name (e.g., "run_moe").
/// @param args        Array of packed arguments for the method.
/// @param num_args    Number of arguments.
/// @param ret         Output: return value from the method call.
/// @return 0 on success, non-zero on error.
extern "C" int tvm_static_ffi_module_call(
    const TVMFFIAny* module_any,
    const char* method_name,
    const TVMFFIAny* args,
    int num_args,
    TVMFFIAny* ret
) {
    try {
        // Reconstruct Module from the packed Any value.
        AnyView view = AnyView::CopyFromTVMFFIAny(*module_any);
        Module mod = view.cast<Module>();

        // Look up the method on the Module.
        Optional<Function> opt_func = mod->GetFunction(String(method_name));
        if (!opt_func.has_value()) {
            TVMFFIErrorSetRaisedFromCStr(
                "ValueError",
                (std::string("Module has no method: ") + method_name).c_str());
            return -1;
        }
        Function func = opt_func.value();

        // Call the method using TVMFFIFunctionCall.
        TVMFFIObjectHandle func_handle =
            details::ObjectUnsafe::MoveObjectRefToTVMFFIObjectPtr(std::move(func));

        // TVMFFIFunctionCall takes non-const args per its signature
        int rc = TVMFFIFunctionCall(func_handle, const_cast<TVMFFIAny*>(args), num_args, ret);
        TVMFFIObjectDecRef(func_handle);
        return rc;
    } catch (const std::exception& e) {
        TVMFFIErrorSetRaisedFromCStr("InternalError", e.what());
        return -1;
    }
}

/// Release a TVM object held by a TVMFFIAny.
/// Must be called when Rust drops a Module/Function object.
extern "C" void tvm_static_ffi_object_dec_ref(const TVMFFIAny* any) {
    if (any->type_index >= 64) {  // kTVMFFIObject = 64
        auto* ptr = reinterpret_cast<TVMFFIObjectHandle>(any->v_ptr);
        if (ptr) {
            TVMFFIObjectDecRef(ptr);
        }
    }
}

/// Simple CUDA DLPack allocator for TVM-FFI internal workspace allocation.
/// Signature matches DLPackManagedTensorAllocator from dlpack.h.
static int cuda_dlpack_alloc(
    DLTensor* prototype, DLManagedTensorVersioned** out,
    void* error_ctx,
    void (*set_error)(void*, const char*, const char*)
) {
    size_t nbytes = 1;
    for (int i = 0; i < prototype->ndim; ++i) {
        nbytes *= prototype->shape[i];
    }
    nbytes *= (prototype->dtype.bits * prototype->dtype.lanes + 7) / 8;
    if (nbytes == 0) nbytes = 1;

    void* data = nullptr;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(
        TVMFFIEnvGetStream(prototype->device.device_type, prototype->device.device_id));
    bool async_alloc = false;
    cudaError_t err = cudaErrorNotSupported;
    if (stream != nullptr) {
        err = cudaMallocAsync(&data, nbytes, stream);
        async_alloc = (err == cudaSuccess);
    }
    if (!async_alloc) {
        err = cudaMalloc(&data, nbytes);
    }
    if (err != cudaSuccess) {
        if (set_error) set_error(error_ctx, "CUDAError", cudaGetErrorString(err));
        return -1;
    }

    auto* managed = new DLManagedTensorVersioned();
    memset(managed, 0, sizeof(DLManagedTensorVersioned));
    managed->version.major = DLPACK_MAJOR_VERSION;
    managed->version.minor = DLPACK_MINOR_VERSION;
    managed->dl_tensor = *prototype;
    managed->dl_tensor.data = data;
    managed->dl_tensor.byte_offset = 0;
    auto* shape_copy = new int64_t[prototype->ndim];
    memcpy(shape_copy, prototype->shape, prototype->ndim * sizeof(int64_t));
    managed->dl_tensor.shape = shape_copy;
    managed->dl_tensor.strides = nullptr;
    managed->manager_ctx = new CudaAllocCtx{stream, async_alloc};
    managed->deleter = [](DLManagedTensorVersioned* self) {
        auto* ctx = static_cast<CudaAllocCtx*>(self->manager_ctx);
        if (ctx != nullptr && ctx->async_alloc && ctx->stream != nullptr) {
            cudaError_t err = cudaFreeAsync(self->dl_tensor.data, ctx->stream);
            if (err != cudaSuccess) {
                cudaFree(self->dl_tensor.data);
            }
        } else {
            cudaFree(self->dl_tensor.data);
        }
        delete ctx;
        delete[] self->dl_tensor.shape;
        delete self;
    };
    *out = managed;
    return 0;
}

/// Register the CUDA DLPack allocator with TVM-FFI.
/// Must be called once before using CUTLASS MoE runner.
extern "C" int tvm_static_ffi_register_cuda_allocator() {
    return TVMFFIEnvSetDLPackManagedTensorAllocator(
        cuda_dlpack_alloc, /*write_to_global=*/1, /*out_original=*/nullptr);
}
