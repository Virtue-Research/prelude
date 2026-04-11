#include <tvm/ffi/c_api.h>

extern "C" const char* tvm_static_ffi_get_last_error(size_t* out_len) {
    TVMFFIObjectHandle handle = nullptr;
    TVMFFIErrorMoveFromRaised(&handle);
    if (!handle) {
        *out_len = 0;
        return nullptr;
    }
    TVMFFIErrorCell* cell = TVMFFIErrorGetCellPtr(handle);
    *out_len = cell->message.size;
    return cell->message.data;
    // NOTE: handle leaked — acceptable for error path only
}
