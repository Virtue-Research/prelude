// Stub for TVMFFIBacktrace — avoids linking libbacktrace.
// We don't need TVM's backtrace in an inference engine.
#include <tvm/ffi/c_api.h>

extern "C" {

const TVMFFIByteArray* TVMFFIBacktrace(const char* /*filename*/, int /*lineno*/,
                                        const char* /*func*/, int /*cross_ffi_boundary*/) {
    // Return a valid empty byte array (not NULL — callers dereference without null check)
    static const TVMFFIByteArray empty = {"", 0};
    return &empty;
}

}  // extern "C"
