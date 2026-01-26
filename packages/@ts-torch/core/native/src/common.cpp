#include "ts_torch/internal.h"

// Thread-local scope stack definition
thread_local std::vector<std::unique_ptr<ts_Scope>> g_scope_stack;

// Version information
const char* ts_version(void) {
    return TS_TORCH_VERSION;
}

// Error handling
void ts_error_clear(ts_Error* error) {
    if (error) {
        error->code = 0;
        error->message[0] = '\0';
    }
}

int ts_error_occurred(const ts_Error* error) {
    return error && error->code != 0;
}
